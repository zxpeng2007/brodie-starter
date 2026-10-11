"""Brodie the dance teacher: teaches a TikTok dance 8 counts at a time, with the song.

DANCE_MODE=poses (the default) runs starter/dance/posegame.py instead: one still target pose at a time over the caller's own
camera (their skeleton + a ghost of the target), no counting, no music; DANCE_MODE=counts is the lesson described below.

Run the worker in dance mode (pose events, no workout script, the voice may speak any time):
  python -m facetime.worker --voice gemini --dance --leave-grace-s -1 dev
Pick the dance with DANCE=<name> (data/dances/<name>.npz, made with starter/dance/reference.py); default: the first one there.

The call:
  0. Watch first: Brodie says hi and plays the whole creator video with the song (the caller is still holding the phone).
     Then setup: "prop your phone up and step back"; the screen shows the caller's live skeleton until the whole body is in
     view for SETUP_HOLD_S.
  1. "Great, I can see you"; Brodie (Gemini) describes part 1.
  2. Each 8-count part plays as ONE clip: that part of the song with "Five, six, seven, eight" and "One" .. "Eight" mixed
     on top (built here from the song and the bank's count clips, so music and counts never drift). The caller's poses are
     recorded while it plays.
  3. The poses go to starter/dance/judge.py (Steven's model, or the angle comparison against the creator's video).
  4. Pass -> praise and the next part's moves. Miss -> one tip about the worst body part, and the part repeats
     (at most MAX_TRIES times, then it moves on anyway). Every miss makes the teacher angrier (verity.variants.ANGER, and with
     --avatar the face's anger stage, up to the red Cruelty look); a pass calms it down.
  5. Then the whole dance once with the music, judged part by part, and a wrap-up with the scores.

on_event runs on the worker's event loop: never block. The judge runs in a thread; timers use loop.call_later.
"""
import asyncio, collections, glob, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from starter.dance.keypoints import mediapipe_to_coco, resample, FPS  # noqa: E402
from starter.dance import judge as judge_mod  # noqa: E402
from starter.dance import reference  # noqa: E402
from verity.variants import ANGER, CRUELTY_INTRO, FINAL_FORM_INTRO  # noqa: E402

PASS = float(os.environ.get('DANCE_PASS', '0.5'))           # score needed to move on (the judge gives the creator's own clip 96-99%, jumping jacks 15-33%)
MAX_TRIES = int(os.environ.get('DANCE_MAX_TRIES', '3'))
COUNT_ALOUD = os.environ.get('DANCE_COUNT_ALOUD', '1') != '0'   # count "One" .. "Eight" over the music while teaching
FULL_RUN = os.environ.get('DANCE_FULL_RUN', '1') != '0'     # the whole dance with music at the end
MUSIC_DB = float(os.environ.get('DANCE_MUSIC_DB', '-24'))   # music loudness (RMS dBFS) under the counts
PREVIEW = os.environ.get('DANCE_PREVIEW', '1') != '0'      # play the whole dance once before the lesson
SETUP_MIN_S = float(os.environ.get('DANCE_SETUP_S', '6'))  # at least this long to prop the phone up and step back
SETUP_HOLD_S = 2.0                                          # whole body in view this long before the lesson starts
HEAR_LAG_S = 0.15                                           # clip pushed -> heard on the phone, roughly (queue + network)
RATE = reference.AUDIO_RATE
COUNT_IN = ['Five', 'Six', 'Seven', 'Eight']
COUNT_GAIN = float(os.environ.get('DANCE_COUNT_GAIN', '0.6'))  # the counts while dancing, quieter than the count-in
COUNT_WORDS = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']
PRAISE = ['Nice work', 'Great rhythm', 'Looking strong', "That's it", 'Keep it up']


def _load_reference():
    name = os.environ.get('DANCE')
    try:
        if name: return reference.load(name)
        found = sorted(glob.glob(os.path.join(reference.DANCES_DIR, '*.npz')))
        return reference.Reference(found[0]) if found else None
    except Exception as e:
        print(f'[dance] could not load the dance {name or ""}: {e!r}', flush=True); return None


class _Practice:
    """Stand-in when there is no reference yet: 100 bpm, 2 chunks, score = how much the caller moved (tests the call flow)."""
    name = 'practice'; bpm = 100.0; first_beat_s = 0.0; n_chunks = 2; audio = None
    beat_s = 60.0 / bpm; chunk_s = 8 * beat_s
    def chunk(self, k): return np.zeros((0, 17, 3), np.float32)
    def step(self, k): return None


REF = _load_reference() or _Practice()
print(f'[dance] dance {REF.name!r}: {REF.n_chunks} parts of 8 counts at {REF.bpm:g} bpm; pass {PASS:.0%}; '
      f'music {"yes" if REF.audio is not None else "no"}; moves described: {sum(1 for k in range(REF.n_chunks) if REF.step(k))}', flush=True)
# DANCE_MODE=poses (default): one still target pose at a time over the caller's own camera, no counting and no music
# (starter/dance/posegame.py); DANCE_MODE=counts: the 8-count lesson with the music below, unchanged.
DANCE_MODE = os.environ.get('DANCE_MODE', 'poses').strip().lower()
POSES_MODE = DANCE_MODE == 'poses' and not isinstance(REF, _Practice)
if DANCE_MODE == 'poses' and not POSES_MODE: print('[dance] DANCE_MODE=poses needs a reference dance: falling back to counts', flush=True)

S = dict(state='idle', visible=False, vis_full=False, vis_upper=False, upper_run=0, upper_miss=0, chunk=0, tries=0, scores={},
         full_scores=None, say=None, ask=None, last_step_back=0.0, play=None, last_pct=None, screen=False, anger=0,
         live=None, visible_since=None, setup_since=None)
SHOW_VIDEO = os.environ.get('DANCE_VIDEO', '1') != '0'      # Brodie's video track: the creator's dance + counts + cards
POSES = collections.deque(maxlen=FPS * 60)                  # (t_wall, coco (17, 3)) of the last minute
# A dancer counts as "in view" when Brodie's rule says the whole body is in frame OR the upper body is: nose, both shoulders and
# both hips inside the frame and confidently seen (MediaPipe indices). The judge needs shoulders and hips (keypoints.normalize,
# the trained judge's ok mask); the reference clips are often framed waist-up themselves, and a laptop webcam or a hand-held
# phone rarely shows the ankles. Brodie's own "whole body" rule (facetime/visibility.py) still applies to workouts.
UPPER_BODY = [0, 11, 12, 23, 24]
UPPER_VIS = float(os.environ.get('DANCE_UPPER_VIS', '0.5'))
UPPER_ON_FRAMES, UPPER_OFF_FRAMES = 5, 30                   # 0.5 s in view to count, 3 s lost to drop (like Brodie's partial_s)


def _upper_body_in_frame(kp):
    """kp: the pose event's (33, 4) x, y in 0..1 of the frame, z, visibility -> True when the upper body is inside and seen."""
    try:
        a = np.asarray(kp, np.float32)[UPPER_BODY]
    except Exception:
        return False
    inside = bool(((a[:, 0] > 0.02) & (a[:, 0] < 0.98) & (a[:, 1] > 0.02) & (a[:, 1] < 0.98)).all())
    return inside and bool((a[:, 3] >= UPPER_VIS).all())


def _loop():
    return asyncio.get_running_loop()


def _player():
    """The worker's cue player, reached through say_now (VoiceLayer.say_now -> job.player); None in events-only mode."""
    try: return S['say'].__self__.job.player
    except Exception: return None


def _ensure_screen():
    """Start Brodie's video once the worker's room is reachable (through say_now -> VoiceLayer.job.room). With a face track
    (--avatar, the Verity characters) the creator's clip, the counts and the cards are drawn INTO that track, so the call has one
    video: the clip with the character in a corner while a part plays, the character with the card between parts. Without a
    face the Screen publishes its own track as before."""
    if S['screen'] or not SHOW_VIDEO: return
    S['screen'] = True
    try:
        job = S['say'].__self__.job; room = job.room
        from starter.dance.screen import Screen
    except Exception as e:
        print(f'[dance] no video track: {e!r}', flush=True); return
    path = getattr(REF, 'video_path', None) or os.path.join(reference.DANCES_DIR, f'{REF.name}.mp4')
    avatar = getattr(job, 'avatar', None)
    screen = None if (POSES_MODE and avatar is not None) else Screen(REF, path)   # poses mode draws the camera, not the clip
    task = _loop().create_task(_compose_into(avatar, screen) if avatar is not None else screen.run(room, lambda: S))
    task.add_done_callback(lambda t: t.cancelled() or t.exception() is None or print(f'[dance] video stopped: {t.exception()!r}', flush=True))


async def _compose_into(avatar, screen):
    """Replace the avatar's draw(): its own face frame, composed with the Screen (clip + counts while a part plays, the card text
    otherwise), or in poses mode with posegame.compose_pose (the caller's camera, skeletons, the match meter). The clip is
    decoded off the loop first. The mouth sync (verity.avatar.tap) is untouched: it feeds render()."""
    from livekit import rtc
    from starter.dance import screen as screen_mod
    if screen is not None: await _loop().run_in_executor(None, screen.load)
    W, H = screen_mod.W, screen_mod.H
    orig_render = avatar.render; failed = [0]
    def draw(now):
        face = orig_render(now)
        if face.shape[0] != H or face.shape[1] != W: face = cv2.resize(face, (W, H))
        try:
            img = posegame.compose_pose(face, S, time.time()) if screen is None else compose(face, screen, S, time.time())
        except Exception as e:                                                 # never let a drawing bug kill the video
            failed[0] += 1
            if failed[0] <= 3: print(f'[dance] compose failed: {e!r}', flush=True)
            img = face[..., :3]
        rgba = np.dstack([img, np.full((H, W), 255, np.uint8)])
        return rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, np.ascontiguousarray(rgba).tobytes())
    avatar.draw = draw
    print('[dance] ' + ('pose game drawn into the face track (camera + skeletons + ghost)' if screen is None else
                        'screen drawn into the face track (clip + counts while a part plays, the card between parts)'), flush=True)


def _blend(dst, src_rgba, x, y):
    """Alpha-blend an RGBA image onto an RGB one at (x, y), clipped."""
    h, w = src_rgba.shape[:2]; H, W = dst.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = src_rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32); a = s[..., 3:4] / 255.0
    dst[y0:y1, x0:x1] = (s[..., :3] * a + dst[y0:y1, x0:x1].astype(np.float32) * (1 - a)).astype(np.uint8)


def compose(face, screen, st, now):
    """face: the character's RGBA frame (W x H, the ball mid-frame); -> RGB W x H for the call."""
    from starter.dance.screen import W, H, VOLT, WHITE, _text
    import textwrap
    if screen.playing(st):                                            # a part is playing: the clip and the count, the face small
        out = screen.render(st, now)
        small = cv2.resize(face, (W // 3, H // 3), interpolation=cv2.INTER_AREA)
        _blend(out, small, W - small.shape[1] - 6, H - small.shape[0] - 6)
        return out
    out = np.zeros((H, W, 3), np.uint8)                               # between parts: the face on top, the card below
    top = face[90:500]                                                # the ball and its gear (verity.avatar: centre 320, radius 151)
    th = 290; tw = int(top.shape[1] * th / top.shape[0])
    _blend(out, cv2.resize(top, (tw, th), interpolation=cv2.INTER_AREA), (W - tw) // 2, 6)
    ref = REF; k = st.get('chunk', 0); n = ref.n_chunks; state = st.get('state'); y = 340
    if state == 'idle' or not st.get('visible'):
        for i, line in enumerate(['Step back a little so', 'I can see your head', 'and your hips']):
            _text(out, line, (0, y + 40 * i), 0.9, WHITE, 2, center=True)
        return out
    if state == 'done':
        fs = st.get('full_scores') or st.get('scores') or {}
        good = bool(fs) and sum(fs.values()) / len(fs) >= PASS * 100
        _text(out, 'YOU DID IT!' if good else 'KEEP PRACTISING', (0, y), 1.4 if good else 1.1, VOLT, 3, center=True)
        for i, (kk, s) in enumerate(sorted(fs.items())[:6]):
            _text(out, f'Part {kk + 1}: {s}%', (0, y + 55 + 40 * i), 0.9, WHITE, 2, center=True)
        return out
    _text(out, 'FULL DANCE NEXT' if state == 'full' else f'PART {k + 1}/{n}', (0, y), 1.2, VOLT, 3, center=True)
    last = st.get('last_pct'); y += 50
    if last is not None: _text(out, f'Last try: {last}%', (0, y), 0.9, WHITE, 2, center=True); y += 44
    steps = ref.step(k) if state != 'full' else None
    if steps:
        for i, line in enumerate(textwrap.wrap(steps, 30)[:7]):
            _text(out, line, (0, y + 30 * i), 0.62, WHITE, 2, center=True)
    return out


NO_COUNT = (" Never count in or count beats aloud (no 'five, six, seven, eight'): " +
            ("the lesson has no counting at all." if DANCE_MODE == 'poses' else "the recorded counts and music do that, you only talk between parts."))
# (Steven, 12:49: Gemini counted over the recorded count-in, two countdowns; in the pose game it counted with nothing to count.)


def _ask(prompt):
    """Ask the character (through Brodie's ask_voice); returns the prompt so when_quiet(..., retry=prompt) can ask it again if lost.
    Every prompt ends with the no-counting rule."""
    if S['ask']: S['ask'](prompt + NO_COUNT)
    return prompt


# The comedy: how each situation is played. The persona has the character; these say what kind of bit the moment is. One bit per
# line, the joke lands at the end, specific beats generic. All PG (the persona's own rules still apply).
def _flavour(pct, passed):
    """A one-line stage direction for the reaction to a score."""
    if passed and pct >= 90: return 'They nailed it: react like a stunned proud coach who did not expect this, one over-the-top compliment.'
    if passed: return 'They passed: one hype line like a dance teacher who secretly had money on them.'
    if pct < 15: return 'They barely moved: tease that the video must have frozen, or that this was a statue impression, then the tip.'
    if pct < 35: return 'They did a confident different dance: say what it looked like instead (one funny guess), then the tip.'
    if pct >= PASS * 100 - 10: return 'So close: play the dramatic almost ("I felt that in my headphones"), then the tip.'
    return 'Mid: a backhanded compliment first ("the energy was there, the arms were somewhere else"), then the tip.'


def _anger(n):
    """Anger stage n (0 calm .. len(ANGER)-1): the face (--avatar) now; the returned mood goes into the next ask."""
    was, S['anger'] = S['anger'], max(0, min(n, len(ANGER) - 1)); n = S['anger']
    try: avatar = S['say'].__self__.job.avatar
    except Exception: avatar = None
    if avatar is not None: avatar.set_stage(n)
    print(f'[dance] anger {n}', flush=True)
    intro = FINAL_FORM_INTRO.get(os.environ.get('VERITY_CHARACTER', '').strip().lower(), CRUELTY_INTRO)   # Brodie: coach mode, not Cruelty
    return (ANGER[n] + (' ' + intro if n == len(ANGER) - 1 and was < n else '')) if n else ''


# ---- audio: one clip per part = the song + the counts ----
def _db_gain(pcm, target_db):
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if len(pcm) else 0.0
    return (10 ** (target_db / 20) * 32768 / rms) if rms > 1 else 1.0


def build_clip(k0, k1, count_dance):
    """Parts k0 .. k1-1 as one int16 clip: 4 count-in beats + 8 beats per part. Returns (pcm, dance_offset_s, t_start)."""
    beat = REF.beat_s
    t_start = REF.first_beat_s + 8 * k0 * beat - 4 * beat           # the count-in starts 4 beats before ONE
    t_end = REF.first_beat_s + 8 * k1 * beat + 0.3
    words = [(i * beat, w) for i, w in enumerate(COUNT_IN)]
    if count_dance:
        words += [(4 * beat + j * beat, COUNT_WORDS[j % 8], COUNT_GAIN) for j in range(8 * (k1 - k0))]
    return build_span(t_start, t_end, words), 4 * beat, t_start


def _fit_word(c, max_s):
    """A spoken count squeezed into its beat: leading/trailing silence dropped, then up to 1.35x faster, then a soft cut."""
    loud = np.flatnonzero(np.abs(c) > 0.02 * 32768)
    if len(loud): c = c[max(0, loud[0] - int(0.01 * RATE)):loud[-1] + int(0.03 * RATE)]
    n_max = int(max_s * RATE)
    if len(c) > n_max:
        if len(c) <= 1.35 * n_max:                                  # a little long: speed it up (slightly higher pitch)
            c = np.interp(np.linspace(0, len(c) - 1, n_max), np.arange(len(c)), c)
        else:                                                       # far too long: keep the start, fade out
            c = c[:n_max].copy()
    f = min(len(c), int(0.03 * RATE)); c[-f:] *= np.linspace(1, 0, f)
    return c


def build_span(t_start, t_end, words):
    """The song from t_start to t_end (video seconds) with bank words [(offset_s, text)] mixed on top: int16 pcm."""
    player = _player()
    n = int((t_end - t_start) * RATE)
    out = np.zeros(n, np.float32)
    if REF.audio is not None:                                       # the song for exactly these beats (zeros before the video starts)
        a = int(round(t_start * RATE)); src = REF.audio.astype(np.float32)
        lo, hi = max(a, 0), min(a + n, len(src))
        if hi > lo:
            seg = src[lo:hi] * _db_gain(src, MUSIC_DB)
            f = min(len(seg), int(0.03 * RATE)); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
            out[lo - a:lo - a + len(seg)] += seg
    words = sorted(words)
    for j, item in enumerate(words):
        t, w = item[0], item[1]; gain = item[2] if len(item) > 2 else 1.0
        c = player.bank.get(w) if player is not None else None
        if c is None: continue
        nxt = words[j + 1][0] if j + 1 < len(words) else t + 2.0
        c = _fit_word(c.astype(np.float32), max(0.15, nxt - t - 0.05)) * gain   # never runs into the next count
        i = int(t * RATE); m = min(len(c), n - i)
        if m > 0: out[i:i + m] += c[:m]
    pcm = np.clip(out, -32768, 32767).astype(np.int16)
    return np.concatenate([pcm, np.zeros((-len(pcm)) % 480, np.int16)])  # whole 20 ms frames


def play_parts(k0, k1, count_dance, then):
    """Play parts k0 .. k1-1 (one clip), then call then(windows): windows[k] = (t0, t1) wall times of part k as heard."""
    player = _player()
    if player is None:
        print('[dance] no cue player (events-only mode): cannot play the clip', flush=True); return
    pcm, dance_off, t_start = build_clip(k0, k1, count_dance)
    _play(pcm, f'[dance {k0}-{k1} {"counts" if count_dance else "quiet"}]',
          dict(kind='full' if k1 - k0 > 1 else 'part', k0=k0, k1=k1, dance_off=dance_off, t_start=t_start), then)


def play_preview(then):
    """The whole creator video with the song, no counts: the caller watches before learning."""
    t_end = (len(REF.audio) / RATE) if REF.audio is not None else REF.first_beat_s + 8 * REF.n_chunks * REF.beat_s
    _play(build_span(0.0, t_end, []), '[dance preview]', dict(kind='preview', k0=0, k1=0, dance_off=0.0, t_start=0.0), lambda wins: then())


def _play(pcm, key, info, then):
    player = _player()
    if player is None:
        print('[dance] no cue player (events-only mode): cannot play the clip', flush=True); return
    player.bank.cache[key] = pcm                                    # the bank serves cached clips by key: say() plays it
    rec = S['say'](key, 'announce')
    S['play'] = dict(info, rec=rec, then=then, dur=len(pcm) / RATE, t_said=time.time())
    print(f'[dance] playing {key} ({len(pcm) / RATE:.1f} s, music {"on" if REF.audio is not None else "off"}): '
          f'{rec.get("status") if isinstance(rec, dict) else rec}', flush=True)
    _loop().call_later(0.2, _watch_play)


def _watch_play():
    p = S['play']
    if p is None: return
    rec = p['rec'] if isinstance(p['rec'], dict) else {}
    if rec.get('status') in ('dropped', 'missing', 'error', 'cut') or rec.get('t_done') or time.time() - p['t_said'] > p['dur'] + 10:
        S['play'] = None
        t_push = rec.get('t_push') or max(p['t_said'], time.time() - p['dur'])
        start = t_push + HEAR_LAG_S + p['dance_off']
        wins = {k: (start + (k - p['k0']) * REF.chunk_s, start + (k - p['k0'] + 1) * REF.chunk_s) for k in range(p['k0'], p['k1'])}
        p['then'](wins); return
    _loop().call_later(0.1, _watch_play)


RETRY_AFTER_S = 3.0                                         # a reply that has not started by then was lost (relay: "no active generation"); replies start in 0.6-1.8 s


def when_quiet(fn, min_wait=2.5, quiet_s=1.0, max_wait=15.0, retry=None):
    """Run fn once Gemini has finished talking: at least min_wait (its reply takes ~1 s to start), then quiet_s with no voice
    audio buffered, at most max_wait (then the leftover voice audio is dropped so it does not play over the dance).
    retry: the prompt that was just asked; if no voice audio at all has played RETRY_AFTER_S after the ask, the reply was lost
    (seen through the relay: "received server content but no active generation") and the prompt is asked once more."""
    player = _player(); t0 = [time.monotonic()]; last = [time.monotonic()]; heard = [False]; retried = [False]
    if retry: min_wait = max(min_wait, RETRY_AFTER_S + 1.0)     # leave room for the retry: fn must not run before the second ask could
    def tick():
        mix = getattr(player, 'mix', None) if player is not None else None
        now = time.monotonic()
        if mix is not None and mix.active: last[0] = now; heard[0] = True
        if retry and not heard[0] and not retried[0] and now - t0[0] >= RETRY_AFTER_S:
            retried[0] = True; t0[0] = now; print('[dance] no reply heard: asking again', flush=True); _ask(retry)
        if now - t0[0] >= max_wait:
            if mix is not None: mix.clear()
            fn(); return
        if now - t0[0] >= min_wait and now - last[0] >= quiet_s:
            fn(); return
        _loop().call_later(0.1, tick)
    _loop().call_later(0.1, tick)


# ---- judging ----
def _window_poses(t0, t1):
    pts = [(t, k) for t, k in POSES if t0 - 0.3 <= t <= t1 + 0.3]
    if not pts: return np.zeros((0, 17, 3), np.float32)
    _, grid = resample(np.array([p[0] for p in pts]), np.array([p[1] for p in pts], np.float32), FPS, t0=t0, t1=t1)
    return grid


def _practice_score(kps):
    """No reference: score by movement, so the flow can be tested by anyone dancing anything."""
    seen = kps[..., 2] > 0.3
    if len(kps) < 3 or seen.mean() < 0.3: return dict(score=0.0, worst_part=None, move=None, source='practice')
    xy = kps[..., :2]; scale = max(float(np.ptp(xy[..., 1][seen])), 1.0)
    motion = float(np.abs(np.diff(xy, axis=0)).mean() / scale)
    return dict(score=float(np.clip(motion * 25, 0, 1)), worst_part=None, move=None, source='practice', details=dict(motion=round(motion, 4)))


def judge_async(items, then):
    """items: [(k, (t0, t1))] -> then({k: result}) on the loop; the judge runs in a thread."""
    data = [(k, _window_poses(*w)) for k, w in items]
    def work():
        return {k: (_practice_score(u) if isinstance(REF, _Practice) else judge_mod.judge(u, REF, k)) for k, u in data}
    fut = _loop().run_in_executor(None, work)
    def done(f):
        try: res = f.result()
        except Exception as e: res = {k: dict(score=0.0, worst_part=None, source='error', details=dict(error=repr(e))) for k, _ in items}
        then(res)
    fut.add_done_callback(done)


# ---- the lesson ----
def _describe(k):
    s = REF.step(k)
    return f' The moves for part {k + 1}: "{s}". Explain them simply in one sentence.' if s else ''


def teach_part():
    """Count in and play the current part (after making sure the caller is in frame)."""
    if S['state'] in ('done', 'full'): return
    if not S['visible']:
        if time.time() - S['last_step_back'] > 6:
            S['say']('Step back so I can see your whole body', 'visibility'); S['last_step_back'] = time.time()
        _loop().call_later(3.0, teach_part); return
    S['state'] = 'dancing'; k = S['chunk']
    print(f'[dance] part {k + 1}/{REF.n_chunks}, try {S["tries"] + 1}', flush=True)
    play_parts(k, k + 1, COUNT_ALOUD, lambda wins: judge_async(list(wins.items()), lambda res: on_result(res[k])))


def on_result(res):
    k = S['chunk']; S['tries'] += 1
    score = float(res.get('score') or 0.0); pct = int(round(score * 100)); S['last_pct'] = pct
    S['scores'][k] = max(S['scores'].get(k, 0), pct)
    print(f'[dance] part {k + 1} try {S["tries"]}: {pct}% ({res.get("source")}) worst {res.get("worst_part")} {res.get("details", "")}', flush=True)
    mood = _anger(0 if score >= PASS else S['anger'] + 1)
    if score >= PASS or S['tries'] >= MAX_TRIES:
        if score >= PASS: S['say'](random.choice(PRAISE))
        if k + 1 >= REF.n_chunks:
            return start_full_run()
        S['chunk'] += 1; S['tries'] = 0; S['last_pct'] = None
        if score >= PASS:
            what = f'passed part {k + 1} with {pct}%. {_flavour(pct, True)}'
        else:
            what = (f'got {pct}% on part {k + 1} after {MAX_TRIES} tries, so you are moving on anyway: one line of theatrical defeat '
                    f'("we will pretend that was on purpose"), no tip')
        p = _ask(f'The caller {what} {mood} In ONE short sentence, then introduce part {k + 2}.' + _describe(k + 1))
    else:
        part = res.get('worst_part'); move = res.get('move')
        tip = f'The body part furthest from the creator was the {part}.' if part else 'Their timing was off.'
        steps = f' The moves are: "{REF.step(k)}".' if REF.step(k) else ''
        again = f' Try {S["tries"] + 1} of {MAX_TRIES} is next.'
        p = _ask(f'The caller scored {pct}% on part {k + 1} of the {REF.name} dance (they need {int(PASS * 100)}%). {tip}'
                 + (f' The move was "{move}".' if move else '') + steps + ' ' + _flavour(pct, False) + again
                 + (f' {mood} Two short sentences at most: the bit, then one concrete tip about that body part said in that mood, then "again".'
                    if mood else ' Two short sentences at most: the bit, then one concrete tip about that body part, then "again".'))
    S['state'] = 'waiting'
    when_quiet(teach_part, retry=p)


def start_full_run():
    if not FULL_RUN or REF.n_chunks < 2:
        return finish()
    S['state'] = 'full'
    mood = ANGER[S['anger']] if S['anger'] else ''
    p = _ask(f'The caller has been through all {REF.n_chunks} parts of the {REF.name} dance. {mood} In ONE short sentence, like a game '
             + ('show host announcing the final round' if not mood else 'furious game show host who still has to announce the final round')
             + ': now the whole dance from the top' + (' with the music' if REF.audio is not None else '') + ', you will count them in.')
    def go():
        print('[dance] full run', flush=True)
        play_parts(0, REF.n_chunks, False, lambda wins: judge_async(sorted(wins.items()), on_full_result))
    when_quiet(go, retry=p)


def on_full_result(res):
    S['full_scores'] = {k: int(round(float(r.get('score') or 0) * 100)) for k, r in res.items()}
    print(f'[dance] full run: {S["full_scores"]}', flush=True)
    finish()


def finish():
    S['state'] = 'done'
    learn = ', '.join(f'part {i + 1} {s}%' for i, s in sorted(S['scores'].items()))
    full = ''
    if S['full_scores']:
        avg = int(round(sum(S['full_scores'].values()) / len(S['full_scores'])))
        full = f' In the full run they scored {avg}% overall (' + ', '.join(f'part {i + 1} {s}%' for i, s in sorted(S['full_scores'].items())) + ').'
    scores = list(S['full_scores'].values()) if S['full_scores'] else list(S['scores'].values())
    good = bool(scores) and sum(scores) / len(scores) >= PASS * 100
    mood = _anger(0 if good else S['anger'])
    _ask(f'The caller finished the {REF.name} dance lesson. Best scores while learning: {learn}.{full} {mood} In two short sentences, '
         + ('celebrate like they just won a dance show, name their strongest part, and tell them to post it and tag you (you will know if they do not).' if good else
            'be honest that it was not there yet, like a loving but disappointed coach, name their best part, and tell them to practise '
            'before they post it, because you will be watching. Lovingly.'))


# ---- the start: watch first, then set up the camera ----
def start_call():
    S['state'] = 'preview'
    if PREVIEW and not isinstance(REF, _Practice):
        p = _ask(f'The call just started. In two short sentences: say hi, you are Brodie and today you will teach them the {REF.name} '
             f'dance; first they should just watch it once on screen.')
        when_quiet(lambda: play_preview(start_setup), min_wait=3.0, retry=p)
    else:
        p = _ask(f'The call just started. In one short sentence: say hi, you are Brodie and today you will teach them the {REF.name} dance.')
        when_quiet(start_setup, min_wait=3.0, retry=p)


def start_setup():
    S['state'] = 'setup'; S['setup_since'] = time.time()
    _ask('Now ask the caller, in two short sentences, to prop the phone up at about knee height and step back two or three '
         'metres until their whole body, head to feet, is on screen.')
    _loop().call_later(1.0, check_setup)


def check_setup():
    if S['state'] != 'setup': return
    now = time.time()
    ready = S['visible'] and S['visible_since'] is not None and now - S['visible_since'] >= SETUP_HOLD_S
    if ready and now - S['setup_since'] >= SETUP_MIN_S:
        S['state'] = 'intro'
        S['say']('Great, I can see you', 'visibility')
        p = _ask(f'The caller is in position. In two short sentences: say we learn the {REF.name} dance in {REF.n_chunks} parts of '
             f'eight counts and that you will count them in.' + _describe(0))
        when_quiet(teach_part, min_wait=4.0, retry=p); return
    if not S['visible'] and now - S['setup_since'] > 20 and now - S['last_step_back'] > 10:
        S['say']('Step back so I can see your whole body', 'visibility'); S['last_step_back'] = now
    _loop().call_later(0.5, check_setup)


# ---- the events ----
def _set_visible():
    """visible = Brodie's whole-body rule OR the upper-body rule; the first time the dancer is in view the lesson starts."""
    was = S['visible']; S['visible'] = bool(S['vis_full'] or S['vis_upper'])
    if S['visible'] and not was: S['visible_since'] = time.time()
    if not S['visible']: S['visible_since'] = None
    if S['visible'] and not was and S['state'] == 'idle':
        if POSES_MODE: S['state'] = 'poses'; posegame.start()            # one pose at a time over the camera
        else: start_call()                                              # counts: watch first, then set up, then the lesson


def on_event(event, say_now, ask_voice):
    S['say'] = say_now; S['ask'] = ask_voice
    _ensure_screen()
    t = event['type']
    if t == 'pose':
        ok = bool(event.get('ok'))
        if ok:
            coco = mediapipe_to_coco(event['kp'], event['wh'])
            POSES.append((event['t_wall'], coco)); S['live'] = (event['t_wall'], coco, event['wh'])
        if POSES_MODE: posegame.on_pose(event)
        if ok and _upper_body_in_frame(event['kp']):
            S['upper_run'] += 1; S['upper_miss'] = 0
            if S['upper_run'] >= UPPER_ON_FRAMES and not S['vis_upper']: S['vis_upper'] = True; _set_visible()
        else:
            S['upper_miss'] += 1; S['upper_run'] = 0
            if S['upper_miss'] >= UPPER_OFF_FRAMES and S['vis_upper']: S['vis_upper'] = False; _set_visible()
        if S['state'] == 'idle' and not POSES_MODE: start_call()    # counts: the caller's video arrived: watch first, then set up (poses: wait to see them)
    elif t == 'visibility':
        S['vis_full'] = event.get('state') == 'full'; _set_visible()
    elif t == 'status' and POSES_MODE and S['state'] == 'idle':
        posegame.no_camera_tick(event)                                   # on the call with no video yet: ask for the camera


if POSES_MODE:                                              # the pose game uses this module's helpers (asks, retries, anger, the loop)
    from starter.dance import posegame
    posegame.bind(dict(S=S, ref=REF, ask=_ask, when_quiet=when_quiet, anger=_anger, loop=_loop))
    print('[dance] mode: poses (one pose at a time over the camera; DANCE_MODE=counts for the 8-count lesson)', flush=True)
else:
    print('[dance] mode: counts (8-count parts with the music)', flush=True)
