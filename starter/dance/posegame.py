"""Pose by pose: the lesson as a series of still target poses, no counting and no music (DANCE_MODE=poses, the default).

The caller sees their OWN camera, mirrored like a selfie, with their skeleton drawn on it (volt) and a ghost skeleton (white)
of where their body should go, placed on their own hips and scaled to their torso. The lesson does not move on until they match
the pose: the match score (segment directions, starter/dance/keypoints.features) must pass POSE_PASS in 6 of the last 8 pose
events. Stuck for 15 s: one voice tip naming the worst body part (the teacher gets angrier); 45 s: "we will come back to that one"
and the next pose. A match calms the teacher down.

Targets: the reference's pose every 2 beats (4 per 8-count part), near-identical neighbours dropped.

The handler (starter/handlers.py) wires it: bind(hooks) once, on_pose(event) for every pose event, start() when the caller is
first in view, compose_pose(face, state, now) for every video frame. Nothing here blocks: timers go through the worker's loop.
"""
import collections, os, random, textwrap, time
import numpy as np
import cv2
from starter.dance.keypoints import mediapipe_to_coco, features, normalize, SEG_NAMES, PART_OF, MP_TO_COCO, MIN_CONF, FPS
from starter.dance.screen import W, H, VOLT, WHITE, BLACK, _text

POSE_PASS = float(os.environ.get('POSE_PASS', '0.65'))                 # score that counts as "in the pose"
NEED, WINDOW = 6, 8                                                      # matched when NEED of the last WINDOW pose events pass
STUCK_S = float(os.environ.get('POSE_STUCK_S', '15'))                   # a tip every so often while stuck on one pose
GIVE_UP_S = float(os.environ.get('POSE_GIVE_UP_S', '45'))               # then move on anyway
MIN_DIFF_DEG = 8.0                                                       # targets closer than this to the previous one are dropped
ERR_FULL_DEG = 40.0                                                      # weighted mean angle error that scores 0
PAUSE_S = 1.2                                                            # praise, then the next pose
PRAISE = ['Nice work', 'Great rhythm', 'Looking strong', "That's it", 'Keep it up']
PRAISE_EVERY_S = 4.0                                                     # at most one bank praise line per 4 s (the test clip matched 4 poses in 6 s)
SEG_W = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.8, 0.6, 1.2], np.float32)   # SEG_NAMES order: arms, legs, torso
LEG_SEGS = [SEG_NAMES.index(n) for n in ('left thigh', 'left shin', 'right thigh', 'right shin')]
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
GREY = (90, 90, 90)


class Target:
    __slots__ = ('part', 'pose', 'step', 'frame', 'dirs', 'ok', 'unit')

    def __init__(self, part, pose, step, frame):
        self.part, self.pose, self.step, self.frame = part, pose, step, frame
        d, ok = features(pose); self.dirs, self.ok = d[0], ok[0]
        self.unit = normalize(pose[None])[0]                             # hip centre at 0, torso 1 long (for the ghost)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_targets(ref):
    """A key pose every 2 beats of every part, skipping poses nearly identical to the previous kept one."""
    out, prev = [], None
    for k in range(ref.n_chunks):
        for j in range(4):
            t = ref.first_beat_s + k * ref.chunk_s + j * 2 * ref.beat_s
            f = int(round(t * FPS))
            if not 0 <= f < len(ref.kps): continue
            pose = ref.kps[f]
            if pose[[5, 6, 11, 12], 2].min() < MIN_CONF: continue        # the ghost needs shoulders and hips
            tg = Target(k, pose, ref.step(k), f)
            if prev is not None:
                both = tg.ok & prev.ok
                if both.any() and float(np.degrees(np.abs(_wrap(tg.dirs[both] - prev.dirs[both]))).mean()) < MIN_DIFF_DEG: continue
            out.append(tg); prev = tg
    return out


def compare(coco, tgt):
    """coco: the caller's (17, 3) pose -> dict(score, worst (segment index), mirror, scores {False: s, True: s}, hint) or None
    when too little of the body is seen. Segment directions compared where both are confident; legs only when the caller's legs
    are in view; as-is and mirrored, the better one wins."""
    use = np.ones(len(SEG_NAMES), bool)
    if int((coco[[13, 14, 15, 16], 2] >= MIN_CONF).sum()) < 2: use[LEG_SEGS] = False
    best, scores = None, {}
    for mirror in (False, True):
        d, ok = features(coco, mirror); d, ok = d[0], ok[0]
        both = ok & tgt.ok & use
        if int(both.sum()) < 3: continue
        diff = np.degrees(np.abs(_wrap(d - tgt.dirs)))
        w = SEG_W * both
        err = float((diff * w).sum() / w.sum())
        score = float(np.clip(1.0 - err / ERR_FULL_DEG, 0.0, 1.0)); scores[mirror] = score
        if best is None or score > best['score']:
            j = int(np.argmax(np.where(both, diff * SEG_W, -1.0)))
            best = dict(score=score, worst=j, mirror=mirror, err=err, hint=_hint(d[j], tgt.dirs[j], SEG_NAMES[j]))
    if best is not None: best['scores'] = scores
    return best


def _hint(a_user, a_tgt, seg):
    """Which way the segment has to go, from its direction now vs the target's (image y points down)."""
    dxu, dyu, dxt, dyt = np.cos(a_user), np.sin(a_user), np.cos(a_tgt), np.sin(a_tgt)
    if seg == 'torso':
        return 'leaning more to the side' if abs(dxt) > abs(dxu) + 0.15 else 'straighter, stand tall'
    if abs(dyt - dyu) >= abs(dxt - dxu):
        return 'higher' if dyt < dyu else 'lower'
    return 'further out to the side' if abs(dxt) > abs(dxu) else 'closer in to your body'


def _user_part(j, mirror):
    """The body part in the caller's own terms: mirrored matching puts their right arm in the 'left' slot."""
    name = SEG_NAMES[j]
    if mirror: name = name.replace('left', 'TMP').replace('right', 'left').replace('TMP', 'right')
    return PART_OF[name]


# ---- state ----
G = dict(targets=[], i=0, phase='idle', hist=collections.deque(maxlen=WINDOW), score=None, worst=None, mirror=False, hint='',
         t_target=0.0, t_tip=0.0, t_step_back=0.0, tips=0, nailed=0, skipped=0, matched=0, user=None, anchor=None, hooks=None,
         ticking=False, cam=None, cam_seq=None, cam_map=None, draw_err=0)


def bind(hooks):
    """hooks: dict(S, ref, ask, when_quiet, anger, loop) from the handler; builds the targets."""
    G['hooks'] = hooks; ref = hooks['ref']
    G['targets'] = build_targets(ref)
    print(f'[pose] {len(G["targets"])} target poses from {ref.name!r} (every 2 beats, {ref.n_chunks} parts); pass {POSE_PASS:.0%} in '
          f'{NEED} of {WINDOW} frames; tip after {STUCK_S:g} s, next pose after {GIVE_UP_S:g} s', flush=True)
    return len(G['targets'])


def _h(name):
    return G['hooks'][name]


def _say(text, intent='other'):
    try: return _h('S')['say'](text, intent)
    except Exception as e: print(f'[pose] say {text!r} failed: {e!r}', flush=True)


def _later(delay, fn):
    _h('loop')().call_later(delay, fn)


def _target():
    t = G['targets']; i = G['i']
    return t[i] if 0 <= i < len(t) else None


# ---- the events ----
def on_pose(event):
    G['seen_pose'] = True
    """Every pose event: remember the caller's pose for the drawing; score it against the current target while playing."""
    kp = event.get('kp')
    if not event.get('ok') or kp is None:
        G['user'] = None; return
    kp = np.asarray(kp, np.float32)
    unit = np.zeros((17, 3), np.float32); unit[:, :2] = kp[MP_TO_COCO, :2]; unit[:, 2] = kp[MP_TO_COCO, 3] if kp.shape[1] >= 4 else 1.0
    G['user'] = unit
    tgt = _target()
    if G['phase'] != 'play' or tgt is None: return
    coco = mediapipe_to_coco(kp, event['wh'])
    res = compare(coco, tgt)
    if res is None:
        G['score'] = None; return
    G['score'] = res['score']; G['worst'] = (res['worst'], res['mirror']); G['hint'] = res['hint']
    cur = res['scores'].get(G['mirror'])                                  # the ghost keeps its orientation unless the other is clearly better
    if res['mirror'] != G['mirror'] and (cur is None or res['score'] - cur > 0.1): G['mirror'] = res['mirror']
    G['hist'].append(res['score'] >= POSE_PASS)
    if sum(G['hist']) >= NEED: _matched()


def start():
    """The caller is in view for the first time: say hi, then the first pose."""
    if G['phase'] != 'idle': return
    G['phase'] = 'intro'; S = _h('S'); ref = _h('ref'); n = len(G['targets'])
    _say('Great, I can see you', 'visibility')
    hi = ('There they are: in one short line react to finally seeing them (no second hello), then explain' if G.get('t_no_camera') else
          'In two short sentences: say hi with your intro line, then explain')
    p = _h('ask')(f'The caller just stepped into frame, ready. {hi} today\'s '
                  f'lesson: the {ref.name} dance one pose at a time, {n} poses, they copy the move on screen and hold it until it locks in. '
                  f'No counting in this lesson, ever.')
    _h('when_quiet')(_begin, min_wait=4.0, retry=p)


def _begin():
    if G['phase'] != 'intro': return
    now = time.time(); G['phase'] = 'play'; G['t_target'] = G['t_tip'] = G['t_step_back'] = now; G['hist'].clear()
    print(f'[pose] lesson started: target 1/{len(G["targets"])}', flush=True)
    _ask_part(_target())
    if not G['ticking']: G['ticking'] = True; _later(1.0, _tick)


def _ask_part(tgt):
    if tgt is None: return
    ref = _h('ref'); what = f' The moves: "{tgt.step}".' if tgt.step else ''
    p = _h('ask')(f'Part {tgt.part + 1} of {ref.n_chunks} of the {ref.name} dance starts now.{what} Say what to do in ONE simple sentence, '
                  f'then tell them to copy the pose on screen and hold it. No counting.')
    _h('when_quiet')(lambda: None, retry=p)


def _matched():
    tgt = _target(); i = G['i']; now = time.time()
    G['matched'] += 1
    if G['tips'] == 0: G['nailed'] += 1
    print(f'[pose] target {i + 1}/{len(G["targets"])} (part {tgt.part + 1}) matched: score {G["score"]:.2f}, '
          f'{"mirrored" if G["mirror"] else "as-is"}, {now - G["t_target"]:.1f} s, tips {G["tips"]}', flush=True)
    _h('anger')(0)
    if _voice_quiet() and now - G.get('t_praise', 0) >= PRAISE_EVERY_S:   # a bank line never over Brodie's own voice, and not on every quick match
        _say(random.choice(PRAISE)); G['t_praise'] = now
    G['phase'] = 'pause'; _later(PAUSE_S, _next)


NO_CAMERA_EVERY_S = 20.0


def no_camera_tick(ev):
    """A status event while nothing has been seen: the caller is on the call without video (camera off or denied). Greet once
    the model session is up and ask for the camera, then remind every NO_CAMERA_EVERY_S. Stops as soon as a pose arrives."""
    if G['phase'] != 'idle' or G.get('seen_pose') or ev.get('voice') != 'on': return
    now = time.time()
    if now - G.get('t_no_camera', 0) < NO_CAMERA_EVERY_S: return
    first = not G.get('t_no_camera'); G['t_no_camera'] = now
    _h('ask')(('The call just connected but their camera is off, so you cannot see them yet. In two short sentences: say hi with your '
               'intro line, then ask them to turn the camera on and prop the phone up so you can see them from head to hips; the lesson '
               'starts when you see them.') if first else
              'Their camera is still off. One short, funny nudge to turn it on (you cannot coach what you cannot see).')
    print('[pose] no camera yet: asked for it', flush=True)


def _voice_quiet():
    """True when no voice audio (the character's reply) is buffered in the cue player's mixer."""
    try:
        player = _h('S')['say'].__self__.job.player; mix = getattr(player, 'mix', None)
        return mix is None or not mix.active
    except Exception:
        return True


def _next():
    now = time.time(); prev = _target()
    G['i'] += 1; G['hist'].clear(); G['score'] = None; G['worst'] = None; G['tips'] = 0; G['t_target'] = G['t_tip'] = now
    tgt = _target()
    if tgt is None: return _finish()
    G['phase'] = 'play'
    print(f'[pose] target {G["i"] + 1}/{len(G["targets"])} (part {tgt.part + 1}, clip frame {tgt.frame})', flush=True)
    if prev is None or tgt.part != prev.part: _ask_part(tgt)


def _tick():
    if G['phase'] == 'done': return
    _later(1.0, _tick)
    if G['phase'] != 'play': return
    now = time.time(); S = _h('S')
    if not S.get('visible'):                                              # nobody in view: the clocks wait, the nag every 8 s
        G['t_target'] += 1.0; G['t_tip'] += 1.0
        if now - G['t_step_back'] > 8: _say('Step back so I can see your whole body', 'visibility'); G['t_step_back'] = now
        return
    if now - G['t_target'] >= GIVE_UP_S: _give_up()
    elif now - G['t_tip'] >= STUCK_S: _tip()


def _tip():
    tgt = _target(); i = G['i']; now = time.time(); S = _h('S'); ref = _h('ref')
    G['tips'] += 1; G['t_tip'] = now
    mood = _h('anger')(S['anger'] + 1)
    sc = G['score']; pct = int(round((sc or 0.0) * 100))
    if G['worst'] is not None and sc is not None:
        part = _user_part(*G['worst']); hint = G['hint'] or 'closer to the pose on screen'
        where = f'Their match is {pct}%; the body part furthest from the target is their {part}: it should go {hint}.'
        example = f'"{part} {hint}"'
    else:
        part = 'body'; where = 'I cannot see enough of their body to score them: they should step back so the whole body is in view.'
        example = '"step back a little"'
    print(f'[pose] target {i + 1} stuck {now - G["t_target"]:.0f} s: score {sc if sc is None else round(sc, 2)}, tip {G["tips"]} about the {part} ({G["hint"]})', flush=True)
    what = f' (part {tgt.part + 1}: "{tgt.step}")' if tgt.step else ''
    p = _h('ask')(f'The caller has been stuck on pose {i + 1} of {len(G["targets"])} for {int(now - G["t_target"])} seconds{what}. {where} '
                  f'{mood} Say ONE short tip naming that body part, like {example}, nothing else.')
    _h('when_quiet')(lambda: None, retry=p)


def _give_up():
    tgt = _target(); i = G['i']; S = _h('S')
    G['skipped'] += 1
    print(f'[pose] target {i + 1} not matched after {GIVE_UP_S:g} s (best recent score {G["score"]}): moving on', flush=True)
    mood = _h('anger')(S['anger'])
    p = _h('ask')(f'The caller could not match pose {i + 1} of {len(G["targets"])} after {GIVE_UP_S:.0f} seconds. {mood} ONE short line, '
                  f'the bit is "we will come back to that one", then "next pose".')
    _h('when_quiet')(lambda: None, retry=p)
    G['phase'] = 'pause'; _later(PAUSE_S, _next)


def _finish():
    G['phase'] = 'done'; S = _h('S'); ref = _h('ref'); n = len(G['targets'])
    good = G['skipped'] == 0 or G['nailed'] >= n / 2
    mood = _h('anger')(0 if good else S['anger'])
    print(f'[pose] done: {n} poses, {G["nailed"]} nailed first try, {G["matched"] - G["nailed"]} after a tip, {G["skipped"]} skipped', flush=True)
    p = _h('ask')(f'The caller finished the {ref.name} dance, all {n} poses: {G["nailed"]} nailed on the first try, '
                  f'{G["matched"] - G["nailed"]} needed a tip, {G["skipped"]} skipped. {mood} Two short sentences: '
                  + ('celebrate like they just won a dance show, say how many poses they nailed first try, then tell them to post it and tag you.' if good else
                     'be honest it was not there yet like a loving but disappointed coach, say how many poses they nailed first try, and tell them to practise before they post it.'))
    _h('when_quiet')(lambda: None, retry=p)


# ---- the picture ----
def _camera():
    """The caller's newest frame (job.latest, RGB24), cover-fitted to W x H (not yet mirrored) and a mapper from the pose
    event's 0..1 coordinates to pixels of that picture. Black and a plain mapper before the first frame."""
    try: latest = _h('S')['say'].__self__.job.latest
    except Exception: latest = None
    if latest is None:
        return np.zeros((H, W, 3), np.uint8), (lambda u, v: (u * W, v * H))
    frame, seq = latest[0], latest[4]
    if G['cam_seq'] != seq:
        w, h = frame.width, frame.height
        buf = np.frombuffer(frame.data, np.uint8)
        if buf.size == w * h * 3: arr = buf.reshape(h, w, 3)
        else:
            stride = buf.size // h; arr = buf[:h * stride].reshape(h, stride)[:, :w * 3].reshape(h, w, 3)
        s = max(W / w, H / h); cw, ch = W / s, H / s; x0 = (w - cw) / 2; y0 = (h - ch) / 2
        crop = arr[int(round(y0)):int(round(y0 + ch)), int(round(x0)):int(round(x0 + cw))]
        G['cam'] = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR); G['cam_seq'] = seq; G['cam_map'] = (w, h, x0, y0, s)
    w, h, x0, y0, s = G['cam_map']
    return G['cam'].copy(), (lambda u, v: ((u * w - x0) * s, (v * h - y0) * s))


def _skeleton(img, pts, conf, color, thick, filled):
    """COCO limbs and joints; pts (17, 2) pixels, conf (17,); hollow joints and a dark outline for the ghost."""
    ok = conf >= MIN_CONF
    p = [(int(round(x)), int(round(y))) for x, y in pts]
    if not filled:
        for a, b in LIMBS:
            if ok[a] and ok[b]: cv2.line(img, p[a], p[b], BLACK, thick + 4, cv2.LINE_AA)
    for a, b in LIMBS:
        if ok[a] and ok[b]: cv2.line(img, p[a], p[b], color, thick, cv2.LINE_AA)
    for j in range(17):
        if not ok[j] or (1 <= j <= 4): continue                              # no dots on the eyes and ears
        if filled: cv2.circle(img, p[j], 5, color, -1, cv2.LINE_AA)
        else:
            r = 12 if j == 0 else 6                                           # the ghost's nose is a head
            cv2.circle(img, p[j], r, BLACK, 4, cv2.LINE_AA); cv2.circle(img, p[j], r, color, 2, cv2.LINE_AA)


def _ghost_points(tgt, pts_u, conf_u):
    """The target placed on the caller's hip centre and torso length (smoothed); mid-screen when nobody is seen."""
    hip = torso = None
    if pts_u is not None and conf_u[[5, 6, 11, 12]].min() >= MIN_CONF:
        hc = (pts_u[11] + pts_u[12]) / 2; sc = (pts_u[5] + pts_u[6]) / 2; L = float(np.linalg.norm(sc - hc))
        if L > 12: hip, torso = hc, L
    if hip is None: hip, torso = np.array([W / 2, H * 0.62], np.float32), 110.0
    a = G['anchor']
    if a is not None:
        hip = 0.6 * hip + 0.4 * a[0]; torso = 0.6 * torso + 0.4 * a[1]                  # a little smoothing against pose jitter
    G['anchor'] = (hip, torso)
    xy = tgt.unit[:, :2].copy()
    if G['mirror']: xy[:, 0] = -xy[:, 0]
    return xy * torso + hip


def _blend(dst, src_rgba, x, y):
    h, w = src_rgba.shape[:2]; HH, WW = dst.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, WW), min(y + h, HH)
    if x1 <= x0 or y1 <= y0: return
    s = src_rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32); a = s[..., 3:4] / 255.0
    dst[y0:y1, x0:x1] = (s[..., :3] * a + dst[y0:y1, x0:x1].astype(np.float32) * (1 - a)).astype(np.uint8)


def _label(img, s, x, y, scale, color, thick):
    """Text centred on x (clamped to the screen)."""
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x = int(min(max(x - tw / 2, 4), W - tw - 4)); y = int(min(max(y, th + 4), H - 4))
    _text(img, s, (x, y), scale, color, thick)


def compose_pose(face, st, now):
    """face: the character's RGBA (H, W, 4) frame -> RGB (H, W, 3) for the call: the selfie camera, the caller's skeleton, the
    ghost of the target, the pose number and the step text, the match meter, the face small in a corner."""
    out, mp = _camera()
    user = G['user']; phase = G['phase']; tgt = _target(); n = len(G['targets'])
    pts_u = conf_u = None
    if user is not None:
        x, y = mp(user[:, 0], user[:, 1]); pts_u = np.column_stack([x, y]); conf_u = user[:, 2]
        _skeleton(out, pts_u, conf_u, VOLT, 4, True)
    ghost = None
    if tgt is not None and phase in ('intro', 'play', 'pause'):
        ghost = _ghost_points(tgt, pts_u, conf_u)
        _skeleton(out, ghost, tgt.pose[:, 2], WHITE, 3, False)
    out = np.ascontiguousarray(out[:, ::-1])                                  # the selfie mirror; text goes on after this
    if ghost is not None:                                                     # the label under the ghost's lowest seen joint
        seen = ghost[tgt.pose[:, 2] >= MIN_CONF]
        gx = W - 1 - float((ghost[11, 0] + ghost[12, 0]) / 2); gy = float(seen[:, 1].max()) + 30 if len(seen) else H * 0.8
        _label(out, 'COPY THIS' if phase == 'play' else 'GET READY', gx, min(gy, H - 74), 0.7, WHITE, 2)
    if phase == 'idle' or (phase == 'play' and not st.get('visible')):
        _text(out, 'Step into frame', (0, 110), 1.0, VOLT, 3, center=True)
        _text(out, 'so I can see your head and hips', (0, 150), 0.6, WHITE, 2, center=True)
    if phase == 'done':
        _text(out, 'YOU DID IT!', (0, 120), 1.4, VOLT, 3, center=True)
        _text(out, f'{G["nailed"]} of {n} poses nailed first try', (0, 170), 0.7, WHITE, 2, center=True)
        if G['skipped']: _text(out, f'{G["skipped"]} to come back to', (0, 200), 0.6, WHITE, 2, center=True)
    elif tgt is not None and phase in ('intro', 'play', 'pause'):
        lines = textwrap.wrap(tgt.step, 30)[:4] if tgt.step else []
        band = 52 + 26 * len(lines)
        out[:band] = (out[:band] * 0.45).astype(np.uint8)                       # a dark band so the text reads over the camera
        _text(out, f'POSE {G["i"] + 1}/{n}', (12, 40), 1.0, VOLT, 3)
        for i, line in enumerate(lines):
            _text(out, line, (12, 72 + 26 * i), 0.6, WHITE, 2)
        sc = G['score']
        if phase in ('play', 'pause'):                                        # the match meter
            x0, x1, y0, y1 = 16, W - 16, H - 30, H - 14
            cv2.rectangle(out, (x0, y0), (x1, y1), BLACK, -1); cv2.rectangle(out, (x0, y0), (x1, y1), GREY, 1)
            if sc is not None:
                good = sc >= POSE_PASS
                cv2.rectangle(out, (x0 + 1, y0 + 1), (x0 + 1 + int((x1 - x0 - 2) * sc), y1 - 1), VOLT if good else (200, 200, 200), -1)
                _text(out, f'MATCH {int(round(sc * 100))}%', (16, H - 38), 0.6, VOLT if good else WHITE, 2)
                if good: _text(out, 'HOLD IT', (W - 110, H - 38), 0.6, VOLT, 2)
            else:
                _text(out, 'MATCH --', (16, H - 38), 0.6, WHITE, 2)
            xp = x0 + 1 + int((x1 - x0 - 2) * POSE_PASS); cv2.line(out, (xp, y0 - 3), (xp, y1 + 3), WHITE, 2)
    small = cv2.resize(face, (120, 213), interpolation=cv2.INTER_AREA)
    _blend(out, small, W - 120 - 6, H - 213 - 52)
    return out
