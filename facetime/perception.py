"""Events-only perception service: FramePipeline + Session exactly as coach_agent runs them, with Session.say replaced by an
event sink and no audio at all (hackathon starter, reports/STARTER_PREREQS_REVIEW.md section 5 items 2 and 3).

Same modes (--plan, --exercise, neither = auto-detect), the same POLICY_DEFAULT, REP_PARAMS_V2, thresholds and trigger defaults,
visibility, gesture and probe as facetime/coach_agent.py and facetime/replay_local.py: the pipeline and the session are built from the
same flags, so the cue events here are the cue timeline the demo coach speaks. Nothing here publishes a track or opens an audio
object; a caller that wants the fast mouth attaches the CuePlayer itself through `say` (facetime/worker.py does, by default).

Event schema (EVENT_SCHEMA_VERSION "1"; one JSON object per event, docs/STARTER_EVENTS.md). Every event has
  v (schema version), t (seconds since the service started, on the caller's monotonic clock: the frame's receive time for frame
  events; a replay passes t0=0 and the clip time), t_wall (epoch seconds when the event was emitted), seq (0, 1, 2, ...), type, and
  frame (index of the processed frame, except status).
Types: rep {exercise, n, confidence {agree, amp, trackers} | null, turning_t (the rep's turning point on the event clock), lag_s
(t - turning_t: how long after the bottom the count landed)}, exercise_seen {name, p}, fault {label, cue_text, p}, cue {text, intent,
why, state, exercise, count, t_session, held_s?, t_decided?} (what the demo coach would have said: counts, corrections, praise,
announces, milestones, visibility lines), visibility {state: full|partial|none, paused, no_pose_s}, gesture {name: tpose}, probe
{arms_up, on_box_ms?, span_ms?, t_first_up?} (a transport may add beep), session {state, exercise, target, count, item, done, mode}, status {fps_recv, fps_processed, pose_ms, clf_ms,
pose_model, recv_to_decision_p50_ms?, dropped?, underruns?, state, exercise, count}.

Use:  per = Perception(Config(plan='squats:10'))            # or Perception(argparse namespace with the coach_agent flag names)
      per.on_event(print); per.log_to('events.jsonl')
      events = per.process_frame(rgb, t_recv=time.monotonic())            # live: one call per sampled frame, any thread
      events = per.process_frame(rgb, t_recv=i / FPS, t_ms=i * 1000 / FPS)  # replay: the pipeline's reproducible t_ms path
      per.close()
"""
import dataclasses, json, logging, os, threading, time
import numpy as np
from facetime import DATA
from facetime.pipeline import FramePipeline, load_json_arg, FPS, COUNTERS, spoken_name
from facetime.session import Session

log = logging.getLogger('perception')
EVENT_SCHEMA_VERSION = "1"
EVENT_TYPES = ('rep', 'exercise_seen', 'fault', 'cue', 'visibility', 'gesture', 'probe', 'session', 'status', 'pose')
MODEL_DEFAULT = os.path.join(DATA, 'models', 'clf_pose.pt')


@dataclasses.dataclass
class Config:
    """The coach_agent / replay_local flags that shape perception and the session (same names, same defaults)."""
    plan: str = None                 # "squats:10,pushups:8,..." (plan mode)
    exercise: str = None             # single-exercise mode; neither = auto-detect
    model: str = MODEL_DEFAULT
    pose: str = 'full'
    params: str = '{}'               # JSON or .json: Coach policy overrides on POLICY_DEFAULT
    thr: str = None                  # per-label thresholds file (default: thresholds_e3c.json)
    counter: str = 'v2'              # COUNTERS key: v2 (demo guards), v2plain, e1
    trigger: str = 'auto'            # learned correction trigger: path, auto, none
    rest_s: float = 5.0
    set_timeout: float = 75.0
    tempo: str = None
    auto_classes: str = 'demo'
    announce: bool = True
    gesture_next: bool = False       # T-pose command (facetime/probe.py TPose)
    latency_probe: bool = False      # probe events are emitted anyway; this only tells a caller with audio to beep
    auto_lite: bool = True           # live only: drop the pose model to lite when full cannot hold 10 fps
    status_every_s: float = 5.0
    seen_every_s: float = 1.0        # exercise_seen at most this often while the top class is unchanged
    dance: bool = False              # dance mode: a `pose` event every frame, no workout script (no counts, no cues; handlers drive the call)

    @classmethod
    def from_args(cls, a):
        """From an argparse namespace (coach_agent.build_parser, replay_local, worker): missing flags keep their defaults."""
        return cls(**{f.name: getattr(a, f.name) for f in dataclasses.fields(cls) if hasattr(a, f.name)})

    @property
    def mode(self):
        return 'plan' if self.plan else ('single' if self.exercise else 'auto')


def add_arguments(ap):
    """The Config flags on an argparse parser (the names coach_agent and replay_local use)."""
    d = Config()
    ap.add_argument('--plan', default=d.plan, help='workout plan mode, e.g. "squats:10,pushups:8,jumping jacks:20,lunges:10"')
    ap.add_argument('--exercise', default=d.exercise, help='single-exercise mode (squats/pushups/lunges/jumping jacks or an LR2CLF key); neither flag = auto-detect')
    ap.add_argument('--model', default=d.model); ap.add_argument('--pose', default=d.pose)
    ap.add_argument('--params', default=d.params); ap.add_argument('--thr', default=d.thr)
    ap.add_argument('--counter', default=d.counter, choices=sorted(COUNTERS))
    ap.add_argument('--trigger', default=d.trigger, help='learned correction trigger: a model path, auto (data/models/trigger_lr.json if present) or none')
    ap.add_argument('--rest-s', type=float, default=d.rest_s); ap.add_argument('--set-timeout', type=float, default=d.set_timeout)
    ap.add_argument('--tempo', default=d.tempo, help='JSON (or .json file) overriding seconds-per-rep targets')
    ap.add_argument('--auto-classes', default=d.auto_classes, choices=['demo', 'all'])
    ap.add_argument('--no-announce', dest='announce', action='store_false', help="skip the \"Let's do <exercise>\" line")
    ap.add_argument('--gesture-next', action='store_true', help='a T-pose held 1.5 s closes the set / skips the rest and emits a gesture event (without it no T-pose detector runs: no gesture events)')
    ap.add_argument('--latency-probe', action='store_true', help='with audio attached: beep the instant both wrists go above the head')
    ap.add_argument('--no-auto-lite', dest='auto_lite', action='store_false')
    ap.add_argument('--dance', action='store_true', help='dance mode: emit a pose event (33 MediaPipe landmarks) every frame and skip the workout script; starter/handlers.py runs the call')
    return ap


def build_pipeline(cfg):
    """The expensive, room-independent part (pose model, classifier, trigger), built exactly as coach_agent.Agent builds it.
    Separate from Perception so a worker can preload it (prewarm) and hand it to Perception(pipe=...) at job start."""
    pipe = FramePipeline(cfg.model, None, cfg.pose, load_json_arg(cfg.params, {}), load_json_arg(cfg.thr),
                         rep_params=COUNTERS[cfg.counter], trigger=cfg.trigger)
    pipe.set_gesture_next(bool(cfg.gesture_next))
    return pipe


def cue_durations(cues_dir=None):
    """text -> clip duration in seconds (None when the bank has no clip), read from the bank manifest only: no PCM is loaded and
    no audio object is created. The Session's rep-aware scheduling needs these so the cue timeline matches the demo coach."""
    from facetime.audio_out import default_cues_dir      # path logic only (module import creates nothing)
    d = cues_dir or default_cues_dir(); mf = os.path.join(d, 'manifest.json')
    try:
        m = json.load(open(mf)) if os.path.exists(mf) else {}
    except (OSError, ValueError) as e:
        log.warning('cue manifest %s unreadable (%s): no clip durations', mf, e); m = {}
    return lambda text: (m[text]['duration_s'] if text in m else None)


class JsonlWriter:
    """One JSON line per event; `extra` keys (room, call id) are added to every line. Safe to call from the perception thread."""
    def __init__(self, path, extra=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        self.path = path; self.f = open(path, 'a'); self.extra = dict(extra or {}); self.n = 0; self._lock = threading.Lock()

    def __call__(self, ev):
        line = json.dumps(dict(ev, **self.extra) if self.extra else ev, sort_keys=False)
        with self._lock:
            if self.f.closed: return
            self.f.write(line + '\n'); self.f.flush(); self.n += 1

    def close(self):
        with self._lock:
            if not self.f.closed: self.f.close()


class Perception:
    def __init__(self, cfg=None, pipe=None, say=None, clip_s=None, t0=None, fps=FPS, **kw):
        """cfg: Config, an argparse namespace with the coach_agent flag names, or keyword fields (Perception(plan='squats:10')).
        pipe: a prebuilt FramePipeline (build_pipeline) for warm starts; else one is built here.
        say(text, intent, meta): optional passthrough of every Session cue, in the order the demo coach would queue them
        (a caller with audio hands it to CuePlayer.say); the cue event is emitted first. Nothing here plays audio.
        clip_s(text) -> seconds | None: cue clip durations for the session's rep-aware scheduling; default: the bank manifest.
        t0: the service start on the caller's clock (None = time.monotonic() now, live); a replay passes 0.0 and clip times."""
        if cfg is None: cfg = Config(**kw)
        elif not isinstance(cfg, Config): cfg = Config.from_args(cfg)
        self.cfg = cfg; self.fps = fps
        self.live = t0 is None; self.t0 = time.monotonic() if t0 is None else float(t0)
        self.pipe = pipe if pipe is not None else build_pipeline(cfg)
        if pipe is not None: self.pipe.set_gesture_next(bool(cfg.gesture_next))   # per session, as the standby agent does per call
        self.clip_s = clip_s if clip_s is not None else cue_durations()
        self.say_cb = say
        self.session = Session(self.pipe, self._say, fps=fps, plan=cfg.plan, exercise=cfg.exercise, announce=cfg.announce,
                               rest_s=cfg.rest_s, set_timeout_s=cfg.set_timeout, tempo=load_json_arg(cfg.tempo, None),
                               auto_classes=cfg.auto_classes, clip_s=self.clip_s, gesture_next=bool(cfg.gesture_next))
        self.mode = self.session.mode
        self._cbs = []; self._writers = []; self.seq = 0; self.n = 0; self.closed = False
        self.frames_received = None; self.frames_dropped = None; self.audio_stats = None   # a transport sets these for the status event
        self._t = 0.0; self._frame = 0; self._frame_events = []
        self._coach_id = None; self._last_count = 0
        self._rc_t = {}; self._rc_last_i = None       # counter frame index -> service time, for turning_t on the event clock
        self._seen = (None, -1e9)                     # exercise_seen: (name, t of the last event)
        self.hold = False                                              # set by the voice layer while a conversation is going; session.py pauses the rest/wait clock
        self._vis_emitted = None; self._vis_cand = (None, 0); self._paused = False; self.vis_hold = 5   # 0.5 s before a state change is reported (paused is hysteretic already)
        self._sess_key = None
        self._recv_hist = []                          # monotonic receive times of the last frames (probe on-box timing)
        self._win = dict(t=None, n=0, pose=0.0, clf=0.0, d=[], recv0=None); self.note = ''
        self.delays = []; self.n_events = {k: 0 for k in EVENT_TYPES}

    # ---- sinks ----
    def on_event(self, cb):
        """cb(event dict), called on the thread that calls process_frame, in emission order. Returns cb."""
        self._cbs.append(cb); return cb

    def log_to(self, path, extra=None):
        """JSONL file sink (one line per event); closed by close(). Returns the writer."""
        w = JsonlWriter(path, extra); self._writers.append(w); self.on_event(w); return w

    def _emit(self, type_, t=None, **fields):
        ev = dict(v=EVENT_SCHEMA_VERSION, t=round(self._t if t is None else t, 3), t_wall=round(time.time(), 3), seq=self.seq, type=type_)
        if type_ != 'status': ev['frame'] = self._frame
        ev.update(fields); self.seq += 1; self.n_events[type_] = self.n_events.get(type_, 0) + 1
        self._frame_events.append(ev)
        for cb in self._cbs:
            try: cb(ev)
            except Exception as e: log.warning('event callback failed: %r', e)
        return ev

    def _say(self, text, intent, meta):
        """Session.say sink: the cue event, then the optional passthrough (the demo coach's player)."""
        extra = {}
        if 'held_s' in meta:                                    # a held correction/praise: released on this frame, decided held_s earlier
            extra['held_s'] = meta['held_s']
            if isinstance(meta.get('t_recv'), (int, float)): extra['t_decided'] = round(meta['t_recv'] - self.t0, 3)
        self._emit('cue', text=text, intent=intent, why=meta.get('why', ''), state=meta.get('session_state'), exercise=spoken_name(meta['session_exercise']) if meta.get('session_exercise') else None,
                   count=meta.get('count', 0), t_session=meta.get('t_session'), **extra)
        if self.say_cb is not None: self.say_cb(text, intent, meta)

    # ---- per frame ----
    def process_frame(self, rgb, t_recv=None, t_ms=None, meta=None):
        """rgb: HxWx3 uint8 at any size. t_recv: the frame's receive time on the service clock (None = time.monotonic() now).
        t_ms: MediaPipe VIDEO-mode timestamp for reproducible replays (i * 1000 / FPS); live callers leave it None.
        meta: extra keys for the cue passthrough (frame_seq, ts_us, t_capture ...). Returns this frame's events (list)."""
        if self.closed: raise RuntimeError('Perception is closed')
        t_mono = time.monotonic()
        if t_recv is None: t_recv = t_mono
        self._t = t_recv - self.t0; self._frame = self.n; self._frame_events = []
        if self.live: self._recv_hist.append(t_recv); del self._recv_hist[:-4]
        r = self.pipe.process(rgb, t_ms=t_ms)                         # an exception here propagates to the caller (ends the run)
        t_dec = time.time(); self.n += 1
        m = dict(frame=self._frame, t_recv=t_recv, t_decision=t_dec, pose_ms=round(r['pose_ms'], 1), clf_ms=round(r['clf_ms'], 1), pose_model=self.pipe.pose_variant)
        if meta: m.update(meta)
        self._perception_events(r, t_mono)
        if self.cfg.dance:                                            # dance mode: the landmarks to the handlers, no workout script
            a = self.pipe.last_landmarks
            self.last_rgb = rgb.copy()                                # the caller's latest frame, for the dance screen (starter/dance/screen.py)
            self._emit('pose', ok=a is not None, wh=[int(rgb.shape[1]), int(rgb.shape[0])],
                       kp=(np.round(a[:, :4], 4).tolist() if a is not None else None))   # (33,4) x,y in 0..1 of the frame, z, visibility
        else:
            r['hold'] = bool(self.hold)
            self.session.on_frame(r, m)                               # cue events come out of _say in the demo coach's order
            self._session_event()
        self._status(r, t_mono)
        return self._frame_events

    def _perception_events(self, r, t_mono):
        coach = self.pipe.coach; cid = id(coach) if coach is not None else None
        if cid != self._coach_id: self._coach_id = cid; self._last_count = 0; self._rc_t = {}; self._rc_last_i = None   # fresh Coach (new exercise, after a rest): counts restart
        rc = coach.rc if coach is not None else None
        if rc is not None and getattr(rc, 'i', None) is not None and rc.i != self._rc_last_i:   # the counter consumed this frame
            self._rc_last_i = rc.i; self._rc_t[rc.i] = self._t
            if len(self._rc_t) > 400: del self._rc_t[min(self._rc_t)]
        c = r['count']
        if c < self._last_count: self._last_count = 0
        if c > self._last_count:
            conf = self._confidence(rc); ex = spoken_name(self.pipe.exercise) if self.pipe.exercise else None
            tt = r.get('turning_t'); turning_t = self._rc_t.get(int(round(tt * self.fps))) if tt is not None else None   # the turning point on the event clock
            for n in range(self._last_count + 1, c + 1):               # the counter can confirm two reps on one frame
                self._emit('rep', exercise=ex, n=n, confidence=conf, turning_t=(round(turning_t, 3) if turning_t is not None else None),
                           lag_s=(round(self._t - turning_t, 3) if turning_t is not None else None))
            self._last_count = c
        if r.get('ex_probs') is not None and self.pipe.clf is not None:
            p = np.asarray(r['ex_probs']); j = int(np.argmax(p)); name = spoken_name(self.pipe.clf.exs[j])
            if name != self._seen[0] or self._t - self._seen[1] >= self.cfg.seen_every_s:
                self._seen = (name, self._t); self._emit('exercise_seen', name=name, p=round(float(p[j]), 3))
        for ev in r['events']:
            if ev['intent'] == 'correction':
                self._emit('fault', label=ev['why'], cue_text=ev['text'], p=self._fault_p(coach, ev['why']))
        state = 'none' if not r['ok'] else ('full' if r.get('full_body') else 'partial'); paused = not r.get('visible', True)
        cand, run = self._vis_cand; self._vis_cand = (state, run + 1 if cand == state else 1)
        if (self._vis_cand[1] >= self.vis_hold and state != self._vis_emitted) or paused != self._paused:
            self._vis_emitted = state; self._paused = paused
            self._emit('visibility', state=state, paused=paused, no_pose_s=round(r.get('no_pose_s', 0.0), 1))
        if r.get('gesture_next'): self._emit('gesture', name='tpose')
        if r.get('probe_fire'):
            ev = dict(arms_up=True)
            if self.live:
                span = int(r.get('probe_span', 1)); t_first = self._recv_hist[-span] if len(self._recv_hist) >= span else self._recv_hist[-1]
                now = time.monotonic(); t_recv = self._t + self.t0
                ev['on_box_ms'] = round(1000 * (now - t_recv), 1)     # this frame received -> this event (coach_agent's receive-to-decision leg; no audio leg)
                ev['span_ms'] = round(1000 * (now - t_first), 1)      # from the FIRST up frame's receipt (the probe confirms over `span` frames)
                ev['t_first_up'] = t_first
            self._emit('probe', **ev)

    @staticmethod
    def _confidence(rc):
        """RepCounter2: agree = share of the vote's slots (top_k, unfired trackers vote 0) whose rebased count reached the new count;
        amp = median reference rep amplitude of the voting trackers (torso lengths / normalised joint angle). E1 counter: None."""
        if rc is None or not hasattr(rc, 'tr'): return None
        try:
            sel = rc.sel if isinstance(rc.sel, list) else ([rc.sel] if isinstance(rc.sel, str) else [])
            if not sel: return None
            counts = [rc.tr[n].rebased(rc.amp_frac) - rc.skip for n in sel if n in rc.tr]
            slots = rc.top_k if getattr(rc, 'pad', True) else max(len(counts), 1)
            agree = sum(1 for x in counts if x >= rc.count) / max(slots, 1)
            amps = [rc.tr[n].A_ref for n in sel if n in rc.tr and rc.tr[n].A_ref is not None]
            return dict(agree=round(min(agree, 1.0), 2), amp=(round(float(np.median(amps)), 3) if amps else None), trackers=len(counts))
        except Exception as e:
            log.debug('confidence failed: %r', e); return None

    @staticmethod
    def _fault_p(coach, label):
        """The Coach's EMA probability of the fault label (None for a side-rule pseudo-label or a tempo cue)."""
        try:
            labs = list(coach.clf.labs)
            return round(float(coach.ema[labs.index(label)]), 3) if (coach.ema is not None and label in labs) else None
        except Exception:
            return None

    def _session_event(self, force=False):
        s = self.session
        ex = spoken_name(self.pipe.exercise) if self.pipe.exercise else (spoken_name(s.name) if s.name and s.state in ('wait', 'active') else None)
        key = (s.state, ex, s.target, s.k, bool(s.done))
        if force or key != self._sess_key:
            self._sess_key = key
            self._emit('session', state=s.state, exercise=ex, target=s.target, count=s.last_count, item=(s.k if s.mode == 'plan' else None), done=bool(s.done), mode=s.mode)

    def _status(self, r, t_mono):
        w = self._win
        if w['t'] is None: w['t'] = self._t; w['recv0'] = self.frames_received
        w['n'] += 1; w['pose'] += r['pose_ms']; w['clf'] += r['clf_ms']
        if self.live: d = t_mono - (self._t + self.t0); w['d'].append(d); self.delays.append(d)
        el = self._t - w['t']
        if el < self.cfg.status_every_s: return
        n = max(w['n'], 1); fps = w['n'] / el; pose_ms = w['pose'] / n; clf_ms = w['clf'] / n
        recv = ((self.frames_received - (w['recv0'] or 0)) / el) if self.frames_received is not None else None
        ev = dict(fps_recv=(round(recv, 1) if recv is not None else None), fps_processed=round(fps, 1), pose_ms=round(pose_ms, 1), clf_ms=round(clf_ms, 1),
                  pose_model=self.pipe.pose_variant, state=self.session.state, exercise=(spoken_name(self.pipe.exercise) if self.pipe.exercise else None), count=self.session.last_count)
        if self.live and w['d']: ev['recv_to_decision_p50_ms'] = round(1000 * float(np.median(w['d'])), 0)
        if self.frames_dropped is not None: ev['dropped'] = self.frames_dropped
        if self.audio_stats is not None:
            try: ev.update(self.audio_stats() or {})
            except Exception as e: log.debug('audio_stats failed: %r', e)
        if self.live and self.cfg.auto_lite and self.pipe.pose_variant != 'lite' and recv is not None and recv >= 8 and fps < 8.5 and pose_ms > 60:
            self.pipe.switch_pose('lite'); self.note = 'POSE->lite (full too slow for 10 fps here)'; ev['note'] = self.note
        self._emit('status', **ev)
        self._win = dict(t=self._t, n=0, pose=0.0, clf=0.0, d=[], recv0=self.frames_received)

    # ---- lifecycle ----
    @property
    def done(self):
        return bool(self.session.done)

    def summary(self):
        return dict(frames=self.n, pose_ok=self.pipe.n_ok, pose_ms=round(self.pipe.pose_ms_sum / max(self.pipe.n, 1), 1), pose_model=self.pipe.pose_variant,
                    clf_ms=round(self.pipe.clf_ms_sum / max(self.pipe.n, 1), 1), events=dict(self.n_events), count=self.session.last_count, state=self.session.state)

    def close(self):
        if self.closed: return
        self.closed = True
        for w in self._writers: w.close()
        self.pipe.close()
