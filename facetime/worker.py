"""LiveKit Agents worker for the hackathon starter: the coach's eyes (facetime/perception.py) and, by default, its fast mouth
(facetime/audio_out.py CuePlayer, unchanged) in ONE participant with ONE audio track, dispatched by name the way the Chert CLI
dispatches agents (reports/STARTER_PREREQS_REVIEW.md section 1: `chert dev --agent NAME` creates a room with a RoomAgentDispatch for
NAME; the call is accepted once a participant of kind AGENT is in the room; the bridge joins as chert-bridge-<call_id>).

Registers with WorkerOptions(agent_name=AGENT_NAME, default "brodie-coach"). Per job: connect (subscribe all), publish the cue audio
track at once unless --events-only / EVENTS_ONLY=1 (the bridge binds to the one participant with an unmuted audio track), then bind
to the first non-agent participant that publishes video, whoever it is (chert-bridge-<call_id>, a phone page, a fake bridge), feed the
newest frame to Perception in a worker thread with the same arrival-aligned 10 fps sampling as coach_agent, and keep the job joined
until the room closes. A job with no caller (`chert doctor --agent`, prewarm: metadata {"chert_warm": true}) does nothing but stay
joined until somebody publishes video; nothing is said or started on job start. Models and the cue bank are loaded in the prewarm
hook, so a job start costs no model load (one idle preloaded process is kept, --idle-processes).
Job end (Mac test 2026-10-09, reports/MAC_TEST_2026-10-08.md section 5): livekit-agents runs shutdown callbacks only AFTER the entrypoint
returns, so run() ends itself: on the room's `disconnected` event, when the bound participant leaves and nobody rebinds within
--leave-grace-s (default 2 s: a rejoin binds again), when connect() fails (a doctor or warm room torn down before the peer connection
completes), on --max-s and on a perception error. It then cleans up with bounded waits (feed thread join 1 s off the loop, VideoStream
closed, player stopped, cue in flight at most 1.5 s and only while still connected) and calls ctx.shutdown() so the framework finalises.

Events (one JSON object each, schema in facetime/perception.py and docs/STARTER_EVENTS.md) go to
  - an in-process asyncio.Queue, job.events, for a conversational layer (register a coroutine with @consumer, below);
  - a JSONL log, --event-log / RTCOACH_EVENT_LOG (default data/tmp/worker_events.jsonl; each line also carries room, call_id, job);
  - optionally the room data channel as JSON (--publish-data, topic --data-topic, off by default: Chert rooms hold two participants,
    the data channel is for rooms we own).
Cues the player finished or dropped go to --cue-log / RTCOACH_CUE_LOG (the coach_agent --log record shape) and to stdout.

Plan: --plan, else env PLAN, else starter/plan.txt (the hackathon script; ships with the demo plan), else the demo plan (DEFAULT_PLAN);
--exercise for one exercise, --auto for auto-detect. Job metadata is read for chert_call and media and logged (never acted on).

Voice (--voice gemini, default off: the default and events-only modes are untouched): facetime/voice.py runs Gemini Live (voice Puck)
in this same participant, its audio mixed into the cue player's one track through CuePlayer.attach_mixer() (ducked about 10 dB under a
count) and gated to the gaps between sets (session state wait/rest/done, a gesture or an armed probe outside a set, a caller's question
during a rest; the caller's audio is not forwarded during a set and a reply still streaming when a set starts is cut: interrupted,
buffers cleared, later frames dropped). The model session starts when a caller is bound (never on a warm job). Persona from
--persona (starter/persona.md); handlers from --handlers, default starter/handlers.py whenever it exists, in EVERY mode (on_event(event,
say_now, ask_voice); without the voice ask_voice returns None); model turns are logged as events of type voice {text, t, role}.
Latency probe: --latency-probe / RTCOACH_LATENCY_PROBE=1 (off by default; probe events are emitted either way; never acted on during
an active set).

Beyond reps (docs/STARTER_README.md "Your own coach: beyond reps"): --perception off runs NO Perception (no pose model in prewarm,
no session script, no rep/cue/fault events): the worker still joins, publishes its audio track, binds to the first remote participant
that publishes video (or, with no video, audio), starts the model session on the bind, emits one synthetic session event
{state: "open", exercise: null, count: 0, mode: "voice"} once the session is up (handlers greet on it) and a status event (no pose
fields) every 5 s, and ends on the same paths (leave grace, disconnect, --max-s, connect failure). The Floor is then ALWAYS OPEN
(--floor open, the default without perception): the caller's audio is always forwarded, ask_voice fires at once, nothing is cut;
say_now still plays bank strings. --vision (with --voice gemini) tees the caller's video track into the model session at about
1 fps (--vision-fps, at most 2): facetime/voice.py VideoTee, a livekit.agents VideoInput set as AgentSession.input.video, pulled by
the framework into the realtime session's push_video (JPEG). A warm job never starts a session or forwards a frame. The plan file
is then appended to the persona as free text (a recipe, a checklist), not parsed as exercises.
  .venv/bin/python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/meal_prep/persona.md \
      --plan-file starter/examples/meal_prep/recipe.md --handlers starter/examples/meal_prep/handlers.py dev
Run:
  .venv/bin/python -m facetime.worker --help
  .venv/bin/python -m facetime.worker [--plan "squats:10,pushups:8"] [--events-only] [--publish-data] dev     # LIVEKIT_URL/API_KEY/API_SECRET in env
  AGENT_NAME=brodie-coach EVENTS_ONLY=1 .venv/bin/python -m facetime.worker start
  GEMINI_API_KEY=... .venv/bin/python -m facetime.worker --voice gemini dev                                      # + GEMINI_LIVE_MODEL, GEMINI_VOICE
The flags before the LiveKit subcommand are ours; `dev`, `start`, `connect`, `download-files` and their options are the LiveKit Agents CLI's.
"""
import argparse, asyncio, collections, json, logging, os, sys, threading, time
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from facetime import DATA, AUDIO_TRACK_NAME
from facetime.pipeline import FPS
from facetime.perception import Perception, Config, add_arguments, build_pipeline, EVENT_SCHEMA_VERSION
from facetime.coach_agent import frame_to_rgb              # the same frame conversion as the demo coach

# livekit-agents lets a plugin register (on import) only on the main thread. On Windows the jobs run in threads, so importing the
# Gemini plugin lazily when a caller binds fails with "Plugins must be registered on the main thread" and the voice never starts.
# Import it here, at module load, which is on the main thread (also in forkserver job processes).
try: import livekit.plugins.google  # noqa: F401
except ImportError: pass

log = logging.getLogger('worker')
ARGS_ENV = 'RTCOACH_WORKER_ARGS'      # main() hands the parsed flags to the job processes as JSON (forkserver children re-import this module)
DEFAULT_PLAN = 'squats:10,pushups:8,jumping jacks:20,lunges:10'
STARTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'starter')
AGENT_KIND = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
BRIDGE_PREFIX = 'chert-bridge-'
LK_COMMANDS = ('dev', 'start', 'connect', 'download-files', 'console')
CONSUMERS = []                        # coroutines consumer(job) started per job; they read job.events (see docs/STARTER_EVENTS.md)
CURRENT_JOB = None                    # the job running in this process (one per process), for handlers that need the room or the avatar


def consumer(fn):
    """Register `async def fn(job)`: started as a task once the job is connected; read events with `await job.events.get()`."""
    CONSUMERS.append(fn); return fn


def truthy(s):
    return str(s or '').strip().lower() in ('1', 'true', 'yes', 'on')


def build_parser():
    ap = argparse.ArgumentParser(prog='python -m facetime.worker', add_help=False, description=__doc__.split('\n')[0],
                                 epilog='After these flags comes the LiveKit Agents CLI subcommand: dev (registers under AGENT_NAME with debug logging; restart it '
                                        'after a code change, livekit-agents 1.8.5 has no hot reload), start (production), connect --room R (join one room directly), '
                                        'download-files. The arms-up latency probe is off unless --latency-probe (env RTCOACH_LATENCY_PROBE=1) and never acts during '
                                        'an active set. Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, AGENT_NAME, PLAN, EVENTS_ONLY, RTCOACH_EVENT_LOG, '
                                        'RTCOACH_CUE_LOG, RTCOACH_LATENCY_PROBE, GEMINI_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE, GEMINI_LANGUAGE.')
    add_arguments(ap)
    for act in ap._actions:                                                 # the perception flag, with the worker's default (env) and what the worker does with it
        if act.dest == 'latency_probe':
            act.default = truthy(os.environ.get('RTCOACH_LATENCY_PROBE'))
            act.help = ('act on the arms-up probe (env RTCOACH_LATENCY_PROBE=1): beep through the cue player and open the voice floor for a few seconds, '
                        'never during an active set; off by default (a jumping jack is an arm raise). Probe events are emitted and logged either way')
    ap.add_argument('--auto', action='store_true', help='auto-detect mode (no plan, no exercise); without it and without --plan/--exercise/PLAN the demo plan runs')
    ap.add_argument('--events-only', action='store_true', default=truthy(os.environ.get('EVENTS_ONLY')), help='publish no audio at all: events only (env EVENTS_ONLY=1)')
    ap.add_argument('--publish-data', action='store_true', help='also publish every event as JSON on the room data channel (off by default)')
    ap.add_argument('--data-topic', default='brodie.events')
    ap.add_argument('--event-log', default=os.environ.get('RTCOACH_EVENT_LOG', os.path.join(DATA, 'tmp', 'worker_events.jsonl')), help='JSONL event log (appended; env RTCOACH_EVENT_LOG)')
    ap.add_argument('--cue-log', default=os.environ.get('RTCOACH_CUE_LOG', os.path.join(DATA, 'tmp', 'worker_cues.jsonl')), help='JSONL log of cues played or dropped (env RTCOACH_CUE_LOG)')
    ap.add_argument('--agent-name', default=os.environ.get('AGENT_NAME', 'brodie-coach'), help='dispatch name (env AGENT_NAME); "" = automatic dispatch to every room (local tests)')
    ap.add_argument('--audio-queue-ms', type=int, default=100, help='AudioSource queue for this worker (100; the Mac saw an underrun at 60 during one-per-second jack counts; coach_agent keeps 60)')
    ap.add_argument('--leave-grace-s', type=float, default=2.0, help='after the bound participant leaves, end the job unless somebody rebinds within this (negative = never end on a leave)')
    ap.add_argument('--no-probe-beep', dest='probe_beep', action='store_false', help='with --latency-probe: open the voice floor on the probe but do not beep')
    ap.add_argument('--video-quality', default='low', choices=['low', 'medium', 'high', 'none'], help='simulcast layer to request when the publisher simulcasts')
    ap.add_argument('--max-s', type=float, default=None, help='leave the room this long after the first video frame (default: stay until the room closes)')
    ap.add_argument('--idle-processes', type=int, default=int(os.environ.get('RTCOACH_IDLE_PROCS', '1')), help='preloaded job processes kept ready (models loaded in prewarm)')
    ap.add_argument('--voice', default=os.environ.get('VOICE', 'off'), choices=['off', 'gemini'], help='conversational layer between sets: gemini = Gemini Live through the cue track (facetime/voice.py; env VOICE)')
    ap.add_argument('--voice-model', default=os.environ.get('GEMINI_LIVE_MODEL') or None, help='Gemini Live model id (env GEMINI_LIVE_MODEL; default facetime.voice.DEFAULT_MODEL)')
    ap.add_argument('--voice-name', default=os.environ.get('GEMINI_VOICE', 'Puck'), help='Gemini Live voice (env GEMINI_VOICE; Puck is the cue bank voice)')
    ap.add_argument('--persona', default=os.environ.get('PERSONA_FILE', os.path.join(STARTER_DIR, 'persona.md')), help='the voice persona prompt file (starter/persona.md)')
    ap.add_argument('--handlers', default=os.environ.get('HANDLERS_FILE') or None, help='a module with on_event(event, say_now, ask_voice); default starter/handlers.py whenever that file exists, in every mode '
                                                                                        '(without --voice gemini, ask_voice returns None); --handlers none loads nothing')
    ap.add_argument('--log-caller', action='store_true', help='also emit and log the caller\'s transcribed turns as voice events (role caller); off by default: the caller\'s words never leave the model session (docs/STARTER_README.md section 8)')
    ap.add_argument('--plan-file', default=os.environ.get('PLAN_FILE', os.path.join(STARTER_DIR, 'plan.txt')), help='the plan when neither --plan nor PLAN is given (starter/plan.txt); '
                                                                                                                      'with --perception off the file is appended to the persona as free text (a recipe)')
    ap.add_argument('--perception', default=os.environ.get('PERCEPTION', 'on'), choices=['on', 'off'], help='off = no pose model, no rep counter, no session script (env PERCEPTION): the worker '
                                                                                                           'joins, binds to the caller, starts the voice and emits one synthetic session {state: open} event; the floor is then always open')
    ap.add_argument('--floor', default=os.environ.get('FLOOR') or None, choices=['gated', 'open'], help='gated = the voice speaks between sets only (the default with perception on); '
                                                                                                        'open = always allowed to speak, nothing is ever cut (the default with --perception off; env FLOOR)')
    ap.add_argument('--barge-in', action='store_true', default=os.environ.get('BARGE_IN') == '1',
                    help="keep the caller's audio flowing while the voice talks (conversation; the phone's echo cancellation keeps the model "
                         'from hearing itself). Default: the echo guard mutes the caller while any audio plays (env BARGE_IN=1)')
    ap.add_argument('--avatar', default=os.environ.get('AVATAR') or None, help='publish a face video track for this Verity character (verity/variants.py key; env AVATAR): Chert expects audio + video from the agent')
    ap.add_argument('--text', action='store_true', default=truthy(os.environ.get('VOICE_TEXT')), help='with --voice gemini: typed chat in (lk.chat) and transcripts out (lk.transcription), for the web app (env VOICE_TEXT=1)')
    ap.add_argument('--vision', action='store_true', default=truthy(os.environ.get('VISION')), help='with --voice gemini: tee the caller\'s video into the model session at --vision-fps (env VISION=1)')
    ap.add_argument('--vision-fps', type=float, default=float(os.environ.get('VISION_FPS', '1') or 1), help='frames per second handed to the model with --vision (default 1, at most 2; JPEG per frame)')
    ap.add_argument('-h', '--help', action='store_true', help='show this help and exit')
    return ap


def perception_off(a):
    return str(getattr(a, 'perception', 'on') or 'on').strip().lower() == 'off'


def read_plan_text(path):
    """--perception off: the plan file as free text (a recipe, a checklist), stripped; empty or missing -> None."""
    if not path: return None
    try:
        with open(path) as f: return f.read().strip() or None
    except OSError:
        return None


def resolve_mode(a):
    """--plan, else PLAN, else starter/plan.txt, else the demo plan; --exercise and --auto take precedence over the default plan.
    Handlers: --handlers PATH, else starter/handlers.py whenever it exists (every mode, so a team's on_event runs with say_now even
    without the voice), `none` for nothing."""
    if perception_off(a):                       # no session script: the plan is free text for the persona (a recipe), kept across the two parses
        a.plan_text = getattr(a, 'plan_text', None) or a.plan or os.environ.get('PLAN') or read_plan_text(getattr(a, 'plan_file', None))
        a.plan = None; a.exercise = None; a.floor = getattr(a, 'floor', None) or 'open'
    else:
        a.plan_text = None; a.floor = getattr(a, 'floor', None) or ('open' if getattr(a, 'dance', False) else 'gated')   # dance: no sets, the voice may speak any time
        if a.exercise: a.plan = None
        elif getattr(a, 'auto', False): a.plan = None
        elif getattr(a, 'dance', False): a.plan = None     # dance mode runs no workout script; the demo plan leaked into the persona
        else: a.plan = a.plan or os.environ.get('PLAN') or read_plan_file(getattr(a, 'plan_file', None)) or DEFAULT_PLAN   # ("let's start with ten squats", 11:43 call)
    a.vision_fps = max(0.1, min(float(getattr(a, 'vision_fps', 1.0) or 1.0), 2.0))
    h = getattr(a, 'handlers', None)
    if h and str(h).strip().lower() in ('none', 'off'): a.handlers = 'none'          # kept as the string: the job process resolves the flags again
    elif not h and os.path.exists(os.path.join(STARTER_DIR, 'handlers.py')): a.handlers = os.path.join(STARTER_DIR, 'handlers.py')
    return a


def handlers_path(a):
    """The handlers module to load, None for --handlers none / nothing."""
    h = getattr(a, 'handlers', None)
    return None if not h or str(h).strip().lower() in ('none', 'off') else h


def read_plan_file(path):
    """starter/plan.txt: one line like "squats:10,pushups:8" (comments with #); empty or missing -> None."""
    if not path: return None
    try:
        lines = [l.strip() for l in open(path) if l.strip() and not l.lstrip().startswith('#')]
        return ','.join(lines) if lines else None
    except OSError:
        return None


def load_args():
    """The flags main() parsed, from the environment (a job process is a fresh interpreter); defaults when run without main()."""
    a = build_parser().parse_args([])
    try: d = json.loads(os.environ.get(ARGS_ENV) or '{}')
    except ValueError: d = {}
    for k, v in d.items(): setattr(a, k, v)
    return resolve_mode(a)


def parse_metadata(md):
    """Job metadata is JSON from the dispatcher (Chert: chert_warm, chert_call, media); anything else is logged and ignored."""
    if not md: return {}
    try:
        d = json.loads(md); return d if isinstance(d, dict) else {'metadata': d}
    except ValueError:
        return {'metadata_raw': md[:200]}


def prewarm(proc: JobProcess):
    """Runs in each job process before any job: models, trigger, thresholds and (unless events-only) the cue bank."""
    a = load_args(); cfg = Config.from_args(a); t = time.time()
    proc.userdata['args'] = a; proc.userdata['cfg'] = cfg
    if perception_off(a):                                      # no pose model, no classifier: nothing to load but the bank
        proc.userdata['pipe'] = None
    else:
        proc.userdata['pipe'] = build_pipeline(cfg)
    if not a.events_only:
        from facetime.audio_out import CueBank
        proc.userdata['bank'] = CueBank()
    if getattr(a, 'avatar', None):                             # the face frames, decoded before a call (1.6 s on the call's loop otherwise)
        from verity.avatar import sprites, real_frames
        for k in {a.avatar, 'cruelty'}: sprites(k) or real_frames(k)
    if perception_off(a):
        print(f'[worker] prewarm: loaded in {time.time() - t:.1f} s (perception OFF: no pose model, no session script; voice {a.voice}, vision {"on" if a.vision else "off"}, '
              f'floor {a.floor}, audio {"OFF (events only)" if a.events_only else "cue bank"})', flush=True)
    else:
        print(f'[worker] prewarm: loaded in {time.time() - t:.1f} s (pose {proc.userdata["pipe"].pose_variant}, counter {cfg.counter}, trigger {proc.userdata["pipe"].trigger_path or "off"}, '
              f'mode {cfg.mode} ({a.plan or a.exercise or "auto-detect"}), audio {"OFF (events only)" if a.events_only else "cue bank"})', flush=True)


def describe_mode(a):
    """The mode words for the console lines: plan/single/auto with perception, 'voice' without."""
    if perception_off(a): return 'voice (perception off' + (', vision on' if getattr(a, 'vision', False) else '') + f', floor {a.floor})'
    return f'{"plan" if a.plan else ("single" if a.exercise else "auto")} ({a.plan or a.exercise or "auto-detect"})'


async def entrypoint(ctx: JobContext):
    ud = ctx.proc.userdata
    a = ud.get('args') or load_args(); cfg = ud.get('cfg') or Config.from_args(a)
    pipe = ud.pop('pipe', None)
    if pipe is None and not perception_off(a): pipe = build_pipeline(cfg)   # single use: a Perception owns it until the job ends
    bank = ud.get('bank')
    if bank is None and not a.events_only:
        from facetime.audio_out import CueBank
        bank = CueBank()
    job = CoachJob(ctx, a, cfg, pipe, bank, parse_metadata(ctx.job.metadata))
    await job.run()


class CoachJob:
    """One dispatched job: the room, the optional cue player, the Perception and its feed thread, the event sinks."""
    def __init__(self, ctx, a, cfg, pipe, bank, md):
        self.ctx = ctx; self.a = a; self.cfg = cfg; self.pipe = pipe; self.bank = bank; self.md = md
        self.room = ctx.room; self.loop = None
        self.warm = bool(md.get('chert_warm'))
        self.events = asyncio.Queue()                 # every event, for an in-process conversational layer
        self.player = None; self.player_task = None; self.cue_log = None
        self.per = None; self.thread = None; self.stop = threading.Event(); self.done = asyncio.Event(); self.closed = False
        self.latest = None; self.seq = 0; self.processed = 0; self.dropped = 0; self.idle = 0; self.frame_event = threading.Event()
        self.video_task = None; self.bound = None; self.call_id = None; self.frame_size = None; self.first_frame_t = None
        self.worker_error = None; self.cues_played = 0; self.ts_mode = None
        self.voice = None; self.voice_seq = 0; self.event_writer = None; self.consumer_tasks = []
        self.end_reason = None; self.leave_task = None; self.ctx_shutdown_called = False; self.t_end = None
        self.session_state = None; self.probes_ignored = 0        # the probe (--latency-probe) never beeps or opens the floor during an active set
        self.voice_mode = perception_off(a)                       # --perception off: no Perception; the worker emits its own session/status events
        self.t0 = time.monotonic(); self.ev_seq = 0; self.vision = None; self.vision_logged = False; self.session_opened = False; self.open_task = None

    # ---- lifecycle ----
    async def run(self):
        self.loop = asyncio.get_running_loop(); ctx = self.ctx; a = self.a
        global CURRENT_JOB; CURRENT_JOB = self; self.avatar = None; self.avatar_task = None
        import facetime.worker as _module; _module.CURRENT_JOB = self      # run as `python -m facetime.worker` this module is __main__; handlers import the package one
        ctx.add_shutdown_callback(self.aclose)
        if not a.events_only:
            from facetime.audio_out import CuePlayer
            self.player = CuePlayer(bank=self.bank, on_record=self.log_cue, queue_ms=a.audio_queue_ms)
            os.makedirs(os.path.dirname(os.path.abspath(a.cue_log)) or '.', exist_ok=True); self.cue_log = open(a.cue_log, 'a')
        if getattr(a, 'voice', 'off') == 'gemini' or handlers_path(a):
            self.voice = self.build_voice()
        self.room.on('track_subscribed', self.on_track_subscribed)
        self.room.on('participant_connected', lambda p: print(f'[worker] participant joined: {p.identity} (kind {rtc.ParticipantKind.Name(p.kind)})', flush=True))
        self.room.on('participant_disconnected', self.on_participant_disconnected)
        self.room.on('disconnected', self.on_room_disconnected)
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError): raise
            print(f'[worker] job {ctx.job.id}: connect failed ({type(e).__name__}: {str(e)[:120]}): the room went away before the peer connection completed '
                  f'(a doctor or warm room torn down early); leaving', flush=True)
            self.end_reason = 'connect failed'; await self.aclose(); self.shutdown_ctx(self.end_reason); return
        md_note = ', '.join(f'{k}={json.dumps(self.md[k])[:80]}' for k in ('chert_warm', 'chert_call', 'media') if k in self.md) or 'none'
        print(f'[worker] job {ctx.job.id}: joined room {self.room.name!r} as {self.room.local_participant.identity!r} (agent {a.agent_name!r}); '
              f'mode {describe_mode(a)}; audio {"OFF (events only)" if a.events_only else "cue player"}; '
              f'metadata: {md_note}; participants {list(self.room.remote_participants)}', flush=True)
        if self.player is not None:
            await self.room.local_participant.publish_track(self.player.track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            self.player_task = asyncio.create_task(self.player.run())
            print(f'[worker] published audio track {AUDIO_TRACK_NAME}', flush=True)
        if getattr(a, 'avatar', None):                         # the face: published with the audio, before any caller (Chert binds on both)
            from verity.avatar import Avatar, tap
            from verity.variants import VARIANTS
            self.avatar = Avatar(VARIANTS[a.avatar]); self.avatar_task = await self.avatar.publish(self.room)
            if self.voice is not None and self.voice.sink is not None: tap(self.avatar, self.voice.sink)   # the mouth follows the model's voice
            print(f'[worker] published video track verity-face ({a.avatar})', flush=True)
        if self.voice_mode:                                       # no Perception: the worker's own session/status events, logged the same way
            from facetime.perception import JsonlWriter
            self.event_writer = JsonlWriter(a.event_log, extra=dict(room=self.room.name, call_id=self.call_id, job=ctx.job.id))
        else:
            self.per = Perception(self.cfg, pipe=self.pipe, say=(self.say_from_worker if self.player is not None else None), clip_s=(self.bank.duration if self.bank is not None else None))
            self.per.on_event(self.on_event)
            self.event_writer = self.per.log_to(a.event_log, extra=dict(room=self.room.name, call_id=self.call_id, job=ctx.job.id))
            if self.player is not None:
                self.per.audio_stats = lambda: dict(underruns=self.player.underruns, cues_played=self.cues_played, cues_dropped=self.player.n_dropped)
        print(f'[worker] events (schema v{EVENT_SCHEMA_VERSION}) -> {a.event_log}' + (f' and data topic {a.data_topic!r}' if a.publish_data else ''), flush=True)
        if self.warm:
            print('[worker] warm job (chert_warm): no caller; staying joined, nothing is said until a participant publishes video' + (' or audio' if self.voice_mode else ''), flush=True)
        if self.voice_mode and getattr(a, 'text', False) and self.room.name.startswith('web-'):   # web app: a caller may only type (mic and camera off)
            def web_caller(p):
                if self.bound is None and not self.closed and p.kind != AGENT_KIND: self.bind(p, 'web caller (typed chat; mic/camera optional)')
            self.room.on('participant_connected', web_caller)
            for p in list(self.room.remote_participants.values()): web_caller(p)
        if self.voice_mode:
            self.consumer_tasks = [asyncio.create_task(self._status_loop(), name='status-loop')]
        else:
            self.thread = threading.Thread(target=self.feed, daemon=True, name='perception-feed'); self.thread.start()
            # dance mode: the worker's own 5 s status tick too, so the handlers hear from a caller who sends no video (camera off)
            self.consumer_tasks = [asyncio.create_task(self._status_loop(), name='status-loop')] if getattr(a, 'dance', False) else []
        self.consumer_tasks += [asyncio.create_task(fn(self), name=f'consumer-{fn.__name__}') for fn in CONSUMERS]
        if self.voice is not None:
            self.consumer_tasks.append(asyncio.create_task(self.voice.run(), name='consumer-voice'))
        await self.done.wait()
        reason = self.end_reason or (f'perception error: {self.worker_error!r}' if self.worker_error is not None else 'done')
        await self.aclose(); self.shutdown_ctx(reason)

    def end(self, reason):
        """End the job (event loop): run() wakes, cleans up and returns, then tells the framework (shutdown_ctx)."""
        if self.end_reason is None: self.end_reason = reason
        if self.t_end is None: self.t_end = time.monotonic()
        self.done.set()

    def shutdown_ctx(self, reason):
        if self.ctx_shutdown_called: return
        self.ctx_shutdown_called = True
        try: self.ctx.shutdown(reason=reason)
        except Exception as e: log.debug('ctx.shutdown: %r', e)

    def on_room_disconnected(self, *args):
        try: why = rtc.DisconnectReason.Name(args[0]) if args else 'unknown'
        except Exception: why = str(args[0]) if args else 'unknown'
        print(f'[worker] room disconnected ({why}): leaving', flush=True); self.end(f'room disconnected ({why})')

    async def _leave_grace(self):
        await asyncio.sleep(max(0.0, self.a.leave_grace_s))
        self.leave_task = None
        if self.bound is None and not self.closed:
            print(f'[worker] caller left and nobody rebound within {self.a.leave_grace_s:g} s: leaving', flush=True); self.end('caller left')

    def room_connected(self):
        try: return self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED
        except Exception: return True

    async def aclose(self, *_):
        """Idempotent cleanup (run() at its end, and the framework's shutdown callback): every wait is bounded, nothing blocks the loop."""
        if self.closed: return
        self.closed = True; self.stop.set(); self.done.set(); t0 = time.monotonic()
        if self.t_end is None: self.t_end = t0
        if self.leave_task is not None: self.leave_task.cancel()
        if getattr(self, 'avatar_task', None) is not None: self.avatar_task.cancel()
        if self.open_task is not None and not self.open_task.done(): self.open_task.cancel()
        for t in getattr(self, 'consumer_tasks', []): t.cancel()
        if self.video_task is not None and not self.video_task.done():                 # closes the VideoStream (read_video's finally)
            self.video_task.cancel()
            try: await asyncio.wait_for(asyncio.gather(self.video_task, return_exceptions=True), 1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError): pass
        if self.thread is not None and self.thread.is_alive():                            # the feed loop polls stop every 50 ms
            await asyncio.get_running_loop().run_in_executor(None, self.thread.join, 1.0)
        if self.voice is not None:
            try: await asyncio.wait_for(self.voice.aclose(), 3)
            except Exception as e: log.warning('voice close: %r', e)
        if self.player is not None:
            if self.room_connected():
                for _ in range(15):                                                       # let the cue in flight finish (max 1.5 s)
                    if not self.player.playing and self.player.pending is None: break
                    await asyncio.sleep(0.1)
            self.player.stop()
            if self.player_task is not None:
                try: await asyncio.wait_for(asyncio.gather(self.player_task, return_exceptions=True), 1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError): self.player_task.cancel()
            src = getattr(self.player, 'source', None)                                    # drop the AudioSource's FFI handle here, not in a __del__
            if hasattr(src, 'aclose'):
                try: await asyncio.wait_for(src.aclose(), 1.0)
                except Exception as e: log.debug('audio source close: %r', e)
        if self.per is not None: s = self.per.summary(); self.per.close()
        else:
            s = {}
            if self.voice_mode:
                s = dict(mode='voice', events=self.ev_seq, session_opened=self.session_opened)
                if self.vision is not None: s.update(vision_offered=self.vision.n_offered, vision_sent=self.vision.n_sent, vision_early=self.vision.n_early)
                if self.event_writer is not None:
                    try: self.event_writer.close()
                    except Exception as e: log.debug('event writer close: %r', e)
            if self.pipe is not None:                                                   # connect failed before Perception took the pipe: close the
                try: self.pipe.close()                                                  # pose landmarker here (an unclosed one hangs in __del__)
                except Exception as e: log.debug('pipe close: %r', e)
        if self.cue_log: self.cue_log.close()
        print(f'[worker] job {self.ctx.job.id} done ({self.end_reason or "shutdown"}; cleanup {1000 * (time.monotonic() - t0):.0f} ms, '
              f'{1000 * (time.monotonic() - self.t_end):.0f} ms after the end trigger): frames received {self.seq}, processed {self.processed}, dropped {self.dropped}, idle ticks {self.idle}; {s}', flush=True)

    # ---- video ----
    def on_track_subscribed(self, track, pub, participant):
        own = participant.kind == AGENT_KIND or participant.identity == self.room.local_participant.identity
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            # --perception off, and dance mode: a caller with audio and no video (camera off, permission denied) is still a caller; the
            # model session starts on it so the character can ask for the camera (12:42 call: 0 frames received, Brodie silent)
            if track.kind == rtc.TrackKind.KIND_AUDIO and (self.voice_mode or getattr(self.a, 'dance', False)) and self.bound is None and not own and not self.closed:
                self.bind(participant, f'audio {pub.name!r} (no video yet)')
            return
        if own:
            print(f'[worker] video from agent participant {participant.identity!r} ignored', flush=True); return
        if self.video_task is not None and not self.video_task.done():
            print(f'[worker] video track {pub.name!r} from {participant.identity!r} ignored: already bound to {self.bound!r}', flush=True); return
        self.bind(participant, f'video {pub.name!r} {pub.width}x{pub.height}, simulcast {getattr(pub, "simulcasted", False)}')
        if self.a.video_quality != 'none' and getattr(pub, 'simulcasted', False):
            try: pub.set_video_quality(rtc.VideoQuality.Value('VIDEO_QUALITY_' + self.a.video_quality.upper()))
            except Exception as e: print(f'[worker] set_video_quality failed: {e}', flush=True)
        self.video_task = asyncio.create_task(self.read_video(track))

    def bind(self, participant, what):
        """A caller: remember it (the leave grace is keyed on it), start the model session once (never on a warm job: this only runs
        when somebody publishes), and in voice mode emit the synthetic session {state: open} event once the session is up."""
        self.bound = participant.identity
        if participant.identity.startswith(BRIDGE_PREFIX): self.call_id = participant.identity[len(BRIDGE_PREFIX):]
        extra = getattr(self.event_writer, 'extra', None)                   # the JSONL lines carry the call id from the bind on (null before it)
        if isinstance(extra, dict): extra['call_id'] = self.call_id
        print(f'[worker] bound to {participant.identity!r} (call id {self.call_id or "n/a"}): {what}', flush=True)
        if self.voice_mode:
            if not self.session_opened and self.open_task is None:
                self.open_task = asyncio.create_task(self._open_session(participant.identity), name='voice-start')
        elif self.voice is not None and self.voice.model is not None and not self.voice.started:
            asyncio.create_task(self.voice.on_caller(participant.identity, room=self.room), name='voice-start')

    async def _open_session(self, identity):
        """Voice mode: the model session (when there is one), then ONE session event {state: open, mode: voice} for the handlers."""
        try:
            if self.voice is not None and self.voice.model is not None:
                await self.voice.on_caller(identity, room=self.room)
            if self.closed or self.session_opened: return
            self.session_opened = True
            self.emit('session', state='open', exercise=None, target=None, count=0, item=None, done=False, mode='voice')
        finally:
            self.open_task = None

    def video_frames(self, track):
        """The VideoStream of the bound track (RGB24 for Perception; the tee hands the same frames to the model). Tests replace it."""
        return rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)

    async def read_video(self, track):
        stream = self.video_frames(track)
        try:
            async for ev in stream:
                t_mono = time.monotonic()
                self.seq += 1; self.latest = (ev.frame, ev.timestamp_us, t_mono, time.time(), self.seq); self.frame_event.set()
                if self.vision is not None and self.vision.offer(ev.frame, t_mono) and not self.vision_logged:
                    self.vision_logged = True
                    print(f'[vision] forwarding video to the model at {self.vision.fps:g} fps ({ev.frame.width}x{ev.frame.height})', flush=True)
                if self.voice_mode:                                                      # no feed thread: first frame, size and --max-s here
                    if self.first_frame_t is None:
                        self.first_frame_t = t_mono; self.frame_size = (ev.frame.width, ev.frame.height)
                        print(f'[worker] first frame {ev.frame.width}x{ev.frame.height}', flush=True)
                    if self.a.max_s and t_mono - self.first_frame_t >= self.a.max_s:
                        print(f'[worker] --max-s {self.a.max_s} reached', flush=True); self.end('max_s reached'); break
        finally:
            await stream.aclose()
        print('[worker] video stream ended (a republished track is picked up again)', flush=True)
        self.video_task = None

    # ---- voice mode (--perception off): the worker's own events ----
    def emit(self, type_, **fields):
        """A synthetic event in the perception schema (v1, t on this job's clock, own seq), through the same sinks as Perception's."""
        ev = dict(v=EVENT_SCHEMA_VERSION, t=round(time.monotonic() - self.t0, 3), t_wall=round(time.time(), 3), seq=self.ev_seq, type=type_)
        if type_ != 'status': ev['frame'] = self.seq
        ev.update(fields); self.ev_seq += 1
        if self.event_writer is not None:
            try: self.event_writer(ev)
            except Exception as e: log.warning('event log failed: %r', e)
        self.on_event(ev)
        return ev

    async def _status_loop(self):
        """Every status_every_s (5 s) while a caller is bound: frames received, the vision tee's counters, the player's stats."""
        every = float(getattr(self.cfg, 'status_every_s', 5.0) or 5.0); last_seq = self.seq; last_t = time.monotonic()
        while True:
            await asyncio.sleep(every)
            now = time.monotonic(); el = max(now - last_t, 1e-6); recv = (self.seq - last_seq) / el; last_seq = self.seq; last_t = now
            if self.bound is None and self.video_task is None: continue
            ev = dict(fps_recv=round(recv, 1), state='open' if self.session_opened else 'init', exercise=None, count=0, mode='voice',
                      voice=('on' if self.voice is not None and self.voice.model is not None and self.voice.started else 'off'))
            if self.vision is not None: ev.update(vision_frames=self.vision.n_sent, vision_fps=self.vision.fps, vision_replaced=self.vision.n_replaced)
            if self.player is not None: ev.update(underruns=self.player.underruns, cues_played=self.cues_played, cues_dropped=self.player.n_dropped)
            self.emit('status', **ev)

    def on_participant_disconnected(self, p):
        print(f'[worker] participant left: {p.identity}', flush=True)
        if p.identity == self.bound:
            self.bound = None
            if self.a.leave_grace_s is not None and self.a.leave_grace_s >= 0 and self.leave_task is None and not self.closed:
                self.leave_task = asyncio.create_task(self._leave_grace(), name='leave-grace')

    def capture_time(self, ts_us, t_wall):
        if self.ts_mode is None:
            self.ts_mode = 'epoch' if abs(ts_us / 1e6 - t_wall) < 5 else 'recv'
        return ts_us / 1e6 if self.ts_mode == 'epoch' else t_wall

    # ---- worker thread: arrival-aligned, >= 90 ms apart, newest frame only (coach_agent._worker) ----
    def feed(self):
        try:
            self._feed()
        except BaseException as e:
            log.exception('perception thread died: %r', e)
            print(f'[worker] perception thread died: {type(e).__name__}: {e}', flush=True)
            self.worker_error = e; self.loop.call_soon_threadsafe(self.done.set)

    def _feed(self):
        last_seq = 0; last_proc = -1e9
        while not self.stop.is_set():
            if not self.frame_event.wait(timeout=0.05): slot = None
            else:
                self.frame_event.clear(); slot = self.latest
                if slot is not None and slot[4] != last_seq:
                    since = time.monotonic() - last_proc
                    if since < 0.09:                              # too soon after the last one: wait for the 90 ms mark, then take the newest frame
                        time.sleep(0.09 - since); slot = self.latest; self.frame_event.clear()
            if slot is None or slot[4] == last_seq:
                self.idle += 1; continue
            frame, ts_us, t_mono, t_wall, seq = slot
            self.dropped += seq - last_seq - 1; last_seq = seq; last_proc = time.monotonic()
            if self.first_frame_t is None:
                self.first_frame_t = t_mono; print(f'[worker] first frame {frame.width}x{frame.height}', flush=True)
            if self.frame_size != (frame.width, frame.height):
                if self.frame_size is not None: print(f'[worker] received resolution changed {self.frame_size} -> {(frame.width, frame.height)}', flush=True)
                self.frame_size = (frame.width, frame.height)
            rgb = frame_to_rgb(frame)
            self.per.frames_received = self.seq; self.per.frames_dropped = self.dropped
            self.per.process_frame(rgb, t_recv=t_mono, meta=dict(frame_seq=seq, ts_us=ts_us, t_capture=self.capture_time(ts_us, t_wall), t_recv=t_wall))
            self.processed += 1
            if self.a.max_s and t_mono - self.first_frame_t >= self.a.max_s:
                print(f'[worker] --max-s {self.a.max_s} reached', flush=True); self.loop.call_soon_threadsafe(self.end, 'max_s reached'); break

    # ---- sinks ----
    def say_from_worker(self, text, intent, meta):
        self.loop.call_soon_threadsafe(self.player.say, text, intent, meta)

    def on_event(self, ev):
        """Perception thread -> event loop: the queue, the console, the optional data channel. The arms-up probe beeps here (say_now
        from the loop, as coach_agent does) only with --latency-probe, the cue player attached and no set active (run B on 10-09: a
        jumping jack is an arm raise), and the event says so (beep) before it is logged; a probe during a set is emitted, not acted on."""
        if ev['type'] == 'session': self.session_state = ev.get('state')
        if ev['type'] == 'probe':
            armed = bool(getattr(self.a, 'latency_probe', False)); active = self.session_state == 'active'
            if active: self.probes_ignored += 1
            beep = armed and not active and self.player is not None and getattr(self.a, 'probe_beep', True)
            ev['beep'] = bool(beep)
            if beep:
                from facetime.audio_out import BEEP
                t_first = ev.get('t_first_up'); now_m = time.monotonic(); now_w = time.time()
                meta = dict(t_recv=(now_w - (now_m - t_first)) if t_first is not None else None, t_detect=now_w, on_box_ms=ev.get('on_box_ms'))
                self.loop.call_soon_threadsafe(self.player.say_now, BEEP, 'probe', meta)
        self.loop.call_soon_threadsafe(self._deliver, ev)

    def _deliver(self, ev):
        self.events.put_nowait(ev)
        t = ev['type']
        if t == 'rep':
            c = ev.get('confidence') or {}
            print(f"[event] rep {ev['exercise']} {ev['n']} (agree {c.get('agree')}, amp {c.get('amp')}) t={ev['t']:.2f}", flush=True)
        elif t in ('session', 'gesture', 'visibility', 'fault', 'probe'):
            note = ''
            if t == 'probe' and not getattr(self.a, 'latency_probe', False): note = '  (probe off: --latency-probe to beep and open the floor)'
            elif t in ('probe', 'gesture') and self.session_state == 'active': note = '  (during a set: no beep, no floor window)'
            print(f"[event] {t} " + ' '.join(f'{k}={v}' for k, v in ev.items() if k not in ('v', 't_wall', 'seq', 'type', 'frame')) + note, flush=True)
        elif t == 'status' and ev.get('mode') == 'voice':
            vis = f"vision {ev['vision_frames']} frames sent ({ev['vision_fps']:g} fps, {ev.get('vision_replaced', 0)} replaced)" if 'vision_frames' in ev else 'vision off'
            print(f"[status] recv {ev.get('fps_recv')} fps, {vis}, voice {ev.get('voice')}, underruns {ev.get('underruns')}, cues {ev.get('cues_played')}, "
                  f"session {ev['state']} (voice mode, no perception)", flush=True)
        elif t == 'status':
            print(f"[status] recv {ev.get('fps_recv')} fps, processed {ev['fps_processed']} fps, pose {ev['pose_ms']} ms ({ev['pose_model']}), clf {ev['clf_ms']} ms, "
                  f"recv->decision p50 {ev.get('recv_to_decision_p50_ms')} ms, dropped {ev.get('dropped')}, underruns {ev.get('underruns')}, session {ev['state']}/{ev['exercise'] or '-'} count {ev['count']}", flush=True)
        if self.a.publish_data:
            asyncio.create_task(self._publish(ev))

    async def _publish(self, ev):
        try:
            await self.room.local_participant.publish_data(json.dumps(ev).encode(), reliable=True, topic=self.a.data_topic)
        except Exception as e:
            log.warning('publish_data failed: %r', e)

    # ---- the conversational layer (facetime/voice.py) ----
    def build_voice(self):
        """--voice gemini: the VoiceLayer with a GeminiModel (session started on bind, never on a warm job) and the mixer input on the
        cue player; --handlers alone: the handlers with say_now only. Returns None (with a message) when the layer cannot run."""
        from facetime.voice import VoiceLayer, GeminiModel, VideoTee, load_handlers, read_text, DEFAULT_PERSONA
        a = self.a; model = None
        if a.voice == 'gemini':
            if self.player is None:
                print('[voice] --voice gemini needs the cue player track (not --events-only): the model speaks through that one AudioSource; voice OFF', flush=True)
            elif not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
                print('[voice] GEMINI_API_KEY is MISSING: voice OFF', flush=True)
            else:
                persona = read_text(a.persona, DEFAULT_PERSONA)
                if a.plan: plan_line = f'\n\nFor your own reference only, never to be read aloud: the session plan, in order, is {a.plan}.'
                elif getattr(a, 'plan_text', None): plan_line = '\n\nThe plan you are coaching, step by step (follow it in order, one step at a time; never read it aloud):\n' + a.plan_text
                else: plan_line = ''
                model = GeminiModel(persona + plan_line, model=a.voice_model, voice=a.voice_name, on_turn=self.emit_voice, text=getattr(a, 'text', False))
        if getattr(a, 'vision', False):
            if model is None: print('[vision] --vision needs --voice gemini with a model session: no video is forwarded', flush=True)
            else: self.vision = VideoTee(fps=a.vision_fps)
        hpath = handlers_path(a); handlers = load_handlers(hpath) if hpath else None
        if model is None and handlers is None: return None
        floor_open = getattr(a, 'floor', None) == 'open'
        if floor_open and not self.voice_mode: print('[voice] --floor open with perception on: the voice is never cut by a set (it will talk over the counts)', flush=True)
        v = VoiceLayer(self, model=model, plan=a.plan, handlers=handlers, probe=bool(getattr(a, 'latency_probe', False)), floor_open=floor_open, vision=self.vision)
        if model is not None:
            if v.floor is not None: v.floor.echo_guard = not getattr(a, 'barge_in', False)
            v.attach()
            print(f'[voice] gemini {model.model_id} voice {model.voice} language {model.language or "model default"}; persona {a.persona} ({len(model.persona)} chars'
                  + (f', plan text {len(a.plan_text)} chars from {a.plan_file}' if getattr(a, 'plan_text', None) else '') + f'); '
                  f'handlers {hpath or "none"}; mixer on the cue track (duck {20 * __import__("math").log10(self.player.mix.duck):.0f} dB); '
                  f'floor {"ALWAYS OPEN" if floor_open else "gated to the gaps between sets"}; probe window {"on" if v.floor.probe else "off"}; '
                  f'vision {f"on ({self.vision.fps:g} fps tee)" if self.vision is not None else "off"}; session starts when a caller is bound', flush=True)
        else:
            print(f'[voice] handlers {hpath} (say_now only; no model: ask_voice returns None)', flush=True)
        return v

    def emit_voice(self, role, text):
        """A model (or caller) turn as an event of type voice {text, t, role}: the queue, the JSONL log, the console (own seq counter)."""
        if role == 'caller' and not getattr(self.a, 'log_caller', False):
            print(f'[voice] caller spoke ({len(text.split())} words; not logged, --log-caller to keep transcripts)', flush=True); return
        t = round(time.monotonic() - (self.per.t0 if self.per is not None else self.t0), 3)
        ev = dict(v=EVENT_SCHEMA_VERSION, t=t, t_wall=round(time.time(), 3), seq=self.voice_seq, type='voice', text=text, role=role, producer='voice')
        self.voice_seq += 1
        print(f'[voice] {role}: {text}', flush=True)
        self.events.put_nowait(ev)
        if self.event_writer is not None:
            try: self.event_writer(ev)
            except Exception as e: log.warning('voice event log failed: %r', e)
        if self.a.publish_data: asyncio.create_task(self._publish(ev))

    def log_cue(self, rec):
        rec = dict(rec, room=self.room.name, call_id=self.call_id)
        if self.cue_log: self.cue_log.write(json.dumps(rec) + '\n'); self.cue_log.flush()
        if rec['status'] == 'played':
            self.cues_played += 1
            lat = '' if rec.get('t_decision') is None else f"  decision->push {1000 * (rec['t_push'] - rec['t_decision']):.0f} ms"
            if rec['intent'] == 'probe' and rec.get('t_recv') is not None:
                lat = f"  first up frame recv -> beep pushed {1000 * (rec['t_push'] - rec['t_recv']):.0f} ms (detect -> push {1000 * (rec['t_push'] - rec['t_detect']):.0f}, on-box {rec.get('on_box_ms')} ms)"
            print(f"[cue] {time.strftime('%H:%M:%S')} [{rec['intent']:10}] {rec['text']!r}{lat}", flush=True)
        else:
            print(f"[cue] {rec['status']} [{rec['intent']}] {rec['text']!r} {rec.get('why_dropped', '')}", flush=True)


def worker_options(a):
    return WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=a.agent_name, num_idle_processes=max(0, int(a.idle_processes)),
                         initialize_process_timeout=90.0, job_memory_warn_mb=3000)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = build_parser(); a, rest = ap.parse_known_args(argv)
    if a.help or (not rest and '--help' in argv):
        ap.print_help(); return 0
    resolve_mode(a)
    if rest and rest[0] not in LK_COMMANDS and rest[0].startswith('-'):
        print(f'[worker] unknown flag {rest[0]!r} (our flags go before the LiveKit subcommand: {", ".join(LK_COMMANDS)})', file=sys.stderr); return 2
    os.environ[ARGS_ENV] = json.dumps({k: v for k, v in vars(a).items() if k != 'help'})
    if a.vision and a.voice != 'gemini':
        print('[worker] --vision only does something with --voice gemini (the video goes to the model session); nothing is forwarded', flush=True)
    print(f'[worker] agent {a.agent_name!r}; mode {describe_mode(a)}; '
          f'audio {"OFF (events only)" if a.events_only else "cue player"}; voice {a.voice}{" (" + (a.voice_model or "default model") + ", " + a.voice_name + ")" if a.voice != "off" else ""}; '
          f'handlers {a.handlers or "none"}; data channel {"on" if a.publish_data else "off"}; events -> {a.event_log}', flush=True)
    sys.argv = [sys.argv[0]] + rest                           # the LiveKit Agents CLI parses the rest (dev | start | connect | download-files)
    cli.run_app(worker_options(a))
    return 0


if __name__ == '__main__':
    sys.exit(main())
