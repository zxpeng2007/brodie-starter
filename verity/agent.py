"""Verity Agent: one -ity character on a FaceTime video call (Gemini Live speech-to-speech, sees the caller's camera at 1 fps).

Each character is a completely separate agent: one call, one character, picked by the room name "web-<key>-..." (the web app,
verity/server.py) or else by VERITY_CHARACTER (default verity; keys in verity/variants.py). AGENT_NAME is the dispatch name.
  python -m verity.agent dev                                 # web app + phone page + smoke test
  VERITY_CHARACTER=dancity python -m verity.agent console    # local mic/speaker, no camera, no avatar
FaceTime/Chert rooms follow facetime/worker.py: on join publish the voice track AND the face at once (Chert's bridge binds to the
participant with an unmuted audio track and expects audio + video from it), then say NOTHING and start no Gemini session until a
caller publishes video or audio (Chert fills empty waiting rooms first; FaceTime Audio calls have no video). Web rooms start as
soon as the caller joins (camera optional, text chat works).

The web caller's token carries attributes: verity.uid (memory key, verity/memory.py) and verity.settings (JSON: asmr "off" |
"mommy", voice (a Gemini voice name or ""), strikes, memory). Settings changed mid-call (attributes) update the instructions live;
a voice change applies from the next call. Mid-call the only character change is Verity -> Cruelty (three strike() calls) and back
(forgive()). The app gets JSON events on data topic "verity" ({type: character|strike|transform|verdict|remember|settings|
memory|dance}) and sends {type: "interrupt"} on topic "verity.control".
"""
import asyncio, json, logging, os, typing
from dataclasses import dataclass, field
from livekit import rtc
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, JobProcess, RunContext, WorkerOptions, cli, function_tool
from livekit.agents.voice.room_io import RoomOptions
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput   # livekit-agents 1.8.5 (pinned)
from livekit.plugins import google
from livekit.plugins.google.realtime.api_proto import Voice
from facetime.voice import VideoTee
from verity import memory
from verity.avatar import Avatar, LevelTap, real_frames, sprites
from verity.dance import DanceDetector
from verity.variants import DEFAULT, VARIANTS, instructions

log = logging.getLogger('verity')


class _QuietLoopMonitor(logging.Filter):
    """livekit-agents warns on every event-loop stall with a long stack trace. On a Windows console, printing those is slow enough
    to cause the next stall, so they are dropped unless VERITY_DEBUG is set."""
    def filter(self, record):
        return not record.getMessage().startswith('event loop blocked')


if not os.environ.get('VERITY_DEBUG'): logging.getLogger('livekit.agents').addFilter(_QuietLoopMonitor())
CHARACTER = os.environ.get('VERITY_CHARACTER', DEFAULT).strip().lower()
VOICES = typing.get_args(Voice)
STRIKES_TO_CRUELTY = 3
DANCE_FPS = 10
PRIVATE = '(Private instruction for you, never say it aloud; answer the caller in your own words.) '   # Brodie's fix (facetime/voice.py): stops the model reading prompts out
DANCE_REACT_S = float(os.environ.get('VERITY_DANCE_REACT_S', '20'))   # Dancity rates a move at most this often, so it doesn't talk over you
MAX_CALL_S = int(os.environ.get('VERITY_MAX_CALL_S', '900'))   # hard cap per call: the team has 240 Gemini Live minutes
LEAVE_GRACE_S = 5.0                                  # a caller who drops and doesn't come back within this ends the call


def parse_settings(raw):
    """verity.settings from the caller's attributes, validated (it comes from the browser)."""
    try: s = json.loads(raw or '{}')
    except ValueError: s = {}
    if not isinstance(s, dict): s = {}
    voice = s.get('voice') if s.get('voice') in VOICES else ''
    return {'asmr': 'mommy' if s.get('asmr') == 'mommy' else 'off', 'voice': voice,
            'strikes': s.get('strikes', True) is not False, 'memory': s.get('memory', True) is not False}


@dataclass
class Call:
    key: str = DEFAULT
    room: rtc.Room | None = None
    avatar: Avatar | None = None
    strikes: int = 0
    settings: dict = field(default_factory=lambda: parse_settings(''))
    uid: str = ''
    mem: str = ''                                        # what we remember about this caller, as an instructions block
    verdict: str = ''
    cruelty: bool = False
    video: bool = False                                  # the caller's camera reached us (web callers may have it off)
    tasks: set = field(default_factory=set)

    def emit(self, type_, **fields):
        """A JSON event for the web app on the data topic "verity" (fire and forget)."""
        if self.room is None or not self.room.isconnected(): return
        log.info('event %s %s', type_, fields)
        t = asyncio.create_task(self.room.local_participant.publish_data(json.dumps({'type': type_, **fields}), reliable=True, topic='verity'))
        self.tasks.add(t); t.add_done_callback(self.tasks.discard)


def realtime(voice):
    kw = dict(model=os.environ.get('GEMINI_LIVE_MODEL', 'gemini-3.8-live'), voice=voice, language='en-US',
              api_key=os.environ['GEMINI_API_KEY'])
    base = os.environ.get('GEMINI_BASE_URL', '').strip().rstrip('/')
    if base:                                             # hackathon relay: GEMINI_API_KEY is the team token (as facetime/voice.py)
        from google.genai import types
        kw['http_options'] = types.HttpOptions(base_url=base)
    return google.beta.realtime.RealtimeModel(**kw)


def build_instructions(key, call, recap=''):
    s = call.settings
    text = instructions(VARIANTS[key], asmr=s['asmr'] == 'mommy', strikes=s['strikes'], memory=call.mem if s['memory'] else '')
    return text + (f'\n- This call so far:\n{recap}' if recap else '')


def recap(session, n=8):
    """The last few turns as plain lines, for a fresh session that has to pick up the conversation."""
    lines = []
    for item in session.history.items[-n * 2:]:
        text = (getattr(item, 'text_content', None) or '').strip()
        if text and getattr(item, 'role', None) in ('user', 'assistant'): lines.append(f'  {"Caller" if item.role == "user" else "You"}: {text[:200]}')
    return '\n'.join(lines[-n:])


class Character(Agent):
    def __init__(self, key, call, opener=None, history=''):
        """opener: what to say on entering instead of the greeting (a live settings change hands off to a fresh session)."""
        self.key, self.v, self.opener = key, VARIANTS[key], opener
        voice = call.settings['voice'] if key == call.key and call.settings['voice'] else self.v['voice']   # Cruelty keeps its own
        super().__init__(instructions=build_instructions(key, call, history), llm=realtime(voice))

    async def on_enter(self):
        call: Call = self.session.userdata
        if call.avatar: call.avatar.set_variant(self.v)
        call.emit('character', key=self.key, name=self.v['name'], color=self.v['color'], face=self.v['face'])
        if self.opener:
            self.session.generate_reply(instructions=PRIVATE + self.opener); return
        see = ('You can mention one thing you see on their camera.' if call.video else
               "Their camera is off, so don't describe them or their room.")
        self.session.generate_reply(instructions=PRIVATE + f'You just picked up the call. Say hi like a friend would, in your own words, and '
                                                 f'work in your signature line "{self.v["intro"]}" naturally. {see} Keep it to one or '
                                                 'two short sentences.')

    @function_tool
    async def verdict(self, ctx: RunContext[Call], name: str):
        """Record the -ity verdict you just gave the caller.

        Args:
            name: the -ity name, e.g. "Procrastinity".
        """
        call = ctx.userdata
        if name.strip()[:40] == call.verdict: return 'Already recorded. Say nothing more about it.'
        call.verdict = name.strip()[:40]
        call.emit('verdict', name=call.verdict)
        return 'Recorded. Do not repeat it; just react to the caller.'

    @function_tool
    async def remember(self, ctx: RunContext[Call], fact: str):
        """Remember something the caller told you about themselves, for next time.

        Args:
            fact: a short note, e.g. "wants to be called Sam" or "has a dance recital on Friday".
        """
        call = ctx.userdata
        if not call.settings['memory'] or not call.uid: return 'Memory is off for this caller. Do not mention it.'
        memory.remember(call.uid, fact)
        call.emit('remember', fact=fact[:200])
        return 'Remembered. Do not announce it.'


class Verity(Character):
    @function_tool
    async def strike(self, ctx: RunContext[Call], reason: str):
        """Give the caller a strike when they mess up (wrong capital of France, rude, ignoring you, hanging up mid-sentence).

        Args:
            reason: what they did, in a few words.
        """
        call = ctx.userdata
        if not call.settings['strikes']: return 'Strikes are off. Let it go, sweetly.'
        call.strikes += 1
        call.emit('strike', count=call.strikes, of=STRIKES_TO_CRUELTY, reason=reason)
        if call.avatar: call.avatar.flash()
        if call.strikes < STRIKES_TO_CRUELTY:
            return f'Strike {call.strikes}/{STRIKES_TO_CRUELTY}. Warn them sweetly, smile a little creepier. Do not explain the rules.'
        call.strikes = 0; call.cruelty = True
        call.emit('transform', to='cruelty', name='Cruelty')
        return make('cruelty', call, history=recap(ctx.session))     # Cruelty knows exactly what they did


class Cruelty(Character):
    @function_tool
    async def forgive(self, ctx: RunContext[Call]):
        """Turn back into Verity once the caller sincerely apologizes."""
        ctx.userdata.emit('transform', to='verity', name='Verity')
        return make('verity', ctx.userdata, history=recap(ctx.session))


def make(key, call, **kw):
    return {'verity': Verity, 'cruelty': Cruelty}.get(key, Character)(key, call, **kw)


def room_character(name):
    """("dancity", True) for a web room "web-dancity-x1y2", else (VERITY_CHARACTER, False)."""
    parts = name.split('-')
    if parts[0] == 'web' and len(parts) > 2 and parts[1] in VARIANTS and not VARIANTS[parts[1]].get('secret'): return parts[1], True
    return CHARACTER, False


async def first_caller(room, need_media):
    """The first non-agent participant (web: as soon as it joins; FaceTime: once it publishes video or audio)."""
    fut = asyncio.get_running_loop().create_future()
    def consider(p, track=None):
        if fut.done() or p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT: return
        if need_media and (track is None or track.kind not in (rtc.TrackKind.KIND_VIDEO, rtc.TrackKind.KIND_AUDIO)): return
        fut.set_result(p)
    room.on('participant_connected', lambda p: consider(p))
    room.on('track_subscribed', lambda t, _pub, p: consider(p, t))
    for p in room.remote_participants.values():
        consider(p)
        for pub in p.track_publications.values():
            if pub.track: consider(p, pub.track)
    return await fut


async def pump(track, tee, session, call):
    """Caller video -> the model (VideoTee, 1 fps) and the dance detector (verity/dance.py, about 10 fps, one frame in flight)."""
    detector, busy, last, said = (DanceDetector() if call.key == 'dancity' else None), None, 0.0, 0.0   # Brodie pose + team judge, Dancity only
    async for ev in rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA):
        tee.offer(ev.frame)
        now = asyncio.get_running_loop().time()
        if detector and busy is None and now - last >= 1 / DANCE_FPS:
            last = now; busy = asyncio.create_task(asyncio.to_thread(detector.feed, ev.frame))
        if busy is not None and busy.done():
            try: move = busy.result()
            except Exception as e: log.warning('dance detector failed: %s', e); move = None   # never stop feeding the model
            busy = None
            if move and now - said >= DANCE_REACT_S:
                said = now; call.emit('dance', **move)
                tip = f" Their {move['tip']} was the furthest off: give one quick tip about it." if move.get('tip') else ''
                session.generate_reply(instructions=PRIVATE + f"You just watched them dance ({move['move']}); your rating: {move['score']}/10. "
                                                    f'React in one short sentence and say the rating.{tip}')


def face_frames(key):
    return sprites(key) or real_frames(key)


def prewarm(proc: JobProcess):
    # Only the faces a FaceTime call can need (its character, and Cruelty mid-call): all six take 4 s here and far longer on a slow
    # disk, and a process that isn't ready within initialize_process_timeout fails. Web calls load their character in a thread.
    for key in {CHARACTER if CHARACTER in VARIANTS else DEFAULT, 'cruelty'}: face_frames(key)


async def entrypoint(ctx: JobContext):
    key, web = room_character(ctx.room.name)
    call = Call(key=key)
    session = AgentSession[Call](userdata=call, user_away_timeout=None, video_sampler=None)
    if ctx.is_fake_job():                                # console: local mic/speaker, nobody to wait for
        await session.start(make(key, call), room=ctx.room)
        return
    await asyncio.gather(ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL), asyncio.to_thread(face_frames, key))   # cached after
    call.room, call.avatar = ctx.room, Avatar(VARIANTS[key])
    call.tasks.add(await call.avatar.publish(ctx.room))
    voice_out = _ParticipantAudioOutput(ctx.room, sample_rate=24000, num_channels=1,
                                        track_publish_options=rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    t = asyncio.create_task(voice_out.start()); call.tasks.add(t)   # published now (then waits for a subscriber); the session speaks into it
    log.info('%s joined %s (%s); waiting for the caller', VARIANTS[key]['name'], ctx.room.name, 'web' if web else 'facetime')
    caller = await first_caller(ctx.room, need_media=not web)
    call.settings = parse_settings(caller.attributes.get('verity.settings'))
    call.uid = caller.attributes.get('verity.uid', '') if call.settings['memory'] else ''
    mem = memory.load(call.uid); call.mem = memory.prompt(mem)
    log.info('caller %s, settings %s, remembers %d facts', caller.identity, call.settings, len(mem['facts']))

    tee = VideoTee(fps=1.0); tee.enable()
    session.input.video = tee
    pumps = {}
    def on_video(track, _pub=None, p=None):              # the caller's camera, whenever it shows up (web: maybe never; FaceTime: may republish)
        if track.kind != rtc.TrackKind.KIND_VIDEO or (p or caller).identity != caller.identity or track.sid in pumps: return
        call.video = True; pumps[track.sid] = t = asyncio.create_task(pump(track, tee, session, call)); call.tasks.add(t)
    def off_video(track, _pub, p):
        if track.sid in pumps: pumps.pop(track.sid).cancel()

    # the call ends when the caller is gone for good (any reason: hang-up, crash, network) or at the time cap
    gone = asyncio.Event()
    def end(reason):
        if gone.is_set(): return
        log.info('ending call: %s', reason); gone.set()
        asyncio.create_task(session.aclose()) if session_started else ctx.shutdown(reason)
    def on_left(p):
        if p.identity != caller.identity: return
        async def grace():
            await asyncio.sleep(LEAVE_GRACE_S)
            if caller.identity not in ctx.room.remote_participants: end('caller left')
        call.tasks.add(asyncio.create_task(grace()))
    session_started = False
    ctx.room.on('participant_disconnected', on_left)
    ctx.room.on('disconnected', lambda *_: end('room disconnected'))
    ctx.room.on('track_unsubscribed', off_video)
    asyncio.get_running_loop().call_later(MAX_CALL_S, end, f'time cap {MAX_CALL_S}s')

    def on_attributes(changed, p):                       # settings flipped mid-call (Mommy ASMR, strikes, memory)
        if p.identity != caller.identity or 'verity.settings' not in changed: return
        old, call.settings = call.settings, parse_settings(p.attributes.get('verity.settings'))
        call.emit('settings', **call.settings)
        if (call.settings['asmr'], call.settings['strikes'], call.settings['memory']) == (old['asmr'], old['strikes'], old['memory']): return
        # Gemini Live can't take new system instructions mid-session (update_instructions arrives as the model's own words and it
        # reads them out), so the same character continues in a fresh session with the new prompt and a recap of the call.
        opener = ('Continue the conversation: one short, soft, whispery line to the caller in your new style.' if call.settings['asmr'] == 'mommy'
                  and old['asmr'] != 'mommy' else 'Continue the conversation with one short line in your normal voice.')
        session.update_agent(make(session.current_agent.key, call, opener=opener, history=recap(session)))
    ctx.room.on('participant_attributes_changed', on_attributes)

    def on_data(packet):                                 # the app's controls
        if packet.topic != 'verity.control': return
        try: msg = json.loads(packet.data)
        except ValueError: return
        if msg.get('type') == 'interrupt': session.interrupt()
    ctx.room.on('data_received', on_data)

    def on_close(_ev):
        try:
            if call.uid and call.settings['memory']:
                memory.log_call(call.uid, character=VARIANTS[key]['name'], verdict=call.verdict, cruelty=call.cruelty)
        except Exception as e: log.warning('memory not saved: %s', e)
        finally: ctx.shutdown()
    session.on('close', on_close)
    session.on('conversation_item_added', lambda ev: log.info('%s: %s', getattr(ev.item, 'role', ev.item.type), getattr(ev.item, 'text_content', '')))
    # does the caller have a camera? (the greeting names something it sees only if so); a web caller publishes it just after joining
    has_cam = lambda: any(pub.kind == rtc.TrackKind.KIND_VIDEO for pub in caller.track_publications.values())
    for _ in range(10 if web else 0):
        if has_cam(): break
        await asyncio.sleep(0.1)
    call.video = has_cam()
    session.output.audio = LevelTap(call.avatar, voice_out)   # before start: RoomIO then publishes no second audio track
    await session.start(make(key, call), room=ctx.room,
                        room_options=RoomOptions(video_input=False, audio_output=False, participant_identity=caller.identity))
    session_started = True
    if caller.identity not in ctx.room.remote_participants: end('caller left before the call started'); return
    ctx.room.on('track_subscribed', on_video)            # after start: a frame can't prompt a session that isn't up yet
    for pub in caller.track_publications.values():
        if pub.track: on_video(pub.track)
    if mem['facts'] or mem['calls']: call.emit('memory', facts=mem['facts'][-15:], calls=len(mem['calls']))


if __name__ == '__main__':
    if CHARACTER not in VARIANTS: raise SystemExit(f'VERITY_CHARACTER={CHARACTER!r}: pick one of {", ".join(VARIANTS)}')
    print(f'[verity] default character {VARIANTS[CHARACTER]["name"]}, agent name {os.environ.get("AGENT_NAME", "")!r}', flush=True)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=os.environ.get('AGENT_NAME', ''),
                              num_idle_processes=1,                                   # a preloaded process: no 7 s cold start on the call
                              initialize_process_timeout=60, shutdown_process_timeout=30))   # slow Windows/OneDrive disks: 10 s defaults time out
