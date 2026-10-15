"""Brodie Dance on the web: the landing page and accounts. Lessons happen over FaceTime only (Brodie's worker on the Chert line).

  python -m brodie.server [--host 127.0.0.1] [--port 8765]
Routes: /  landing (web/index.html)
        POST /api/start {email}  the landing page's one field: creates the account (no password) or signs into it
        POST /api/signup {name, email, password}   POST /api/login {email, password}   POST /api/logout   GET /api/me
        GET /api/line   signed in only: Brodie's line, open | busy (seconds_left) | mine | offline, the FaceTime number when open or mine
        POST /api/line/claim   the visitor clicked "FaceTime Brodie": a 15-minute session is theirs, or 409 busy
        POST /api/line/done    "I'm done dancing": the visitor's session ends, the line is free for everyone
        GET /api/config  public settings (dance, fork; no number, no status)   GET /api/dances   GET /api/health
        POST /api/heartbeat  Brodie's worker saying it is up (X-Brodie-Key: HEARTBEAT_KEY); "offline" on the page comes from it
        GET /api/admin/line   POST /api/admin/line/busy?minutes=N | /free | /extend?minutes=5   the stats page (?key=ADMIN_KEY)
The line is website-only (brodie/line.py): Brodie's worker reports nothing but its heartbeat.
Sessions are a signed cookie (brodie/users.py); HOST/PORT come from the environment on a hosting service (docs/HOSTING.md).
"""
import argparse, datetime, glob, hmac, json, os, time
from aiohttp import web
from brodie import line, users

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, 'web')
COOKIE = 'bd_session'
# Brodie's worker runs on another machine; it POSTs /api/heartbeat every ~20 s (facetime/worker.py, BRODIE_SERVER +
# BRODIE_HEARTBEAT_KEY). With HEARTBEAT_KEY set here, "online" means a heartbeat in the last HEARTBEAT_TTL seconds.
HEARTBEAT_TTL = float(os.environ.get('HEARTBEAT_TTL', '75'))
_LAST_BEAT = {'t': 0.0, 'agent': ''}


def worker_online():
    """True / False when the worker heartbeat is configured (HEARTBEAT_KEY), None when it is not (unknown)."""
    if not os.environ.get('HEARTBEAT_KEY'): return None
    return time.time() - _LAST_BEAT['t'] < HEARTBEAT_TTL


# ---- dev only: ?preview=free|busy|offline|loggedout shows each state of the landing page without a real call ----
# Off unless DEV_PREVIEW=1 is set (never on the host): without it the parameter is ignored everywhere.
PREVIEW_BUSY_S = 402                      # the busy preview counts down from 6:42, then turns free (the switch-over), then starts again
_PREVIEW = {'t0': 0.0}


def facetime_number():
    """FACETIME_NUMBER as (shown, dialled): any formatting is stripped for the link (+16282892384); a US number is shown as
    +1 (628) 289-2384, anything else as given. ('', '') when unset."""
    raw = os.environ.get('FACETIME_NUMBER', '').strip()
    digits = ''.join(ch for ch in raw if ch.isdigit())
    if not digits: return raw, raw                       # an Apple ID email is dialled as is
    dial = '+' + digits if raw.startswith('+') or len(digits) == 11 else digits
    if len(digits) == 11 and digits[0] == '1': return f'+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}', '+' + digits
    if len(digits) == 10: return f'+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}', '+1' + digits
    return raw, dial


def preview(req):
    p = req.query.get('preview', '')
    return p if os.environ.get('DEV_PREVIEW') == '1' and p in ('free', 'busy', 'mine', 'offline', 'loggedout') else None


def _preview_line(p):
    st = {'cap_s': line.cap_s(), 'preview': p}
    if p in ('busy', 'mine'):
        now = time.time()
        if now - _PREVIEW['t0'] > PREVIEW_BUSY_S + 15: _PREVIEW['t0'] = now
        left = PREVIEW_BUSY_S - (now - _PREVIEW['t0'])
        if left > 0 and p == 'busy': return dict(st, state='busy', seconds_left=int(round(left)))
        if left > 0:
            shown, dial = facetime_number()
            return dict(st, state='mine', seconds_left=int(round(left)), facetime=shown or '+1 (555) 010-0000', facetime_dial=dial or '+15550100000')
        p = 'free'
    if p == 'offline': return dict(st, state='offline', seconds_left=0)
    return dict(st, state='open', seconds_left=0)


def _public(u):
    return {'id': u['id'], 'name': u['name'], 'email': u['email'], 'calls': u['calls']}


def current_user(req):
    return users.get(users.check(req.cookies.get(COOKIE, '')))


def _with_session(resp, req, uid):
    secure = req.scheme == 'https' or req.headers.get('X-Forwarded-Proto', '') == 'https'
    resp.set_cookie(COOKIE, users.sign(uid), max_age=users.SESSION_DAYS * 86400, httponly=True, samesite='Lax', secure=secure, path='/')
    return resp


async def _body(req):
    try: b = await req.json()
    except Exception: b = {}
    return b if isinstance(b, dict) else {}


async def signup(req):
    b = await _body(req)
    try: u = users.create(b.get('name', ''), b.get('email', ''), b.get('password', ''))
    except ValueError as e:
        if str(e) == 'exists': return web.json_response({'error': 'That email already has an account. Log in instead.'}, status=409)
        return web.json_response({'error': str(e).replace('invalid: ', '')}, status=400)
    return _with_session(web.json_response({'ok': True, 'user': _public(u)}), req, u['id'])


async def login(req):
    b = await _body(req)
    u = users.verify(b.get('email', ''), b.get('password', ''))
    if u is None: return web.json_response({'error': 'Wrong email or password.'}, status=401)
    return _with_session(web.json_response({'ok': True, 'user': _public(u)}), req, u['id'])


async def logout(_req):
    resp = web.json_response({'ok': True}); resp.del_cookie(COOKIE, path='/'); return resp


async def start(req):
    """One field: an email signs the visitor in (a new email creates the account). Then the page shows Brodie's line."""
    b = await _body(req)
    try: u, created = users.by_email(b.get('email', ''))
    except ValueError as e: return web.json_response({'error': str(e).replace('invalid: ', '')}, status=400)
    return _with_session(web.json_response({'ok': True, 'new': created, 'user': _public(u)}), req, u['id'])


async def me(req):
    p = preview(req)
    if p == 'loggedout': return web.json_response({'error': 'not logged in'}, status=401)
    if p: return web.json_response({'user': {'id': 0, 'name': 'preview', 'email': 'preview@example.com', 'calls': 0}})
    u = current_user(req)
    if u is None: return web.json_response({'error': 'not logged in'}, status=401)
    return web.json_response({'user': _public(u)})


async def health(req):
    """Public: ok. With ?key=<ADMIN_KEY>: the worker's heartbeat and the line too (Brodie's status is for signed-in visitors)."""
    out = {'ok': True, 'dance': os.environ.get('DANCE', '')}
    key = os.environ.get('ADMIN_KEY', '')
    if key and hmac.compare_digest(req.query.get('key', ''), key):
        beat = _LAST_BEAT['t']
        out.update(users=users.count(), worker_online=worker_online(), line=line.info(worker_online()),
                   last_heartbeat_s_ago=round(time.time() - beat, 1) if beat else None)
    return web.json_response(out)


async def heartbeat(req):
    """Brodie's worker says it is up: POST /api/heartbeat with header X-Brodie-Key: <HEARTBEAT_KEY>."""
    key = os.environ.get('HEARTBEAT_KEY', '')
    if not key or not hmac.compare_digest(req.headers.get('X-Brodie-Key', ''), key):
        return web.json_response({'error': 'forbidden'}, status=403)
    b = await _body(req)
    _LAST_BEAT['t'] = time.time(); _LAST_BEAT['agent'] = str(b.get('agent', ''))[:64]
    return web.json_response({'ok': True})


def _line_view(u):
    """What this visitor sees. The FaceTime number only once the session is theirs (after their click on "FaceTime Brodie"),
    so almost every caller starts the timer; a free line shows the button only."""
    st = line.view(u['id'], worker_online())
    if st['state'] == 'mine': st['facetime'], st['facetime_dial'] = facetime_number()
    return st


NO_STORE = {'Cache-Control': 'no-store'}


async def line_status(req):
    """Signed in only (logged out: 401 and nothing else)."""
    p = preview(req)
    if p and p != 'loggedout': return web.json_response(_preview_line(p), headers=NO_STORE)
    u = None if p == 'loggedout' else current_user(req)
    if u is None: return web.json_response({'error': 'sign up first'}, status=401, headers=NO_STORE)
    return web.json_response(_line_view(u), headers=NO_STORE)


async def line_claim(req):
    """The visitor clicked "FaceTime Brodie": the line is theirs for 15 minutes (200 with the number), or 409 busy (no number).
    Offline: 503. Two clicks at once: one sqlite write transaction, the first wins."""
    u = current_user(req)
    if u is None: return web.json_response({'error': 'sign up first'}, status=401, headers=NO_STORE)
    if worker_online() is False: return web.json_response(dict(line.view(u['id'], False), error='offline'), status=503, headers=NO_STORE)
    ok, _ = line.claim(u['id'])
    return web.json_response(_line_view(u), status=200 if ok else 409, headers=NO_STORE)


async def line_done(req):
    """ "I'm done dancing": the visitor's own session ends at once; the line is free for everyone."""
    u = current_user(req)
    if u is None: return web.json_response({'error': 'sign up first'}, status=401, headers=NO_STORE)
    freed = line.done(u['id'])
    return web.json_response(dict(_line_view(u), freed=freed), headers=NO_STORE)


def _admin_ok(req):
    key = os.environ.get('ADMIN_KEY', '')
    return bool(key) and hmac.compare_digest(req.query.get('key', ''), key)


def _forbidden(): return web.json_response({'error': 'forbidden'}, status=403)


def _admin_view():
    beat = _LAST_BEAT['t']
    return dict(line.info(worker_online()), heartbeat_s_ago=round(time.time() - beat) if beat else None, log=line.log(25))


async def admin_line(req):
    """The stats page: GET /api/admin/line?key=<ADMIN_KEY>: the state, who started the session, time left, the log."""
    if not _admin_ok(req): return _forbidden()
    return web.json_response(_admin_view(), headers=NO_STORE)


def _minutes(req, default):
    try: return float(req.query.get('minutes', default))
    except ValueError: return float(default)


async def admin_busy(req):
    """Mark busy: POST /api/admin/line/busy?key=…&minutes=N (1 to 15, default 15): a session from now, replacing any other."""
    if not _admin_ok(req): return _forbidden()
    line.admin_busy(_minutes(req, 15))
    return web.json_response(_admin_view(), headers=NO_STORE)


async def admin_free(req):
    """Mark free: POST /api/admin/line/free?key=…: the session ends now, whoever started it."""
    if not _admin_ok(req): return _forbidden()
    line.admin_free()
    return web.json_response(_admin_view(), headers=NO_STORE)


async def admin_extend(req):
    """Add time: POST /api/admin/line/extend?key=…&minutes=5: never past 15 minutes from now. 409 when the line is free."""
    if not _admin_ok(req): return _forbidden()
    ok = line.admin_extend(_minutes(req, 5))
    return web.json_response(_admin_view(), status=200 if ok else 409, headers=NO_STORE)


async def config(_req):
    """Public settings for the landing page. No FaceTime number and no status here: those are for signed-in visitors (/api/line)."""
    return web.json_response({'ok': True, 'dance': os.environ.get('DANCE', ''),
                              'fork': os.environ.get('FORK_URL', 'https://github.com/zxpeng2007/ivyhacks-brodie-dance')})


async def dances(_req):
    """The lessons in data/dances (newest first, as the handler picks them): name, parts, length, bpm, steps."""
    out = []
    try:
        from starter.dance.reference import Reference, DANCES_DIR
        live = os.environ.get('DANCE', '')
        paths = sorted(glob.glob(os.path.join(DANCES_DIR, '*.npz')), key=os.path.getmtime, reverse=True)
        for i, p in enumerate(paths):
            try:
                r = Reference(p)
                out.append({'name': r.name, 'parts': r.n_chunks, 'bpm': r.bpm, 'seconds': round(len(r.kps) / 10, 1),
                            'steps': [s for s in r.steps if s.strip()], 'video': bool(r.video_path), 'music': r.audio is not None,
                            'live': r.name == live or (not live and i == 0)})
            except Exception as e:
                out.append({'name': os.path.basename(p)[:-4], 'error': repr(e)})
    except Exception as e:
        return web.json_response({'dances': out, 'error': repr(e)})
    return web.json_response({'dances': out})


# ---- waitlist: open to the public landing page wherever it is hosted (CORS on these routes only) ----
CORS = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'POST, GET, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type'}


async def waitlist_join(req):
    b = await _body(req)
    try: pos, created = users.waitlist_add(b.get('email', ''), b.get('name', ''), req.headers.get('Origin', '') or req.headers.get('Referer', '')[:80])
    except ValueError as e: return web.json_response({'error': str(e)}, status=400, headers=CORS)
    return web.json_response({'ok': True, 'position': pos, 'new': created, 'count': users.waitlist_count()}, headers=CORS)


async def waitlist_count(_req):
    return web.json_response({'count': users.waitlist_count()}, headers=CORS)


async def waitlist_options(_req):
    return web.Response(status=204, headers=CORS)


async def waitlist_export(req):
    """CSV of the waitlist for the team: GET /api/waitlist/export?key=<ADMIN_KEY>."""
    key = os.environ.get('ADMIN_KEY', '')
    if not key or req.query.get('key', '') != key: return web.json_response({'error': 'forbidden'}, status=403)
    rows = users.waitlist_all()
    body = 'id,email,name,source,created\n' + ''.join(f'{r["id"]},{r["email"]},"{r["name"]}","{r["source"]}",{datetime.datetime.utcfromtimestamp(r["created"]).isoformat()}\n' for r in rows)
    return web.Response(text=body, content_type='text/csv')


async def stats(req):
    """Numbers for the team (stats page at /stats.html): GET /api/stats?key=<ADMIN_KEY>.
    TEAM_EMAILS (comma-separated) are left out so our own test calls don't count."""
    key = os.environ.get('ADMIN_KEY', '')
    if not key or not hmac.compare_digest(req.query.get('key', ''), key): return web.json_response({'error': 'forbidden'}, status=403)
    return web.json_response(users.stats(exclude=os.environ.get('TEAM_EMAILS', '').split(',')), headers={'Cache-Control': 'no-store'})


async def landing(_req): return web.FileResponse(os.path.join(WEB, 'index.html'))


def app():
    users.init(); line.init()
    a = web.Application(client_max_size=64 * 1024)
    a.add_routes([web.post('/api/signup', signup), web.post('/api/login', login), web.post('/api/logout', logout), web.post('/api/start', start), web.get('/api/me', me),
                  web.get('/api/health', health), web.post('/api/heartbeat', heartbeat), web.get('/api/config', config),
                  web.get('/api/line', line_status), web.post('/api/line/claim', line_claim), web.post('/api/line/done', line_done),
                  web.get('/api/admin/line', admin_line), web.post('/api/admin/line/busy', admin_busy), web.post('/api/admin/line/free', admin_free),
                  web.post('/api/admin/line/reset', admin_free), web.post('/api/admin/line/extend', admin_extend), web.get('/api/dances', dances),
                  web.post('/api/waitlist', waitlist_join), web.get('/api/waitlist', waitlist_count), web.options('/api/waitlist', waitlist_options),
                  web.get('/api/waitlist/export', waitlist_export), web.get('/api/stats', stats),
                  web.get('/', landing), web.get('/index.html', landing), web.static('/', WEB)])
    return a


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'), help='0.0.0.0 on a hosting service')
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8765')), help='hosting services set PORT')
    a = ap.parse_args()
    users.init(); line.init()
    print(f'[brodie] web on http://{a.host}:{a.port}: landing at /, {users.count()} users in {users.DB}', flush=True)
    web.run_app(app(), host=a.host, port=a.port, print=None)
