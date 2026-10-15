"""The landing page's flow and Brodie's one line (brodie/line.py, website-only) end to end against the real aiohttp app, with a
fake clock and a throwaway sqlite file. A session starts when a signed-in visitor clicks "FaceTime Brodie" (or an admin marks
the line busy); Brodie's worker only sends its heartbeat.

    python scripts/test_line.py
"""
import asyncio, os, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
TMP = tempfile.mkdtemp(prefix='brodie-line-')
NUMBER = '+1 (628) 289-2384'
os.environ.update(BRODIE_DB=os.path.join(TMP, 'brodie.db'), SESSION_SECRET='test', HEARTBEAT_KEY='k3y', FACETIME_NUMBER='+16282892384',
                  LINE_CAP_S='900', ADMIN_KEY='adm')
os.environ.pop('DEV_PREVIEW', None)

from aiohttp.test_utils import TestServer, TestClient
from brodie import server, line

CLOCK = [1_000_000.0]
line.now = lambda: CLOCK[0]
RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'   ({detail})' if detail != '' else ''), flush=True)


def mmss(s): return f'{int(s) // 60:02d}:{int(s) % 60:02d}'
def beat(): server._LAST_BEAT['t'] = time.time()


async def j(r):
    try: return await r.json()
    except Exception: return {}


async def line_of(c):
    r = await c.get('/api/line'); return r.status, await j(r)


async def visitor(srv, email=None):
    c = TestClient(srv); await c.start_server()
    if email: assert (await c.post('/api/start', json={'email': email})).status == 200
    return c


async def admin(c, path, key='adm'):
    r = await c.post(f'/api/admin/line/{path}' + ('&' if '?' in path else '?') + f'key={key}' if key else f'/api/admin/line/{path}')
    return r.status, await j(r)


async def main():
    srv = TestServer(server.app()); await srv.start_server()
    beat()

    print('\n1. logged out: no number, no status anywhere')
    anon = await visitor(srv)
    code, d = await line_of(anon); check('GET /api/line is 401 with nothing in it', code == 401 and set(d) == {'error'}, d)
    r = await anon.post('/api/line/claim'); check('a logged-out click cannot take the line (401)', r.status == 401)
    leaks = []
    for path in ('/api/config', '/api/health', '/api/dances', '/', '/index.html', '/line.js'):
        body = await (await anon.get(path)).text()
        if '289-2384' in body or '2892384' in body or 'worker_online' in body or '"state":' in body: leaks.append(path)
    check('the number and the status appear in no public response', not leaks, leaks or 'config, health, dances, page, line.js')

    print('\n2. email sign-up, returning visitor')
    r = await anon.post('/api/start', json={'email': 'not-an-email'}); check('a bad email is refused', r.status == 400)
    ana = await visitor(srv, 'Ana@Example.com'); ben = await visitor(srv, 'ben@example.com'); cy = await visitor(srv, 'cy@example.com')
    code, d = await line_of(ana)
    check('signed in: Brodie is free, the button only: no number before the click', code == 200 and d['state'] == 'open'
          and 'facetime' not in d and 'facetime_dial' not in d, d)
    r = await ana.get('/api/me'); check('returning visitor: the session cookie skips sign-up', r.status == 200 and (await r.json())['user']['email'] == 'ana@example.com')
    again = await visitor(srv); r = await again.post('/api/start', json={'email': 'ana@example.com'})
    check('the same email on a new device signs into the same account', (await r.json())['new'] is False)
    r = await ana.post('/api/login', json={'email': 'ana@example.com', 'password': 'guess'}); check('no password login into an email-only account', r.status == 401)

    print('\n3. a click on "FaceTime Brodie" starts a 15-minute session')
    r = await ana.post('/api/line/claim'); d = await j(r)
    check('Ana clicks: the line is hers (mine) and the click reveals the number (+1 (628) 289-2384, dialled +16282892384), 15:00',
          r.status == 200 and d['state'] == 'mine' and d['facetime'] == NUMBER and d['facetime_dial'] == '+16282892384' and d['seconds_left'] == 900, mmss(d.get('seconds_left', 0)))
    CLOCK[0] += 42
    code, d = await line_of(ben)
    check('everyone else: busy, real time left (14:18), no number', d['state'] == 'busy' and d['seconds_left'] == 900 - 42 and 'facetime' not in d, mmss(d['seconds_left']))
    code, d = await line_of(ana); check('Ana still sees her session, the number and the countdown', d['state'] == 'mine' and d['facetime'] == NUMBER, mmss(d['seconds_left']))
    r = await ben.post('/api/line/claim'); d = await j(r)
    check('Ben clicks while Ana is on: 409 busy, no number', r.status == 409 and d['state'] == 'busy' and 'facetime' not in d, r.status)
    r = await ana.post('/api/line/claim'); d = await j(r)
    check('Ana clicks again: still hers, the clock is not restarted', r.status == 200 and d['seconds_left'] == 900 - 42, mmss(d['seconds_left']))
    r = await ben.post('/api/line/done'); d = await j(r); code, d2 = await line_of(cy)
    check('"I\'m done dancing" from someone else changes nothing', d['freed'] is False and d2['state'] == 'busy')

    print('\n4. "I\'m done dancing" frees the line for everyone at once')
    CLOCK[0] += 300
    r = await ana.post('/api/line/done'); d = await j(r)
    code, b = await line_of(ben); code, c = await line_of(cy)
    check('Ana is done: free for Ana, Ben and Cy right away (button only, the number hidden again)', d['freed'] and d['state'] == 'open'
          and b['state'] == 'open' and c['state'] == 'open' and 'facetime' not in d and 'facetime' not in c, b['state'])

    print('\n5. two clicks at the same moment: the first gets the line')
    rs = await asyncio.gather(ben.post('/api/line/claim'), cy.post('/api/line/claim'))
    states = sorted([(r.status, (await j(r))['state']) for r in rs])
    check('one 200 mine, one 409 busy', states == [(200, 'mine'), (409, 'busy')], states)
    holder = ben if rs[0].status == 200 else cy
    await holder.post('/api/line/done')

    print('\n6. the 15 minutes run out: free on its own')
    await ana.post('/api/line/claim'); CLOCK[0] += 899
    code, d = await line_of(ben); check('busy one second before the end', d['state'] == 'busy' and d['seconds_left'] == 1, mmss(d['seconds_left']))
    CLOCK[0] += 1
    code, d = await line_of(ben); check('at 15:00: free (no reload needed: the page polls), number hidden until the next click', d['state'] == 'open' and 'facetime' not in d, d['state'])
    r = await ben.post('/api/line/claim'); d = await j(r); check("and the next visitor's click takes it and gets the number", r.status == 200 and d['facetime'] == NUMBER)
    log = (await j(await ben.get('/api/admin/line?key=adm')))['log']
    check('the expiry is logged at the end time', any(e['event'] == 'expired' and e['t'] == CLOCK[0] for e in log), [e['event'] for e in log[:3]])
    await ben.post('/api/line/done')

    print('\n7. a session survives a server restart')
    await ana.post('/api/line/claim'); CLOCK[0] += 100
    _, before = await line_of(ben)
    srv2 = TestServer(server.app()); await srv2.start_server(); c2 = await visitor(srv2, 'ben@example.com')
    _, after = await line_of(c2)
    check('busy with the same time left on a fresh server', after['state'] == 'busy' and after['seconds_left'] == before['seconds_left'], mmss(after['seconds_left']))
    await c2.close(); await srv2.close(); await ana.post('/api/line/done')

    print('\n8. the coach goes offline (worker heartbeat stops)')
    server._LAST_BEAT['t'] = 0
    code, d = await line_of(ana); check('offline: no number, no button', d['state'] == 'offline' and 'facetime' not in d, d['state'])
    r = await ana.post('/api/line/claim'); check('no session can start while offline (503)', r.status == 503)
    beat()

    print('\n9. admin controls on the stats page (ADMIN_KEY on the server)')
    before = (await j(await ana.get('/api/admin/line?key=adm')))['log'][0]['id']
    results = []
    for path in ('busy?minutes=10', 'free', 'extend?minutes=5', 'reset'):
        for key in (None, 'wrong'):
            code, _ = await admin(ana, path, key); results.append(code)
    code_get = (await ana.get('/api/admin/line')).status
    code, d = await line_of(ana); after = (await j(await ana.get('/api/admin/line?key=adm')))['log'][0]['id']
    check('without the key (or a wrong one) every control is 403 and nothing changes', set(results) == {403} and code_get == 403 and d['state'] == 'open' and after == before, results)

    code, a = await admin(ana, 'busy?minutes=10'); code, b = await line_of(ben)
    check('Mark busy 10 min: everyone sees busy 10:00, started by admin', code == 200 and a['busy'] and a['seconds_left'] == 600 and a['started_by'] == 'admin'
          and b['state'] == 'busy' and b['seconds_left'] == 600, (a['started_by'], mmss(b['seconds_left'])))
    code, a = await admin(ana, 'busy'); check('Mark busy default: 15 min', a['seconds_left'] == 900, mmss(a['seconds_left']))
    code, a = await admin(ana, 'busy?minutes=40'); check('Mark busy is capped at 15 min', a['seconds_left'] == 900, mmss(a['seconds_left']))
    CLOCK[0] += 600
    code, a = await admin(ana, 'extend?minutes=5'); check('Add 5 minutes: 05:00 -> 10:00', code == 200 and a['seconds_left'] == 600, mmss(a['seconds_left']))
    code, a = await admin(ana, 'extend?minutes=5'); code, a = await admin(ana, 'extend?minutes=5')
    check('Add 5 minutes never goes past 15 minutes from now', a['seconds_left'] == 900, mmss(a['seconds_left']))
    code, a = await admin(ana, 'free'); code, b = await line_of(ben)
    check('Mark free: free for everyone at once', code == 200 and not a['busy'] and b['state'] == 'open')
    code, a = await admin(ana, 'extend?minutes=5'); check('Add 5 minutes on a free line: 409, nothing changes', code == 409 and not a['busy'])
    await ana.post('/api/line/claim')
    code, a = await admin(ana, 'reset'); code, b = await line_of(ben)
    check('Mark free (reset) clears a visitor\'s stuck session', not a['busy'] and b['state'] == 'open')
    log = (await j(await ana.get('/api/admin/line?key=adm')))['log']
    admin_events = [e for e in log if e['actor'] == 'admin']
    details = ' | '.join(e['detail'] for e in admin_events)
    check('every admin action is logged with the time (busy 10, busy 15, add 5 min, mark free)',
          all(x in details for x in ('mark busy 10 min', 'mark busy 15 min', 'add 5 min: 300 s -> 600 s left', 'mark free (before: visitor ana@example.com'))
          and all(e['t'] > 0 for e in admin_events) and len(admin_events) >= 8, f'{len(admin_events)} admin entries')
    check('the visitor click is logged with who clicked', any(e['event'] == 'busy' and e['actor'] == 'visitor ana@example.com' for e in log))
    a = await j(await ana.get('/api/admin/line?key=adm'))
    await ana.post('/api/line/claim'); a = await j(await ana.get('/api/admin/line?key=adm'))
    check('the stats page shows who started the session (a visitor click)', a['started_by'] == 'visitor ana@example.com' and a['source'] == 'visitor', a['started_by'])
    await admin(ana, 'free')

    print('\n10. the web call is gone; the landing page only describes the FaceTime flow')
    for path in ('/app', '/api/token', '/api/line/start', '/api/line/end'):
        r = await ana.get(path) if path in ('/app', '/api/token') else await ana.post(path)
        check(f'{path} is gone (404/405)', r.status in (404, 405), r.status)
    page = await (await ana.get('/')).text(); low = page.lower()
    old = ['href="/app', "'/app", '"/app', '/api/token', 'in the browser', 'in your browser', 'nothing to install', 'body tracking',
           'your camera', 'livekit', 'phone_test', 'mint_token', 'web call']
    hits = [p for p in old if p in low]
    check('no link to /app and none of the old browser-call phrases', not hits, hits or 'clean')
    hero = page[page.index('<header class="hero">'):page.index('</header>')]; nav = page[page.index('<nav'):page.index('</nav>')]
    check('one email field in the hero; FaceTime label; github not in the nav; "I\'m done dancing" button',
          hero.count('type="email"') == 1 and 'FaceTime Brodie, your dance coach' in hero and 'github' not in nav.lower() and "I'm done dancing" in hero)

    print('\n11. the dev preview switch is off unless DEV_PREVIEW=1')
    for q in ('free', 'busy', 'mine', 'offline'):
        r = await ben.get('/api/line?preview=' + q); d = await j(r)
        check(f'?preview={q} ignored on the live site', 'preview' not in d)
    os.environ['DEV_PREVIEW'] = '1'
    d = await j(await anon.get('/api/line?preview=busy'))
    check('with DEV_PREVIEW=1: the busy preview', d.get('state') == 'busy', d.get('seconds_left'))
    os.environ.pop('DEV_PREVIEW')

    for c in (anon, ana, ben, cy, again): await c.close()
    await srv.close()


if __name__ == '__main__':
    asyncio.run(main())
    bad = [n for n, ok in RESULTS if not ok]
    print(f'\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed' + (f'; FAILED: {bad}' if bad else ''))
    sys.exit(1 if bad else 0)
