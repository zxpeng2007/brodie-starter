# Hosting Brodie Dance

## Handoff checklist (deploying the website on a remote host)

1. Build from the repo root with the `Dockerfile` (`main` branch, Python 3.12 slim, `requirements-web.txt`; no torch, no MediaPipe).
   The container listens on `PORT` (default 8765) and `HOST=0.0.0.0`.
2. Set these environment variables on the host (values come from Steven privately, never from the repo):
   `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `AGENT_NAME=brodie-dance`, `DANCE=sbbyj`,
   `SESSION_SECRET` (any long random string), `ADMIN_KEY` (any string, for the waitlist CSV),
   optional `FACETIME_NUMBER`, `FORK_URL=https://github.com/zxpeng2007/ivyhacks-brodie-dance`.
3. Give the container a persistent disk and point `BRODIE_DB` at it (e.g. `/data/brodie.db`), or accounts and the
   waitlist vanish on every redeploy. To carry over the signups collected so far, copy Steven's `brodie.db` file there
   before the first start (or merge the CSV later).
4. Check: `GET /api/health` returns `"livekit": true`; `GET /api/waitlist` returns the count; the landing page at `/`.
5. Tell Steven the public https URL. The Vercel copy of the landing page posts its waitlist to whatever
   `<meta name="brodie-api">` in `web/index.html` says; it must change from the temporary tunnel to the host URL
   (one commit; Vercel redeploys itself). Once the host serves the landing page too, the Vercel copy is optional.
6. The coach (the worker) keeps running on Steven's PC for now; it needs the organisers' data bundle and the models,
   which are not in git. Moving it later: README steps 1-2 plus `bash scripts/demo_worker.sh` with the same
   `LIVEKIT_*` and `AGENT_NAME`.

Three pieces. Only the first two go on a hosting service.

| Piece | What | Where |
| --- | --- | --- |
| Website (`brodie/server.py`) | Landing page at `/`, email sign-in (`/api/start`), Brodie's line (`/api/line`, `/api/line/claim`, `/api/line/done`), the stats page's line controls (`/api/admin/line`), `/api/config`, `/api/dances` | Any Docker host: Render, Railway, Fly, Cloud Run |
| LiveKit Cloud | Rooms, media, agent dispatch | cloud.livekit.io (free project) |
| Brodie's worker (`facetime.worker --dance`) | The coach: pose, judge, voice, the lesson video | A machine with the data bundle and the Python env (a laptop is fine); it connects out to LiveKit, no inbound port |

## 1. Deploy the website

Build from the repo root with the `Dockerfile` (it copies `brodie/`, `web/`, the reference loader and `data/dances/*.npz`; no torch, no MediaPipe).

Environment variables on the host:

| Variable | Value |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | from the LiveKit project (Settings, then Keys) |
| `AGENT_NAME` | `brodie-dance` (must match the worker) |
| `DANCE` | `sbbyj` (the lesson the worker runs; "live now" on the landing page) |
| `HEARTBEAT_KEY` | any long random string, the same as the worker's `BRODIE_HEARTBEAT_KEY`: the worker's heartbeat decides online / offline on the page (unset: never shown as offline) |
| `SESSION_SECRET` | any long random string, so logins survive a redeploy |
| `BRODIE_DB` | path of the sqlite file, on a persistent disk if the host has one (default `data/brodie.db`) |
| `FACETIME_NUMBER` | the Chert FaceTime line, e.g. `+16282892384` (any formatting is fine: the link uses the digits, the page shows `+1 (628) 289-2384`); shown only after a signed-in visitor clicks "FaceTime Brodie" (that click starts their 15-minute session) |
| `LINE_CAP_S` | session length in seconds (optional, default 900 = 15 minutes) |
| `ADMIN_KEY` on `/stats.html` | the stats page also shows Brodie's line and has Mark busy (1 to 15 minutes), Add 5 minutes and Mark free; all need the key |
| `DEV_PREVIEW` | never on the host. `1` on a dev machine turns on `?preview=free`, `busy`, `mine`, `offline`, `loggedout` |
| `FORK_URL` | the team's GitHub fork (optional) |
| `HOST` / `PORT` | `0.0.0.0` and whatever the host assigns (the Dockerfile sets both; most hosts override `PORT`) |

Accounts: visitors enter their email on the landing page (one field, `/api/start`) and get a signed, HttpOnly session cookie;
a new email creates the account. Only signed-in visitors see Brodie's status and the FaceTime number (`/api/line`). Lessons
are FaceTime only: the web call page and `/api/token` are gone.

**Email-only sign-in has no check that the visitor owns the address** (anyone who types an email is signed into that account).
That is acceptable only while accounts hold nothing but name, email and call count. Don't store anything personal on accounts
(scores tied to a person, phone numbers, recordings) unless sign-in gets a real check, such as an emailed link or code.
The cookie is marked Secure when the request arrives over https (directly or via `X-Forwarded-Proto`).

## 2. Run the worker

On the machine that has the data bundle, with the same `LIVEKIT_*` and `AGENT_NAME` in `.env.local`:

```
bash scripts/demo_worker.sh
```

Add to that machine's `.env.local` so the website knows Brodie is up: `BRODIE_SERVER=https://<server>` and
`BRODIE_HEARTBEAT_KEY=<the server's HEARTBEAT_KEY>`. The worker then POSTs `/api/heartbeat` every 20 s; when it stops (laptop
closed, worker stopped) the page says "Brodie is offline right now" within about a minute.

### Brodie's line (website only)

The worker is not involved beyond its heartbeat. A session starts when a signed-in visitor clicks "FaceTime Brodie" on the
landing page: the line is theirs for 15 minutes (they keep the number, the button and an "I'm done dancing" button), and
every other visitor sees "Brodie is in a session with another person right now" with a live countdown and no number. The
session ends when that visitor clicks "I'm done dancing", when the 15 minutes run out, or from the stats page (Mark free).
Two clicks at once: the first gets the line. Sessions are stored in the sqlite file (`BRODIE_DB`), so they survive a restart.

The line follows clicks, not calls: someone who calls the number without clicking (or after their session ran out) is not
seen by it. The stats page's Mark busy / Add 5 minutes / Mark free correct it by hand; every change is logged there.

LiveKit dispatches that worker into the Chert FaceTime line's rooms (the website creates no rooms). The
worker's `DANCE` should match the website's.

## 3. Check it

```
curl https://<server>/api/health     # ok, agent_name, dance, livekit (keys present), users
curl https://<server>/api/config     # live, facetime number, fork
curl https://<server>/api/dances     # the lessons with parts, bpm, steps
```

## 4. The landing page on Vercel (launch day)

The landing page is static, so it can go up on Vercel in a minute while the server runs elsewhere:

1. In `web/index.html` set `<meta name="brodie-api" content="https://<your server>">` to the public https URL of the
   server (a Cloudflare quick tunnel to this PC works: `cloudflared tunnel --url http://localhost:8099`, or the hosted
   server from section 1). The waitlist posts there across origins; the server allows it (CORS on `/api/waitlist` only).
2. Deploy the `web/` folder: `cd web && vercel --prod` (or drag the folder onto vercel.com). `web/vercel.json` serves
   the landing page at `/`.
3. On the Vercel copy, accounts and the call page are hidden (they need the server's own origin for the session cookie);
   visitors join the waitlist. For a quick check without redeploying, open `https://<vercel>/?api=https://<server>`.

The waitlist lives in the server's sqlite file: `GET /api/waitlist` is the public count, `GET /api/waitlist/export?key=<ADMIN_KEY>`
is the CSV for the team (set `ADMIN_KEY` in the server's environment).

## Stats

Every call is logged (`call_log` table: who, when, which dance) from the moment this is deployed. Open `/stats.html` on the server
or the Vercel link and enter `ADMIN_KEY`: waitlist, sign-ups, callers, calls, who came back on another day, D1/D3/D7 return and a
per-day table (days in New York time). `GET /api/stats?key=<ADMIN_KEY>` is the same as JSON. Set `TEAM_EMAILS` (comma-separated)
so the team's own test accounts are left out of every number. Keep the key out of the group chat: the page shows nothing personal,
but the same key unlocks the waitlist CSV.
