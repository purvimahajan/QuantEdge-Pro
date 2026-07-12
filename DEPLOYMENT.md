# Deploying Optika Publicly

## Recommended path: Streamlit Community Cloud (free, fastest, built for this)

Since Optika is a pure Streamlit app with no paid API keys required, Community
Cloud is the natural fit — it's free, deploys straight from GitHub, and
handles HTTPS/hosting for you.

### 1. Push the repo to GitHub
```bash
cd optika
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/optika.git
git push -u origin main
```
Make sure `app.py`, `requirements.txt`, and `.streamlit/config.toml` are all
in the repo root (the `.gitignore` included already excludes local caches
and the SQLite track-record file — see the note on persistence below).

### 2. Deploy
1. Go to **share.streamlit.io** and sign in with GitHub
2. Click **"New app"** → **"Use existing repo"**
3. Select your repo, branch (`main`), and main file path (`app.py`)
4. Click **"Deploy"**

Most apps go live within a few minutes. Every time you push to `main`
afterward, the live app updates automatically.

### 3. Set your URL
By default you get `<something-random>.streamlit.app`. In your app's
**Settings** (via the "Manage app" menu), you can customize the subdomain,
e.g. `optika.streamlit.app` (subject to availability) — this is a
**subdomain of streamlit.app**, not a fully custom domain like
`optika.com`. See "Custom domain" below if you need the latter.

### 4. Secrets (not needed yet, but for later)
Optika doesn't currently use any API keys — Yahoo Finance, NSE India, and
Stooq are all called without authentication. If you later add a paid data
source (e.g. a broker API), add it via the **"Secrets"** field in your app
settings (paste `secrets.toml`-formatted content) rather than hardcoding it.

---

## Pre-launch checklist specific to Optika

**1. SQLite track record won't persist long-term.**
Community Cloud's filesystem is ephemeral — `optika_track_record.db` gets
wiped on redeploys and periodic app restarts. Fine for a demo/portfolio
launch; if you want a durable track record, swap it for a free-tier hosted
Postgres (e.g. Supabase) via `st.connection` — the `log_recommendation()` /
`get_recommendation_history()` functions are isolated enough to swap out
without touching the rest of the app.

**2. NSE India fallback may get blocked more under public load.**
More traffic = more requests hitting NSE's bot detection, and it already
blocks a meaningful fraction of requests (see the earlier fix). This is
exactly why the CSV/Excel upload + manual entry option exists — consider
surfacing it more prominently (e.g. a banner) if you expect a lot of Indian
equity traffic, since it sidesteps the NSE reliability issue entirely.

**3. Resource limits on the free tier.**
Community Cloud's free tier gives ~1 GB RAM per app. The Heston Monte Carlo
(20k paths × 100 steps) and Random Forest vol forecaster are the heaviest
operations — they should run fine, but if you notice slowness under load,
reduce `n_paths`/`n_steps` defaults in `heston_mc_price()` and
`merton_jump_price()`, or add `st.cache_data` around the advanced-model
calls keyed on (ticker, strike, expiry).

**4. Legal/disclaimer visibility.**
The "educational tool, not financial advice" badge is in the hero header and
final recommendation — for a public launch, consider also adding a one-time
disclaimer acknowledgment (e.g. a checkbox or modal on first visit) since
financial-adjacent tools draw more scrutiny once publicly discoverable.

**5. Rate limiting your own outbound calls.**
`st.cache_data(ttl=...)` is already applied to all external data calls,
which naturally throttles repeat requests for the same ticker. Worth
monitoring Yahoo Finance's response if traffic grows — `yfinance` is
unofficial and can rate-limit aggressively-polling apps.

---

## Alternative: a host with real custom-domain + persistent storage support

If you want `optika.yourdomain.com` with a proper CNAME and durable local
storage (so SQLite actually persists), Streamlit Community Cloud isn't the
right fit long-term. Reasonable next steps, roughly in order of effort:

| Platform | Custom domain | Persistent disk | Free tier | Notes |
|---|---|---|---|---|
| **Render** | Yes (native) | Yes (paid add-on) | Yes (spins down when idle) | Easiest "real" host; same GitHub-push workflow |
| **Railway** | Yes (native) | Yes | Small free credit | Similar to Render, slightly more generous free compute |
| **Hugging Face Spaces** | Via HF subdomain, custom domain on paid | Limited | Yes | Popular for ML-adjacent public demos |
| **Fly.io** | Yes | Yes (volumes) | Small free tier | More control, requires a `Dockerfile` |
| **Your own VPS + Docker** | Yes, fully | Yes, fully | N/A (pay for VPS) | Most control, most setup work |

For any of these, the app itself doesn't need to change — just add a
`Dockerfile` if the platform requires one:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Quick recommendation

**Start on Streamlit Community Cloud today** — it's free, takes 10 minutes,
and gives you a shareable public link immediately (great for a portfolio/
resume link right now). Migrate to Render or similar later only if you
specifically need a custom domain or durable storage for the track record.
