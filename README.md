# 🎛️ Wavy Outreach

One app for the whole pipeline: collect artists from YouTube playlists, enrich
them with Apify, write personalized messages with AI, and send + track your
outreach. Replaces the separate `youtube-artist-scraper` and
`Artist-AI-Outreach` apps — same features, one database, no CSV shuffling
between apps.

## Pages

- **Dashboard** — pipeline meters (Leads → Contactable → Drafted → Contacted → Replied), reply rate, and "Next up" actions computed from your data.
- **Collect** — sync a playlist, then bulk-scrape Instagram bios/emails and Google-search missing IG profiles via Apify.
- **Write** — Gemini cleans artist/song names, Claude drafts an email + DM per artist.
- **Send** — review queue: one artist at a time, editable email with an "Open in Mail" button (pre-filled), copy-ready DM, one-click "Mark as reached out". Follow-ups due after 7 days surface below.
- **Leads** — the full editable table with search, filters, and status colors.
- **Settings** — API keys, backup/restore, blacklist, factory reset.

## Deploy on Streamlit Cloud

1. Create a new GitHub repo and add these files, keeping the folder structure:
   ```
   app.py
   requirements.txt
   .streamlit/config.toml   <- the theme lives here
   ```
   Do **not** commit `secrets.toml` — the example file shows the format only.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   YOUTUBE_API_KEY = "..."
   APIFY_API_TOKEN = "..."
   GEMINI_API_KEY = "..."
   ANTHROPIC_API_KEY = "..."
   ```
   Use a **fresh** YouTube key — the one that was hardcoded in the old repo is
   public and should be deleted in Google Cloud Console.

## Migrating your data from the old apps

1. In each old app, download the CSV backup.
2. In Wavy Outreach: **Settings → Backup & restore → Restore from a backup**.
   Both old export formats are accepted; missing columns are filled in
   automatically and old `[Claude Error]` strings are scrubbed.

## Important: storage is temporary

Streamlit Cloud wipes the app's disk whenever it restarts (redeploys,
inactivity, resource resets). **Download a backup in Settings after every
session** and restore it when needed. If this gets old, the next upgrade is
moving storage to Google Sheets or Supabase — the app's `save_db`/`load_db`
functions are the only two places that would change.

## Models

- Messages: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- Name cleaning: Gemini 2.5 Flash via the `google-genai` SDK
