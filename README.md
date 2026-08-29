🎛️ Wavy Outreach

One app for the whole pipeline: collect artists from YouTube playlists, enrich them with Apify, write personalized messages with AI, and send + track your outreach. Replaces the separate youtube-artist-scraper and Artist-AI-Outreach apps — same features, one database, no CSV shuffling between apps.

Pages
Dashboard — pipeline meters (Leads → Contactable → Drafted → Contacted → Replied), reply rate, and "Next up" actions computed from your data.
Collect — sync a playlist, then bulk-scrape Instagram bios/emails and Google-search missing IG profiles via Apify.
Write — Gemini cleans artist/song names. Messages are written through Claude.ai for free: download a batch CSV, paste the built-in prompt into claude.ai with the file attached, import the finished CSV back. Batches self-advance — the next download always contains whoever is still pending.
Send — review queue: one artist at a time, one editable message, "Open in Mail" pre-filled for email leads, copy-ready block for DMs, one-click "Mark as reached out". Follow-ups due after 7 days surface below.
Leads — the full editable table with search, filters, and status colors.
Settings — API keys, backup/restore, blacklist, factory reset.
Deploy on Streamlit Cloud
Create a new GitHub repo and add these files, keeping the folder structure:
   app.py
   requirements.txt
   .streamlit/config.toml   <- the theme lives here

Do not commit secrets.toml — the example file shows the format only. 2. On share.streamlit.io, create a new app pointing at app.py. 3. In the app's Settings → Secrets, paste:

toml
   YOUTUBE_API_KEY = "..."
   APIFY_API_TOKEN = "..."
   GEMINI_API_KEY = "..."

Use a fresh YouTube key — the one that was hardcoded in the old repo is public and should be deleted in Google Cloud Console.

Migrating your data from the old apps
In each old app, download the CSV backup.
In Wavy Outreach: Settings → Backup & restore → Restore from a backup. Both old export formats are accepted; missing columns are filled in automatically and old [Claude Error] strings are scrubbed.
Important: storage is temporary

Streamlit Cloud wipes the app's disk whenever it restarts (redeploys, inactivity, resource resets). Download a backup in Settings after every session and restore it when the data disappears. The backup is a single .zip containing your leads plus the blacklist, seen-video history, and scrape memory, so a restore puts everything back exactly as it was. Dedupe is also restart-proof on its own: the leads table itself records what has been scraped, so even without a restore the app won't re-scrape or re-search artists it already processed. If manual backups get old, the next upgrade is moving storage to Google Sheets or Supabase — the app's save_db/load_db functions are the only two places that would change.

Models

Name cleaning runs on Gemini 3.6 Flash via the google-genai SDK; if Google retires the model again, the app auto-discovers a current Flash model your key supports. Outreach messages are written through Claude.ai using the app's built-in prompt — no message-writing API, no rate limits, and you review every batch before it enters the database.
