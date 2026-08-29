"""
Wavy Outreach — one app for the whole pipeline.

Collect leads from YouTube playlists, enrich them with Apify, write
personalized messages with AI, and send + track outreach — all against a
single database. Replaces the separate youtube-artist-scraper and
Artist-AI-Outreach apps.
"""

import io
import os
import uuid
import re
import json
import time
import zipfile
from urllib.parse import quote

import pandas as pd
import streamlit as st

from google import genai as google_genai
from google.genai import types as genai_types

# ----------------------------------------------------------------------------
# App config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wavy Outreach",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#F0A93B"  # VU-meter amber (matches .streamlit/config.toml)

# Storage files (note: on Streamlit Cloud this disk is temporary — the app
# nudges you to download backups and can restore them in Settings)
DB_FILE = "wavy_outreach_db.csv"
SEEN_VIDEOS_FILE = "seen_videos.json"
BLACKLIST_FILE = "blacklist.json"
LAST_PLAYLIST_FILE = "last_playlist.txt"
SCRAPED_IGS_FILE = "scraped_igs.json"
SCRAPED_GOOGLES_FILE = "scraped_googles.json"

GEMINI_MODEL = "gemini-3.6-flash"  # auto-falls back via resolve_gemini_model if retired
APIFY_POLL_TIMEOUT_SECS = 15 * 60
FOLLOW_UP_DAYS = 7

# ----------------------------------------------------------------------------
# Keys — secrets/env, with optional per-session override from Settings
# ----------------------------------------------------------------------------
KEY_NAMES = {
    "YOUTUBE_API_KEY": "YouTube",
    "APIFY_API_TOKEN": "Apify",
    "GEMINI_API_KEY": "Gemini",
}

def get_secret(name):
    try:
        val = st.secrets.get(name)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(name, "")

def get_key(name):
    override = str(st.session_state.get(f"key_{name}", "") or "").strip()
    return override or get_secret(name)

# ----------------------------------------------------------------------------
# Schema — one lead table for the whole pipeline
# ----------------------------------------------------------------------------
COLUMN_ORDER = [
    "🗑️ Blacklist", "❌ Remove", "Reached Out", "Replied", "Reached_Out_Date",
    "Channel_ID", "Channel Name", "Cleaned Artist", "Cleaned Song",
    "Song Name", "Subscribers", "Email Address", "Instagram", "IG Followers",
    "IG Bio", "Draft Message", "🔄 Regenerate",
    "Channel_URL", "Google Search Status", "🔍 Quick Search",
]

TEXT_DEFAULTS = {
    "Channel_ID": "", "Channel Name": "", "Cleaned Artist": "", "Cleaned Song": "",
    "Song Name": "Unknown", "Email Address": "None Found", "Instagram": "None",
    "IG Bio": "Not Scanned", "Draft Message": "",
    "Channel_URL": "", "Google Search Status": "Not Searched", "🔍 Quick Search": "",
}
BOOL_COLS = ["🗑️ Blacklist", "❌ Remove", "Reached Out", "Replied", "🔄 Regenerate"]
INT_COLS = ["Subscribers", "IG Followers"]

def quick_search_url(name):
    return f"https://www.google.com/search?q=%22{str(name).replace(' ', '+')}%22+AND+(email+OR+instagram)"

def ensure_schema(df):
    """Add missing columns, fix dtypes, and order columns. Never raises."""
    if df is None:
        return pd.DataFrame(columns=COLUMN_ORDER)
    df = df.copy()

    # Migrate the legacy two-draft schema (Draft Email / Draft DM) into the
    # single Draft Message column, scrubbing old error strings on the way.
    if "Draft Message" not in df.columns:
        df["Draft Message"] = ""
    df["Draft Message"] = df["Draft Message"].fillna("").astype(str).replace("nan", "")
    for legacy in ["Draft DM", "Draft Email"]:
        if legacy in df.columns:
            vals = df[legacy].fillna("").astype(str).replace("nan", "")
            vals = vals.apply(lambda v: "" if v.strip().startswith("[Claude Error") else v)
            empty = df["Draft Message"].str.strip() == ""
            df.loc[empty, "Draft Message"] = vals[empty]
            df = df.drop(columns=[legacy])

    for col, default in TEXT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        # Empty text columns reload from CSV as NaN (float dtype), which
        # crashes st.data_editor's TextColumn config — force strings.
        df[col] = df[col].fillna(default).astype(str).replace("nan", default)

    for col in BOOL_COLS:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].map(lambda v: str(v).strip().lower() == "true" if not isinstance(v, bool) else v)
        df[col] = df[col].fillna(False).astype(bool)

    for col in INT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "Reached_Out_Date" not in df.columns:
        df["Reached_Out_Date"] = pd.NaT
    df["Reached_Out_Date"] = pd.to_datetime(df["Reached_Out_Date"], errors="coerce")

    # Old "[Claude Error]" strings must never block regeneration
    df["Draft Message"] = df["Draft Message"].apply(lambda v: "" if str(v).strip().startswith("[Claude Error") else v)

    # Keep only the first IG link if a comma list slipped in
    df["Instagram"] = df["Instagram"].apply(
        lambda x: str(x).split(",")[0].strip() if is_valid_data(x) else "None"
    )

    needs_qs = ~df["🔍 Quick Search"].apply(is_valid_data)
    if needs_qs.any():
        df.loc[needs_qs, "🔍 Quick Search"] = df.loc[needs_qs, "Channel Name"].apply(quick_search_url)

    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]

def load_db():
    if os.path.exists(DB_FILE):
        try:
            return ensure_schema(pd.read_csv(DB_FILE))
        except Exception as e:
            st.warning(f"Could not read {DB_FILE} ({e}). Starting with an empty database.")
    return ensure_schema(pd.DataFrame())

UI_ONLY_COLS = ["🗑️ Blacklist", "❌ Remove"]

def save_db(df):
    df.drop(columns=UI_ONLY_COLS, errors="ignore").to_csv(DB_FILE, index=False)

def load_json_set(filepath):
    # Values are stored as-is: YouTube video/channel IDs are case-sensitive.
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return set(str(item) for item in json.load(f))
        except Exception:
            return set()
    return set()

def save_json_set(data_set, filepath):
    with open(filepath, "w") as f:
        json.dump(sorted(str(item) for item in data_set), f)

def load_last_playlist():
    if os.path.exists(LAST_PLAYLIST_FILE):
        try:
            with open(LAST_PLAYLIST_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""

def save_last_playlist(url):
    with open(LAST_PLAYLIST_FILE, "w") as f:
        f.write(url)

# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------
def is_valid_data(val):
    v = str(val).lower().strip()
    return v not in ["none", "none found", "", "nan", "not scanned", "not searched", "empty bio"]

def safe_int(val, default=0):
    try:
        v = pd.to_numeric(val, errors="coerce")
        return default if pd.isna(v) else int(v)
    except Exception:
        return default

FAKE_EMAIL_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp3", ".mp4", ".wav")

def extract_emails(text):
    if not text:
        return []
    text = re.sub(
        r'\b([a-zA-Z0-9._%+-]+)\s+(?:at|\(at\)|\[at\])\s+([a-zA-Z0-9.-]+)\s+(?:dot|\(dot\)|\[dot\])\s+([a-zA-Z]{2,})\b',
        r'\1@\2.\3', text, flags=re.IGNORECASE,
    )
    found = re.findall(r'[a-zA-Z0-9%._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return [e for e in set(found) if not e.lower().endswith(FAKE_EMAIL_SUFFIXES)]

def extract_instagram(text):
    if not text:
        return "None"
    all_urls = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', text)
    ig_handles = re.findall(r'(?:ig|instagram):\s*@?([a-zA-Z0-9_.-]+)', text, re.IGNORECASE)

    ig_links = []
    for url in all_urls:
        url_lower = url.lower()
        if 'instagram.com' in url_lower:
            bad = ['/p/', '/reel/', '/reels/', '/explore/', '/tags/', '/audio/', 'instagram.com/https']
            if not any(b in url_lower for b in bad):
                clean_url = url.split('?')[0].strip('/').rstrip('/')
                if clean_url.lower().startswith("www."):
                    clean_url = "https://" + clean_url
                ig_links.append(clean_url)

    for h in ig_handles:
        if h.lower() != 'https':
            ig_links.append(f"https://instagram.com/{h}")

    return ig_links[0] if ig_links else "None"

def extract_playlist_id(url_or_id):
    if "list=" in url_or_id:
        return url_or_id.split("list=")[1].split("&")[0]
    return url_or_id.strip()

def extract_ig_username(ig_url):
    match = re.search(r'instagram\.com/([a-zA-Z0-9_.-]+)', ig_url, re.IGNORECASE)
    if match:
        return match.group(1).split("?")[0].strip("/")
    return ig_url.replace("@", "").strip()

def aggressive_regex_clean(channel_name, video_title):
    title = str(video_title)
    channel = str(channel_name).strip()
    title = re.sub(r'\(.*?\)', '', title)
    title = re.sub(r'\[.*?\]', '', title)
    noise = [
        r'(?i)official\s*(music)?\s*(video)?', r'(?i)lyric\s*(video)?', r'(?i)visualizer',
        r'(?i)vevo', r'(?i)shot\s*by.*', r'(?i)dir(ected)?\s*by.*', r'(?i)prod(uced)?\s*by.*',
        r'(?i)feat\..*', r'(?i)ft\..*', r'(?i)featuring.*', r'(?i)audio',
    ]
    for pattern in noise:
        title = re.sub(pattern, '', title)
    if channel:
        title = re.sub(f'(?i){re.escape(channel)}', '', title)
    title = re.sub(r'[-:|~]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'[,.\/\\]+$', '', title).strip()
    return title.replace('"', '').replace("'", "")

# ----------------------------------------------------------------------------
# AI helpers
# ----------------------------------------------------------------------------
ALEX_WAVY_PERSONA = """You are Alex Wavy, a mixing and mastering engineer from Europe who lives in trap, drill, and hip-hop. Over 5 years of experience on a hybrid analog/digital setup.

VOICE:
- Real recognizes real: casual, direct, respectful. Like one artist texting another.
- Short sentences. No corporate talk, no marketing language, no fake friendliness.
- One genuine line about their music is worth more than five compliments.
- Mild slang and profanity are fine when natural ("I fuck with your sound").
- Never use emojis, subject lines, sign-offs, links, or placeholders like [Name]."""

def resolve_gemini_model(client):
    """Pick a Gemini model that actually works on this account.

    Google retires model names regularly (1.5 -> 2.5 -> 3.x). Try the preferred
    model first; if it's gone, discover a current Flash model from the API's
    own model list. The result is cached for the session."""
    cached = st.session_state.get("_gemini_model")
    if cached:
        return cached

    candidates = [GEMINI_MODEL]
    try:
        discovered = []
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").split("/")[-1]
            if "flash" in name and not any(x in name for x in
                    ["image", "live", "tts", "audio", "embedding", "lite", "veo", "nano"]):
                discovered.append(name)
        candidates += sorted(set(discovered) - set(candidates), reverse=True)
    except Exception:
        pass

    for cand in candidates:
        try:
            client.models.generate_content(model=cand, contents="ok")
            st.session_state["_gemini_model"] = cand
            return cand
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "exhaust" in msg or "rate" in msg:
                # Model exists — the account just hit a limit. Use it anyway.
                st.session_state["_gemini_model"] = cand
                return cand
            continue
    return GEMINI_MODEL

def gemini_config(model):
    kwargs = {"response_mime_type": "application/json"}
    if str(model).startswith("gemini-3"):
        try:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel.LOW  # fast + cheap for extraction
            )
        except Exception:
            pass
    return genai_types.GenerateContentConfig(**kwargs)

def clean_with_gemini(client, model, channel_name, video_title):
    """Returns (artist, song, error). Error is None on success."""
    pre_cleaned = aggressive_regex_clean(channel_name, video_title)
    prompt = f"""
    Analyze this YouTube data to extract the REAL Artist Name and Song Name.
    Channel Name: "{channel_name}"
    Pre-Cleaned Video Title: "{pre_cleaned}"

    Rules:
    1. The "song" value MUST NOT contain the artist's name. Remove it if it is still there.
    2. The "song" value MUST NOT contain hyphens (-), colons (:), or other separator symbols.
    3. If there are multiple songs listed, separate them with a comma in the song field.
    4. Return ONLY a valid JSON object in this exact format:
    {{"artist": "Cleaned Artist Name", "song": "Cleaned Song Name"}}
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gemini_config(model),
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        artist = str(data.get("artist") or channel_name).strip()
        song = str(data.get("song") or pre_cleaned).strip()
        return artist, song, None
    except Exception as e:
        return str(channel_name).strip(), pre_cleaned, str(e)

CLAUDE_PROMPT = """I'm Alex Wavy, a mixing and mastering engineer from Europe specializing in trap, drill, and hip-hop — 5+ years of experience on a hybrid analog/digital setup. You're going to write my cold outreach messages.

I've attached a CSV with the columns: Channel_ID, Artist, Song, Draft Message.

TASK: For every row, write ONE outreach message in the Draft Message column. Each message gets sent exactly as written — sometimes as an email body, sometimes as an Instagram DM — so it must read naturally as both.

STRUCTURE every message like this, in fresh wording each time:
1. GREETING with their name: "Hey {Artist}," / "What's up {Artist}," / "Yo {Artist}," — vary it. If the Artist value is clearly a channel name or contains junk, clean it up or greet without a name.
2. THE HOOK: I listened to {Song} and a few of their other tracks, and I really mess with their art — say it like I mean it, because I do. Reference the actual track title naturally — strip any junk like hashtags, director tags, or feature lists when mentioning it. ("really mess with" counts as the one allowed casual touch.)
3. WHO I AM: Alex Wavy, mixing and mastering engineer specializing in trap, drill, and hip-hop. 5+ years, hybrid analog/digital studio.
4. THE OFFER — be specific about what to send: they reply with ONE track they're working on (a rough bounce or the session files, whatever they have) and I send back a finished professional mix within 48 hours, completely free.
5. THE INTENTION + WHY IT'S FREE — make it clear I genuinely want to work with this artist, and the free mix is how I'd like to start. One honest line: no catch — if they like the result, I want to keep working with them; if not, they keep the mix anyway. It must read like I picked them on purpose, not like a mass blast.
6. CLEAR CTA + SIGN-OFF: tell them exactly what to do ("just reply with the track") and sign off "– Alex Wavy".

STYLE RULES:
- 70 to 100 words. Short sentences. Clear and easy to read on a phone.
- Professional but human — a real engineer reaching out, not a marketing bot and not a try-hard.
- Keep slang minimal (one casual touch max) and no profanity.
- BANNED: "opportunity", "elevate", "next level", "game-changer", "incredible", "amazing", "I'd love to", "hope this finds you well", emojis, links, fake flattery.
- Vary the structure and wording between rows so no two messages look templated.
- Do NOT change Channel_ID, Artist, or Song. Fill ONLY the Draft Message column.
- If a row is clearly a media channel rather than an artist, adapt the offer: I'll mix a track free for any artist they work with.

OUTPUT: Give me back the complete CSV as a downloadable file — same columns, same rows, same order, Draft Message filled for every row. Use proper CSV quoting so commas inside messages don't break the format."""

def merge_message_csv(df, incoming):
    """Merge Draft Message values from a finished batch CSV back into the leads.
    Matches by Channel_ID first, then by Artist + Song. Returns (df, updated)."""
    incoming = incoming.copy()
    incoming.columns = [str(c).strip() for c in incoming.columns]
    colmap = {c.lower(): c for c in incoming.columns}

    msg_col = next((colmap[k] for k in ["draft message", "message", "draft_message", "outreach message"] if k in colmap), None)
    if not msg_col:
        raise ValueError("Couldn't find a 'Draft Message' column in that CSV.")
    id_col = colmap.get("channel_id")
    artist_col = next((colmap[k] for k in ["artist", "cleaned artist"] if k in colmap), None)
    song_col = next((colmap[k] for k in ["song", "cleaned song"] if k in colmap), None)

    updated, skipped = 0, 0
    for _, r in incoming.iterrows():
        text = str(r[msg_col]).strip()
        if not text or text.lower() == "nan":
            continue
        mask = None
        if id_col is not None and is_valid_data(r.get(id_col, "")):
            mask = df["Channel_ID"].astype(str) == str(r[id_col]).strip()
        if (mask is None or not mask.any()) and artist_col and song_col:
            mask = (
                (df["Cleaned Artist"].astype(str).str.strip().str.lower() == str(r[artist_col]).strip().lower())
                & (df["Cleaned Song"].astype(str).str.strip().str.lower() == str(r[song_col]).strip().lower())
            )
        if mask is not None and mask.any():
            df.loc[mask, "Draft Message"] = text
            df.loc[mask, "🔄 Regenerate"] = False
            updated += int(mask.sum())
        else:
            skipped += 1
    return df, updated, skipped

# ----------------------------------------------------------------------------
# Apify helper
# ----------------------------------------------------------------------------
def run_apify_and_poll(client, actor_id, run_input, total_targets, progress_bar, status_text, label):
    """Start an Apify actor and poll until it finishes. Returns (status, dataset_id)."""
    run_obj = client.actor(actor_id).start(run_input=run_input)

    run_id = run_obj.get("id") if isinstance(run_obj, dict) else getattr(run_obj, "id", None)
    dataset_id = run_obj.get("defaultDatasetId") if isinstance(run_obj, dict) else getattr(run_obj, "default_dataset_id", getattr(run_obj, "defaultDatasetId", ""))

    if not run_id or not dataset_id:
        return "FAILED (could not read run/dataset id from Apify)", dataset_id

    start_time = time.time()
    status = "UNKNOWN"
    while True:
        run_info = client.run(run_id).get()
        status = run_info.get("status") if isinstance(run_info, dict) else getattr(run_info, "status", "UNKNOWN")

        dataset_info = client.dataset(dataset_id).get()
        if not dataset_info:
            item_count = 0
        elif isinstance(dataset_info, dict):
            item_count = dataset_info.get("itemCount", 0)
        else:
            item_count = getattr(dataset_info, "item_count", getattr(dataset_info, "itemCount", 0))

        elapsed = int(time.time() - start_time)
        progress_val = min(0.95, item_count / total_targets) if total_targets > 0 else 0.5
        progress_bar.progress(progress_val)
        status_text.info(f"{label}: {item_count}/{total_targets} done · {elapsed}s · {status}")

        if status in ["SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"]:
            break
        if elapsed > APIFY_POLL_TIMEOUT_SECS:
            status = "TIMED-OUT (gave up waiting — check the run in your Apify console)"
            break
        time.sleep(1.5)

    return status, dataset_id

# ----------------------------------------------------------------------------
# Backup: one ZIP with the leads plus every memory file
# ----------------------------------------------------------------------------
MEMORY_FILES = [SEEN_VIDEOS_FILE, BLACKLIST_FILE, SCRAPED_IGS_FILE, SCRAPED_GOOGLES_FILE, LAST_PLAYLIST_FILE]

def build_backup_zip(df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(DB_FILE, df.drop(columns=UI_ONLY_COLS, errors="ignore").to_csv(index=False))
        for f in MEMORY_FILES:
            if os.path.exists(f):
                z.write(f, arcname=f)
    return buf.getvalue()

def restore_memory_from_zip(z, replace):
    """Restore the JSON memory files from a backup ZIP.
    Replace mode overwrites; merge mode unions with what's already there."""
    names = set(z.namelist())
    for f in [SEEN_VIDEOS_FILE, BLACKLIST_FILE, SCRAPED_IGS_FILE, SCRAPED_GOOGLES_FILE]:
        if f in names:
            try:
                incoming = set(str(i) for i in json.loads(z.read(f).decode("utf-8")))
            except Exception:
                continue
            merged = incoming if replace else (load_json_set(f) | incoming)
            save_json_set(merged, f)
    if LAST_PLAYLIST_FILE in names and (replace or not os.path.exists(LAST_PLAYLIST_FILE)):
        with open(LAST_PLAYLIST_FILE, "wb") as out:
            out.write(z.read(LAST_PLAYLIST_FILE))

# ----------------------------------------------------------------------------
# Flash messages that survive st.rerun()
# ----------------------------------------------------------------------------
def flash(level, msg):
    st.session_state["_flash"] = (level, msg)

def show_flash():
    if "_flash" in st.session_state:
        level, msg = st.session_state.pop("_flash")
        getattr(st, level, st.info)(msg)

# ----------------------------------------------------------------------------
# Pipeline stats
# ----------------------------------------------------------------------------
def ig_already_scanned(row):
    """True if this row's IG was already scraped — judged from the data itself,
    so dedupe survives even if the memory files are lost to a restart."""
    return str(row.get("IG Bio", "")).strip().lower() not in ["not scanned", "", "nan"]

def google_already_searched(row):
    return str(row.get("Google Search Status", "")).strip().lower().startswith("searched")

def has_any_draft(row):
    return bool(str(row.get("Draft Message", "")).strip())

def pipeline_stats(df):
    if df.empty:
        return {"total": 0, "contactable": 0, "drafted": 0, "contacted": 0, "replied": 0}
    contactable = df.apply(lambda r: is_valid_data(r["Email Address"]) or is_valid_data(r["Instagram"]), axis=1).sum()
    drafted = df.apply(has_any_draft, axis=1).sum()
    return {
        "total": len(df),
        "contactable": int(contactable),
        "drafted": int(drafted),
        "contacted": int(df["Reached Out"].sum()),
        "replied": int(df["Replied"].sum()),
    }

def compute_write_targets(df):
    clean_targets, msg_targets = [], []
    for idx, row in df.iterrows():
        if not is_valid_data(row.get("Cleaned Artist", "")) or not is_valid_data(row.get("Cleaned Song", "")):
            clean_targets.append(idx)
        has_contact = is_valid_data(row.get("Email Address", "")) or is_valid_data(row.get("Instagram", ""))
        force = bool(row.get("🔄 Regenerate", False))
        dm = str(row.get("Draft Message", "")).strip()
        if has_contact and (not dm or dm == "nan" or force):
            msg_targets.append(idx)
    return clean_targets, msg_targets

def enrich_targets(df, scraped_igs, scraped_googles):
    ig_pending, google_pending = 0, 0
    for _, row in df.iterrows():
        ig_val = str(row["Instagram"])
        if is_valid_data(ig_val):
            if (not is_valid_data(row["Email Address"]) or safe_int(row["IG Followers"]) <= 0) \
               and not ig_already_scanned(row) \
               and ig_val.split(",")[0].strip().rstrip("/").lower() not in scraped_igs:
                ig_pending += 1
        else:
            if not google_already_searched(row) \
               and str(row["Channel Name"]).strip().lower() not in scraped_googles:
                google_pending += 1
    return ig_pending, google_pending

# ----------------------------------------------------------------------------
# State init (runs on every rerun)
# ----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_db()

seen_videos = load_json_set(SEEN_VIDEOS_FILE)
blacklist = load_json_set(BLACKLIST_FILE)
scraped_igs = load_json_set(SCRAPED_IGS_FILE)
scraped_googles = load_json_set(SCRAPED_GOOGLES_FILE)

# ----------------------------------------------------------------------------
# Light global styling (theme lives in .streamlit/config.toml)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
""", unsafe_allow_html=True)

def vu_meter(label, count, total):
    pct = 0 if total <= 0 or count <= 0 else max(6, int(round(100 * count / total)))
    return f"""
    <div style="text-align:center;">
      <div style="height:110px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.10);
                  border-radius:8px;display:flex;align-items:flex-end;overflow:hidden;">
        <div style="width:100%;height:{pct}%;background:linear-gradient(180deg,{ACCENT},#7a5210);"></div>
      </div>
      <div style="margin-top:6px;font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;">{count}</div>
      <div style="font-size:0.72rem;letter-spacing:.09em;text-transform:uppercase;opacity:.6;">{label}</div>
    </div>"""

def highlight_rows(row):
    color = ''
    if row.get('Replied', False):
        color = 'background-color: rgba(39, 174, 96, 0.35)'
    elif row.get('Reached Out', False):
        date_val = row.get('Reached_Out_Date')
        if pd.notna(date_val):
            days_passed = (pd.Timestamp.now().normalize() - pd.to_datetime(date_val).normalize()).days
            color = 'background-color: rgba(231, 76, 60, 0.35)' if days_passed >= FOLLOW_UP_DAYS else 'background-color: rgba(240, 169, 59, 0.28)'
        else:
            color = 'background-color: rgba(240, 169, 59, 0.28)'
    return [color] * len(row)

def display_name(row):
    artist = str(row.get("Cleaned Artist", "")).strip()
    return artist if is_valid_data(artist) else str(row.get("Channel Name", "Unknown")).strip()

def display_song(row):
    song = str(row.get("Cleaned Song", "")).strip()
    return song if is_valid_data(song) else str(row.get("Song Name", "")).strip()

# ============================================================================
# PAGE: Dashboard
# ============================================================================
def page_dashboard():
    show_flash()
    st.title("🎛️ Wavy Outreach")
    st.caption("Find artists → write messages → track replies. One database, start to finish.")

    df = st.session_state.df
    stats = pipeline_stats(df)

    if stats["total"] == 0:
        st.info("No leads yet. Sync a playlist in **Collect** to load your first artists — or restore a backup in **Settings**.")
        c1, c2 = st.columns(2)
        if c1.button("🎧 Go to Collect", width="stretch"):
            st.switch_page(pg_collect)
        if c2.button("⚙️ Restore a backup", width="stretch"):
            st.switch_page(pg_settings)
        return

    # Signature: the pipeline as five channel-strip meters
    cols = st.columns(5)
    meters = [
        ("Leads", stats["total"]), ("Contactable", stats["contactable"]),
        ("Drafted", stats["drafted"]), ("Contacted", stats["contacted"]),
        ("Replied", stats["replied"]),
    ]
    for col, (label, count) in zip(cols, meters):
        col.markdown(vu_meter(label, count, stats["total"]), unsafe_allow_html=True)

    st.write("")
    m1, m2, m3 = st.columns(3)
    reply_rate = f"{(stats['replied'] / stats['contacted'] * 100):.0f}%" if stats["contacted"] else "—"
    with m1, st.container(border=True):
        st.metric("Reply rate", reply_rate)
    queue_count = int(df.apply(lambda r: has_any_draft(r) and not r["Reached Out"], axis=1).sum())
    with m2, st.container(border=True):
        st.metric("Ready to send", queue_count)
    due = follow_ups_due(df)
    with m3, st.container(border=True):
        st.metric("Follow-ups due", len(due))

    # Next actions, computed from the data
    st.subheader("Next up")
    clean_targets, msg_targets = compute_write_targets(df)
    ig_pending, google_pending = enrich_targets(df, scraped_igs, scraped_googles)

    actions = []
    if ig_pending:
        actions.append((f"Scrape **{ig_pending}** Instagram profiles for bios, followers, and emails.", "🎧 Open Collect", pg_collect))
    if google_pending:
        actions.append((f"Search Google for **{google_pending}** missing Instagram profiles.", "🎧 Open Collect", pg_collect))
    if clean_targets:
        actions.append((f"Clean artist & song names for **{len(clean_targets)}** leads.", "✍️ Open Write", pg_write))
    if msg_targets:
        actions.append((f"Write messages for **{len(msg_targets)}** artists.", "✍️ Open Write", pg_write))
    if queue_count:
        actions.append((f"**{queue_count}** drafts are ready to send.", "📤 Open Send", pg_send))
    if len(due):
        actions.append((f"**{len(due)}** contacts are {FOLLOW_UP_DAYS}+ days without a reply.", "📤 Open Send", pg_send))

    if not actions:
        st.success("All caught up. Sync your playlist again when new tracks drop.")
    for i, (text, label, page) in enumerate(actions):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(text)
            if c2.button(label, key=f"act_{i}", width="stretch"):
                st.switch_page(page)

def follow_ups_due(df):
    if df.empty:
        return df
    mask = (df["Reached Out"]) & (~df["Replied"]) & df["Reached_Out_Date"].notna()
    sub = df[mask].copy()
    if sub.empty:
        return sub
    days = (pd.Timestamp.now().normalize() - sub["Reached_Out_Date"].dt.normalize()).dt.days
    return sub[days >= FOLLOW_UP_DAYS]

# ============================================================================
# PAGE: Collect
# ============================================================================
def page_collect():
    show_flash()
    st.title("🎧 Collect")
    st.caption("Pull artists from a YouTube playlist, then enrich them with Instagram and Google data.")

    df = st.session_state.df

    # ---- Step 1: Sync playlist -------------------------------------------
    with st.container(border=True):
        st.subheader("1 · Sync playlist")
        playlist_input = st.text_input(
            "YouTube playlist URL or ID",
            value=load_last_playlist(),
            placeholder="https://www.youtube.com/playlist?list=PL...",
        )
        if st.button("Sync playlist", type="primary"):
            yt_key = get_key("YOUTUBE_API_KEY")
            if not yt_key:
                st.error("YouTube API key is missing. Add it in Settings.")
            elif not playlist_input:
                st.warning("Paste a playlist URL or ID first.")
            else:
                save_last_playlist(playlist_input)
                sync_playlist(playlist_input, yt_key)

    # ---- Step 2: Enrich with Apify ---------------------------------------
    ig_pending, google_pending = enrich_targets(df, scraped_igs, scraped_googles)
    with st.container(border=True):
        st.subheader("2 · Enrich")
        if not get_key("APIFY_API_TOKEN"):
            st.warning("Apify token is missing — add it in Settings to enable enrichment.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scrape Instagram bios**")
            st.caption(f"Pulls bio, follower count, and emails. {ig_pending} profiles pending.")
            if st.button("Scrape Instagram bios", disabled=(ig_pending == 0 or not get_key("APIFY_API_TOKEN"))):
                scrape_instagram()
        with c2:
            st.markdown("**Find missing Instagrams**")
            st.caption(f"Googles each artist for a profile link. {google_pending} artists pending.")
            if st.button("Search Google for IGs", disabled=(google_pending == 0 or not get_key("APIFY_API_TOKEN"))):
                scrape_google()

def sync_playlist(playlist_input, yt_key):
    from googleapiclient.discovery import build
    youtube = build("youtube", "v3", developerKey=yt_key, cache_discovery=False)

    playlist_id = extract_playlist_id(playlist_input)
    new_videos_found = 0
    new_channel_video_text = {}
    new_channel_video_titles = {}
    new_unique_channel_ids = []
    next_page_token = None

    df = st.session_state.df
    existing_channels = set(df["Channel_ID"].tolist()) if not df.empty else set()

    with st.spinner("1/2 · Checking playlist for newly added videos..."):
        try:
            while True:
                playlist_response = youtube.playlistItems().list(
                    part="snippet", playlistId=playlist_id, maxResults=50, pageToken=next_page_token
                ).execute()

                for item in playlist_response.get("items", []):
                    snippet = item.get("snippet", {})
                    video_id = snippet.get("resourceId", {}).get("videoId")
                    channel_id = snippet.get("videoOwnerChannelId")
                    video_desc = snippet.get("description", "")
                    video_title = snippet.get("title", "Unknown Title")

                    if video_id in seen_videos or channel_id in blacklist:
                        continue

                    seen_videos.add(video_id)
                    new_videos_found += 1

                    if channel_id and channel_id not in existing_channels:
                        if channel_id not in new_channel_video_text:
                            new_unique_channel_ids.append(channel_id)
                            new_channel_video_text[channel_id] = video_desc
                            new_channel_video_titles[channel_id] = video_title
                        else:
                            new_channel_video_text[channel_id] += " \n " + video_desc
                            if video_title not in new_channel_video_titles[channel_id]:
                                new_channel_video_titles[channel_id] += f", {video_title}"

                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break
        except Exception as e:
            st.error(f"Could not read that playlist: {e}")
            return

    save_json_set(seen_videos, SEEN_VIDEOS_FILE)

    if new_videos_found == 0:
        st.info("No new videos found — everything is up to date.")
        return

    if not new_unique_channel_ids:
        flash("success", f"Found {new_videos_found} new videos, all from channels you already track.")
        st.rerun()

    new_channels_data = []
    with st.spinner("2/2 · Extracting channel info, socials, and subscribers..."):
        try:
            for i in range(0, len(new_unique_channel_ids), 50):
                batch_ids = new_unique_channel_ids[i:i + 50]
                channel_response = youtube.channels().list(
                    part="snippet,statistics", id=",".join(batch_ids)
                ).execute()

                for ch in channel_response.get("items", []):
                    ch_id = ch["id"]
                    ch_snippet = ch.get("snippet", {})
                    ch_stats = ch.get("statistics", {})

                    channel_name = ch_snippet.get("title", "Unknown")
                    channel_desc = ch_snippet.get("description", "")
                    custom_url = ch_snippet.get("customUrl", "")
                    channel_url = f"https://www.youtube.com/{custom_url}" if custom_url else f"https://www.youtube.com/channel/{ch_id}"

                    full_text = channel_desc + " \n " + new_channel_video_text.get(ch_id, "")
                    emails = extract_emails(full_text)

                    new_channels_data.append({
                        "Channel_ID": ch_id,
                        "Channel Name": channel_name,
                        "Song Name": new_channel_video_titles.get(ch_id, "Unknown Title"),
                        "Subscribers": safe_int(ch_stats.get("subscriberCount", 0)),
                        "Email Address": ", ".join(emails) if emails else "None Found",
                        "Instagram": extract_instagram(full_text),
                        "Channel_URL": channel_url,
                        "🔍 Quick Search": quick_search_url(channel_name),
                    })
        except Exception as e:
            st.error(f"Could not fetch channel details: {e}")
            return

    new_df = ensure_schema(pd.DataFrame(new_channels_data))
    st.session_state.df = ensure_schema(pd.concat([st.session_state.df, new_df], ignore_index=True))
    save_db(st.session_state.df)
    flash("success", f"Added {len(new_channels_data)} new artists from {new_videos_found} new videos.")
    st.rerun()

def scrape_instagram():
    df = st.session_state.df
    target_usernames, idx_mapping = [], {}

    for idx in df.index:
        ig_val = str(df.loc[idx, "Instagram"])
        if is_valid_data(ig_val) and not ig_already_scanned(df.loc[idx]) \
           and (not is_valid_data(df.loc[idx, "Email Address"]) or safe_int(df.loc[idx, "IG Followers"]) <= 0):
            ig_url = ig_val.split(",")[0].strip().rstrip("/")
            ig_url_lower = ig_url.lower()
            if ig_url_lower not in scraped_igs:
                username = extract_ig_username(ig_url)
                username_lower = username.lower()
                if username_lower not in idx_mapping:
                    idx_mapping[username_lower] = {"indices": [], "url": ig_url_lower}
                    target_usernames.append(username)
                idx_mapping[username_lower]["indices"].append(idx)

    if not target_usernames:
        st.info("No unscanned Instagram profiles found.")
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.info(f"Starting Apify job for {len(target_usernames)} Instagram profiles...")

    try:
        from apify_client import ApifyClient
        client = ApifyClient(get_key("APIFY_API_TOKEN"))
        status, dataset_id = run_apify_and_poll(
            client, "apify/instagram-profile-scraper",
            {"usernames": target_usernames},
            len(target_usernames), progress_bar, status_text, "Scraping Instagram",
        )
        if status != "SUCCEEDED":
            status_text.error(f"Apify run ended with status: {status}")
            return

        status_text.info("Saving results to your leads...")
        emails_found, profiles_scraped = 0, 0
        for item in client.dataset(dataset_id).iterate_items():
            returned_username = str(item.get("username", "")).lower()
            if returned_username in idx_mapping:
                mapped = idx_mapping[returned_username]
                bio_text = str(item.get("biography", ""))
                followers_count = safe_int(item.get("followersCount", 0))

                emails = extract_emails(bio_text)
                if item.get("public_email"): emails.append(item["public_email"])
                if item.get("business_email"): emails.append(item["business_email"])
                clean_emails = [e for e in emails if "example" not in e.lower()]
                found_email = ", ".join(set(clean_emails)) if clean_emails else "None Found"

                for i in mapped["indices"]:
                    if i not in st.session_state.df.index:
                        continue
                    st.session_state.df.loc[i, "IG Bio"] = bio_text if bio_text.strip() else "Empty Bio"
                    st.session_state.df.loc[i, "IG Followers"] = followers_count
                    if found_email != "None Found" and not is_valid_data(st.session_state.df.loc[i, "Email Address"]):
                        st.session_state.df.loc[i, "Email Address"] = found_email
                        emails_found += 1

                scraped_igs.add(mapped["url"])
                profiles_scraped += 1

        save_json_set(scraped_igs, SCRAPED_IGS_FILE)
        save_db(st.session_state.df)
        flash("success", f"Scraped {profiles_scraped} Instagram bios and found {emails_found} new emails.")
        st.rerun()
    except Exception as e:
        status_text.error(f"Instagram scrape failed: {e}")

def scrape_google():
    df = st.session_state.df
    target_queries, idx_mapping = [], {}

    for idx in df.index:
        ig_val = str(df.loc[idx, "Instagram"])
        channel_name = str(df.loc[idx, "Channel Name"]).strip()
        channel_lower = channel_name.lower()
        if not is_valid_data(ig_val) and not google_already_searched(df.loc[idx]) and channel_lower not in scraped_googles:
            search_query = f'"{channel_name}" "instagram.com"'
            query_lower = search_query.lower()
            query_no_quotes = query_lower.replace('"', '')
            if query_lower not in idx_mapping:
                idx_mapping[query_lower] = {"indices": [], "channel": channel_lower}
                idx_mapping[query_no_quotes] = idx_mapping[query_lower]
                target_queries.append(search_query)
            if idx not in idx_mapping[query_lower]["indices"]:
                idx_mapping[query_lower]["indices"].append(idx)

    if not target_queries:
        st.info("No artists need a Google search right now.")
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.info(f"Starting Google search for {len(target_queries)} artists...")

    try:
        from apify_client import ApifyClient
        client = ApifyClient(get_key("APIFY_API_TOKEN"))
        status, dataset_id = run_apify_and_poll(
            client, "apify/google-search-scraper",
            {"queries": "\n".join(target_queries), "maxPagesPerQuery": 1, "resultsPerPage": 10},
            len(target_queries), progress_bar, status_text, "Searching Google",
        )
        if status != "SUCCEEDED":
            status_text.error(f"Apify run ended with status: {status}")
            return

        status_text.info("Saving results to your leads...")
        igs_found = 0
        for item in client.dataset(dataset_id).iterate_items():
            original_query = item.get("searchQuery", {}).get("term", "").lower()
            if original_query in idx_mapping:
                mapped = idx_mapping[original_query]
                organic_results = item.get("organicResults", [])
                combined = " ".join([r.get("description", r.get("snippet", "")) + " " + r.get("url", "") for r in organic_results])
                found_ig = extract_instagram(combined)

                for i in mapped["indices"]:
                    if i not in st.session_state.df.index:
                        continue
                    st.session_state.df.loc[i, "Google Search Status"] = f"Searched ({len(organic_results)} results)"
                    if found_ig != "None" and not is_valid_data(st.session_state.df.loc[i, "Instagram"]):
                        st.session_state.df.loc[i, "Instagram"] = found_ig
                        igs_found += 1

        for _, mapped in idx_mapping.items():
            scraped_googles.add(mapped["channel"])
            for i in mapped["indices"]:
                if i in st.session_state.df.index and not is_valid_data(st.session_state.df.loc[i, "Google Search Status"]):
                    st.session_state.df.loc[i, "Google Search Status"] = "Searched (0 results)"

        save_json_set(scraped_googles, SCRAPED_GOOGLES_FILE)
        save_db(st.session_state.df)
        flash("success", f"Google search finished — found {igs_found} missing Instagram profiles.")
        st.rerun()
    except Exception as e:
        status_text.error(f"Google search failed: {e}")

# ============================================================================
# PAGE: Write
# ============================================================================
def page_write():
    show_flash()
    st.title("✍️ Write")
    st.caption("Clean up messy titles, then draft a personal email and DM for every contactable artist.")

    df = st.session_state.df
    if df.empty:
        st.info("No leads yet. Sync a playlist in **Collect** first.")
        return

    clean_targets, msg_targets = compute_write_targets(df)
    cleaned_msg_targets = [i for i in msg_targets if i not in set(clean_targets)]

    # ---- Step 1: Clean names ----------------------------------------------
    with st.container(border=True):
        st.subheader("1 · Clean names (Gemini)")
        st.caption(f"Turns \"Lil X - Song (Official Video)\" into a clean artist + song. {len(clean_targets)} leads pending.")
        reclean_all = st.checkbox(
            "Re-clean every lead (overwrite existing names)",
            help="Use after a failed run or a model upgrade to redo names that were already filled in.",
        )
        run_targets = list(df.index) if reclean_all else clean_targets
        if st.button(f"Clean names ({len(run_targets)})", disabled=(len(run_targets) == 0)):
            key = get_key("GEMINI_API_KEY")
            if not key:
                st.error("Gemini API key is missing. Add it in Settings.")
            else:
                try:
                    gclient = google_genai.Client(api_key=key)
                except Exception as e:
                    st.error(f"Gemini client failed to start: {e}")
                    gclient = None
                if gclient:
                    progress = st.progress(0.0)
                    status = st.empty()
                    status.info("Checking which Gemini model your key can use...")
                    model = resolve_gemini_model(gclient)
                    errors, last_error, cleaned, consecutive = 0, "", 0, 0
                    total = len(run_targets)
                    aborted = False
                    for pos, idx in enumerate(run_targets):
                        row = st.session_state.df.loc[idx]
                        status.info(f"Cleaning {row['Channel Name']}... ({pos + 1}/{total})")
                        artist, song, err = clean_with_gemini(gclient, model, row["Channel Name"], row["Song Name"])
                        if err:
                            # Leave the row untouched so it stays pending for retry
                            errors += 1
                            last_error = err
                            consecutive += 1
                            if consecutive >= 3:
                                aborted = True
                                break
                        else:
                            consecutive = 0
                            cleaned += 1
                            st.session_state.df.loc[idx, "Cleaned Artist"] = artist
                            st.session_state.df.loc[idx, "Cleaned Song"] = song
                        progress.progress(min(1.0, (pos + 1) / total))
                    save_db(st.session_state.df)
                    if aborted:
                        flash("warning", f"Stopped early: Gemini failed 3 times in a row, so the remaining {total - pos - 1} leads weren't attempted. Nothing was overwritten — fix the issue and run again. Last error: {last_error}")
                    elif errors:
                        flash("warning", f"Cleaned {cleaned} leads; {errors} failed and stayed pending for retry. Last error: {last_error}")
                    else:
                        flash("success", f"Cleaned names for {cleaned} leads with {model}.")
                    st.rerun()

    # ---- Step 2: Messages via Claude.ai (no API) ---------------------------
    with st.container(border=True):
        st.subheader("2 · Write messages (Claude.ai — no API)")
        st.caption(f"Export a batch, have Claude.ai write the messages, import the finished CSV back. {len(cleaned_msg_targets)} artists pending.")
        skipped = len(msg_targets) - len(cleaned_msg_targets)
        if skipped:
            st.warning(f"{skipped} artists will be skipped until their names are cleaned in step 1.")

        if not cleaned_msg_targets:
            st.info("No artists are waiting for messages right now — tick 🔄 on rows in Leads to rewrite them, and the batch download will reappear here.")
        else:
            batch_n = st.number_input(
                "Artists in this batch",
                min_value=1,
                max_value=len(cleaned_msg_targets),
                value=min(100, len(cleaned_msg_targets)),
                help="After you import a finished batch, the next download automatically contains whoever is still pending.",
            )
            batch_idx = cleaned_msg_targets[: int(batch_n)]
            batch = st.session_state.df.loc[batch_idx, ["Channel_ID", "Cleaned Artist", "Cleaned Song"]].copy()
            batch.columns = ["Channel_ID", "Artist", "Song"]
            batch["Draft Message"] = ""

            st.markdown("**Step A — download the batch**")
            st.download_button(
                f"📥 Download batch CSV ({len(batch_idx)} artists)",
                data=batch.to_csv(index=False).encode("utf-8"),
                file_name="wavy_message_batch.csv",
                mime="text/csv",
            )

        st.markdown("**Step B — copy this prompt into Claude.ai and attach the CSV**")
        st.code(CLAUDE_PROMPT, language=None, wrap_lines=True)

        st.markdown("**Step C — import the finished CSV**")
        finished = st.file_uploader("Finished batch from Claude.ai", type=["csv"], key="msg_import", label_visibility="collapsed")
        if finished is not None and st.button("Import messages", type="primary"):
            try:
                incoming = pd.read_csv(finished)
                st.session_state.df, updated, skipped = merge_message_csv(st.session_state.df, incoming)
                save_db(st.session_state.df)
                if updated:
                    note = f" ({skipped} rows matched no lead and were skipped.)" if skipped else ""
                    flash("success", f"Imported {updated} messages into your leads database — saved permanently and waiting in Send.{note}")
                else:
                    flash("warning", "That CSV imported, but no rows matched your leads — check that Channel_ID or Artist + Song columns are intact.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not import that CSV: {e}")

# ============================================================================
# PAGE: Send
# ============================================================================
def page_send():
    show_flash()
    st.title("📤 Send")
    st.caption("Review one artist at a time, send, and mark it done. Follow-ups surface here too.")

    df = st.session_state.df
    queue_idx = [i for i in df.index if has_any_draft(df.loc[i]) and not df.loc[i, "Reached Out"]] if not df.empty else []

    if not queue_idx:
        st.info("The send queue is empty. Draft messages in **Write** and they'll show up here.")
    else:
        # Keep the selection stable across edits, reset it after marking sent
        prev_id = st.session_state.get("send_current_id")
        default_pos = 0
        if prev_id:
            for pos, i in enumerate(queue_idx):
                if df.loc[i, "Channel_ID"] == prev_id:
                    default_pos = pos
                    break

        choice = st.selectbox(
            f"Up next · {len(queue_idx)} in queue",
            options=queue_idx,
            index=default_pos,
            format_func=lambda i: f"{display_name(df.loc[i])} — {display_song(df.loc[i])}",
        )
        st.session_state["send_current_id"] = df.loc[choice, "Channel_ID"]
        row = df.loc[choice]

        with st.container(border=True):
            left, right = st.columns([1, 2])

            with left:
                st.subheader(display_name(row))
                st.caption(f"“{display_song(row)}”")
                st.write(f"**{row['Subscribers']:,}** subscribers")
                if safe_int(row["IG Followers"]) > 0:
                    st.write(f"**{safe_int(row['IG Followers']):,}** IG followers")
                if is_valid_data(row["Channel_URL"]):
                    st.link_button("▶️ YouTube channel", row["Channel_URL"], width="stretch")
                if is_valid_data(row["Instagram"]):
                    st.link_button("📸 Instagram profile", row["Instagram"], width="stretch")
                if is_valid_data(row.get("IG Bio", "")):
                    with st.expander("IG bio"):
                        st.write(row["IG Bio"])

            with right:
                email_addr = str(row["Email Address"]).split(",")[0].strip()
                draft = str(row["Draft Message"]).strip()

                st.markdown("**💬 Message** · one text, works as the email and the DM")
                msg = st.text_area("Message", value=draft, height=150, key=f"msg_{choice}", label_visibility="collapsed")

                if is_valid_data(email_addr):
                    subject = st.text_input("Subject", value=f"Your track \"{display_song(row)}\"", key=f"subj_{choice}")

                b1, b2 = st.columns(2)
                if b1.button("Save edits", key=f"save_msg_{choice}", width="stretch"):
                    st.session_state.df.loc[choice, "Draft Message"] = msg
                    save_db(st.session_state.df)
                    st.toast("Message saved.")
                if is_valid_data(email_addr):
                    mailto = f"mailto:{email_addr}?subject={quote(subject, safe='')}&body={quote(msg, safe='')}"
                    b2.link_button("Open in Mail", mailto, type="primary", width="stretch")

                if is_valid_data(row["Instagram"]):
                    st.caption("DM: tap the copy icon, then paste it on their profile")
                    st.code(msg, language=None, wrap_lines=True)

            st.divider()
            a1, a2, a3 = st.columns([2, 1, 1])
            if a1.button("✅ Mark as reached out", type="primary", key=f"sent_{choice}", width="stretch"):
                st.session_state.df.loc[choice, "Reached Out"] = True
                st.session_state.df.loc[choice, "Reached_Out_Date"] = pd.Timestamp.now().normalize()
                save_db(st.session_state.df)
                st.session_state["send_current_id"] = None
                flash("success", f"Marked {display_name(row)} as reached out.")
                st.rerun()
            if a2.button("🔄 Rewrite next run", key=f"regen_{choice}", width="stretch", help="Flags this artist so Write drafts fresh messages next time."):
                st.session_state.df.loc[choice, "🔄 Regenerate"] = True
                save_db(st.session_state.df)
                st.toast("Flagged for regeneration.")
            if a3.button("🚫 Blacklist", key=f"bl_{choice}", width="stretch", help="Removes this artist and bans the channel from future syncs."):
                blacklist.add(str(row["Channel_ID"]))
                save_json_set(blacklist, BLACKLIST_FILE)
                st.session_state.df = st.session_state.df.drop(index=choice)
                save_db(st.session_state.df)
                st.session_state["send_current_id"] = None
                flash("success", f"Blacklisted {display_name(row)}.")
                st.rerun()

    # ---- Follow-ups ---------------------------------------------------------
    due = follow_ups_due(df)
    st.subheader(f"⏰ Follow-ups due ({len(due)})")
    if due.empty:
        st.caption(f"Contacts with no reply after {FOLLOW_UP_DAYS} days will appear here.")
    else:
        for idx in due.index[:25]:
            row = due.loc[idx]
            days = (pd.Timestamp.now().normalize() - pd.to_datetime(row["Reached_Out_Date"]).normalize()).days
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2])
                c1.markdown(f"**{display_name(row)}** · contacted {days} days ago")
                if is_valid_data(row["Instagram"]):
                    c2.link_button("Instagram", row["Instagram"], width="stretch")
                email_addr = str(row["Email Address"]).split(",")[0].strip()
                if is_valid_data(email_addr):
                    c3.link_button("Email", f"mailto:{email_addr}", width="stretch")
                if c4.button("Replied ✅", key=f"fu_replied_{idx}", width="stretch"):
                    st.session_state.df.loc[idx, "Replied"] = True
                    save_db(st.session_state.df)
                    flash("success", f"Marked {display_name(row)} as replied.")
                    st.rerun()

# ============================================================================
# PAGE: Leads
# ============================================================================
def page_leads():
    show_flash()
    st.title("📋 Leads")
    st.caption("Every artist in one table. Edit cells directly — changes save automatically.")

    with st.expander("➕ Add a lead manually"):
        with st.form("add_lead", clear_on_submit=True):
            c1, c2 = st.columns(2)
            new_name = c1.text_input("Artist / channel name (required)")
            new_song = c2.text_input("Song")
            c3, c4, c5 = st.columns(3)
            new_email = c3.text_input("Email")
            new_ig = c4.text_input("Instagram URL or @handle")
            new_subs = c5.number_input("Subscribers", min_value=0, value=0)
            if st.form_submit_button("Add lead"):
                if not new_name.strip():
                    st.warning("Name is required.")
                else:
                    ig = new_ig.strip()
                    if ig.startswith("@"):
                        ig = f"https://instagram.com/{ig[1:]}"
                    row = {
                        "Channel_ID": "manual-" + uuid.uuid4().hex[:10],
                        "Channel Name": new_name.strip(),
                        "Song Name": new_song.strip() or "Unknown",
                        "Email Address": new_email.strip() or "None Found",
                        "Instagram": ig or "None",
                        "Subscribers": int(new_subs),
                    }
                    st.session_state.df = ensure_schema(
                        pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)
                    )
                    save_db(st.session_state.df)
                    flash("success", f"Added {new_name.strip()} to your leads.")
                    st.rerun()

    if st.session_state.df.empty:
        st.info("No leads yet. Sync a playlist in **Collect** — or add one manually above.")
        return

    search = st.text_input("Search", placeholder="Search by artist or channel name...", label_visibility="collapsed")
    col1, col2, col3 = st.columns(3)
    with col1: email_filter = st.radio("Email", ["All", "Has email", "No email"], horizontal=True)
    with col2: ig_filter = st.radio("Instagram", ["All", "Has IG", "No IG"], horizontal=True)
    with col3: crm_filter = st.radio("Pipeline", ["All", "Not contacted", "Contacted", "Replied"], horizontal=True)

    filtered_df = st.session_state.df.copy()

    if search.strip():
        s = search.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Channel Name"].str.lower().str.contains(s, na=False)
            | filtered_df["Cleaned Artist"].str.lower().str.contains(s, na=False)
        ]

    if email_filter == "Has email": filtered_df = filtered_df[filtered_df["Email Address"].apply(is_valid_data)]
    elif email_filter == "No email": filtered_df = filtered_df[~filtered_df["Email Address"].apply(is_valid_data)]
    if ig_filter == "Has IG": filtered_df = filtered_df[filtered_df["Instagram"].apply(is_valid_data)]
    elif ig_filter == "No IG": filtered_df = filtered_df[~filtered_df["Instagram"].apply(is_valid_data)]
    if crm_filter == "Not contacted": filtered_df = filtered_df[filtered_df["Reached Out"] == False]
    elif crm_filter == "Contacted": filtered_df = filtered_df[(filtered_df["Reached Out"] == True) & (filtered_df["Replied"] == False)]
    elif crm_filter == "Replied": filtered_df = filtered_df[filtered_df["Replied"] == True]

    st.caption(f"Showing {len(filtered_df)} of {len(st.session_state.df)} artists · amber = contacted · red = follow-up due · green = replied · 🗑️ removes AND bans, ❌ just removes")

    edited_df = st.data_editor(
        filtered_df.style.apply(highlight_rows, axis=1),
        column_config={
            "Channel_ID": None,
            "Channel_URL": None,
            "Song Name": None,
            "Draft Message": st.column_config.TextColumn("Message", width="large", max_chars=2000),
            "🗑️ Blacklist": st.column_config.CheckboxColumn("🗑️", help="Remove this lead AND ban the channel from future syncs."),
            "❌ Remove": st.column_config.CheckboxColumn("❌", help="Delete this lead without banning — it can come back on a future sync."),
            "Reached Out": st.column_config.CheckboxColumn("Reached Out"),
            "Replied": st.column_config.CheckboxColumn("Replied"),
            "🔄 Regenerate": st.column_config.CheckboxColumn("🔄", help="Rewrite this artist's messages on the next Write run."),
            "Reached_Out_Date": st.column_config.DateColumn("Contacted", disabled=True, format="MMM DD, YYYY"),
            "Subscribers": st.column_config.NumberColumn("Subs", format="%d"),
            "IG Followers": st.column_config.NumberColumn("IG Followers", format="%d"),
            "Cleaned Artist": st.column_config.TextColumn("Artist"),
            "Cleaned Song": st.column_config.TextColumn("Song"),
            "Google Search Status": st.column_config.TextColumn("Google", disabled=True),
            "IG Bio": st.column_config.TextColumn("IG Bio", max_chars=1000),
            "🔍 Quick Search": st.column_config.LinkColumn("Find Contacts", display_text="Search web", disabled=True),
        },
        width="stretch",
        hide_index=True,
    )

    changes_made = False

    to_delete = edited_df[edited_df["🗑️ Blacklist"] == True]
    if not to_delete.empty:
        banned_ids = to_delete["Channel_ID"].tolist()
        blacklist.update(banned_ids)
        save_json_set(blacklist, BLACKLIST_FILE)
        st.session_state.df = st.session_state.df[~st.session_state.df["Channel_ID"].isin(banned_ids)]
        changes_made = True

    to_remove = edited_df[(edited_df["❌ Remove"] == True) & (edited_df["🗑️ Blacklist"] != True)]
    removed_ids = to_remove["Channel_ID"].tolist()
    if removed_ids:
        st.session_state.df = st.session_state.df[~st.session_state.df["Channel_ID"].isin(removed_ids)]
        changes_made = True

    for idx in edited_df.index:
        ch_id = edited_df.loc[idx, "Channel_ID"]
        if ch_id in to_delete["Channel_ID"].values or ch_id in removed_ids:
            continue

        was_reached = filtered_df.loc[idx, "Reached Out"]
        is_reached = edited_df.loc[idx, "Reached Out"]
        was_replied = filtered_df.loc[idx, "Replied"]
        is_replied = edited_df.loc[idx, "Replied"]

        if is_reached != was_reached or is_replied != was_replied:
            changes_made = True
            st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, "Reached Out"] = is_reached
            st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, "Replied"] = is_replied
            if is_reached and not was_reached:
                st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, "Reached_Out_Date"] = pd.Timestamp.now().normalize()
            elif not is_reached and was_reached:
                st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, "Reached_Out_Date"] = pd.NaT

        for col in ["Channel Name", "Subscribers", "Email Address", "Instagram", "IG Bio", "IG Followers", "Cleaned Artist", "Cleaned Song", "Draft Message", "🔄 Regenerate"]:
            old_val = filtered_df.loc[idx, col]
            new_val = edited_df.loc[idx, col]
            if str(old_val) != str(new_val):
                changes_made = True
                st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, col] = new_val

    if changes_made:
        save_db(st.session_state.df)
        st.rerun()

    export_cols = [c for c in filtered_df.columns if c not in ["Channel_ID", "🗑️ Blacklist", "❌ Remove", "🔍 Quick Search"]]
    st.download_button(
        "📥 Download filtered leads (.csv)",
        data=filtered_df[export_cols].to_csv(index=False).encode("utf-8"),
        file_name="wavy_outreach_leads.csv",
        mime="text/csv",
    )
    st.caption("⚠️ Streamlit Cloud storage is temporary — download a full backup in Settings after each session.")

# ============================================================================
# PAGE: Settings
# ============================================================================
def page_settings():
    show_flash()
    st.title("⚙️ Settings")

    # ---- API keys -----------------------------------------------------------
    with st.container(border=True):
        st.subheader("API keys")
        st.caption("Keys load from Streamlit secrets. Paste one below to use it for this session only — it isn't saved anywhere.")
        for name, label in KEY_NAMES.items():
            c1, c2 = st.columns([1, 2])
            in_secrets = bool(get_secret(name))
            in_session = bool(str(st.session_state.get(f"key_{name}", "") or "").strip())
            status = "✅ Set in secrets" if in_secrets else ("🟡 Set for this session" if in_session else "❌ Missing")
            c1.markdown(f"**{label}**  \n{status}")
            c2.text_input(f"{label} key", type="password", key=f"key_{name}",
                          placeholder="Paste to override for this session", label_visibility="collapsed")
        st.caption("To set keys permanently: App → Settings → Secrets on Streamlit Cloud, using the names "
                   + ", ".join(f"`{n}`" for n in KEY_NAMES))

    # ---- Backup & restore ---------------------------------------------------
    with st.container(border=True):
        st.subheader("Backup & restore")
        st.caption("Streamlit Cloud wipes this app's disk whenever it restarts. Download a backup after every session; restore it here when the data disappears.")
        df = st.session_state.df
        st.download_button(
            "📥 Download full backup (.zip)",
            data=build_backup_zip(df),
            file_name="wavy_outreach_backup.zip",
            mime="application/zip",
            type="primary",
            disabled=df.empty,
            help="Includes your leads plus the blacklist, seen-videos, and scrape memory.",
        )
        restore_file = st.file_uploader("Restore a backup (.zip) or an old app's export (.csv)", type=["zip", "csv"])
        mode = st.radio("Restore mode", ["Merge with current leads", "Replace everything"], horizontal=True)
        if restore_file is not None and st.button("Restore now"):
            try:
                if restore_file.name.lower().endswith(".zip"):
                    z = zipfile.ZipFile(restore_file)
                    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                    if not csv_names:
                        raise ValueError("No CSV found inside that ZIP.")
                    csv_name = DB_FILE if DB_FILE in csv_names else csv_names[0]
                    incoming = ensure_schema(pd.read_csv(z.open(csv_name)))
                    restore_memory_from_zip(z, replace=(mode == "Replace everything"))
                else:
                    incoming = ensure_schema(pd.read_csv(restore_file))
                if mode == "Replace everything" or st.session_state.df.empty:
                    st.session_state.df = incoming
                    added = len(incoming)
                else:
                    current = st.session_state.df
                    if incoming["Channel_ID"].apply(is_valid_data).any() and current["Channel_ID"].apply(is_valid_data).any():
                        existing = set(current["Channel_ID"])
                        incoming = incoming[~incoming["Channel_ID"].isin(existing)]
                    else:
                        existing = set(current["Song Name"])
                        incoming = incoming[~incoming["Song Name"].isin(existing)]
                    added = len(incoming)
                    st.session_state.df = ensure_schema(pd.concat([current, incoming], ignore_index=True))
                save_db(st.session_state.df)
                flash("success", f"Restored {added} leads.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not restore that file: {e}")

    # ---- Blacklist ----------------------------------------------------------
    with st.container(border=True):
        st.subheader("Blacklist")
        st.write(f"**{len(blacklist)}** channels are banned from future syncs.")
        if len(blacklist) > 0:
            to_unban = st.multiselect("Unban specific channels", sorted(blacklist))
            c1, c2 = st.columns(2)
            if c1.button("Unban selected", disabled=(not to_unban)):
                for b in to_unban:
                    blacklist.discard(b)
                save_json_set(blacklist, BLACKLIST_FILE)
                flash("success", f"Unbanned {len(to_unban)} channel(s) — they can return on the next sync.")
                st.rerun()
            if c2.button("Clear blacklist (unban all)"):
                blacklist.clear()
                save_json_set(blacklist, BLACKLIST_FILE)
                flash("success", "Blacklist cleared.")
                st.rerun()

    with st.container(border=True):
        st.subheader("Scrape memory")
        st.caption("These lists stop the app from re-processing what it already handled. Forget one to run it again. Note: leads that already hold scraped data still won't re-scrape unless you also clear their IG Bio / Google status cells in Leads.")
        m1, m2, m3 = st.columns(3)
        if m1.button(f"Forget seen videos ({len(seen_videos)})"):
            seen_videos.clear()
            save_json_set(seen_videos, SEEN_VIDEOS_FILE)
            flash("success", "Seen-video memory cleared — the next sync rescans the whole playlist.")
            st.rerun()
        if m2.button(f"Forget IG scrapes ({len(scraped_igs)})"):
            scraped_igs.clear()
            save_json_set(scraped_igs, SCRAPED_IGS_FILE)
            flash("success", "Instagram scrape memory cleared.")
            st.rerun()
        if m3.button(f"Forget Google searches ({len(scraped_googles)})"):
            scraped_googles.clear()
            save_json_set(scraped_googles, SCRAPED_GOOGLES_FILE)
            flash("success", "Google search memory cleared.")
            st.rerun()

    # ---- Danger zone --------------------------------------------------------
    with st.container(border=True):
        st.subheader("⚠️ Danger zone")
        st.write("Deletes all leads, sync history, blacklist, scrape memory, and the saved playlist.")
        confirm = st.checkbox("Yes, permanently delete all data.")
        if st.button("Wipe all data", type="primary"):
            if confirm:
                st.session_state.df = ensure_schema(pd.DataFrame())
                for f in [DB_FILE, SEEN_VIDEOS_FILE, BLACKLIST_FILE, LAST_PLAYLIST_FILE, SCRAPED_IGS_FILE, SCRAPED_GOOGLES_FILE]:
                    if os.path.exists(f):
                        os.remove(f)
                flash("success", "All data wiped.")
                st.rerun()
            else:
                st.warning("Tick the confirmation box first.")

# ============================================================================
# Navigation
# ============================================================================
pg_dashboard = st.Page(page_dashboard, title="Dashboard", icon="📊", default=True)
pg_collect = st.Page(page_collect, title="Collect", icon="🎧")
pg_write = st.Page(page_write, title="Write", icon="✍️")
pg_send = st.Page(page_send, title="Send", icon="📤")
pg_leads = st.Page(page_leads, title="Leads", icon="📋")
pg_settings = st.Page(page_settings, title="Settings", icon="⚙️")

nav = st.navigation({
    "Overview": [pg_dashboard],
    "Workflow": [pg_collect, pg_write, pg_send],
    "Data": [pg_leads, pg_settings],
})

with st.sidebar:
    stats = pipeline_stats(st.session_state.df)
    st.divider()
    st.caption("🎛️ **Wavy Outreach**")
    st.caption(f"{stats['total']} leads · {stats['contacted']} contacted · {stats['replied']} replied")
    dots = " · ".join(("🟢" if get_key(n) else "⚪") + " " + lbl for n, lbl in KEY_NAMES.items())
    st.caption(dots)

nav.run()
