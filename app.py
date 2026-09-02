"""
Wavy Outreach — one app for the whole pipeline.

Collect leads from YouTube playlists, enrich them with Apify, write
personalized messages with AI, and send + track outreach — all against a
single database. Replaces the separate youtube-artist-scraper and
Artist-AI-Outreach apps.
"""

import io
import os
import base64
import hashlib
import html as html_escape
import uuid
import re
import json
import time
import threading
import zipfile
import urllib.request
import urllib.error
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

ACCENT = "#F0A93B"      # brand accent (keep in sync with .streamlit/config.toml primaryColor)
BRAND_URL = "https://www.wavymixing.com"
BRAND_LOGO = "https://wavymixing.com/logo.webp"

try:
    st.logo(BRAND_LOGO, link=BRAND_URL, size="large")
except Exception:
    pass  # older Streamlit or offline — branding is cosmetic, never fatal

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
    "🗑️ Blacklist", "❌ Remove", "Followed", "Emailed", "DM'd", "Reached Out", "Replied", "Free Mix Sent", "Paid Customer", "Revenue", "Followed_Date", "Reached_Out_Date", "Email_Date", "DM_Date", "Mix_Date", "Paid_Date", "Added_Date",
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
BOOL_COLS = ["🗑️ Blacklist", "❌ Remove", "Followed", "Emailed", "DM'd", "Reached Out", "Replied", "Free Mix Sent", "Paid Customer", "🔄 Regenerate"]
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

    if "Revenue" not in df.columns:
        df["Revenue"] = 0.0
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)

    for _flag in ["Emailed", "DM'd"]:
        if _flag not in df.columns:
            df[_flag] = False
        df[_flag] = df[_flag].map(lambda v: str(v).strip().lower() == "true" if not isinstance(v, bool) else v).fillna(False).astype(bool)

    for dcol in ["Followed_Date", "Reached_Out_Date", "Email_Date", "DM_Date", "Mix_Date", "Paid_Date", "Added_Date"]:
        if dcol not in df.columns:
            df[dcol] = pd.NaT
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Reached Out is derived: emailed OR DM'd. Migrate legacy single-flag data.
    legacy_reached = df["Reached Out"].copy() if "Reached Out" in df.columns else pd.Series(False, index=df.index)
    legacy_reached = legacy_reached.map(lambda v: str(v).strip().lower() == "true" if not isinstance(v, bool) else v).fillna(False).astype(bool)
    orphan = legacy_reached & ~df["Emailed"] & ~df["DM'd"]
    if orphan.any():
        has_email = df["Email Address"].apply(is_valid_data).astype(bool)
        df.loc[orphan & has_email, "Emailed"] = True
        df.loc[orphan & ~has_email, "DM'd"] = True
        carry = df["Reached_Out_Date"] if "Reached_Out_Date" in df.columns else pd.Series(pd.NaT, index=df.index)
        df.loc[orphan & has_email & df["Email_Date"].isna(), "Email_Date"] = carry
        df.loc[orphan & ~has_email & df["DM_Date"].isna(), "DM_Date"] = carry
    # Reached Out ticks as soon as the artist has been contacted on ANY channel.
    df["Reached Out"] = df["Emailed"] | df["DM'd"]

    # Old "[Claude Error]" strings must never block regeneration
    df["Draft Message"] = df["Draft Message"].apply(lambda v: "" if str(v).strip().startswith("[Claude Error") else v)

    # Keep only the first IG link if a comma list slipped in
    df["Instagram"] = df["Instagram"].apply(
        lambda x: str(x).split(",")[0].strip() if is_valid_data(x) else "None"
    )

    needs_qs = ~df["🔍 Quick Search"].apply(is_valid_data).astype(bool)
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
# Cloud save — keeps the database + memory files on a GitHub branch so they
# survive Streamlit Cloud reboots (this app's local disk is temporary).
#
# One-time setup:
#   1. On GitHub: Settings → Developer settings → Fine-grained tokens →
#      new token with access to ONLY this repo, permission Contents: Read & write.
#   2. On Streamlit Cloud: App → Settings → Secrets → add
#         GITHUB_TOKEN = "github_pat_..."
# The app then auto-saves every change to the `app-data` branch and restores
# everything on boot. Saving to a separate branch keeps your data out of the
# code and, importantly, does NOT trigger a redeploy (Streamlit watches main).
# Optional secrets: GITHUB_REPO ("owner/name"), GITHUB_DATA_BRANCH.
# ----------------------------------------------------------------------------
PERSIST_FILES = [DB_FILE, SEEN_VIDEOS_FILE, BLACKLIST_FILE, LAST_PLAYLIST_FILE,
                 SCRAPED_IGS_FILE, SCRAPED_GOOGLES_FILE]
CLOUD_STATE_FILE = ".cloud_state.json"     # what we last pushed (md5 + git sha per file)
CLOUD_RESTORED_MARKER = ".cloud_restored"  # written once per container boot

def _cloud_token():
    return str(get_secret("GITHUB_TOKEN") or "").strip()

def _cloud_repo():
    return str(get_secret("GITHUB_REPO") or "alexsamey95/wavyout").strip()

def _cloud_branch():
    return str(get_secret("GITHUB_DATA_BRANCH") or "app-data").strip()

def cloud_enabled():
    return bool(_cloud_token())

def _gh_headers(accept="application/vnd.github+json"):
    return {
        "Authorization": f"Bearer {_cloud_token()}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "wavy-outreach",
    }

def _gh(method, path, payload=None):
    req = urllib.request.Request(
        f"https://api.github.com/{path}",
        method=method,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=_gh_headers(),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else None

def _gh_fetch_raw(path):
    """Raw bytes of a file on the data branch, or None if it isn't there."""
    url = (f"https://api.github.com/repos/{_cloud_repo()}/contents/{quote(path)}"
           f"?ref={quote(_cloud_branch())}")
    req = urllib.request.Request(url, headers=_gh_headers("application/vnd.github.raw+json"))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

def _git_blob_sha(data):
    """The sha GitHub assigns a file's content (git blob sha)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

def _ensure_data_branch():
    repo, branch = _cloud_repo(), _cloud_branch()
    try:
        _gh("GET", f"repos/{repo}/branches/{quote(branch)}")
        return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    base = (_gh("GET", f"repos/{repo}") or {}).get("default_branch", "main")
    ref = _gh("GET", f"repos/{repo}/git/ref/heads/{quote(base)}")
    _gh("POST", f"repos/{repo}/git/refs",
        {"ref": f"refs/heads/{branch}", "sha": ref["object"]["sha"]})

def _gh_put_file(path, data, sha):
    payload = {"message": f"wavy data: update {path}",
               "content": base64.b64encode(data).decode("ascii"),
               "branch": _cloud_branch()}
    if sha:
        payload["sha"] = sha
    try:
        _gh("PUT", f"repos/{_cloud_repo()}/contents/{quote(path)}", payload)
    except urllib.error.HTTPError as e:
        if e.code not in (409, 422):
            raise
        # Our remembered sha is stale — re-read the real one and retry once.
        remote = _gh_fetch_raw(path)
        if remote is None:
            payload.pop("sha", None)
        else:
            payload["sha"] = _git_blob_sha(remote)
        _gh("PUT", f"repos/{_cloud_repo()}/contents/{quote(path)}", payload)

def _gh_delete_file(path, sha):
    payload = {"message": f"wavy data: delete {path}",
               "sha": sha or "", "branch": _cloud_branch()}
    try:
        _gh("DELETE", f"repos/{_cloud_repo()}/contents/{quote(path)}", payload)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return  # already gone
        if e.code not in (409, 422):
            raise
        remote = _gh_fetch_raw(path)
        if remote is None:
            return
        payload["sha"] = _git_blob_sha(remote)
        _gh("DELETE", f"repos/{_cloud_repo()}/contents/{quote(path)}", payload)

def _cloud_state_load():
    if os.path.exists(CLOUD_STATE_FILE):
        try:
            with open(CLOUD_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _cloud_state_save(state):
    with open(CLOUD_STATE_FILE, "w") as f:
        json.dump(state, f)

def _cloud_err_msg(e):
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 401:
            return "GitHub token is invalid or expired — update GITHUB_TOKEN in the app secrets."
        if e.code == 403:
            return "GitHub token lacks permission — it needs Contents: Read and write on the repo."
        if e.code == 404:
            return (f"GitHub repo `{_cloud_repo()}` not found or the token can't see it — "
                    "check GITHUB_REPO / the token's repository access.")
        return f"GitHub error {e.code}"
    return str(e) or type(e).__name__

def cloud_pending_count():
    """How many data files differ from what's saved on GitHub."""
    state = _cloud_state_load()
    n = 0
    for path in PERSIST_FILES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                if state.get(path, {}).get("md5") != hashlib.md5(f.read()).hexdigest():
                    n += 1
        elif path in state:
            n += 1
    return n

CLOUD_BRANCH_MARKER = ".cloud_branch_ok"  # data branch verified once per container

def _cloud_sync_core():
    """Session-free sync core — safe to call from background threads.
    Pushes every locally-changed data file to GitHub. Returns files pushed."""
    if not cloud_enabled():
        return 0
    if not os.path.exists(CLOUD_BRANCH_MARKER):
        _ensure_data_branch()
        with open(CLOUD_BRANCH_MARKER, "w") as f:
            f.write("ok")
    state = _cloud_state_load()
    pushed = 0
    for path in PERSIST_FILES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            digest = hashlib.md5(data).hexdigest()
            if state.get(path, {}).get("md5") == digest:
                continue
            _gh_put_file(path, data, state.get(path, {}).get("sha"))
            state[path] = {"md5": digest, "sha": _git_blob_sha(data)}
            pushed += 1
        elif path in state:
            _gh_delete_file(path, state[path].get("sha"))
            del state[path]
            pushed += 1
    if pushed:
        _cloud_state_save(state)
    return pushed

def cloud_sync():
    """UI wrapper around the sync core. Returns (pushed, error_message)."""
    if not cloud_enabled():
        return 0, None
    try:
        pushed = _cloud_sync_core()
        if pushed:
            st.session_state["_cloud_last_sync"] = time.time()
        st.session_state["_cloud_error"] = None
        return pushed, None
    except Exception as e:
        msg = _cloud_err_msg(e)
        st.session_state["_cloud_error"] = msg
        return 0, msg

def cloud_restore_on_boot():
    """On a fresh container, pull the data files back down from GitHub.
    Never overwrites a file that already exists locally, and never raises."""
    if not cloud_enabled() or os.path.exists(CLOUD_RESTORED_MARKER):
        return
    try:
        _ensure_data_branch()
        with open(CLOUD_BRANCH_MARKER, "w") as f:
            f.write("ok")
        state = _cloud_state_load()
        restored = 0
        for path in PERSIST_FILES:
            if os.path.exists(path):
                continue
            data = _gh_fetch_raw(path)
            if data is None:
                continue
            with open(path, "wb") as f:
                f.write(data)
            state[path] = {"md5": hashlib.md5(data).hexdigest(),
                           "sha": _git_blob_sha(data)}
            restored += 1
        _cloud_state_save(state)
        if restored:
            st.toast(f"☁️ Restored {restored} data file(s) from GitHub.")
    except Exception as e:
        st.warning(f"Could not restore data from GitHub ({_cloud_err_msg(e)}). "
                   "Starting with what's on disk.")
    finally:
        with open(CLOUD_RESTORED_MARKER, "w") as f:
            f.write(str(time.time()))

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
    try:
        cached = st.session_state.get("_gemini_model")
    except Exception:
        cached = None  # running in a background thread — no session cache
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
            try:
                st.session_state["_gemini_model"] = cand
            except Exception:
                pass  # background thread — skip the session cache
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

THE VOICE — read this first, it matters most:
Write like a real person who genuinely respects the artist and wants to connect — warm, polite, and human, but still street and friendly, not corporate. Full, comprehensible sentences that flow when read aloud. Do NOT cram the offer into a rushed, comma-spliced checklist like "rough bounce is fine, back within two days, free" — that reads robotic. Slow down and say it like you'd say it to someone you actually admire. Sympathetic and easygoing, never pushy, never a sales pitch.

THE INTENTION behind every message (weave it in naturally, don't state it mechanically):
I genuinely want to work with this artist. I'm reaching out to connect with real talent and offer to mix and master one of their tracks for free — not as a gimmick, but to show what I can do. My honest hope is that if they like the result, it turns into a long-term working relationship. It should feel like I chose them on purpose and would be glad to have them as a long-term client.

STRUCTURE every message like this, in fresh wording each time:
1. GREETING with their name: "Hey {Artist}," / "What's up {Artist}," / "Yo {Artist}," — vary it. If the Artist value is clearly a channel name or junk, clean it up or greet warmly without a name.
2. THE HOOK — appreciation comes FIRST, right after the greeting. The very first sentence of the body must open with a warm line of genuine appreciation for their music, BEFORE anything else. Lead with a varied phrase that sounds like how rappers actually talk to each other: "I fw your music heavy", "I really mess with your sound", "your music go crazy", "I been rocking with your music", "you got a crazy sound", "your shit hard", "I really rate your music". Then tie it to the specific track — simply and naturally. Good: "I fw your music heavy, {Song} especially." / "I really mess with your sound — {Song} go crazy." / "Been rocking with your music, {Song} on repeat." Keep it short, plain, and street. Vary the opener across rows.

DO NOT use soft, formal, or corny phrasing — you're talking to rappers, not writing a greeting card. BANNED appreciation phrasing: "your music genuinely moves me", "in particular", "a great record", "keep coming back to", "the one I keep coming back to", "your artistry", "truly", "such a talent", "I must say", "a masterpiece", "sonically", anything that sounds like a music review or a formal compliment. Say it plain and real: you like the music, name the track, keep it moving. Reference the real track title naturally, stripping any junk (hashtags, director tags, feature lists). Do NOT bury the appreciation mid-sentence — it leads.

KEEP THE APPRECIATION SIMPLE — no corny filler. Say you like the music and name the track, then move on. BANNED filler phrases (these sound like a bot padding): "kept me listening", "kept me there", "your other tracks hold up", "hold up too", "went through your whole catalog", "the rest of your catalog kept me", "pulled me straight in", "had it on repeat along with a few of your other tracks". One clean sentence of genuine appreciation is better than a stacked pile of them. Do not list multiple reactions to the music — one honest line is enough.
3. WHO I AM: Alex Wavy, a mixing and mastering engineer specializing in trap, drill, and hip-hop — 5+ years on a hybrid analog/digital setup.
4. THE OFFER, said like a human: I'd genuinely like to work with them, and I'd love to mix and master one of their tracks for free to show what I can bring to their sound. Explain gently what to send — just one track they're working on, whatever they have, even a rough version — and that I'll take care of it and get a finished, professional mix back to them within a day or two. Let it breathe across a sentence or two; do not compress it into fragments.
5. WHY IT'S FREE + THE LONG GAME: one honest, unpushy line — there's no catch, this is just how I like to introduce myself to artists I believe in, and if they're happy with the mix I'd love to keep working together going forward. If not, the mix is theirs to keep either way.
6. WARM CLOSE + SIGN-OFF: end like a real person hoping to connect — "If you're interested, please let me know." / "Let me know your thoughts." / "I hope to hear from you." / "Looking forward to hearing from you." Vary it. Then sign off "– Alex Wavy".

LAYOUT (exact — every row must follow this): put the greeting on its own line, then a BLANK line, then the whole body (hook, who I am, offer, why-free) as one flowing paragraph, then a BLANK line, then the closing line and sign-off. Use real line breaks. The finished message looks like:

Hey NAME,

[body opens with the appreciation line first — e.g. "I really love your music, been playing {Song} on repeat." — then who I am, then the free mix offer in full sentences, then the no-catch/long-term line]

Looking forward to hearing from you.
– Alex Wavy

Keep those blank lines in every message so there is space to read and room for a signature block.

STYLE RULES:
- #1 RULE — NEVER RUSH THE OFFER INTO A CHECKLIST. Every message must read like flowing, spoken English, not a list of terms crammed together with commas. This applies to EVERY row without exception, no matter how large the batch.
  BAD (rushed, robotic, banned): "send one track, rough bounce is fine, back within two days, free."
  GOOD (warm, complete sentences): "If you're up for it, just send over one track you're working on — even a rough version is totally fine. I'll mix and master it properly and get it back to you within a day or two, completely free."
  If a sentence has three or more comma-separated fragments in a row, rewrite it as full sentences before moving on.
- 80 to 110 words. Complete, readable sentences — err toward warmth and clarity over brevity. Never sacrifice readability to save words.
- Polite, sympathetic, street-but-professional. A real engineer who respects the artist, not a marketing bot and not a hustler.
- Minimal slang (one casual touch max) and no profanity.
- BANNED: "opportunity", "elevate", "next level", "game-changer", "incredible", "amazing", "I'd love to" (as filler), "hope this finds you well", "let's get started", "let's do this", "don't miss", "act now", "Interested?" as a one-word hook, emojis, links, fake flattery, and rushed comma-spliced fragments like "rough bounce is fine, back within two days, free".
- Vary structure and wording between rows so no two messages look templated.
- Do NOT change Channel_ID, Artist, or Song. Fill ONLY the Draft Message column.
- If a row is clearly a media channel or label rather than an artist, adapt warmly: I'd love to work with the artists they back and mix a track for free for any of them.

OUTPUT: Give me back the complete CSV as a downloadable file — same columns, same rows, same order, Draft Message filled for every row. Each Draft Message contains line breaks (see LAYOUT), so wrap every message in double quotes so the newlines and commas stay inside one cell and don't break the CSV."""

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
def run_apify_and_poll(client, actor_id, run_input, total_targets, report, label):
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
        report(progress_val, f"{label}: {item_count}/{total_targets} done · {elapsed}s · {status}")

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

GREETING_RE = re.compile(r"^\s*((?:Hey|What's up|What's good|Yo|Hi|Hi there)\b[^,\u2014]*[,\u2014])\s*", re.I)

def reformat_message(text):
    """Greeting on line 1, blank line, body, blank line, close + sign-off.
    Idempotent: if already formatted with blank lines, leaves it alone."""
    t = str(text or "").strip()
    if not t:
        return t
    if "\n\n" in t:  # already has blank-line layout
        return t
    # split off greeting
    m = GREETING_RE.match(t)
    greeting, rest = ("", t)
    if m:
        greeting = m.group(1).strip()
        rest = t[m.end():].strip()
    # split off sign-off (– Alex Wavy or - Alex Wavy), keeping the closing line with it
    sign = ""
    for marker in ["\u2013 Alex Wavy", "- Alex Wavy", "\u2014 Alex Wavy"]:
        idx = rest.rfind(marker)
        if idx != -1:
            sign = rest[idx:].strip()
            rest = rest[:idx].strip()
            break
    # pull the final sentence before the sign-off up as the closing line
    close = ""
    if sign and rest:
        parts = re.split(r"(?<=[.!?])\s+", rest)
        if len(parts) > 1:
            close = parts[-1].strip()
            rest = " ".join(parts[:-1]).strip()
    body = rest
    out = []
    if greeting: out.append(greeting)
    if greeting and body: out.append("")           # blank line under greeting
    if body: out.append(body)
    if (close or sign): out.append("")             # blank line above sign-off block
    tail = []
    if close: tail.append(close)
    if sign: tail.append(sign)
    if tail: out.append("\n".join(tail))
    return "\n".join(out)

def pending_channels(row):
    """Channels this artist can still be contacted on."""
    ch = []
    if is_valid_data(row.get("Email Address", "")) and not bool(row.get("Emailed", False)):
        ch.append("email")
    if is_valid_data(row.get("Instagram", "")) and not bool(row.get("DM'd", False)):
        ch.append("ig")
    return ch

def mark_channel_sent(idx, flag_col, date_col, row):
    st.session_state.df.loc[idx, flag_col] = True
    st.session_state.df.loc[idx, date_col] = pd.Timestamp.now().normalize()
    # One touch on ANY channel = contacted. Stamp the date on first contact only.
    if not bool(st.session_state.df.loc[idx, "Reached Out"]):
        st.session_state.df.loc[idx, "Reached_Out_Date"] = pd.Timestamp.now().normalize()
    st.session_state.df.loc[idx, "Reached Out"] = True
    save_db(st.session_state.df)
    st.session_state["send_current_id"] = None
    flash("success", f"{display_name(row)} marked as contacted — next up.")
    st.rerun()

def pipeline_stats(df):
    if df.empty:
        return {"total": 0, "contactable": 0, "drafted": 0, "contacted": 0, "replied": 0,
                "free_mix": 0, "paid": 0, "revenue": 0.0, "followed": 0}
    contactable = df.apply(lambda r: is_valid_data(r["Email Address"]) or is_valid_data(r["Instagram"]), axis=1).sum()
    drafted = df.apply(has_any_draft, axis=1).sum()
    return {
        "total": len(df),
        "contactable": int(contactable),
        "drafted": int(drafted),
        "contacted": int(df["Reached Out"].sum()),
        "followed": int(df["Followed"].sum()) if "Followed" in df.columns else 0,
        "replied": int(df["Replied"].sum()),
        "free_mix": int(df["Free Mix Sent"].sum()) if "Free Mix Sent" in df.columns else 0,
        "paid": int(df["Paid Customer"].sum()) if "Paid Customer" in df.columns else 0,
        "revenue": float(df["Revenue"].sum()) if "Revenue" in df.columns else 0.0,
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
# One-click pipeline — Google-search IGs -> scrape bios -> clean names, in a
# background thread on the server. The job keeps running after the browser
# tab is closed; progress lives in shared state that any session can read.
# ----------------------------------------------------------------------------
@st.cache_resource
def _pipeline_job():
    """Shared job state: survives reruns, page switches, and closed tabs."""
    return {"running": False, "stage": "", "detail": "", "frac": 0.0,
            "log": [], "error": None, "started": 0.0, "finished": 0.0}

def run_full_pipeline(job, apify_token, gemini_key):
    """Background worker. Must never touch st.session_state or draw UI.
    Reads the database from disk, saves after every stage, then cloud-syncs."""
    def report(frac, text):
        job["frac"] = float(min(1.0, max(0.0, frac)))
        job["detail"] = str(text)

    def begin(n, name):
        job["stage"] = f"Step {n}/3 · {name}"
        job["frac"] = 0.0
        job["detail"] = ""
        job["log"].append(f"▶ {name}")

    def run_stage(fn, *args):
        try:
            df2, summary = fn(*args)
            job["log"].append("✅ " + (summary or "Nothing pending — skipped."))
            return df2
        except Exception as e:
            job["log"].append(f"⚠️ Failed: {e}")
            return args[0]  # keep whatever partial work landed in df

    try:
        df = ensure_schema(pd.read_csv(DB_FILE)) if os.path.exists(DB_FILE) else ensure_schema(pd.DataFrame())

        begin(1, "Google-searching missing Instagrams")
        if apify_token:
            df = run_stage(core_scrape_google, df, load_json_set(SCRAPED_GOOGLES_FILE), apify_token, report)
            save_db(df)
        else:
            job["log"].append("⏭️ Skipped — Apify token missing (add it in Settings).")

        begin(2, "Scraping Instagram bios & emails")
        if apify_token:
            df = run_stage(core_scrape_instagram, df, load_json_set(SCRAPED_IGS_FILE), apify_token, report)
            save_db(df)
        else:
            job["log"].append("⏭️ Skipped — Apify token missing (add it in Settings).")

        begin(3, "Cleaning names & titles")
        if gemini_key:
            df = run_stage(core_clean_names, df, gemini_key, report)
            save_db(df)
        else:
            job["log"].append("⏭️ Skipped — Gemini key missing (add it in Settings).")

        job["stage"] = "Backing up to GitHub"
        job["detail"] = ""
        try:
            if _cloud_sync_core():
                job["log"].append("☁️ Saved everything to GitHub.")
        except Exception as e:
            job["log"].append(f"☁️⚠️ Cloud save failed: {_cloud_err_msg(e)}")
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["running"] = False
        job["finished"] = time.time()

def start_full_pipeline():
    job = _pipeline_job()
    if job["running"]:
        return False
    job.update({"running": True, "stage": "Starting...", "detail": "", "frac": 0.0,
                "log": [], "error": None, "started": time.time(), "finished": 0.0})
    threading.Thread(
        target=run_full_pipeline,
        args=(job, get_key("APIFY_API_TOKEN"), get_key("GEMINI_API_KEY")),
        daemon=True,
    ).start()
    return True

@st.fragment(run_every=2.0)
def _pipeline_status_live():
    """Auto-refreshing status card. When the job ends, refresh the whole page."""
    job = _pipeline_job()
    if job["running"]:
        st.progress(min(1.0, max(0.0, float(job.get("frac") or 0.0))))
        st.info(f"**{job.get('stage') or 'Working...'}**  \n{job.get('detail') or ''}")
        st.caption("🟢 Running on the server — you can close this tab and come back anytime.")
    else:
        st.rerun(scope="app")

def render_pipeline_box():
    job = _pipeline_job()
    with st.container(border=True):
        st.subheader("2 · Run everything (one click)")
        st.caption("Finds missing Instagrams, scrapes bios & emails, then cleans names and titles — "
                   "in that order, automatically. It runs on the server, so once it starts you can "
                   "close the browser and come back when it's done.")
        if job["running"]:
            _pipeline_status_live()
            return
        if job.get("finished"):
            when = time.strftime("%H:%M", time.localtime(job["finished"]))
            with st.expander(f"Last run · finished {when}"):
                for line in job["log"]:
                    st.write(line)
                if job.get("error"):
                    st.error(f"Run crashed: {job['error']}")
        missing = [lbl for k, lbl in [("APIFY_API_TOKEN", "Apify"), ("GEMINI_API_KEY", "Gemini")] if not get_key(k)]
        if missing:
            st.warning("Missing API keys: " + ", ".join(missing) + " — those steps would be skipped. Add them in Settings.")
        if st.button("🚀 Run everything", type="primary", width="stretch"):
            start_full_pipeline()
            st.rerun()

# ----------------------------------------------------------------------------
# State init (runs on every rerun)
# ----------------------------------------------------------------------------
cloud_restore_on_boot()  # fresh container -> pull saved data back from GitHub first

_job_boot = _pipeline_job()
if "df" not in st.session_state:
    st.session_state.df = load_db()
    st.session_state["_df_loaded_at"] = time.time()
elif (not _job_boot["running"]) and _job_boot.get("finished", 0) > st.session_state.get("_df_loaded_at", 0):
    # A background pipeline finished since this tab last loaded — pick up its results
    st.session_state.df = load_db()
    st.session_state["_df_loaded_at"] = time.time()

seen_videos = load_json_set(SEEN_VIDEOS_FILE)
blacklist = load_json_set(BLACKLIST_FILE)
scraped_igs = load_json_set(SCRAPED_IGS_FILE)
scraped_googles = load_json_set(SCRAPED_GOOGLES_FILE)

# ----------------------------------------------------------------------------
# Light global styling (theme lives in .streamlit/config.toml)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

/* Self-healing icons: load Material Symbols ourselves and pin it to icon
   elements, so icons render even if Streamlit's own font copy fails to load
   (ad blockers, network hiccups). Raw text like "expand_more" = font missing. */
span[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal;
    letter-spacing: normal;
    text-transform: none;
    line-height: 1;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}

/* Narrow selectors only — anything broader breaks Streamlit's icon font
   and renders icons as raw text like "keyboard_double_arrow_right". */
p, li, label, input, textarea, [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', -apple-system, sans-serif;
}
h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.01em;
}
h1 { text-transform: uppercase; letter-spacing: 0.03em; }
.stButton button, .stDownloadButton button, .stLinkButton a, .stFormSubmitButton button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
}
[data-testid="stCaptionContainer"], .stCaption, code, .stCode, [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
}
.block-container {padding-top: 2.2rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
""", unsafe_allow_html=True)

STAGE_COLORS = {
    "Leads":       ("#9AA3B2", "#3E434D"),  # slate — raw material
    "Contactable": ("#37B8C4", "#14555C"),  # teal — reachable
    "Drafted":     ("#F0A93B", "#7A5210"),  # brand amber — work done
    "Contacted":   ("#E8763A", "#77320F"),  # hot orange — out the door
    "Replied":     ("#3BC474", "#155A33"),  # green — success
    "Free Mix":    ("#9B8CFF", "#3F3878"),  # violet — proof delivered
    "Paid":        ("#EFC94C", "#6E5716"),  # gold — money
}

def vu_meter(label, count, total):
    pct = 0 if total <= 0 or count <= 0 else max(6, int(round(100 * count / total)))
    hi, lo = STAGE_COLORS.get(label, (ACCENT, "#7A5210"))
    return f"""
    <div style="text-align:center;">
      <div style="height:110px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.10);
                  border-radius:8px;display:flex;align-items:flex-end;overflow:hidden;">
        <div style="width:100%;height:{pct}%;background:linear-gradient(180deg,{hi},{lo});"></div>
      </div>
      <div style="margin-top:6px;font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;">{count}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:.12em;text-transform:uppercase;opacity:.6;">{label}</div>
    </div>"""

REPORT_CSS = """
body{margin:0;background:#0A0A0C;color:#F4F2ED;font-family:'Inter',-apple-system,sans-serif;}
.wrap{max-width:820px;margin:0 auto;padding:48px 28px;}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.14em;color:#9AA3B2;text-transform:uppercase;}
h1{font-family:'Space Grotesk',sans-serif;font-size:40px;letter-spacing:.02em;text-transform:uppercase;margin:6px 0 28px;}
h2{font-family:'Space Grotesk',sans-serif;font-size:18px;margin:36px 0 14px;color:#F0A93B;text-transform:uppercase;letter-spacing:.06em;}
.cards{display:flex;gap:14px;flex-wrap:wrap;}
.card{flex:1;min-width:150px;background:#151518;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:18px;}
.card .v{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:700;}
.card .l{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#9AA3B2;margin-top:6px;}
.frow{display:flex;align-items:center;gap:12px;margin:9px 0;}
.frow .fl{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#9AA3B2;width:110px;flex-shrink:0;}
.bar{flex:1;height:22px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:5px;overflow:hidden;}
.fill{height:100%;}
.frow .fc{font-family:'Space Grotesk',sans-serif;font-weight:700;width:90px;text-align:right;flex-shrink:0;}
table{width:100%;border-collapse:collapse;margin-top:8px;}
th{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#9AA3B2;text-align:left;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.12);}
td{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,.06);font-size:14px;}
td.money{color:#EFC94C;font-family:'Space Grotesk',sans-serif;font-weight:700;}
.foot{margin-top:44px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5b616c;letter-spacing:.1em;text-transform:uppercase;}
"""

def build_report_html(df, start=None, end=None, period_label="All time"):
    now = pd.Timestamp.now()
    ranged = start is not None

    def pct(a, b):
        return "—" if not b else f"{a / b * 100:.0f}%"

    if ranged:
        dates = pd.to_datetime(df["Reached_Out_Date"], errors="coerce")
        cohort = df[(dates >= start) & (dates <= end)] if not df.empty else df
        subtitle = f"Cohort: artists contacted {period_label} · outcomes to date"
        contacted = len(cohort)
        replied = int(cohort["Replied"].sum()) if contacted else 0
        free_mix = int(cohort["Free Mix Sent"].sum()) if contacted else 0
        paid = int(cohort["Paid Customer"].sum()) if contacted else 0
        revenue = float(cohort["Revenue"].sum()) if contacted else 0.0
        stages = [("Contacted", contacted), ("Replied", replied), ("Free Mix", free_mix), ("Paid", paid)]
        base = contacted
        clients_src = cohort

        mix_dates = pd.to_datetime(df["Mix_Date"], errors="coerce")
        paid_dates = pd.to_datetime(df["Paid_Date"], errors="coerce")
        mixes_in_period = int(((mix_dates >= start) & (mix_dates <= end)).sum())
        closed_in_period = int(((paid_dates >= start) & (paid_dates <= end)).sum())
        activity = [("Contacted in period", contacted),
                    ("Mixes delivered in period", mixes_in_period),
                    ("New paying clients", closed_in_period)]
    else:
        stats = pipeline_stats(df)
        subtitle = "All time"
        contacted, replied = stats["contacted"], stats["replied"]
        free_mix, paid, revenue = stats["free_mix"], stats["paid"], stats["revenue"]
        stages = [("Leads", stats["total"]), ("Contactable", stats["contactable"]), ("Drafted", stats["drafted"]),
                  ("Contacted", contacted), ("Replied", replied), ("Free Mix", free_mix), ("Paid", paid)]
        base = stats["total"]
        clients_src = df

        contacted_dates = pd.to_datetime(df["Reached_Out_Date"], errors="coerce") if not df.empty else pd.Series(dtype="datetime64[ns]")
        c7 = int((contacted_dates >= now - pd.Timedelta(days=7)).sum()) if not df.empty else 0
        c30 = int((contacted_dates >= now - pd.Timedelta(days=30)).sum()) if not df.empty else 0
        due = len(follow_ups_due(df)) if not df.empty else 0
        activity = [("Contacted · 7 days", c7), ("Contacted · 30 days", c30), ("Follow-ups due", due)]

    avg_client = f"${revenue / paid:,.0f}" if paid else "—"
    rev_per_contact = f"${revenue / contacted:,.2f}" if contacted else "—"

    funnel = ""
    for label, count in stages:
        hi, lo = STAGE_COLORS.get(label, (ACCENT, "#7A5210"))
        width = 0 if base == 0 or count == 0 else max(2, round(100 * count / base))
        funnel += (f'<div class="frow"><div class="fl">{label}</div>'
                   f'<div class="bar"><div class="fill" style="width:{width}%;background:linear-gradient(90deg,{lo},{hi});"></div></div>'
                   f'<div class="fc">{count} · {pct(count, base)}</div></div>')

    clients_rows = ""
    if not clients_src.empty and clients_src["Paid Customer"].any():
        top = clients_src[clients_src["Paid Customer"]].sort_values("Revenue", ascending=False).head(15)
        for _, r in top.iterrows():
            name = html_escape.escape(display_name(r))
            mix = "✓" if r.get("Free Mix Sent", False) else "—"
            clients_rows += f'<tr><td>{name}</td><td>{mix}</td><td class="money">${float(r["Revenue"]):,.0f}</td></tr>'
    else:
        clients_rows = '<tr><td colspan="3" style="color:#9AA3B2;">No paying clients in this period yet.</td></tr>'

    activity_cards = "".join(
        f'<div class="card"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in activity
    )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>WavyMixing — Outreach Report {now:%Y-%m-%d}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{REPORT_CSS}</style></head><body><div class="wrap">
<div class="eyebrow">WAVYMIXING · OUTREACH REPORT · {period_label.upper()} · {now:%b %d, %Y}</div>
<h1>Pipeline Report</h1>
<div class="eyebrow" style="margin:-16px 0 24px;">{subtitle}</div>
<div class="cards">
  <div class="card"><div class="v" style="color:#EFC94C;">${revenue:,.0f}</div><div class="l">Revenue</div></div>
  <div class="card"><div class="v">{paid}</div><div class="l">Paying clients</div></div>
  <div class="card"><div class="v">{avg_client}</div><div class="l">Avg per client</div></div>
  <div class="card"><div class="v">{free_mix}</div><div class="l">Free mixes done</div></div>
</div>
<h2>Funnel</h2>
{funnel}
<h2>Conversion rates</h2>
<div class="cards">
  <div class="card"><div class="v">{pct(replied, contacted)}</div><div class="l">Reply rate</div></div>
  <div class="card"><div class="v">{pct(free_mix, replied)}</div><div class="l">Replies → mixes</div></div>
  <div class="card"><div class="v">{pct(paid, free_mix)}</div><div class="l">Mixes → paid</div></div>
  <div class="card"><div class="v">{rev_per_contact}</div><div class="l">Revenue / contact</div></div>
</div>
<h2>Activity</h2>
<div class="cards">{activity_cards}</div>
<h2>Paying clients</h2>
<table><tr><th>Artist</th><th>Free mix</th><th>Revenue</th></tr>{clients_rows}</table>
<div class="foot">Generated by Wavy Outreach · wavymixing.com</div>
</div></body></html>"""

def highlight_rows(row):
    color = ''
    if row.get('Paid Customer', False):
        color = 'background-color: rgba(239, 201, 76, 0.30)'
    elif row.get('Replied', False):
        color = 'background-color: rgba(39, 174, 96, 0.35)'
    elif row.get('Emailed', False) or row.get("DM'd", False):
        date_val = last_touch_date(row)
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
    st.caption("WAVYMIXING · OUTREACH CONSOLE")
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
    cols = st.columns(7)
    meters = [
        ("Leads", stats["total"]), ("Contactable", stats["contactable"]),
        ("Drafted", stats["drafted"]), ("Contacted", stats["contacted"]),
        ("Replied", stats["replied"]), ("Free Mix", stats["free_mix"]),
        ("Paid", stats["paid"]),
    ]
    for col, (label, count) in zip(cols, meters):
        col.markdown(vu_meter(label, count, stats["total"]), unsafe_allow_html=True)

    st.write("")
    r1a, r1b, r1c = st.columns(3)
    with r1a, st.container(border=True):
        st.metric("💰 Revenue", f"${stats['revenue']:,.0f}")
    with r1b, st.container(border=True):
        conv = f"{(stats['paid'] / stats['free_mix'] * 100):.0f}%" if stats["free_mix"] else "—"
        st.metric("Paying clients", stats["paid"], help="Conversion from free mixes: " + conv)
    with r1c, st.container(border=True):
        st.metric("Free mixes delivered", stats["free_mix"])

    m1, m2, m3 = st.columns(3)
    reply_rate = f"{(stats['replied'] / stats['contacted'] * 100):.0f}%" if stats["contacted"] else "—"
    with m1, st.container(border=True):
        st.metric("Reply rate", reply_rate)
    queue_count = int(df.apply(lambda r: has_any_draft(r) and bool(pending_channels(r)) and not bool(r["Reached Out"]), axis=1).sum())
    with m2, st.container(border=True):
        st.metric("Ready to send", queue_count)
    due = follow_ups_due(df)
    with m3, st.container(border=True):
        st.metric("Follow-ups due", len(due))

    with st.container(border=True):
        rc1, rc2, rc3 = st.columns([2, 3, 2], vertical_alignment="bottom")
        with rc1:
            period = st.selectbox(
                "📄 CRM report period",
                ["All time", "Last 7 days", "Last 30 days", "Last 90 days", "Last year", "Custom range"],
                key="report_period",
            )
        start = end = None
        label = period.lower() if period != "All time" else "All time"
        if period == "Custom range":
            with rc2:
                d1, d2 = st.columns(2)
                start_d = d1.date_input("From", value=pd.Timestamp.now().date() - pd.Timedelta(days=30), key="rep_from")
                end_d = d2.date_input("To", value=pd.Timestamp.now().date(), key="rep_to")
            start = pd.Timestamp(start_d)
            end = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            label = f"{start_d:%b %d} – {end_d:%b %d, %Y}"
        elif period != "All time":
            days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "Last year": 365}[period]
            start = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
            end = pd.Timestamp.now()
        slug = period.lower().replace(" ", "-")
        with rc3:
            st.download_button(
                "Download report",
                data=build_report_html(df, start, end, label).encode("utf-8"),
                file_name=f"wavymixing_report_{slug}_{pd.Timestamp.now():%Y-%m-%d}.html",
                mime="text/html",
                width="stretch",
            )

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
    mixes_owed = int(((df["Replied"]) & (~df["Free Mix Sent"])).sum())
    if mixes_owed:
        actions.append((f"**{mixes_owed}** artists replied and are waiting on their free mix — deliver these first.", "📋 Open Leads", pg_leads))
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

def last_touch_date(row):
    dates = [d for d in [row.get("Email_Date"), row.get("DM_Date"), row.get("Reached_Out_Date")] if pd.notna(d)]
    return max(dates) if dates else pd.NaT

def follow_ups_due(df):
    if df.empty:
        return df
    mask = (df["Emailed"] | df["DM'd"]) & (~df["Replied"])
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub["_touch"] = sub.apply(last_touch_date, axis=1)
    sub = sub[sub["_touch"].notna()]
    if sub.empty:
        return sub
    days = (pd.Timestamp.now().normalize() - sub["_touch"].dt.normalize()).dt.days
    return sub[days >= FOLLOW_UP_DAYS].drop(columns=["_touch"])

# ============================================================================
# PAGE: Collect
# ============================================================================
def page_collect():
    show_flash()
    st.title("🎧 Collect")
    st.caption("Pull artists from a YouTube playlist, then enrich them with Instagram and Google data.")

    df = st.session_state.df
    pipeline_running = _pipeline_job()["running"]

    # ---- Step 1: Sync playlist -------------------------------------------
    with st.container(border=True):
        st.subheader("1 · Sync playlist")
        playlist_input = st.text_input(
            "YouTube playlist URL or ID",
            value=load_last_playlist(),
            placeholder="https://www.youtube.com/playlist?list=PL...",
        )
        if st.button("Sync playlist", type="primary", disabled=pipeline_running,
                      help="Locked while the one-click pipeline is running." if pipeline_running else None):
            yt_key = get_key("YOUTUBE_API_KEY")
            if not yt_key:
                st.error("YouTube API key is missing. Add it in Settings.")
            elif not playlist_input:
                st.warning("Paste a playlist URL or ID first.")
            else:
                save_last_playlist(playlist_input)
                sync_playlist(playlist_input, yt_key)

    # ---- Step 2: One-click pipeline ---------------------------------------
    render_pipeline_box()

    # ---- Step 3: Enrich manually ------------------------------------------
    ig_pending, google_pending = enrich_targets(df, scraped_igs, scraped_googles)
    with st.container(border=True):
        st.subheader("3 · Or run each step yourself")
        if not get_key("APIFY_API_TOKEN"):
            st.warning("Apify token is missing — add it in Settings to enable enrichment.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scrape Instagram bios**")
            st.caption(f"Pulls bio, follower count, and emails. {ig_pending} profiles pending.")
            if st.button("Scrape Instagram bios", disabled=(ig_pending == 0 or not get_key("APIFY_API_TOKEN") or pipeline_running)):
                scrape_instagram()
        with c2:
            st.markdown("**Find missing Instagrams**")
            st.caption(f"Googles each artist for a profile link. {google_pending} artists pending.")
            if st.button("Search Google for IGs", disabled=(google_pending == 0 or not get_key("APIFY_API_TOKEN") or pipeline_running)):
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
                        "Added_Date": pd.Timestamp.now().normalize(),
                    })
        except Exception as e:
            st.error(f"Could not fetch channel details: {e}")
            return

    new_df = ensure_schema(pd.DataFrame(new_channels_data))
    st.session_state.df = ensure_schema(pd.concat([st.session_state.df, new_df], ignore_index=True))
    save_db(st.session_state.df)
    flash("success", f"Added {len(new_channels_data)} new artists from {new_videos_found} new videos.")
    st.rerun()

def core_scrape_instagram(df, igs_set, token, report):
    """Scrape IG bios/followers/emails for pending profiles. Session-free so it
    can run in a background thread. Mutates df in place, saves the memory set,
    and returns (df, summary) — summary is None when there was nothing to do.
    Raises RuntimeError if the Apify run does not succeed."""
    target_usernames, idx_mapping = [], {}

    for idx in df.index:
        ig_val = str(df.loc[idx, "Instagram"])
        if is_valid_data(ig_val) and not ig_already_scanned(df.loc[idx]) \
           and (not is_valid_data(df.loc[idx, "Email Address"]) or safe_int(df.loc[idx, "IG Followers"]) <= 0):
            ig_url = ig_val.split(",")[0].strip().rstrip("/")
            ig_url_lower = ig_url.lower()
            if ig_url_lower not in igs_set:
                username = extract_ig_username(ig_url)
                username_lower = username.lower()
                if username_lower not in idx_mapping:
                    idx_mapping[username_lower] = {"indices": [], "url": ig_url_lower}
                    target_usernames.append(username)
                idx_mapping[username_lower]["indices"].append(idx)

    if not target_usernames:
        return df, None

    report(0.0, f"Starting Apify job for {len(target_usernames)} Instagram profiles...")
    from apify_client import ApifyClient
    client = ApifyClient(token)
    status, dataset_id = run_apify_and_poll(
        client, "apify/instagram-profile-scraper",
        {"usernames": target_usernames},
        len(target_usernames), report, "Scraping Instagram",
    )
    if status != "SUCCEEDED":
        raise RuntimeError(f"Apify run ended with status: {status}")

    report(0.98, "Saving results to your leads...")
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
                if i not in df.index:
                    continue
                df.loc[i, "IG Bio"] = bio_text if bio_text.strip() else "Empty Bio"
                df.loc[i, "IG Followers"] = followers_count
                if found_email != "None Found" and not is_valid_data(df.loc[i, "Email Address"]):
                    df.loc[i, "Email Address"] = found_email
                    emails_found += 1

            igs_set.add(mapped["url"])
            profiles_scraped += 1

    save_json_set(igs_set, SCRAPED_IGS_FILE)
    return df, f"Scraped {profiles_scraped} Instagram bios and found {emails_found} new emails."

def scrape_instagram():
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    def report(frac, text):
        progress_bar.progress(min(1.0, max(0.0, float(frac))))
        status_text.info(text)
    summary, ok = None, False
    try:
        df, summary = core_scrape_instagram(st.session_state.df, scraped_igs, get_key("APIFY_API_TOKEN"), report)
        if summary is None:
            st.info("No unscanned Instagram profiles found.")
            return
        st.session_state.df = df
        save_db(st.session_state.df)
        ok = True
    except Exception as e:
        status_text.error(f"Instagram scrape failed: {e}")
    if ok:
        flash("success", summary)
        st.rerun()

def core_scrape_google(df, googles_set, token, report):
    """Google-search artists with no Instagram yet. Session-free (see above).
    Returns (df, summary) — summary is None when there was nothing to do."""
    target_queries, idx_mapping = [], {}

    for idx in df.index:
        ig_val = str(df.loc[idx, "Instagram"])
        channel_name = str(df.loc[idx, "Channel Name"]).strip()
        channel_lower = channel_name.lower()
        if not is_valid_data(ig_val) and not google_already_searched(df.loc[idx]) and channel_lower not in googles_set:
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
        return df, None

    report(0.0, f"Starting Google search for {len(target_queries)} artists...")
    from apify_client import ApifyClient
    client = ApifyClient(token)
    status, dataset_id = run_apify_and_poll(
        client, "apify/google-search-scraper",
        {"queries": "\n".join(target_queries), "maxPagesPerQuery": 1, "resultsPerPage": 10},
        len(target_queries), report, "Searching Google",
    )
    if status != "SUCCEEDED":
        raise RuntimeError(f"Apify run ended with status: {status}")

    report(0.98, "Saving results to your leads...")
    igs_found = 0
    for item in client.dataset(dataset_id).iterate_items():
        original_query = item.get("searchQuery", {}).get("term", "").lower()
        if original_query in idx_mapping:
            mapped = idx_mapping[original_query]
            organic_results = item.get("organicResults", [])
            combined = " ".join([r.get("description", r.get("snippet", "")) + " " + r.get("url", "") for r in organic_results])
            found_ig = extract_instagram(combined)

            for i in mapped["indices"]:
                if i not in df.index:
                    continue
                df.loc[i, "Google Search Status"] = f"Searched ({len(organic_results)} results)"
                if found_ig != "None" and not is_valid_data(df.loc[i, "Instagram"]):
                    df.loc[i, "Instagram"] = found_ig
                    igs_found += 1

    for _, mapped in idx_mapping.items():
        googles_set.add(mapped["channel"])
        for i in mapped["indices"]:
            if i in df.index and not is_valid_data(df.loc[i, "Google Search Status"]):
                df.loc[i, "Google Search Status"] = "Searched (0 results)"

    save_json_set(googles_set, SCRAPED_GOOGLES_FILE)
    return df, f"Google search finished — found {igs_found} missing Instagram profiles."

def scrape_google():
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    def report(frac, text):
        progress_bar.progress(min(1.0, max(0.0, float(frac))))
        status_text.info(text)
    summary, ok = None, False
    try:
        df, summary = core_scrape_google(st.session_state.df, scraped_googles, get_key("APIFY_API_TOKEN"), report)
        if summary is None:
            st.info("No artists need a Google search right now.")
            return
        st.session_state.df = df
        save_db(st.session_state.df)
        ok = True
    except Exception as e:
        status_text.error(f"Google search failed: {e}")
    if ok:
        flash("success", summary)
        st.rerun()

def core_clean_names(df, gemini_key, report):
    """Clean artist/song names for every pending lead. Session-free (see above).
    Returns (df, summary) — summary is None when there was nothing to do.
    Stops (raises) after 3 consecutive Gemini failures; partial work stays in df."""
    clean_targets, _ = compute_write_targets(df)
    if not clean_targets:
        return df, None

    gclient = google_genai.Client(api_key=gemini_key)
    report(0.0, "Checking which Gemini model your key can use...")
    model = resolve_gemini_model(gclient)
    errors, last_error, cleaned, consecutive = 0, "", 0, 0
    total = len(clean_targets)
    for pos, idx in enumerate(clean_targets):
        row = df.loc[idx]
        report(pos / total, f"Cleaning {row['Channel Name']}... ({pos + 1}/{total})")
        artist, song, err = clean_with_gemini(gclient, model, row["Channel Name"], row["Song Name"])
        if err:
            errors += 1
            last_error = err
            consecutive += 1
            if consecutive >= 3:
                raise RuntimeError(
                    f"Gemini failed 3 times in a row after cleaning {cleaned} of {total} leads — "
                    f"the rest stayed pending for retry. Last error: {last_error}"
                )
        else:
            consecutive = 0
            cleaned += 1
            df.loc[idx, "Cleaned Artist"] = artist
            df.loc[idx, "Cleaned Song"] = song
    if errors:
        return df, f"Cleaned {cleaned} names; {errors} failed and stayed pending for retry. Last error: {last_error}"
    return df, f"Cleaned names for {cleaned} leads with {model}."

# ============================================================================
# PAGE: Write
# ============================================================================
def page_write():
    show_flash()
    st.title("✍️ Write")
    st.caption("Clean up messy titles, then draft a personal email and DM for every contactable artist.")

    df = st.session_state.df
    if _pipeline_job()["running"]:
        st.info("🚀 The one-click pipeline is running (see **Collect** for live progress). "
                "Write is paused until it finishes so the two don't overwrite each other.")
        return
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
                        err_low = str(last_error).lower()
                        if "prepayment" in err_low or ("credit" in err_low and "429" in err_low):
                            flash("warning", f"Stopped: your Google AI prepaid credits are used up — this is a billing balance issue, not a rate limit, so the usage dashboard will still look empty. Top up at ai.studio/projects (a small amount covers thousands of cleanings), then press Clean names again. The remaining {total - pos - 1} leads are still pending and nothing was overwritten.")
                        else:
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

    if _pipeline_job()["running"]:
        st.info("🚀 The one-click pipeline is running (see **Collect** for live progress). "
                "Send is paused until it finishes so nothing gets overwritten mid-run.")
        return

    df = st.session_state.df
    # Contacted artists (emailed OR DM'd) drop out of the queue.
    queue_all = [i for i in df.index
                 if has_any_draft(df.loc[i]) and pending_channels(df.loc[i])
                 and not bool(df.loc[i, "Reached Out"])] if not df.empty else []

    fc1, fc2 = st.columns(2)
    chan = fc1.radio("Channel", ["All", "✉️ Email pending", "📸 DM pending"], horizontal=True, key="send_chan")
    fstat = fc2.radio("Follow status", ["All", "➕ Not followed", "✅ Followed"], horizontal=True, key="send_follow",
                      help="Follow-first flow: pass 1 — follow everyone here. Pass 2 — filter to Followed and DM the ones who followed back.")
    if chan == "✉️ Email pending":
        queue_idx = [i for i in queue_all if "email" in pending_channels(df.loc[i])]
    elif chan == "📸 DM pending":
        queue_idx = [i for i in queue_all if "ig" in pending_channels(df.loc[i])]
    else:
        queue_idx = queue_all
    if fstat == "➕ Not followed":
        queue_idx = [i for i in queue_idx if not bool(df.loc[i, "Followed"])]
    elif fstat == "✅ Followed":
        queue_idx = [i for i in queue_idx if bool(df.loc[i, "Followed"])]

    if not queue_idx:
        if queue_all:
            st.info("Nothing pending on this channel — switch the filter above.")
        else:
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

        nav_l, nav_mid, nav_r = st.columns([1, 10, 1], vertical_alignment="bottom")
        with nav_l:
            if st.button("◀", key="send_prev", help="Previous artist", width="stretch"):
                new_pos = (default_pos - 1) % len(queue_idx)
                st.session_state["send_current_id"] = df.loc[queue_idx[new_pos], "Channel_ID"]
                st.rerun()
        with nav_mid:
            choice = st.selectbox(
                f"Up next · {default_pos + 1} of {len(queue_idx)} in queue",
                options=queue_idx,
                index=default_pos,
                format_func=lambda i: f"{display_name(df.loc[i])} — {display_song(df.loc[i])}",
            )
        with nav_r:
            if st.button("▶", key="send_next", help="Next artist", width="stretch"):
                new_pos = (default_pos + 1) % len(queue_idx)
                st.session_state["send_current_id"] = df.loc[queue_idx[new_pos], "Channel_ID"]
                st.rerun()
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
                    if bool(row.get("Followed", False)):
                        fd = row.get("Followed_Date")
                        if pd.notna(fd):
                            days_ago = (pd.Timestamp.now().normalize() - pd.to_datetime(fd)).days
                            st.caption(f"➕ Followed {pd.to_datetime(fd):%b %d} · {days_ago}d ago — check if they follow back before you DM.")
                        else:
                            st.caption("➕ Followed — check if they follow back before you DM.")
                    elif st.button("➕ Mark followed", key=f"fol_{choice}", width="stretch",
                                   help="Tick after you tap Follow on their profile — the date stamps automatically."):
                        st.session_state.df.loc[choice, "Followed"] = True
                        st.session_state.df.loc[choice, "Followed_Date"] = pd.Timestamp.now().normalize()
                        save_db(st.session_state.df)
                        st.rerun()
                if is_valid_data(row.get("IG Bio", "")):
                    with st.expander("IG bio"):
                        st.write(row["IG Bio"])
                sent_bits = []
                if bool(row.get("Emailed", False)):
                    d = row.get("Email_Date")
                    sent_bits.append("✉️ Emailed" + (f" {pd.to_datetime(d):%b %d}" if pd.notna(d) else ""))
                if bool(row.get("DM'd", False)):
                    d = row.get("DM_Date")
                    sent_bits.append("📸 " + "DM'd" + (f" {pd.to_datetime(d):%b %d}" if pd.notna(d) else ""))
                if sent_bits:
                    st.caption(" · ".join(sent_bits))

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
            a1, a2, a3, a4 = st.columns([1.4, 1.4, 1, 1])
            can_email = is_valid_data(email_addr) and not bool(row.get("Emailed", False))
            can_dm = is_valid_data(row["Instagram"]) and not bool(row.get("DM'd", False))
            if a1.button("✉️ Mark emailed", type="primary", key=f"em_{choice}", width="stretch", disabled=not can_email,
                         help="Stamps today as the email date." if can_email else "No email, or already emailed."):
                mark_channel_sent(choice, "Emailed", "Email_Date", row)
            if a2.button("📸 Mark " + "DM'd", key=f"dm_{choice}", width="stretch", disabled=not can_dm,
                         help="Stamps today as the DM date." if can_dm else "No Instagram, or already sent."):
                mark_channel_sent(choice, "DM'd", "DM_Date", row)
            if a3.button("🔄 Rewrite next run", key=f"regen_{choice}", width="stretch", help="Flags this artist so Write drafts fresh messages next time."):
                st.session_state.df.loc[choice, "🔄 Regenerate"] = True
                save_db(st.session_state.df)
                st.toast("Flagged for regeneration.")
            if a4.button("🚫 Blacklist", key=f"bl_{choice}", width="stretch", help="Removes this artist and bans the channel from future syncs."):
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
            days = (pd.Timestamp.now().normalize() - pd.to_datetime(last_touch_date(row)).normalize()).days
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
                        "Added_Date": pd.Timestamp.now().normalize(),
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

    search = st.text_input("Search", placeholder="Search by artist name, Instagram, or email...", label_visibility="collapsed")
    col1, col2, col3 = st.columns(3)
    with col1: email_filter = st.radio("Email", ["All", "Has email", "No email"], horizontal=True)
    with col2: ig_filter = st.radio("Instagram", ["All", "Has IG", "No IG"], horizontal=True)
    with col3: crm_filter = st.radio("Pipeline", ["All", "Not contacted", "Contacted", "Emailed", "DM'd", "Replied", "Free mix sent", "Paid"], horizontal=True)

    filtered_df = st.session_state.df.copy()

    if search.strip():
        s = search.strip().lower()
        def _match(col):
            return filtered_df[col].astype(str).str.lower().str.contains(s, na=False, regex=False) if col in filtered_df.columns else False
        mask = (
            _match("Channel Name")
            | _match("Cleaned Artist")
            | _match("Instagram")
            | _match("Email Address")
            | _match("Song Name")
            | _match("Cleaned Song")
            | _match("IG Bio")
        )
        filtered_df = filtered_df[mask]

    if email_filter == "Has email": filtered_df = filtered_df[filtered_df["Email Address"].apply(is_valid_data).astype(bool)]
    elif email_filter == "No email": filtered_df = filtered_df[~filtered_df["Email Address"].apply(is_valid_data).astype(bool)]
    if ig_filter == "Has IG": filtered_df = filtered_df[filtered_df["Instagram"].apply(is_valid_data).astype(bool)]
    elif ig_filter == "No IG": filtered_df = filtered_df[~filtered_df["Instagram"].apply(is_valid_data).astype(bool)]
    if crm_filter == "Not contacted": filtered_df = filtered_df[filtered_df["Reached Out"] == False]
    elif crm_filter == "Contacted": filtered_df = filtered_df[(filtered_df["Reached Out"] == True) & (filtered_df["Replied"] == False)]
    elif crm_filter == "Emailed": filtered_df = filtered_df[filtered_df["Emailed"] == True]
    elif crm_filter == "DM'd": filtered_df = filtered_df[filtered_df["DM'd"] == True]
    elif crm_filter == "Replied": filtered_df = filtered_df[filtered_df["Replied"] == True]
    elif crm_filter == "Free mix sent": filtered_df = filtered_df[filtered_df["Free Mix Sent"] == True]
    elif crm_filter == "Paid": filtered_df = filtered_df[filtered_df["Paid Customer"] == True]

    st.caption(f"Showing {len(filtered_df)} of {len(st.session_state.df)} artists · amber = contacted · red = follow-up due · green = replied · gold = paying client · 🗑️ removes AND bans, ❌ just removes")

    if _pipeline_job()["running"]:
        st.warning("🚀 The one-click pipeline is running — the table is read-only until it finishes, "
                   "so your edits and the incoming data can't overwrite each other. Live progress is in **Collect**.")
        st.dataframe(
            filtered_df.drop(columns=["🗑️ Blacklist", "❌ Remove", "🔄 Regenerate", "Channel_ID", "Channel_URL"], errors="ignore"),
            width="stretch", hide_index=True,
        )
        return

    edited_df = st.data_editor(
        filtered_df.style.apply(highlight_rows, axis=1),
        column_config={
            "Channel_ID": None,
            "Channel_URL": None,
            "Song Name": None,
            "Draft Message": st.column_config.TextColumn("Message", width="large", max_chars=2000),
            "🗑️ Blacklist": st.column_config.CheckboxColumn("🗑️", help="Remove this lead AND ban the channel from future syncs."),
            "❌ Remove": st.column_config.CheckboxColumn("❌", help="Delete this lead without banning — it can come back on a future sync."),
            "Followed": st.column_config.CheckboxColumn("➕ Followed", help="Tick when you follow them on IG — the date stamps automatically."),
            "Emailed": st.column_config.CheckboxColumn("✉️ Emailed", help="Tick when the email goes out — the date stamps automatically."),
            "DM'd": st.column_config.CheckboxColumn("📸 DM'd", help="Tick when the DM goes out — the date stamps automatically."),
            "Reached Out": st.column_config.CheckboxColumn("Reached", disabled=True, help="Automatic: ticks as soon as you've emailed or DM'd this artist."),
            "Replied": st.column_config.CheckboxColumn("Replied"),
            "Free Mix Sent": st.column_config.CheckboxColumn("Free Mix", help="Tick when you've delivered their free mix."),
            "Paid Customer": st.column_config.CheckboxColumn("Paid 💰", help="Tick when they become a paying client."),
            "Revenue": st.column_config.NumberColumn("Revenue $", min_value=0, step=10, format="$%d", help="Total earned from this artist — update it every time they pay."),
            "🔄 Regenerate": st.column_config.CheckboxColumn("🔄", help="Rewrite this artist's messages on the next Write run."),
            "Reached_Out_Date": st.column_config.DateColumn("First contact", disabled=True, format="MMM DD, YYYY"),
            "Email_Date": st.column_config.DateColumn("Emailed on", disabled=True, format="MMM DD"),
            "DM_Date": st.column_config.DateColumn("DM'd on", disabled=True, format="MMM DD"),
            "Followed_Date": st.column_config.DateColumn("Followed on", disabled=True, format="MMM DD"),
            "Mix_Date": None,
            "Paid_Date": None,
            "Added_Date": None,
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

        was_replied = filtered_df.loc[idx, "Replied"]
        is_replied = edited_df.loc[idx, "Replied"]
        if is_replied != was_replied:
            changes_made = True
            st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, "Replied"] = is_replied

        for flag_col, date_col in [("Followed", "Followed_Date"), ("Emailed", "Email_Date"), ("DM'd", "DM_Date"), ("Free Mix Sent", "Mix_Date"), ("Paid Customer", "Paid_Date")]:
            was_f = filtered_df.loc[idx, flag_col]
            is_f = edited_df.loc[idx, flag_col]
            if is_f != was_f:
                changes_made = True
                st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, flag_col] = is_f
                st.session_state.df.loc[st.session_state.df["Channel_ID"] == ch_id, date_col] = (
                    pd.Timestamp.now().normalize() if is_f else pd.NaT
                )

        # Reached Out is derived: ticks as soon as either channel is contacted
        mask = st.session_state.df["Channel_ID"] == ch_id
        sub = st.session_state.df.loc[mask]
        if not sub.empty:
            r0 = sub.iloc[0]
            contacted_now = bool(r0["Emailed"] or r0["DM'd"])
            if bool(r0["Reached Out"]) != contacted_now:
                changes_made = True
                st.session_state.df.loc[mask, "Reached Out"] = contacted_now
                st.session_state.df.loc[mask, "Reached_Out_Date"] = pd.Timestamp.now().normalize() if contacted_now else pd.NaT

        for col in ["Channel Name", "Subscribers", "Email Address", "Instagram", "IG Bio", "IG Followers", "Cleaned Artist", "Cleaned Song", "Draft Message", "Revenue", "🔄 Regenerate"]:
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
    if not cloud_enabled():
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
        if cloud_enabled():
            st.caption(f"☁️ Cloud save is ON — every change is auto-saved to the `{_cloud_branch()}` "
                       f"branch of `{_cloud_repo()}` on GitHub and restored whenever the app reboots. "
                       "Downloading a backup below is an optional extra safety net.")
        else:
            st.caption("☁️ Cloud save is OFF, and Streamlit Cloud wipes this app's disk whenever it "
                       "restarts. To keep your data automatically: on GitHub create a fine-grained "
                       "token (Settings → Developer settings → Fine-grained tokens) with access to "
                       "only this repo and permission **Contents: Read and write**, then on Streamlit "
                       "Cloud add it under App → Settings → Secrets as `GITHUB_TOKEN = \"github_pat_...\"` "
                       "and reboot. Until then, download a backup after every session and restore it here.")
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
                    if incoming["Channel_ID"].apply(is_valid_data).astype(bool).any() and current["Channel_ID"].apply(is_valid_data).astype(bool).any():
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
        st.subheader("Message layout")
        st.caption("Add spacing to existing drafts: greeting on its own line, a blank line, the message, a blank line, then the sign-off — leaving room for your signature. Safe to run repeatedly; already-spaced messages are left as they are.")
        drafts = st.session_state.df["Draft Message"].apply(lambda x: bool(str(x).strip())).sum() if not st.session_state.df.empty else 0
        if st.button(f"Reformat {int(drafts)} existing messages", disabled=(drafts == 0)):
            n = 0
            for i in st.session_state.df.index:
                cur = st.session_state.df.loc[i, "Draft Message"]
                if str(cur).strip():
                    new_v = reformat_message(cur)
                    if new_v != cur:
                        st.session_state.df.loc[i, "Draft Message"] = new_v
                        n += 1
            save_db(st.session_state.df)
            flash("success", f"Reformatted {n} messages with spacing for your signature.")
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
    st.caption(f"{stats['total']} leads · {stats['followed']} followed · {stats['contacted']} contacted · {stats['replied']} replied")
    dots = " · ".join(("🟢" if get_key(n) else "⚪") + " " + lbl for n, lbl in KEY_NAMES.items())
    st.caption(dots)

nav.run()

# ----------------------------------------------------------------------------
# Keep scroll position — Streamlit rebuilds the page (and the leads table) on
# every edit, which normally throws you back to the top. This invisible script
# remembers where you were, per page, and puts you back there after each rerun.
# Wrapped defensively: if a future Streamlit removes this API, the app keeps
# working and only the scroll-restore quietly turns off.
# ----------------------------------------------------------------------------
try:
    components.html("""
<script>
(function () {
  try {
    var P = window.parent;
    var D = P.document;
    var KEY = "wavy-scroll:" + P.location.pathname;
    var restoring = true;

    function targets() {
      var out = [];
      var main = D.querySelector('section[data-testid="stMain"]') ||
                 D.querySelector("section.stMain") ||
                 D.querySelector("section.main") ||
                 D.querySelector('[data-testid="stAppViewContainer"]');
      if (main) out.push(["page", main]);
      var grids = D.querySelectorAll(".dvn-scroller");
      for (var i = 0; i < grids.length; i++) out.push(["grid" + i, grids[i]]);
      return out;
    }

    function save() {
      if (restoring) return;
      var pos = { win: [P.scrollX || 0, P.scrollY || 0] };
      var ts = targets();
      for (var i = 0; i < ts.length; i++) pos[ts[i][0]] = [ts[i][1].scrollLeft, ts[i][1].scrollTop];
      try { P.sessionStorage.setItem(KEY, JSON.stringify(pos)); } catch (e) {}
    }

    function hook() {
      var ts = targets();
      for (var i = 0; i < ts.length; i++) {
        var el = ts[i][1];
        if (!el.dataset.wavyHook) {
          el.dataset.wavyHook = "1";
          el.addEventListener("scroll", save, { passive: true });
        }
      }
      if (!D.body.dataset.wavyWinHook) {
        D.body.dataset.wavyWinHook = "1";
        P.addEventListener("scroll", save, { passive: true });
      }
    }

    var saved = {};
    try { saved = JSON.parse(P.sessionStorage.getItem(KEY) || "{}"); } catch (e) {}

    var tries = 0;
    var timer = setInterval(function () {
      tries += 1;
      hook();
      var ts = targets();
      for (var i = 0; i < ts.length; i++) {
        var p = saved[ts[i][0]];
        if (p) { ts[i][1].scrollLeft = p[0]; ts[i][1].scrollTop = p[1]; }
      }
      if (saved.win) P.scrollTo(saved.win[0], saved.win[1]);
      if (tries >= 8) { clearInterval(timer); restoring = false; }
    }, 120);
    setInterval(hook, 700);
  } catch (e) {}
})();
</script>
""", height=0)
except Exception:
    pass  # scroll restore unavailable — never block the app over it

# ----------------------------------------------------------------------------
# Cloud save — runs after the page so it captures everything this run changed.
# ----------------------------------------------------------------------------
if cloud_enabled():
    cloud_sync()

with st.sidebar:
    st.divider()
    if cloud_enabled():
        _cloud_err = st.session_state.get("_cloud_error")
        _pending = cloud_pending_count()
        if _cloud_err:
            st.caption(f"☁️⚠️ Cloud save error: {_cloud_err}")
        elif _pending:
            st.caption(f"☁️ {_pending} change(s) waiting to save")
        else:
            _ts = st.session_state.get("_cloud_last_sync")
            _when = f" · {time.strftime('%H:%M', time.localtime(_ts))}" if _ts else ""
            st.caption(f"☁️ All changes saved to GitHub{_when}")
        if st.button("💾 Save to cloud now", use_container_width=True):
            _n, _err = cloud_sync()
            if _err:
                st.error(_err)
            else:
                st.toast("Saved to GitHub ✓" if _n else "Already up to date ✓")
    else:
        st.caption("☁️ Cloud save is OFF — data is lost when the app reboots. "
                   "Add a `GITHUB_TOKEN` secret to turn it on (see Settings → Backup).")
