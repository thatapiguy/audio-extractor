"""
Libby Audio Extractor – Streamlit Web App
==========================================
Upload a HAR file captured from Libby, then download all audio parts
combined into a single MP3. Also detects chapters via ID3 tags and
the OverDrive Openbook API.
"""

import base64
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import zipfile
from urllib.parse import unquote

import requests
import streamlit as st

try:
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.mp3 import MP3
    MUTAGEN_OK = True
except ImportError:
    MUTAGEN_OK = False

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Tool detection (ffmpeg / ffprobe)
# ─────────────────────────────────────────────────────────────────────────────

def _find_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for path in [
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/local/bin/{name}",
        f"/usr/bin/{name}",
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None

FFMPEG  = _find_binary("ffmpeg")
FFPROBE = _find_binary("ffprobe")


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Libby Audio Extractor",
    page_icon="🎧",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────────────────────
# HAR parsing — audio parts
# ─────────────────────────────────────────────────────────────────────────────

def parse_har(har_bytes: bytes) -> dict:
    har = json.loads(har_bytes)
    entries = har["log"]["entries"]

    result = {
        "title": None,
        "author": None,
        "parts": [],
        "headers": {},
        "referer": None,
    }

    for entry in entries:
        url = entry["request"]["url"]
        if "thunder.api.overdrive.com" in url and "/media/" in url:
            body = entry["response"]["content"].get("text", "")
            if body:
                try:
                    meta = json.loads(body)
                    result["title"] = meta.get("title") or meta.get("sortTitle")
                    creators = meta.get("creators", [])
                    if creators:
                        result["author"] = creators[0].get("name")
                except json.JSONDecodeError:
                    pass

    dewey_parts = {}
    cdn_parts   = {}

    for i, entry in enumerate(entries):
        url    = entry["request"]["url"]
        mime   = entry["response"]["content"].get("mimeType", "")
        status = entry["response"]["status"]

        if "listen.libbyapp.com" in url and status in (301, 302):
            match = re.search(r"Part(\d+)\.mp3", unquote(url))
            if match:
                part_num = int(match.group(1))
                dewey_parts[part_num] = url
                for h in entry["request"]["headers"]:
                    if h["name"].lower() == "referer":
                        result["referer"] = h["value"]

        if mime == "audio/mpeg" or (
            "audioclips.cdn.overdrive.com" in url and status in (200, 206)
        ):
            size = entry["response"]["content"].get("size", 0)
            cdn_parts[i] = {"url": url, "size": size}
            if not result["headers"]:
                for h in entry["request"]["headers"]:
                    if h["name"].startswith(":"):
                        continue
                    if h["name"].lower() in (
                        "user-agent", "accept", "accept-language",
                        "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
                    ):
                        result["headers"][h["name"]] = h["value"]

    cdn_list = sorted(cdn_parts.keys())
    for idx, part_num in enumerate(sorted(dewey_parts.keys())):
        cdn_url  = cdn_parts[cdn_list[idx]]["url"]  if idx < len(cdn_list) else None
        cdn_size = cdn_parts[cdn_list[idx]]["size"] if idx < len(cdn_list) else 0
        result["parts"].append({
            "part_num":  part_num,
            "dewey_url": dewey_parts[part_num],
            "cdn_url":   cdn_url,
            "size":      cdn_size,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_download_session(audio_info: dict) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": audio_info["headers"].get(
            "user-agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Sec-Fetch-Dest": "audio",
        "Sec-Fetch-Mode": "cors",
    })
    if audio_info.get("referer"):
        session.headers["Referer"] = audio_info["referer"]
    return session


def download_part(
    session: requests.Session,
    part: dict,
    dest_dir: str,
    progress_callback=None,
) -> str:
    part_num = part["part_num"]
    filepath = os.path.join(dest_dir, f"part_{part_num:03d}.mp3")

    urls_to_try = []
    if part.get("cdn_url"):
        urls_to_try.append(part["cdn_url"])
    if part.get("dewey_url"):
        urls_to_try.append(part["dewey_url"])

    for url in urls_to_try:
        try:
            resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
            resp.raise_for_status()
            content_type   = resp.headers.get("Content-Type", "")
            content_length = int(resp.headers.get("Content-Length", 0))
            if "audio" not in content_type and content_length < 100_000:
                continue
            downloaded = 0
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and content_length:
                        progress_callback(downloaded / content_length)
            return filepath
        except requests.RequestException:
            continue

    raise RuntimeError(f"Could not download Part {part_num} from any URL")


def concat_with_ffmpeg(part_files: list, output_path: str):
    list_path = output_path + ".filelist.txt"
    try:
        with open(list_path, "w") as f:
            for pf in part_files:
                f.write(f"file '{pf.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'\n")
        result = subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", output_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error:\n{result.stderr}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def concat_pure_python(part_files: list, output_path: str):
    with open(output_path, "wb") as out:
        for pf in part_files:
            with open(pf, "rb") as inp:
                shutil.copyfileobj(inp, out)


def combine_parts(part_files: list, output_path: str) -> str:
    if FFMPEG:
        concat_with_ffmpeg(part_files, output_path)
        return "ffmpeg (lossless stream copy)"
    else:
        concat_pure_python(part_files, output_path)
        return "pure Python binary concatenation"


# ─────────────────────────────────────────────────────────────────────────────
# Chapter detection — helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ms(ms: int) -> str:
    """Milliseconds → H:MM:SS"""
    total_s, ms_r = divmod(int(ms), 1000)
    h, rem = divmod(total_s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"

def _fmt_s(seconds: float) -> str:
    """Seconds → H:MM:SS"""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Chapter detection — Option 1: ID3 tags
# ─────────────────────────────────────────────────────────────────────────────

def _mp3_duration_s(filepath: str) -> float:
    if MUTAGEN_OK:
        try:
            return MP3(filepath).info.length
        except Exception:
            pass
    if FFPROBE:
        try:
            r = subprocess.run(
                [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", filepath],
                capture_output=True, text=True, timeout=30,
            )
            return float(json.loads(r.stdout)["format"]["duration"])
        except Exception:
            pass
    return 0.0


def _chapters_via_ffprobe(filepath: str) -> list[dict]:
    if not FFPROBE:
        return []
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_chapters", filepath],
            capture_output=True, text=True, timeout=30,
        )
        chapters = []
        for ch in json.loads(r.stdout).get("chapters", []):
            chapters.append({
                "title":    ch.get("tags", {}).get("title", f"Chapter {ch['id']+1}"),
                "start_ms": int(float(ch["start_time"]) * 1000),
                "end_ms":   int(float(ch["end_time"])   * 1000),
            })
        return chapters
    except Exception:
        return []


def _chapters_via_mutagen(filepath: str) -> list[dict]:
    if not MUTAGEN_OK:
        return []
    try:
        tags        = ID3(filepath)
        chap_frames = tags.getall("CHAP")
        if not chap_frames:
            return []
        chapters = []
        for frame in sorted(chap_frames, key=lambda f: f.start_time):
            title = "Chapter"
            for sub in getattr(frame, "sub_frames", []):
                if sub.FrameID == "TIT2":
                    title = str(sub)
                    break
            chapters.append({
                "title":    title,
                "start_ms": frame.start_time,
                "end_ms":   frame.end_time,
            })
        return chapters
    except (ID3NoHeaderError, Exception):
        return []


def detect_id3_chapters(part_files: list[str]) -> list[dict]:
    """
    Scan downloaded MP3 parts for embedded ID3 CHAP frames.
    Returns a flat list with timestamps relative to the full combined file.
    """
    all_chapters  = []
    cumulative_ms = 0

    for filepath in sorted(part_files):
        fname      = os.path.basename(filepath)
        dur_s      = _mp3_duration_s(filepath)
        dur_ms     = int(dur_s * 1000)

        chapters = _chapters_via_ffprobe(filepath) or _chapters_via_mutagen(filepath)

        for ch in chapters:
            all_chapters.append({
                "Title":     ch["title"],
                "Start":     _fmt_ms(cumulative_ms + ch["start_ms"]),
                "End":       _fmt_ms(cumulative_ms + ch["end_ms"]),
                "Part file": fname,
                "_start_ms": cumulative_ms + ch["start_ms"],
                "_end_ms":   cumulative_ms + ch["end_ms"],
            })

        cumulative_ms += dur_ms

    return all_chapters


# ─────────────────────────────────────────────────────────────────────────────
# Chapter detection — Option 2: OverDrive Openbook API
# ─────────────────────────────────────────────────────────────────────────────

def extract_har_credentials(har_bytes: bytes) -> dict:
    har     = json.loads(har_bytes)
    entries = har["log"]["entries"]

    creds = {
        "dewey_base":  None,
        "token":       None,
        "buid":        None,
        "reserve_id":  None,
        "library_key": None,
        "m_param":     None,
    }

    for entry in entries:
        url = entry["request"]["url"]

        m = re.search(r"https://(dewey-([a-f0-9]+)\.listen\.libbyapp\.com)", url)
        if m and not creds["dewey_base"]:
            creds["dewey_base"] = f"https://{m.group(1)}"
            creds["buid"]       = m.group(2)

        if "sentry.libbyapp.com/open/audiobook" in url:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "t" in qs:
                try:
                    raw     = qs["t"][0] + "=" * (4 - len(qs["t"][0]) % 4)
                    decoded = json.loads(base64.b64decode(raw))
                    codex   = decoded.get("codex", {})
                    creds["title_id"]    = (codex.get("title")   or {}).get("titleId")
                    creds["library_key"] = (codex.get("library") or {}).get("key")
                except Exception:
                    pass

        if "listen.libbyapp.com/?" in url:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            for key in ("m", "d"):
                if key in qs:
                    try:
                        raw     = qs[key][0]
                        padded  = raw + "=" * (4 - len(raw) % 4)
                        decoded = json.loads(base64.b64decode(padded))
                        if decoded.get("token"):
                            creds["token"]   = decoded["token"]
                            creds["m_param"] = raw
                        if decoded.get("buid"):
                            creds["buid"] = decoded["buid"]
                    except Exception:
                        pass

        if "thunder.api.overdrive.com" in url and "/media/" in url:
            m = re.search(r"/media/([a-f0-9-]+)", url, re.IGNORECASE)
            if m and not creds["reserve_id"]:
                creds["reserve_id"] = m.group(1)

    return creds


def fetch_openbook_chapters(har_bytes: bytes) -> tuple[list[dict], str]:
    """
    Try to fetch the OverDrive openbook JSON and extract chapter data.
    Returns (chapters_list, status_message).
    """
    creds = extract_har_credentials(har_bytes)

    if not creds.get("dewey_base"):
        return [], "Dewey base URL not found in HAR."

    base    = creds["dewey_base"]
    m_param = creds.get("m_param", "")
    token   = creds.get("token", "")

    candidates = []
    if m_param:
        candidates += [
            f"{base}/openbook?m={m_param}",
            f"{base}/_d/openbook?m={m_param}",
        ]
    candidates += [
        f"{base}/openbook",
        f"{base}/_d/openbook",
        f"{base}/openbook.json",
    ]

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         base + "/",
    })
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    tried = []
    for url in candidates:
        short = url[:80] + "..." if len(url) > 80 else url
        try:
            r = session.get(url, timeout=15, allow_redirects=True)
            tried.append(f"• `{short}` → HTTP {r.status_code}")
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                openbook = r.json()
                chapters = _parse_openbook(openbook)
                return chapters, f"Retrieved from `{short}`"
        except requests.RequestException as e:
            tried.append(f"• `{short}` → Error: {e}")

    status = (
        "Could not retrieve openbook data. Tried:\n" + "\n".join(tried) + "\n\n"
        "The session token may have expired. Re-capture a fresh HAR and try again immediately."
    )
    return [], status


def _parse_openbook(openbook: dict) -> list[dict]:
    chapters = []

    spine = openbook.get("spine", [])
    if spine:
        cumulative = 0.0
        for i, item in enumerate(spine):
            dur    = float(item.get("duration") or 0)
            offset = float(item.get("offset")   or cumulative)
            chapters.append({
                "Title":    item.get("title") or item.get("label") or f"Part {i+1}",
                "Start":    _fmt_s(offset),
                "Duration": _fmt_s(dur),
                "_offset_s": offset,
                "_dur_s":    dur,
            })
            cumulative += dur
        return chapters

    toc = (openbook.get("nav") or {}).get("toc") or openbook.get("toc", [])
    if toc:
        for item in toc:
            chapters.append({
                "Title":  item.get("title") or item.get("label") or "Chapter",
                "Start":  _fmt_s(float(item.get("offset") or 0)),
                "_offset_s": float(item.get("offset") or 0),
            })
        return chapters

    for i, item in enumerate(openbook.get("readingOrder", [])):
        dur = float(item.get("duration") or 0)
        chapters.append({
            "Title":    item.get("title") or f"Part {i+1}",
            "Duration": _fmt_s(dur),
            "_dur_s":   dur,
        })

    return chapters


# ─────────────────────────────────────────────────────────────────────────────
# Chapter detection — Option 3: Audio analysis (silence + Whisper)
# ─────────────────────────────────────────────────────────────────────────────

def detect_silences(mp3_path: str, noise_db: int = -40, min_dur_s: float = 2.0) -> list[dict]:
    """
    Run ffmpeg silencedetect on the combined MP3.
    Returns list of silence windows with start, end, duration, and the
    timestamp where audio resumes (= candidate chapter start).
    """
    if not FFMPEG:
        return []

    result = subprocess.run(
        [FFMPEG, "-i", mp3_path,
         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur_s}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )

    silences = []
    current  = {}
    for line in result.stderr.split("\n"):
        sm = re.search(r"silence_start: ([\d.]+)", line)
        em = re.search(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", line)
        if sm:
            current = {"silence_start_s": float(sm.group(1))}
        if em and current:
            current["silence_end_s"] = float(em.group(1))
            current["silence_dur_s"] = float(em.group(2))
            current["chapter_start_s"] = float(em.group(1))
            silences.append(current)
            current = {}

    return silences


def confirm_chapters_whisper(
    mp3_path: str,
    silences: list[dict],
    clip_duration_s: int = 20,
    progress_callback=None,
) -> list[dict]:
    """
    For each silence candidate, extract a short clip from where the audio
    resumes and transcribe it with Whisper tiny.en.
    Marks entries where "chapter" is spoken, and captures the opening text.
    """
    if not WHISPER_OK or not FFMPEG:
        return silences

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    tmp_clip = tempfile.mktemp(suffix=".wav")
    confirmed = []

    for i, silence in enumerate(silences):
        start = silence["chapter_start_s"]
        try:
            subprocess.run(
                [FFMPEG, "-y", "-i", mp3_path,
                 "-ss", str(start), "-t", str(clip_duration_s),
                 "-ar", "16000", "-ac", "1", tmp_clip],
                capture_output=True, timeout=30,
            )
            segments, _ = model.transcribe(
                tmp_clip, language="en", beam_size=1, vad_filter=True,
            )
            transcript = " ".join(s.text for s in segments).strip()
            result = dict(silence)
            result["transcript"] = transcript
            result["has_chapter_word"] = "chapter" in transcript.lower()
            confirmed.append(result)
        except Exception as e:
            result = dict(silence)
            result["transcript"] = f"(transcription error: {e})"
            result["has_chapter_word"] = False
            confirmed.append(result)

        if progress_callback:
            progress_callback((i + 1) / len(silences))

    if os.path.exists(tmp_clip):
        os.remove(tmp_clip)

    return confirmed


# ─────────────────────────────────────────────────────────────────────────────
# Chapter splitting
# ─────────────────────────────────────────────────────────────────────────────

def _chapter_filename(idx: int, transcript: str) -> str:
    """
    Derive a clean filename from a Whisper transcript snippet.
    e.g. "Chapter 1, California One summer..." → "01_Chapter_1_California.mp3"
    """
    t = transcript.strip()

    m = re.match(
        r"(chapter\s+[\w]+)[,\.]?\s*([\w][\w\s]{0,30}?)(?:[,\.\!]|$)",
        t, re.IGNORECASE
    )
    if m:
        chapter_part = re.sub(r"\s+", "_", m.group(1).strip())
        title_part   = re.sub(r"[^\w\s]", "", m.group(2)).strip()
        title_part   = re.sub(r"\s+", "_", title_part)
        return f"{idx:02d}_{chapter_part}_{title_part}.mp3"

    clean = re.sub(r"[^\w\s]", "", t[:40]).strip()
    clean = re.sub(r"\s+", "_", clean)
    return f"{idx:02d}_{clean or 'Chapter'}.mp3"


def split_into_chapters(
    mp3_path: str,
    confirmed: list[dict],
    book_safe_title: str,
    progress_callback=None,
) -> list[tuple[str, bytes]]:
    """
    Cut the combined MP3 at the confirmed chapter boundaries using ffmpeg.
    Returns a list of (filename, audio_bytes) tuples — ready to zip.
    """
    if not FFMPEG or not confirmed:
        return []

    total_s = 0.0
    if FFPROBE:
        try:
            r = subprocess.run(
                [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", mp3_path],
                capture_output=True, text=True, timeout=30,
            )
            total_s = float(json.loads(r.stdout)["format"]["duration"])
        except Exception:
            pass

    chapters = []

    first_start = confirmed[0]["chapter_start_s"]
    if first_start > 10:
        chapters.append({
            "start_s":   0.0,
            "end_s":     first_start,
            "filename":  f"00_{book_safe_title}_Intro.mp3",
        })

    for i, ch in enumerate(confirmed):
        end_s = confirmed[i + 1]["chapter_start_s"] if i + 1 < len(confirmed) else total_s or None
        chapters.append({
            "start_s":   ch["chapter_start_s"],
            "end_s":     end_s,
            "filename":  _chapter_filename(i + 1, ch.get("transcript", "")),
        })

    tmp_dir = tempfile.mkdtemp(prefix="libby_split_")
    result_files = []

    try:
        for i, ch in enumerate(chapters):
            out_path = os.path.join(tmp_dir, ch["filename"])
            cmd = [FFMPEG, "-y", "-i", mp3_path,
                   "-ss", str(ch["start_s"])]
            if ch["end_s"]:
                cmd += ["-to", str(ch["end_s"])]
            cmd += ["-c", "copy", out_path]

            subprocess.run(cmd, capture_output=True, timeout=120)

            with open(out_path, "rb") as f:
                result_files.append((ch["filename"], f.read()))

            if progress_callback:
                progress_callback((i + 1) / len(chapters))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result_files


def build_zip(chapter_files: list[tuple[str, bytes]]) -> bytes:
    """Pack (filename, audio_bytes) pairs into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for filename, data in chapter_files:
            zf.writestr(filename, data)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# YOTO integration
# ─────────────────────────────────────────────────────────────────────────────

YOTO_AUTH_URL    = "https://login.yotoplay.com"
YOTO_API_URL     = "https://api.yotoplay.com"
YOTO_SCOPE       = "profile offline_access"
YOTO_AUDIENCE    = "https://api.yotoplay.com"
YOTO_ICON        = "yoto:#aUm9i3ex3qqAMYBv-i-O-pYMKuMJGICtR3Vhf289u2Q"
YOTO_TOKEN_FILE  = os.path.expanduser("~/.yoto_credentials.json")


def yoto_generate_pkce() -> tuple[str, str]:
    code_verifier  = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return code_verifier, code_challenge


def yoto_build_auth_url(client_id: str, redirect_uri: str, code_challenge: str) -> str:
    params = urllib.parse.urlencode({
        "audience":              YOTO_AUDIENCE,
        "scope":                 YOTO_SCOPE,
        "response_type":         "code",
        "client_id":             client_id,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
        "redirect_uri":          redirect_uri,
    })
    return f"{YOTO_AUTH_URL}/authorize?{params}"


def yoto_exchange_code(
    client_id: str, code: str, code_verifier: str, redirect_uri: str
) -> dict:
    resp = requests.post(
        f"{YOTO_AUTH_URL}/oauth/token",
        data={
            "grant_type":    "authorization_code",
            "client_id":     client_id,
            "code":          code,
            "code_verifier": code_verifier,
            "redirect_uri":  redirect_uri,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if not resp.ok:
        body = resp.json()
        raise RuntimeError(body.get("error_description") or body.get("error") or resp.text)
    return resp.json()


def yoto_refresh_access_token(client_id: str, refresh_token: str) -> dict:
    resp = requests.post(
        f"{YOTO_AUTH_URL}/oauth/token",
        data={
            "grant_type":    "refresh_token",
            "client_id":     client_id,
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if not resp.ok:
        body = resp.json()
        raise RuntimeError(body.get("error_description") or body.get("error") or resp.text)
    return resp.json()


def yoto_save_credentials(client_id: str, access_token: str, refresh_token: str) -> None:
    data = {
        "client_id":     client_id,
        "access_token":  access_token,
        "refresh_token": refresh_token,
    }
    with open(YOTO_TOKEN_FILE, "w") as f:
        json.dump(data, f, indent=2)


def yoto_load_credentials() -> dict | None:
    if not os.path.exists(YOTO_TOKEN_FILE):
        return None
    try:
        with open(YOTO_TOKEN_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def yoto_get_valid_token() -> tuple[str | None, str | None]:
    creds = yoto_load_credentials()
    if not creds:
        return None, None
    client_id     = creds.get("client_id", "")
    access_token  = creds.get("access_token", "")
    refresh_token = creds.get("refresh_token", "")
    try:
        r = requests.get(
            f"{YOTO_API_URL}/content/mine",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=8,
        )
        if r.ok:
            return access_token, client_id
    except Exception:
        pass
    try:
        tokens = yoto_refresh_access_token(client_id, refresh_token)
        new_access  = tokens["access_token"]
        new_refresh = tokens.get("refresh_token", refresh_token)
        yoto_save_credentials(client_id, new_access, new_refresh)
        return new_access, client_id
    except Exception:
        return None, client_id


# ─────────────────────────────────────────────────────────────────────────────
# Book cover search (Open Library — no API key required)
# ─────────────────────────────────────────────────────────────────────────────

def find_book_cover_url(title: str, author: str = "") -> str | None:
    try:
        params: dict = {"title": title, "limit": 3, "fields": "cover_i,title"}
        if author:
            params["author"] = author
        r = requests.get(
            "https://openlibrary.org/search.json", params=params, timeout=8
        )
        for doc in r.json().get("docs", []):
            if doc.get("cover_i"):
                return f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-L.jpg"
    except Exception:
        pass
    return None


def yoto_upload_cover_from_url(access_token: str, image_url: str) -> str | None:
    """
    Tell YOTO to fetch an image from `image_url` and store it as a cover.
    Returns the mediaUrl (full CDN URL) to use in metadata.cover.imageL,
    or None on failure.
    """
    resp = requests.post(
        f"{YOTO_API_URL}/media/coverImage/user/me/upload",
        params={
            "imageUrl":  image_url,
            "coverType": "myo",
        },
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    if resp.ok:
        return resp.json().get("coverImage", {}).get("mediaUrl")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# YOTO icon library + per-chapter icon matching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def yoto_fetch_public_icons(_access_token: str) -> list[dict]:
    resp = requests.get(
        f"{YOTO_API_URL}/media/displayIcons/user/yoto",
        headers={"Authorization": f"Bearer {_access_token}"},
        timeout=15,
    )
    if resp.ok:
        return resp.json().get("displayIcons", [])
    return []


_ICON_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "to", "is", "it", "for",
    "on", "with", "as", "at", "by", "from", "chapter", "part", "one",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}


def yoto_match_icon(chapter_title: str, icons: list[dict]) -> str:
    words = [
        w for w in re.findall(r"[a-z]+", chapter_title.lower())
        if w not in _ICON_STOPWORDS and len(w) > 2
    ]
    if not words:
        return YOTO_ICON

    best_score, best_ref = 0, YOTO_ICON
    for icon in icons:
        tags  = " ".join(icon.get("publicTags", [])).lower()
        title = icon.get("title", "").lower()
        score = sum(1 for w in words if w in tags or w in title)
        if score > best_score:
            best_score = score
            best_ref   = f"yoto:#{icon['mediaId']}"

    return best_ref


# ─────────────────────────────────────────────────────────────────────────────
# YOTO playlist management (fetch / update existing cards)
# ─────────────────────────────────────────────────────────────────────────────

def yoto_get_my_content(access_token: str) -> list[dict]:
    resp = requests.get(
        f"{YOTO_API_URL}/content/mine",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("cards", [])


def yoto_get_card_details(access_token: str, card_id: str) -> dict:
    resp = requests.get(
        f"{YOTO_API_URL}/content/{card_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("card", {})


def yoto_update_card_chapters(
    access_token: str, card_id: str, chapters: list[dict]
) -> dict:
    existing   = yoto_get_card_details(access_token, card_id)
    ex_content = dict(existing.get("content", {}))
    ex_content["chapters"] = chapters

    resp = requests.post(
        f"{YOTO_API_URL}/content",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
        },
        json={"cardId": card_id, "content": ex_content},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("card", {})


def yoto_update_card_cover(
    access_token: str, card_id: str, cover_media_url: str
) -> dict:
    existing   = yoto_get_card_details(access_token, card_id)
    ex_content = dict(existing.get("content", {}))
    ex_content["cover"] = {"imageL": cover_media_url}

    resp = requests.post(
        f"{YOTO_API_URL}/content",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
        },
        json={
            "cardId":   card_id,
            "content":  ex_content,
            "metadata": {"cover": {"imageL": cover_media_url}},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("card", {})


def yoto_upload_card(
    access_token: str,
    card_title: str,
    chapter_files: list[tuple[str, bytes]],
    book_title: str = "",
    book_author: str = "",
    progress_callback=None,
) -> dict:
    """
    Upload chapter MP3s to YOTO and create a MYO card/playlist.
    """
    auth_headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept":        "application/json",
    }

    chapters = []
    total    = len(chapter_files)

    for i, (filename, audio_bytes) in enumerate(chapter_files):
        up_resp = requests.get(
            f"{YOTO_API_URL}/media/transcode/audio/uploadUrl",
            headers=auth_headers,
            timeout=20,
        )
        up_resp.raise_for_status()
        upload_info = up_resp.json()["upload"]
        upload_url  = upload_info["uploadUrl"]
        upload_id   = upload_info["uploadId"]

        put_resp = requests.put(
            upload_url,
            data=audio_bytes,
            headers={"Content-Type": "audio/mpeg"},
            timeout=300,
        )
        put_resp.raise_for_status()

        transcoded = None
        for _ in range(120):
            tc_resp = requests.get(
                f"{YOTO_API_URL}/media/upload/{upload_id}/transcoded?loudnorm=false",
                headers=auth_headers,
                timeout=15,
            )
            if tc_resp.ok:
                data = tc_resp.json()
                if data.get("transcode", {}).get("transcodedSha256"):
                    transcoded = data["transcode"]
                    break
            time.sleep(0.5)

        if not transcoded:
            raise RuntimeError(
                f"Transcoding timed out for chapter {i + 1} ({filename}). "
                "Try again — YOTO servers are sometimes slow."
            )

        media_info = transcoded.get("transcodedInfo", {})

        ch_title = os.path.splitext(filename)[0].replace("_", " ").strip()
        parts    = ch_title.split()
        if parts and parts[0].isdigit():
            ch_title = " ".join(parts[1:])

        ch_key = f"{i + 1:02d}"
        chapters.append({
            "key":          ch_key,
            "title":        ch_title,
            "overlayLabel": str(i + 1),
            "display":      {"icon16x16": YOTO_ICON},
            "tracks": [{
                "key":          "01",
                "title":        ch_title,
                "trackUrl":     f"yoto:#{transcoded['transcodedSha256']}",
                "duration":     media_info.get("duration"),
                "fileSize":     media_info.get("fileSize"),
                "channels":     media_info.get("channels"),
                "format":       media_info.get("format", "mp3"),
                "type":         "audio",
                "overlayLabel": str(i + 1),
                "display":      {"icon16x16": YOTO_ICON},
            }],
        })

        if progress_callback:
            progress_callback((i + 1) / total)

    icons = yoto_fetch_public_icons(access_token)
    if icons:
        for ch in chapters:
            icon_ref = yoto_match_icon(ch["title"], icons)
            ch["display"]["icon16x16"] = icon_ref
            ch["tracks"][0]["display"]["icon16x16"] = icon_ref

    cover_media_url = None
    search_title    = book_title or card_title
    cover_img_url   = find_book_cover_url(search_title, book_author)
    if cover_img_url:
        cover_media_url = yoto_upload_cover_from_url(access_token, cover_img_url)

    total_duration = sum(
        (ch["tracks"][0].get("duration") or 0) for ch in chapters
    )
    total_size = sum(
        (ch["tracks"][0].get("fileSize") or 0) for ch in chapters
    )

    content_body = {
        "title": card_title,
        "content": {
            "chapters": chapters,
            **({"cover": {"imageL": cover_media_url}} if cover_media_url else {}),
        },
        "metadata": {
            "media": {
                "duration":         total_duration,
                "fileSize":         total_size,
                "readableFileSize": round((total_size / 1024 / 1024) * 10) / 10,
            },
            **({"cover": {"imageL": cover_media_url}} if cover_media_url else {}),
        },
    }

    create_resp = requests.post(
        f"{YOTO_API_URL}/content",
        headers={**auth_headers, "Content-Type": "application/json"},
        json=content_body,
        timeout=30,
    )
    create_resp.raise_for_status()
    card = create_resp.json().get("card", {})

    card_id = card.get("cardId", "")
    return {
        "card_id":        card_id,
        "card_url":       f"https://yotoplay.com/my-library/{card_id}" if card_id else None,
        "cover_found":    cover_img_url is not None,
        "cover_uploaded": cover_media_url is not None,
        "icons_matched":  icons != [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Sidebar — YOTO Settings ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🎴 YOTO Settings")

    if "_yoto_access_token" not in st.session_state:
        _saved_token, _saved_cid = yoto_get_valid_token()
        st.session_state["_yoto_access_token"] = _saved_token or ""
        st.session_state["_yoto_client_id"]    = _saved_cid  or ""

    _token = st.session_state.get("_yoto_access_token", "")

    if _token:
        st.success("✅ Logged in to YOTO")
        if st.button("Sign out", key="sb_signout"):
            st.session_state["_yoto_access_token"] = ""
            st.session_state["_yoto_client_id"]    = ""
            st.session_state.pop("_yoto_pkce", None)
            if os.path.exists(YOTO_TOKEN_FILE):
                os.remove(YOTO_TOKEN_FILE)
            st.experimental_rerun()
    else:
        st.caption(
            "Log in once — credentials are saved and auto-refreshed. "
            "Requires a [YOTO Developer](https://yoto.dev) Public Client app "
            "with `http://localhost:8501/` as an Allowed Callback URL."
        )

        _client_id = st.text_input(
            "Client ID",
            value=st.session_state.get("_yoto_client_id", ""),
            key="sb_client_id",
            placeholder="paste-your-client-id",
        )
        _redirect_uri = "http://localhost:8501/"
        _pkce = st.session_state.get("_yoto_pkce")

        if not _pkce:
            if st.button("🔑 Log in with YOTO", disabled=not _client_id, key="sb_login"):
                _verifier, _challenge = yoto_generate_pkce()
                _auth_url = yoto_build_auth_url(_client_id, _redirect_uri, _challenge)
                st.session_state["_yoto_pkce"] = {
                    "verifier":  _verifier,
                    "client_id": _client_id,
                    "auth_url":  _auth_url,
                }
                st.experimental_rerun()
        else:
            _auth_url = _pkce["auth_url"]
            st.markdown(f"**Step 1** — [Click here to log in with YOTO]({_auth_url})")
            st.caption(
                "After approving, copy the full URL from the address bar "
                "(even if it shows an error page) and paste it below."
            )
            _pasted = st.text_input(
                "Paste the redirect URL here",
                key="sb_pasted_url",
                placeholder="http://localhost:8501/?code=...",
            )
            col_ok, col_cancel = st.columns([3, 1])
            with col_ok:
                if st.button("✅ Complete login", disabled=not _pasted, key="sb_complete"):
                    try:
                        _parsed = urllib.parse.urlparse(_pasted)
                        _code   = urllib.parse.parse_qs(_parsed.query).get("code", [None])[0]
                        if not _code:
                            st.error("No `code` in URL — copy the full address bar URL.")
                        else:
                            _tokens = yoto_exchange_code(
                                _pkce["client_id"], _code,
                                _pkce["verifier"],  _redirect_uri,
                            )
                            _at = _tokens["access_token"]
                            _rt = _tokens.get("refresh_token", "")
                            yoto_save_credentials(_pkce["client_id"], _at, _rt)
                            st.session_state["_yoto_access_token"] = _at
                            st.session_state["_yoto_client_id"]    = _pkce["client_id"]
                            st.session_state.pop("_yoto_pkce", None)
                            st.experimental_rerun()
                    except Exception as _e:
                        st.error(f"Login failed: {_e}")
            with col_cancel:
                if st.button("Cancel", key="sb_cancel"):
                    st.session_state.pop("_yoto_pkce", None)
                    st.experimental_rerun()

    st.divider()

    # Tool status in sidebar (compact)
    with st.expander("🔧 Tool status"):
        st.caption(f"ffmpeg:  {'✅' if FFMPEG  else '❌ missing'}")
        st.caption(f"ffprobe: {'✅' if FFPROBE else '❌ missing'}")
        st.caption(f"mutagen: {'✅' if MUTAGEN_OK else '❌ missing'}")
        st.caption(f"whisper: {'✅' if WHISPER_OK else '❌ missing'}")

    st.caption("v1.1 · Libby Audio Extractor")


# ── Main header ───────────────────────────────────────────────────────────────
st.title("🎧 Libby Audio Extractor")
st.caption(
    "Upload a HAR file captured while playing an audiobook in Libby — "
    "combine all audio parts into one MP3, detect chapters, and send to your YOTO player."
)

with st.expander("📖 How to capture a HAR file from Libby"):
    st.markdown("""
1. Open **Chrome** or **Edge** → [libbyapp.com](https://libbyapp.com)
2. Open **DevTools** → **Network** tab (`F12` / `Cmd+Option+I`)
3. Tick **Preserve log**
4. Start playing your audiobook
5. Wait until a few parts load (`Part01.mp3`, `Part02.mp3`, …)
6. Right-click in the Network tab → **Save all as HAR with content**
7. Upload the `.har` file below
    """)

st.divider()


# ── Step 1 — Upload ───────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload your HAR file", type=["har"])

# When a NEW file is uploaded, clear any previous results
if uploaded:
    if st.session_state.get("_libby_har_name") != uploaded.name:
        for _k in [k for k in st.session_state if k.startswith("_libby_")]:
            del st.session_state[_k]
        st.session_state["_libby_har_name"] = uploaded.name
        # Parse and cache immediately so reruns don't re-read the file object
        try:
            _raw = uploaded.read()
            _ai  = parse_har(_raw)
            st.session_state["_libby_har_bytes"]  = _raw
            st.session_state["_libby_audio_info"] = _ai
        except Exception as _ex:
            st.error(f"Failed to parse HAR file: {_ex}")
            st.stop()

if uploaded:
    audio_info = st.session_state.get("_libby_audio_info")
    har_bytes  = st.session_state.get("_libby_har_bytes", b"")

    if not audio_info or not audio_info["parts"]:
        st.warning(
            "No audio parts found in this HAR file. Make sure you captured "
            "network traffic while the audiobook was actively playing in Libby."
        )
        st.stop()

    title     = audio_info["title"]  or "Unknown Title"
    author    = audio_info["author"] or "Unknown Author"
    num_parts = len(audio_info["parts"])
    total_mb  = sum(p["size"] for p in audio_info["parts"]) / (1024 * 1024)

    col1, col2, col3 = st.columns(3)
    col1.metric("📖 Title",  title)
    col2.metric("✍️ Author", author)
    col3.metric("🗂 Parts",  f"{num_parts}  ({total_mb:.0f} MB)")

    with st.expander("View audio parts"):
        for p in audio_info["parts"]:
            st.write(f"Part {p['part_num']:02d} — {p['size']/(1024*1024):.1f} MB")

    st.divider()

    # ── Step 2 — Process ─────────────────────────────────────────────────────
    _already_done = bool(st.session_state.get("_libby_mp3_bytes"))

    _btn_label = "🔄  Re-process" if _already_done else "⬇️  Download & Combine"
    if _already_done:
        st.caption("✅ Already processed — downloads and chapter info are below.")

    _process_clicked = st.button(_btn_label, use_container_width=True, key="process_btn")

    if _process_clicked:
        # Clear previous results before re-processing
        for _k in [k for k in st.session_state if k.startswith("_libby_result_")]:
            del st.session_state[_k]

        _dl_session      = build_download_session(audio_info)
        _tmp_dir         = tempfile.mkdtemp(prefix="libby_")
        _downloaded      = []
        _errors          = []

        _overall_bar = st.progress(0.0, text="Starting downloads…")
        _status      = st.empty()

        for idx, part in enumerate(audio_info["parts"]):
            pnum = part["part_num"]
            _status.write(f"Downloading Part {pnum:02d} of {num_parts}…")
            _pbar = st.progress(0.0)

            def _cb(frac, _b=_pbar):
                _b.progress(min(frac, 1.0))

            try:
                fp = download_part(_dl_session, part, _tmp_dir, progress_callback=_cb)
                _downloaded.append(fp)
                _pbar.progress(1.0)
            except RuntimeError as e:
                _errors.append(str(e))
                _pbar.empty()
                st.warning(f"⚠️ Part {pnum}: {e}")

            _overall_bar.progress(
                (idx + 1) / num_parts,
                text=f"Downloaded {idx + 1} / {num_parts} parts",
            )

        if not _downloaded:
            st.error(
                "No parts could be downloaded. The CDN URLs in your HAR may have "
                "expired — re-capture a fresh HAR while the audiobook is playing."
            )
            st.stop()

        _status.write("Scanning for embedded chapter tags…")
        _id3_ch = detect_id3_chapters(_downloaded)

        _status.write("Querying OverDrive openbook API…")
        _api_ch, _api_status = fetch_openbook_chapters(har_bytes)

        _status.write("Combining parts (may take a moment for large books)…")
        _safe_title  = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        _safe_author = re.sub(r"[^\w\s-]", "", author).strip().replace(" ", "_")
        _out_path    = os.path.join(_tmp_dir, f"{_safe_author}-{_safe_title}.mp3")

        try:
            _method = combine_parts(_downloaded, _out_path)
        except RuntimeError as e:
            st.error(f"Concatenation failed: {e}")
            st.stop()

        _status.write("Detecting silences…")
        _silences       = []
        _whisper_res    = []

        if FFMPEG:
            _silences = detect_silences(_out_path, noise_db=-40, min_dur_s=2.0)

        if WHISPER_OK and _silences:
            _status.write(
                f"Running Whisper on {len(_silences)} silence break(s)… "
                "(downloads ~75 MB model on first run)"
            )
            _wbar = st.progress(0.0, text="Transcribing chapter boundaries…")
            _whisper_res = confirm_chapters_whisper(
                _out_path, _silences,
                progress_callback=lambda f: _wbar.progress(f),
            )
            _wbar.empty()

        _confirmed = [r for r in _whisper_res if r.get("has_chapter_word")]
        _ch_files  = []
        _zip_bytes = None

        if _confirmed and FFMPEG:
            _status.write(f"Splitting into {len(_confirmed)} chapter files…")
            _sbar = st.progress(0.0, text="Splitting…")
            _ch_files = split_into_chapters(
                _out_path, _confirmed, _safe_title,
                progress_callback=lambda f: _sbar.progress(f),
            )
            _sbar.empty()
            if _ch_files:
                _zip_bytes = build_zip(_ch_files)

        with open(_out_path, "rb") as _f:
            _mp3_bytes = _f.read()

        # Persist everything to session state
        st.session_state["_libby_mp3_bytes"]          = _mp3_bytes
        st.session_state["_libby_chapter_files"]      = _ch_files
        st.session_state["_libby_zip_bytes"]          = _zip_bytes
        st.session_state["_libby_id3_chapters"]       = _id3_ch
        st.session_state["_libby_api_chapters"]       = _api_ch
        st.session_state["_libby_api_status"]         = _api_status
        st.session_state["_libby_silence_candidates"] = _silences
        st.session_state["_libby_whisper_results"]    = _whisper_res
        st.session_state["_libby_safe_title"]         = _safe_title
        st.session_state["_libby_safe_author"]        = _safe_author
        st.session_state["_libby_card_title"]         = title
        st.session_state["_libby_output_filename"]    = f"{_safe_author}-{_safe_title}.mp3"
        st.session_state["_libby_output_mb"]          = len(_mp3_bytes) / (1024 * 1024)
        st.session_state["_libby_method_used"]        = _method
        st.session_state["_libby_errors"]             = _errors

        # Clean up temp files
        for _fp in _downloaded:
            try: os.remove(_fp)
            except OSError: pass
        try:
            os.remove(_out_path)
            os.rmdir(_tmp_dir)
        except OSError:
            pass

        _overall_bar.empty()
        _status.empty()

        # Rerun so results render cleanly outside this button block
        st.experimental_rerun()


# ── Results — always visible once processed (reads session state) ─────────────
_mp3_bytes    = st.session_state.get("_libby_mp3_bytes")
_ch_files     = st.session_state.get("_libby_chapter_files", [])
_zip_bytes    = st.session_state.get("_libby_zip_bytes")
_errors       = st.session_state.get("_libby_errors", [])
_method_used  = st.session_state.get("_libby_method_used", "")
_output_mb    = st.session_state.get("_libby_output_mb", 0.0)
_output_fn    = st.session_state.get("_libby_output_filename", "audiobook.mp3")
_safe_title   = st.session_state.get("_libby_safe_title", "")
_safe_author  = st.session_state.get("_libby_safe_author", "")
_card_title   = st.session_state.get("_libby_card_title", "My Audiobook")

if _mp3_bytes:
    st.success(f"✅ **{_card_title}** — {_output_mb:.1f} MB · {_method_used}")

    if _errors:
        with st.expander(f"⚠️ {len(_errors)} part(s) had download errors"):
            for _e in _errors:
                st.write(_e)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("⬇️ Download")
    _dl1, _dl2 = st.columns(2)

    with _dl1:
        st.download_button(
            label=f"🎵  Combined MP3  ({_output_mb:.0f} MB)",
            data=_mp3_bytes,
            file_name=_output_fn,
            mime="audio/mpeg",
            use_container_width=True,
            key="dl_mp3",
        )
        st.caption("Single file — all chapters in sequence.")

    with _dl2:
        if _zip_bytes:
            _zip_mb = len(_zip_bytes) / (1024 * 1024)
            st.download_button(
                label=f"✂️  Chapter Files ZIP  ({len(_ch_files)} files, {_zip_mb:.0f} MB)",
                data=_zip_bytes,
                file_name=f"{_safe_author}-{_safe_title}-chapters.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip",
            )
            st.caption("One MP3 per chapter, named by title.")
        else:
            st.button(
                "✂️  Chapter Files ZIP  (unavailable)",
                disabled=True,
                use_container_width=True,
                key="dl_zip_disabled",
            )
            st.caption(
                "Requires **faster-whisper** to confirm chapter boundaries. "
                "`pip install faster-whisper`, restart and re-process."
            )

    # ── Chapter Detection ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Chapter Detection")

    _id3_ch     = st.session_state.get("_libby_id3_chapters", [])
    _api_ch     = st.session_state.get("_libby_api_chapters", [])
    _api_status = st.session_state.get("_libby_api_status", "")
    _silences   = st.session_state.get("_libby_silence_candidates", [])
    _wres       = st.session_state.get("_libby_whisper_results", [])

    _confirmed_count  = sum(1 for r in _wres if r.get("has_chapter_word"))

    _tab1, _tab2, _tab3 = st.tabs([
        f"ID3 Tags {'✅ ' + str(len(_id3_ch)) if _id3_ch else '—'}",
        f"OverDrive API {'✅ ' + str(len(_api_ch)) if _api_ch else '—'}",
        f"Audio Analysis {'✅ ' + str(_confirmed_count) + ' confirmed' if _confirmed_count else ('⚠️ ' + str(len(_silences)) + ' breaks' if _silences else '—')}",
    ])

    with _tab1:
        if not FFPROBE and not MUTAGEN_OK:
            st.warning(
                "Neither ffprobe nor mutagen is available. "
                "Install ffmpeg (`brew install ffmpeg`) or mutagen (`pip install mutagen`)."
            )
        elif _id3_ch:
            st.dataframe(
                [{"#": i+1, "Title": c["Title"], "Start": c["Start"], "End": c["End"], "File": c["Part file"]}
                 for i, c in enumerate(_id3_ch)],
                use_container_width=True,
            )
        else:
            st.info(
                "No ID3 chapter frames found. Not all publishers embed tags — "
                "check the OverDrive API or Audio Analysis tabs."
            )

    with _tab2:
        if _api_ch:
            st.dataframe(
                [{k: v for k, v in c.items() if not k.startswith("_")} for c in _api_ch],
                use_container_width=True,
            )
        else:
            st.info(_api_status or "No chapter data from the OverDrive API.")

    with _tab3:
        if not FFMPEG:
            st.warning("ffmpeg is required for audio analysis. `brew install ffmpeg`")
        elif not _silences:
            st.info("No silences of 2+ seconds found in the combined audio.")
        else:
            if not WHISPER_OK:
                st.info(
                    "💡 Install **faster-whisper** to confirm true chapter boundaries "
                    "via speech recognition:\n\n"
                    "```\npip install faster-whisper\n```\n\nThen restart and re-process."
                )

            _rows = []
            _src  = _wres if _wres else _silences
            for i, r in enumerate(_src):
                row = {
                    "#":             i + 1,
                    "Chapter start": _fmt_s(r["chapter_start_s"]),
                    "Silence dur":   f"{r['silence_dur_s']:.1f}s",
                }
                if _wres:
                    row["'Chapter' spoken"] = "✅ Yes" if r.get("has_chapter_word") else "—"
                    row["Opening words"]    = (r.get("transcript") or "")[:80]
                _rows.append(row)

            st.dataframe(_rows, use_container_width=True)

            if _wres:
                if _confirmed_count:
                    st.success(
                        f"✅ **{_confirmed_count} chapter(s) confirmed** by Whisper. "
                        f"{len(_silences) - _confirmed_count} other silence(s) are likely section pauses."
                    )
                else:
                    st.warning(
                        "Whisper didn't detect the word 'chapter' after any silence. "
                        "Check the 'Opening words' column for clues."
                    )

    # ── Send to YOTO ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎴 Send to YOTO")

    _access_token = st.session_state.get("_yoto_access_token", "")

    if not _ch_files:
        st.info(
            "Chapter-split files aren't available yet. "
            "YOTO upload works best with individual chapter tracks — "
            "install **faster-whisper** and re-process to enable this.",
            icon="ℹ️",
        )
    elif not _access_token:
        st.warning("Log in to YOTO first using the sidebar →", icon="👈")
    else:
        st.caption(f"{len(_ch_files)} chapter track(s) ready to upload.")

        _yoto_title = st.text_input(
            "Card title",
            value=_card_title,
            key="yoto_card_title_input",
            help="This will be the name of the playlist on your YOTO player.",
        )
        st.caption(
            "Each chapter becomes a separate track. "
            "Tap any physical MYO card on your YOTO player to link it to this playlist."
        )

        if st.button("🚀  Create YOTO Card", use_container_width=True, key="yoto_create"):
            _yoto_bar    = st.progress(0.0, text="Starting upload…")
            _yoto_status = st.empty()
            _total_ch    = len(_ch_files)

            def _yoto_prog(frac):
                _done = int(frac * _total_ch)
                _yoto_bar.progress(frac, text=f"Uploading chapter {_done}/{_total_ch}…")

            try:
                _result = yoto_upload_card(
                    access_token      = _access_token,
                    card_title        = _yoto_title,
                    chapter_files     = _ch_files,
                    book_title        = _card_title,
                    book_author       = _safe_author.replace("_", " "),
                    progress_callback = _yoto_prog,
                )
                _yoto_bar.progress(1.0, text="Done!")
                st.success(f"✅ Card created!  ID: `{_result.get('card_id')}`")
                if _result.get("card_url"):
                    st.markdown(f"[Open in your YOTO library]({_result['card_url']})")
                _notes = []
                if _result.get("cover_uploaded"):
                    _notes.append("📖 Cover uploaded")
                elif _result.get("cover_found"):
                    _notes.append("📖 Cover found but upload failed")
                else:
                    _notes.append("📖 No cover found on Open Library")
                if _result.get("icons_matched"):
                    _notes.append("🎨 Chapter icons matched")
                _yoto_status.info("  ·  ".join(_notes))
            except Exception as _ex:
                _yoto_bar.empty()
                st.error(f"YOTO upload failed: {_ex}")


# ── My YOTO Playlists ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📚 My YOTO Playlists")

_tok = st.session_state.get("_yoto_access_token", "")

if not _tok:
    st.info("Log in to YOTO via the sidebar to manage your playlists.", icon="👈")
else:
    _col_load, _ = st.columns([1, 3])
    with _col_load:
        _load_btn = st.button("🔄 Load my playlists", key="load_playlists")

    if _load_btn or st.session_state.get("_yoto_cards"):
        if _load_btn:
            with st.spinner("Fetching your YOTO library…"):
                try:
                    st.session_state["_yoto_cards"] = yoto_get_my_content(_tok)
                except Exception as _ex:
                    st.error(f"Failed to load playlists: {_ex}")

        _cards = st.session_state.get("_yoto_cards", [])
        if not _cards:
            st.info("No MYO cards found in your library.")
        else:
            _card_opts    = {c["title"]: c["cardId"] for c in _cards}
            _sel_title    = st.selectbox("Select a playlist", list(_card_opts.keys()), key="playlist_select")
            _sel_id       = _card_opts[_sel_title]

            _col_det, _col_cov = st.columns([2, 1])
            with _col_det:
                _load_detail = st.button("📋 Load chapters", key="load_chapters")
            with _col_cov:
                _cover_btn = st.button("🖼️ Find & upload cover", key="find_cover")

            if _cover_btn:
                with st.spinner(f"Searching for cover: '{_sel_title}'…"):
                    _src_url = find_book_cover_url(_sel_title)
                    if _src_url:
                        _cdn_url = yoto_upload_cover_from_url(_tok, _src_url)
                        if _cdn_url:
                            yoto_update_card_cover(_tok, _sel_id, _cdn_url)
                            st.success("✅ Cover uploaded and applied!")
                            st.image(_src_url, width=160, caption="Cover applied")
                        else:
                            st.warning("Cover found but YOTO upload failed.")
                    else:
                        st.warning(f"No cover found on Open Library for '{_sel_title}'.")

            if _load_detail:
                with st.spinner("Loading chapter details…"):
                    try:
                        _det = yoto_get_card_details(_tok, _sel_id)
                        st.session_state["_yoto_card_detail"]    = _det
                        st.session_state["_yoto_card_detail_id"] = _sel_id
                    except Exception as _ex:
                        st.error(f"Failed to load card: {_ex}")

            _detail    = st.session_state.get("_yoto_card_detail")
            _detail_id = st.session_state.get("_yoto_card_detail_id")

            if _detail and _detail_id == _sel_id:
                _chapters = _detail.get("content", {}).get("chapters", [])
                if not _chapters:
                    st.info("No chapters found in this card.")
                else:
                    st.markdown(f"**{len(_chapters)} chapter(s)**")
                    with st.spinner("Loading YOTO icon library…"):
                        _icons = yoto_fetch_public_icons(_tok)
                    _icon_by_id = {i["mediaId"]: i.get("title", i["mediaId"][:12] + "…") for i in _icons}

                    _rows = []
                    for _ch in _chapters:
                        _cur_ref  = _ch.get("display", {}).get("icon16x16", "")
                        _cur_mid  = _cur_ref.replace("yoto:#", "") if _cur_ref else ""
                        _cur_name = _icon_by_id.get(_cur_mid, "default" if not _cur_mid else _cur_mid[:12] + "…")
                        _sug_ref  = yoto_match_icon(_ch.get("title", ""), _icons)
                        _sug_mid  = _sug_ref.replace("yoto:#", "")
                        _sug_name = _icon_by_id.get(_sug_mid, "default")
                        _rows.append({
                            "Chapter":        _ch.get("title", "—"),
                            "Current icon":   _cur_name,
                            "Suggested icon": _sug_name,
                            "Match?":         "✅" if _cur_ref == _sug_ref else "🔄",
                        })

                    st.dataframe(_rows, use_container_width=True)

                    if st.button(
                        "🎨 Apply suggested icons to all chapters",
                        key="apply_icons",
                        use_container_width=True,
                    ):
                        with st.spinner("Updating icons…"):
                            try:
                                _upd = []
                                for _ch in _chapters:
                                    _ch = dict(_ch)
                                    _iref = yoto_match_icon(_ch.get("title", ""), _icons)
                                    _ch.setdefault("display", {})["icon16x16"] = _iref
                                    for _tr in _ch.get("tracks", []):
                                        _tr.setdefault("display", {})["icon16x16"] = _iref
                                    _upd.append(_ch)
                                yoto_update_card_chapters(_tok, _sel_id, _upd)
                                st.session_state["_yoto_card_detail"] = yoto_get_card_details(_tok, _sel_id)
                                st.success("✅ Icons updated! Refresh the YOTO app to see changes.")
                            except Exception as _ex:
                                st.error(f"Update failed: {_ex}")

st.divider()
st.caption(
    "HAR file URLs expire after ~48 hours. "
    "If downloads fail, re-capture a fresh HAR while the audiobook is actively playing."
)
