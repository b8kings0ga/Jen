from __future__ import annotations

import re
from typing import Any
from urllib import parse

from voice_assistant.json_utils import parse_jsonish_value


def normalized_url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = parse.urlparse(raw)
    host = parsed.netloc.lower().removeprefix("www.")
    raw_path = parsed.path.rstrip("/")
    path = raw_path.lower()
    query = parse.parse_qs(parsed.query)
    if host == "youtu.be":
        video_id = raw_path.strip("/").split("/", 1)[0]
        return f"youtube:{video_id}" if video_id else f"{host}{path}"
    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        video_id = (query.get("v") or [""])[0].strip()
        if video_id:
            return f"youtube:{video_id}"
    normalized_query = parse.urlencode(sorted((key, values[-1]) for key, values in query.items() if values), doseq=False)
    return f"{host}{path}?{normalized_query}" if normalized_query else f"{host}{path}"


def urls_match(left: str, right: str) -> bool:
    left_key = normalized_url_key(left)
    right_key = normalized_url_key(right)
    return bool(left_key and right_key and left_key == right_key)


def url_explicitly_in_user_text(url: str, user_text: str) -> bool:
    text = str(user_text or "")
    if not text:
        return False
    raw = str(url or "").strip()
    if raw and raw in text:
        return True
    parsed = parse.urlparse(raw)
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path.rstrip("/")
    if host and f"{host}{path}" in text:
        return True
    return bool(host and host in text and path and path in text)


def extract_urls_from_value(value: Any) -> set[str]:
    payload = parse_jsonish_value(value)
    urls: set[str] = set()
    if isinstance(payload, dict):
        for key in ("url", "href", "link", "canonical_url"):
            candidate = str(payload.get(key) or "").strip()
            if candidate.startswith(("http://", "https://")):
                urls.add(candidate)
        for nested in payload.values():
            if isinstance(nested, (dict, list, tuple, str)):
                urls.update(extract_urls_from_value(nested))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            urls.update(extract_urls_from_value(item))
    elif isinstance(payload, str):
        urls.update(re.findall(r"https?://[^\s\"'<>，,。!！?？、；;]+", payload))
    return urls


def extract_verified_urls_from_tool_result(tool_name: str, arguments: dict[str, Any], result: Any) -> set[str]:
    name = str(tool_name or "").split(":", 1)[-1]
    urls: set[str] = set()
    if name in {"web_search", "search_news", "fetch_url"}:
        urls.update(extract_urls_from_value(result))
    if name == "fetch_url":
        url = str((arguments or {}).get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            urls.add(url)
    return urls


def open_url_verification_error(arguments: dict[str, Any], user_text: str, verified_urls: set[str]) -> dict[str, Any] | None:
    url = str((arguments or {}).get("url") or "").strip()
    if not url.startswith(("http://", "https://")) or not looks_like_video_url(url):
        return None
    if url_explicitly_in_user_text(url, user_text):
        return None
    if any(urls_match(url, verified) for verified in verified_urls):
        return None
    return {
        "ok": False,
        "error": "video URL not verified in current turn",
        "url": url,
        "instruction": "Do not invent video URLs. Search or fetch the requested video/page first, then open a URL returned by the current-turn tool result.",
    }


def _compact_text(text: str) -> str:
    return re.sub(r"[\s，,。.!！?？、；;：:《》<>「」『』\"'“”‘’]+", "", str(text or "").strip())


def user_requested_multiple_video_opens(user_text: str) -> bool:
    compact = _compact_text(user_text).lower()
    if not any(token in compact for token in ["视频", "youtube", "video", "mtv"]):
        return False
    return any(
        token in compact
        for token in [
            "两个视频",
            "2个视频",
            "两个youtube",
            "2个youtube",
            "多个视频",
            "几个视频",
            "都打开",
            "全部打开",
            "分别打开",
            "一起打开",
        ]
    )


def duplicate_video_open_url(user_text: str, url: str, opened_video_urls: set[str]) -> str:
    if user_requested_multiple_video_opens(user_text):
        return ""
    if not opened_video_urls or not looks_like_video_url(str(url or "")):
        return ""
    return sorted(opened_video_urls)[0]


def first_openable_url(urls: list[str] | set[str]) -> str:
    ordered = [str(url or "").strip() for url in urls if str(url or "").strip().startswith(("http://", "https://"))]
    for url in ordered:
        if looks_like_video_url(url):
            return url
    return ordered[0] if ordered else ""


def looks_like_video_url(url: str) -> bool:
    parsed = parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if any(domain in host for domain in ["youtube.com", "youtu.be", "bilibili.com", "vimeo.com", "twitch.tv"]):
        return True
    return any(token in path for token in ["/video", "/watch", "/live"])


def video_search_retry_query(query: str, results: list[dict[str, Any]]) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    lower = text.lower()
    intent_tokens = [
        "youtube",
        "youtu.be",
        "视频",
        "影片",
        "播放",
        "音乐",
        "歌曲",
        "mv",
        "mtv",
        "music video",
        "official video",
        "full video",
    ]
    if not any(token in lower for token in intent_tokens):
        return ""
    for row in results or []:
        if isinstance(row, dict) and looks_like_video_url(str(row.get("url") or "")):
            return ""
    cleaned = re.sub(r"\b(site:youtube\.com/watch|site:youtu\.be)\b", "", text, flags=re.IGNORECASE).strip()
    return f"site:youtube.com/watch {cleaned}"
