from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any
from urllib import parse


def parse_front_note_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", "hide", ""}


def sanitize_front_note_html(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = re.sub(r"(?is)<\s*(script|style|iframe|object|embed|meta|link)\b.*?<\s*/\s*\1\s*>", "", text)
    text = re.sub(r"(?is)<\s*(script|style|iframe|object|embed|meta|link)\b[^>]*>", "", text)
    text = re.sub(r"\s+on[a-zA-Z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", text)
    text = re.sub(r"(?i)(href|src)\s*=\s*(['\"])\s*javascript:[^'\"]*\2", r"\1=\"#\"", text)
    allowed = {
        "a", "audio", "b", "blockquote", "br", "code", "div", "em", "figcaption", "figure", "h1", "h2", "h3",
        "hr", "i", "img", "li", "ol", "p", "pre", "small", "span", "strong", "u", "ul",
    }
    allowed_attrs = {"href", "src", "alt", "title", "class", "controls", "target", "rel"}

    def clean_tag(match: re.Match[str]) -> str:
        slash = match.group(1) or ""
        tag = match.group(2).lower()
        attrs = match.group(3) or ""
        if tag not in allowed:
            return ""
        if slash:
            return f"</{tag}>"
        cleaned_attrs: list[str] = []
        for attr_match in re.finditer(r"([a-zA-Z:-]+)(?:\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s\"'>]+))?", attrs):
            name = attr_match.group(1).lower()
            raw_value = attr_match.group(2)
            if name not in allowed_attrs:
                continue
            if raw_value is None:
                if name == "controls":
                    cleaned_attrs.append("controls")
                continue
            attr_value = raw_value.strip().strip("\"'")
            if name in {"href", "src"} and re.match(r"(?i)javascript:", attr_value):
                continue
            cleaned_attrs.append(f'{name}="{html.escape(attr_value, quote=True)}"')
        if tag == "a":
            if not any(item.startswith("target=") for item in cleaned_attrs):
                cleaned_attrs.append('target="_blank"')
            if not any(item.startswith("rel=") for item in cleaned_attrs):
                cleaned_attrs.append('rel="noreferrer"')
        if tag == "audio" and not any(item == "controls" for item in cleaned_attrs):
            cleaned_attrs.append("controls")
        suffix = " " + " ".join(cleaned_attrs) if cleaned_attrs else ""
        return f"<{tag}{suffix}>"

    return re.sub(r"<\s*(/)?\s*([a-zA-Z0-9]+)([^>]*)>", clean_tag, text)


def front_note_html_to_text(html_text: str) -> str:
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", str(html_text or ""))
    text = re.sub(r"(?i)</\s*(p|div|h1|h2|h3|li|blockquote)\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_front_note_media(media: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if not isinstance(media, list):
        return items
    for raw in media[:8]:
        if isinstance(raw, str):
            raw = {"type": "link", "url": raw, "title": raw}
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or raw.get("kind") or "").strip().lower()
        url = str(raw.get("url") or raw.get("src") or raw.get("path") or "").strip()
        title = str(raw.get("title") or raw.get("label") or url).strip()[:200]
        caption = str(raw.get("caption") or raw.get("description") or "").strip()[:500]
        if not url:
            continue
        if not kind:
            lower = url.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")):
                kind = "image"
            elif lower.endswith((".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac")):
                kind = "audio"
            else:
                kind = "link" if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) else "file"
        if kind not in {"image", "audio", "file", "link"}:
            kind = "file"
        if kind in {"image", "audio", "file"}:
            parsed = parse.urlparse(url)
            if not parsed.scheme:
                try:
                    path = Path(url).expanduser().resolve()
                    if path.exists():
                        url = path.as_uri()
                    else:
                        kind = "link"
                except Exception:
                    kind = "link"
        items.append({"type": kind, "url": url[:2000], "title": title, "caption": caption})
    return items


def front_note_markdown_to_html(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "<p class='empty'>空便签</p>"
    lines = text.splitlines()
    out: list[str] = []
    in_list = False
    for line in lines[:120]:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{html.escape(stripped[2:].strip())}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if stripped.startswith("#"):
            level = min(3, len(stripped) - len(stripped.lstrip("#")))
            value = stripped[level:].strip()
            out.append(f"<h{level}>{html.escape(value)}</h{level}>")
        else:
            value = html.escape(stripped)
            value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
            out.append(f"<p>{value}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def front_note_content_to_html(content: str) -> str:
    content = str(content or "")
    if not content:
        return ""
    if re.search(r"<\s*[a-zA-Z][^>]*>", content):
        return sanitize_front_note_html(content)
    return sanitize_front_note_html(front_note_markdown_to_html(content))


def render_front_note_media_cards(media: Any) -> str:
    media_html: list[str] = []
    for item in sanitize_front_note_media(media or []):
        title = html.escape(item.get("title") or item.get("url") or "")
        caption = html.escape(item.get("caption") or "")
        url = html.escape(item.get("url") or "")
        kind = item.get("type")
        if kind == "image":
            media_html.append(f"<figure><img src='{url}' alt='{title}'><figcaption>{title}{('<br>' + caption) if caption else ''}</figcaption></figure>")
        elif kind == "audio":
            media_html.append(f"<div class='attachment'><strong>{title}</strong><audio controls src='{url}'></audio><small>{caption or url}</small></div>")
        elif kind == "file":
            media_html.append(f"<a class='link-card' href='{url}'><span>{title}</span><small>{caption or url}</small></a>")
        else:
            subtitle = caption or html.escape(item.get("url") or "")
            media_html.append(f"<a class='link-card' href='{url}'><span>{title}</span><small>{subtitle}</small></a>")
    return "".join(media_html)

