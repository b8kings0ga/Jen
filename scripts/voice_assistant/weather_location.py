from __future__ import annotations

import re


_ENGLISH_NON_LOCATION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "current",
    "currently",
    "for",
    "forecast",
    "how",
    "in",
    "is",
    "it",
    "like",
    "near",
    "now",
    "s",
    "the",
    "today",
    "tomorrow",
    "weather",
    "what",
    "whats",
}


def plausible_weather_location(location: str) -> bool:
    value = re.sub(r"\s+", " ", str(location or "")).strip(" ：:，,。.!！?？")
    if not value:
        return False
    compact = re.sub(r"[\s,，。！？!?、；;：:'\"`]+", "", value).lower()
    if not compact:
        return False
    if compact in {"这边", "这里", "当地", "本地", "当前位置", "我这边"}:
        return False
    if re.search(r"[\u4e00-\u9fff]", value):
        return True
    if "," in value or "，" in value or re.search(r"\d", value):
        return True
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", value.lower())
    if not tokens:
        return False
    meaningful = [token for token in tokens if token not in _ENGLISH_NON_LOCATION_WORDS]
    if not meaningful:
        return False
    return any(len(token) >= 3 for token in meaningful)
