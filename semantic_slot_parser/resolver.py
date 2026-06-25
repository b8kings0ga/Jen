#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from typing import Any

ACTION_HINTS = [
    "天气怎么样",
    "温度怎么样",
    "现在几度",
    "几度",
    "会不会下雨",
    "會不會下雨",
    "会下雨吗",
    "会下雨么",
    "會下雨嗎",
    "下不下雨",
    "有雨吗",
    "有雨么",
    "有雨嗎",
    "下雨了吗",
    "下雨了嗎",
    "下雨了么",
    "下雨了麼",
    "下雨了吧",
    "下雨嘛",
    "下雨吗",
    "下雨嗎",
    "下雨么",
    "下雨麼",
    "要不要带伞",
    "查一下",
    "搜索一下",
    "打开",
    "关闭",
    "播放",
    "提醒我",
    "记一下",
    "排一下窗口",
]

ACTION_NORMALIZE = [
    ("天气", "weather"),
    ("温度", "weather"),
    ("几度", "weather"),
    ("雨吗", "weather"),
    ("雨嗎", "weather"),
    ("雨么", "weather"),
    ("雨麼", "weather"),
    ("下雨", "weather"),
    ("雨", "weather"),
    ("带伞", "weather"),
    ("地图", "map"),
    ("路线", "map"),
    ("怎么走", "map"),
    ("在哪", "map"),
    ("哪里", "map"),
    ("地址", "map"),
    ("提醒", "reminder_create"),
    ("闹钟", "reminder_create"),
    ("日历", "calendar_list"),
    ("行程", "calendar_list"),
    ("便签", "note_live"),
    ("note", "note_live"),
    ("上下文", "note_context"),
    ("context", "note_context"),
    ("记住", "memory"),
    ("记得", "memory"),
    ("时间", "time"),
    ("几点", "time"),
]


def _clean(text: str) -> str:
    return re.sub(r"[\s,，。！？!?、；;：:]+", "", text or "")


def normalize_daily_action(action: str, text: str = "") -> str:
    compact = _clean(action)
    for needle, normalized in ACTION_NORMALIZE:
        if needle.lower() in compact.lower():
            return normalized
    text_compact = _clean(text)
    if compact in {"去", "到", "查", "查一下", "看看", "搜索", "搜索一下", "打开", "关闭", "播放"} or not compact:
        for needle, normalized in ACTION_NORMALIZE:
            if needle.lower() in text_compact.lower():
                return normalized
    return compact


def _target_from_text(text: str, action: str) -> str:
    compact = _clean(text)
    if not compact:
        return ""
    if action == "map":
        value = re.sub(r"^(查一下|查查|看看|搜索|搜|打开地图|地图|去|到)", "", compact, flags=re.IGNORECASE)
        value = re.sub(r"(怎么走|路线|地图|地址|在哪|哪里)$", "", value, flags=re.IGNORECASE)
        return value[:80]
    if action == "time":
        value = re.sub(r"(现在几点|几点|现在时间|当地时间|时间)$", "", compact, flags=re.IGNORECASE)
        value = re.sub(r"^(查一下|看看|查查)", "", value, flags=re.IGNORECASE)
        return value[:80]
    if action == "weather":
        value = re.sub(r"(天气怎么样|天氣怎麼樣|天气|天氣|温度怎么样|溫度怎麼樣|温度|溫度|现在几度|幾度|几度|多少度|会不会下雨|會不會下雨|会下雨吗|会下雨么|會下雨嗎|下不下雨|有雨吗|有雨么|有雨嗎|下雨了吗|下雨了嗎|下雨了么|下雨了麼|下雨了吧|下雨嘛|下雨吗|下雨嗎|下雨么|下雨麼|雨了吗|雨了嗎|雨了么|雨了麼|雨吗|雨嗎|雨么|雨麼|要不要带伞)$", "", compact, flags=re.IGNORECASE)
        value = re.sub(r"^(查一下|看看|查查)", "", value, flags=re.IGNORECASE)
        for token in ("今天", "明天", "后天", "现在", "当前", "当地"):
            value = value.replace(token, "")
        if any(token in value for token in ("不是", "不对", "说错了", "改成", "等一下")):
            chunks = re.split(r"(?:不是不是|不是|不对|说错了|改成|等一下|是)", value)
            value = chunks[-1] if chunks else value
        return value[:80]
    return ""


def _target_needs_repair(target: str, action: str) -> bool:
    if action not in {"map", "time", "weather"}:
        return False
    if not target:
        return True
    if len(target) <= 1:
        return True
    if re.fullmatch(r"[A-Za-z]{1,2}", target):
        return True
    return False


def _merge_adjacent_place_spans(spans: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for span in spans:
        typ = str(span.get("type") or "")
        text = str(span.get("text") or "")
        if typ in {"TARGET", "LOCATION"} and merged and merged[-1].get("type") in {"TARGET", "LOCATION"}:
            previous_type = str(merged[-1].get("type") or "")
            merged[-1]["text"] = str(merged[-1].get("text") or "") + text
            if typ == "LOCATION" or previous_type == "LOCATION":
                merged[-1]["type"] = "LOCATION"
            continue
        merged.append(dict(span))
    return merged


def resolve(spans: list[dict[str, str]], text: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {
        "domain": "daily",
        "action": "",
        "target": "",
        "content": "",
        "time": "",
        "location": "",
        "object": "",
        "modifiers": [],
        "cancelled": False,
    }
    after_correction = False
    for span in _merge_adjacent_place_spans(spans):
        typ = span.get("type", "")
        value = _clean(span.get("text", ""))
        if not value:
            continue
        if typ == "CORRECTION":
            after_correction = True
            continue
        if typ == "NEGATION":
            result["cancelled"] = True
            continue
        key = typ.lower()
        if key in {"time", "location", "object", "target", "content", "action"}:
            result[key] = value
            if typ == "LOCATION":
                result["target"] = value
            elif typ == "OBJECT" and not result["content"]:
                result["content"] = value
            if after_correction:
                after_correction = False
        elif typ == "MODIFIER":
            result["modifiers"].append(value)

    if not result["action"] and text:
        compact = _clean(text)
        for hint in ACTION_HINTS:
            if hint in compact:
                result["action"] = hint
                break
        if not result["action"]:
            if "天气" in compact:
                result["action"] = "天气怎么样"
            elif "温度" in compact:
                result["action"] = "温度怎么样"
            elif "下雨" in compact:
                result["action"] = "会不会下雨"
            elif "带伞" in compact:
                result["action"] = "要不要带伞"

    result["action"] = normalize_daily_action(str(result["action"]), text)
    if not result["target"] and result["location"]:
        result["target"] = result["location"]
    if not result["content"] and result["object"]:
        result["content"] = result["object"]
    if text and _target_needs_repair(str(result["target"]), str(result["action"])):
        repaired_target = _target_from_text(text, str(result["action"]))
        if repaired_target:
            result["target"] = repaired_target
    result["daily_action_call"] = {
        "action": result["action"],
        "target": result["target"] or result["content"],
        "args": {
            "time": result["time"],
            "content": result["content"],
            "modifiers": result["modifiers"],
        },
    }

    if result["cancelled"] and not any(result[k] for k in ("time", "location", "object", "target", "content", "action")):
        result["action"] = "cancel"
        result["daily_action_call"]["action"] = "cancel"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spans_json", help="JSON spans list")
    parser.add_argument("--text", default="")
    args = parser.parse_args()
    print(json.dumps(resolve(json.loads(args.spans_json), args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
