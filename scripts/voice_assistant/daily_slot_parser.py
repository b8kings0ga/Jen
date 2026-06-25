from __future__ import annotations

import importlib
import re
import sys
import threading
from pathlib import Path
from typing import Any

from voice_assistant.voice_text import normalize_chinese_asr_variants


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"loaded": False, "available": None, "error": ""}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _semantic_dir() -> Path:
    return _repo_root() / "semantic_slot_parser"


def _artifact_dir() -> Path:
    return _semantic_dir() / "artifacts"


def available() -> bool:
    return bool(_artifact_dir().joinpath("model_mlx.npz").exists())


def _load() -> dict[str, Any]:
    with _LOCK:
        if _STATE.get("loaded"):
            return _STATE
        _STATE["loaded"] = True
        if not available():
            _STATE.update({"available": False, "error": "semantic slot parser artifacts missing"})
            return _STATE
        semantic_path = str(_semantic_dir())
        if semantic_path not in sys.path:
            sys.path.insert(0, semantic_path)
        try:
            infer_mlx = importlib.import_module("infer_mlx")
            model, vocab, id_to_label = infer_mlx.load_model(_artifact_dir(), 64, 64)
            _STATE.update(
                {
                    "available": True,
                    "error": "",
                    "predict": infer_mlx.predict,
                    "model": model,
                    "vocab": vocab,
                    "id_to_label": id_to_label,
                }
            )
        except Exception as exc:
            _STATE.update({"available": False, "error": str(exc)[:240]})
        return _STATE


def parse_daily_slots(text: str) -> dict[str, Any] | None:
    raw = normalize_chinese_asr_variants(text).strip()
    if not raw:
        return None
    state = _load()
    if not state.get("available"):
        return None
    try:
        result = state["predict"](raw, state["model"], state["vocab"], state["id_to_label"])
    except Exception as exc:
        with _LOCK:
            _STATE["error"] = str(exc)[:240]
        return None
    resolved = result.get("resolved") if isinstance(result, dict) else None
    if not isinstance(resolved, dict):
        return None
    call = resolved.get("daily_action_call")
    if not isinstance(call, dict):
        return None
    action = str(call.get("action") or "").strip()
    target = str(call.get("target") or "").strip()
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    if _reminder_requested(raw):
        action = "reminder_create"
        target = _reminder_target_from_text(raw) or target
        reminder_time = _reminder_time_from_text(raw)
        args = {**args, "content": target}
        if reminder_time:
            args["time"] = reminder_time
    if action == "weather" and _deictic_weather_target(target):
        target = ""
    if not action or action == "cancel":
        return None
    return {
        "action": action,
        "target": target,
        "args": args,
        "spans": result.get("spans") or [],
        "resolved": resolved,
        "cancelled": bool(resolved.get("cancelled")),
    }


REMINDER_TIME_HEAD_RE = re.compile(
    r"^(?:"
    r"今天|明天|后天|今晚|明早|明晚|"
    r"今天上午|今天下午|今天晚上|明天上午|明天下午|明天晚上|后天上午|后天下午|后天晚上|"
    r"上午|下午|晚上|早上|中午|"
    r"[0-9零一二两三四五六七八九十半个]+(?:个)?(?:分钟|小时|钟头)(?:后|之后)|"
    r"(?:今天|明天|后天)?[零一二两三四五六七八九十0-9]{1,3}(?:点|[:：])"
    r")"
)


def _reminder_requested(text: str) -> bool:
    return bool(re.search(r"(提醒我|提醒|设个提醒|加个提醒|闹钟)", str(text or "")))


def _has_non_reminder_daily_intent(text: str) -> bool:
    return bool(re.search(r"(天气|温度|几度|下雨|带伞|冷|热|几点|时间|日历|calendar|地图|路线|地址|在哪|哪里|note|便签|记住|记得)", str(text or ""), flags=re.IGNORECASE))


def _parse_explicit_reminder_actions(text: str) -> list[dict[str, Any]]:
    raw = normalize_chinese_asr_variants(text).strip(" ，,。；;")
    if not _reminder_requested(raw):
        return []
    match = re.search(
        r"(?P<prefix>(?:今天|明天|后天|今晚|明早|明晚|今天上午|今天下午|今天晚上|明天上午|明天下午|明天晚上|后天上午|后天下午|后天晚上|上午|下午|晚上|早上|中午|[0-9零一二两三四五六七八九十半个]+(?:个)?(?:分钟|小时|钟头)(?:后|之后))?)"
        r"(?:记得)?(?:提醒我|提醒|设个提醒|加个提醒|闹钟)(?P<body>.+)$",
        raw,
    )
    if not match:
        return []
    prefix = str(match.group("prefix") or "")
    body = str(match.group("body") or "").strip(" ，,。；;")
    if not body:
        return []
    body = re.sub(r"(?:，|,|。|；|;)?\s*(?:然后|再|接着|顺便|并且|同时)\s*", "，", body)
    pieces = [part.strip(" ，,。；;") for part in re.split(r"[，,。；;]+", body) if part.strip(" ，,。；;")]
    if not pieces:
        return []
    items: list[str] = []
    for index, piece in enumerate(pieces):
        candidate = (prefix + piece).strip() if index == 0 and prefix else piece
        if index > 0 and not REMINDER_TIME_HEAD_RE.search(candidate):
            items[-1] = f"{items[-1]}，{candidate}"
            continue
        items.append(candidate)
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        target = item.strip(" ，,。；;")
        if not target:
            continue
        reminder_time = _reminder_time_from_text(target)
        args = {"content": target}
        if reminder_time:
            args["time"] = reminder_time
        actions.append(
            {
                "action": "reminder_create",
                "target": target,
                "args": args,
                "spans": [],
                "resolved": {
                    "domain": "daily",
                    "action": "reminder_create",
                    "target": target,
                    "content": target,
                    "time": reminder_time,
                    "daily_action_call": {"action": "reminder_create", "target": target, "args": args},
                },
                "cancelled": False,
                "segment": {"text": target, "start": 0, "end": len(target), "index": index},
            }
        )
    return actions


def _reminder_target_from_text(text: str) -> str:
    raw = str(text or "").strip(" ，,。；;")
    match = re.search(
        r"((?:今天|明天|后天|今晚|明早|明天下午|明天上午|明天晚上)?(?:上午|下午|晚上|早上|中午)?|(?:[0-9零一二两三四五六七八九十半个]+(?:个)?(?:分钟|小时|钟头)(?:后|之后))?)"
        r"(?:记得)?(?:提醒我|提醒|设个提醒|加个提醒|闹钟)(.+)$",
        raw,
    )
    if not match:
        return raw
    prefix = str(match.group(1) or "")
    content = str(match.group(2) or "").strip(" ，,。；;")
    return (prefix + content).strip() or raw


def _reminder_time_from_text(text: str) -> str:
    match = re.search(r"([0-9零一二两三四五六七八九十半个]+(?:个)?(?:分钟|小时|钟头)(?:后|之后))", str(text or ""))
    if match:
        return str(match.group(1) or "")
    match = re.search(
        r"(今天上午|今天下午|今天晚上|明天上午|明天下午|明天晚上|后天上午|后天下午|后天晚上|今晚|明早|明晚|今天|明天|后天|上午|下午|晚上|早上|中午)"
        r"(?:[零一二两三四五六七八九十0-9]{1,3}(?:点|[:：])(?:[零一二两三四五六七八九十0-9]{1,3})?)?",
        str(text or ""),
    )
    return str(match.group(0) or "") if match else ""


def _deictic_weather_target(target: str) -> bool:
    compact = re.sub(r"[\s,，。！？!?、；;：:]+", "", str(target or ""))
    return compact in {"这边", "这里", "当地", "本地", "当前位置", "我这边"} or compact.startswith(("这边", "这里", "当地", "本地"))


def split_daily_segments(text: str) -> list[dict[str, Any]]:
    raw = normalize_chinese_asr_variants(text).strip()
    if not raw:
        return []
    parts: list[dict[str, Any]] = []
    start = 0
    pattern = re.compile(
        r"(?:，|,|。|；|;)?\s*(然后|再|接着|顺便|并且|同时)\s*"
        r"|(?:，|,|。|；|;|？|\?)\s*(?=(?:打开|启动|播放|放|open|launch|play))"
        r"|(?:，|,|。|；|;|？|\?)\s*(?=(?:(?:今天|明天|后天|今晚|明早|明天下午|明天上午|明天晚上|上午|下午|晚上|早上|中午).{0,16})?(?:天气|温度|几度|下雨|带伞|冷|热))"
        r"|(?:，|,|。|；|;|？|\?)\s*(?=(?:记得)?(?:提醒我|提醒|设个提醒|加个提醒|闹钟|写到|写下|记到|记下|记住|帮我记一下|放进上下文|去|查一下|查查|搜索|搜|看看|地图))"
        r"|(?<!^)(?<![，,。；;？?\s])(?=(?:今天|明天|后天|今晚|明早|明天下午|明天上午|明天晚上|上午|下午|晚上|早上|中午)?(?:记得)?(?:提醒我|提醒|设个提醒|加个提醒|闹钟))"
    )
    for match in pattern.finditer(raw):
        chunk = raw[start : match.start()].strip(" ，,。；;？?")
        if chunk:
            parts.append({"text": chunk, "start": start, "end": match.start()})
        start = match.end()
    tail = raw[start:].strip(" ，,。；;？?")
    if tail:
        parts.append({"text": tail, "start": start, "end": len(raw)})
    merged: list[dict[str, Any]] = []
    pending_prefix = ""
    pending_start: int | None = None
    for part in parts:
        text_part = str(part.get("text") or "")
        if re.fullmatch(r"(?:今天|明天|后天|今晚|明早|上午|下午|晚上|早上|中午|明天下午|明天上午|明天晚上|[0-9零一二两三四五六七八九十半个]+(?:个)?(?:分钟|小时|钟头)(?:后|之后))+", text_part):
            pending_prefix += text_part
            pending_start = int(part.get("start") or 0) if pending_start is None else pending_start
            continue
        if pending_prefix:
            part = {**part, "text": pending_prefix + text_part, "start": pending_start if pending_start is not None else part.get("start", 0)}
            pending_prefix = ""
            pending_start = None
        merged.append(part)
    if pending_prefix:
        merged.append({"text": pending_prefix, "start": pending_start or 0, "end": len(raw)})
    return merged or [{"text": raw, "start": 0, "end": len(raw)}]


def parse_daily_actions(text: str) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    explicit_reminders = _parse_explicit_reminder_actions(raw)
    if explicit_reminders and not _has_non_reminder_daily_intent(raw):
        return explicit_reminders
    segments = split_daily_segments(raw)
    if len(segments) <= 1:
        one = parse_daily_slots(raw)
        if one:
            one["segment"] = {"text": raw, "start": 0, "end": len(raw), "index": 0}
            return [one]
        return []
    results: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        segment_text = str(segment.get("text") or "")
        reminder_items = _parse_explicit_reminder_actions(segment_text)
        if reminder_items:
            for reminder in reminder_items:
                reminder["segment"] = {**segment, "index": idx}
                results.append(reminder)
            continue
        parsed = parse_daily_slots(segment_text)
        if not parsed:
            continue
        parsed["segment"] = {**segment, "index": idx}
        results.append(parsed)
    return results


def parser_status() -> dict[str, Any]:
    state = _load()
    return {
        "available": bool(state.get("available")),
        "artifact_dir": str(_artifact_dir()),
        "error": str(state.get("error") or ""),
    }
