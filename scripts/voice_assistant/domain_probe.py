from __future__ import annotations

import difflib
import datetime as dt
import json
import plistlib
import re
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import parse, request

from voice_assistant.daily_slot_parser import parse_daily_actions
from voice_assistant.location_helper import current_address
from voice_assistant.weather_location import plausible_weather_location
from voice_assistant.voice_text import normalize_chinese_asr_variants


PROBE_ANSWER_TIMEOUT_SECONDS = 0.2
PROBE_ANSWER_TIMEOUT_BY_ACTION = {
    "weather": 4.2,
    "time": 0.2,
    "map": 0.5,
}
_ANSWER_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="domain-probe-answer")
_ANSWER_CACHE: dict[str, dict[str, Any]] = {}
_ANSWER_INFLIGHT: dict[str, Future[dict[str, Any] | None]] = {}
_ANSWER_LOCK = threading.Lock()


@dataclass
class DomainProbeResult:
    domain: str
    confidence: float
    intent: str
    context: str
    matched_entities: list[dict[str, Any]] = field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 2),
            "intent": self.intent,
            "context": self.context,
            "matched_entities": self.matched_entities,
            "suggested_actions": self.suggested_actions,
        }


DOMAIN_DESCRIPTIONS = {
    "computer": "本机 App、窗口、文件系统、shell、osascript、屏幕、摄像头和剪贴板操作。",
    "search": "公开信息检索、网页、新闻、deep search、URL 抓取和视频结果发现。",
    "daily": "时间、天气、地图、日历、提醒、note、长期记忆和待办。",
    "python": "Python 代码执行、数据处理、画图、脚本生成和可复现分析；开发应用默认属于 computer domain。",
    "mim": "Mimir/Nomad/Ratatoskr 集群、部署、egress、日志和节点健康。",
    "communication": "邮件、微信、Telegram、联系人和消息收发。",
}


WEATHER_INTENT_KEYWORDS = [
    "天气",
    "气温",
    "温度",
    "几度",
    "多少度",
    "体感",
    "冷不冷",
    "热不热",
    "会冷",
    "会热",
    "下雨",
    "降雨",
    "雨",
    "湿度",
    "风速",
    "刮风",
    "风大",
    "空气",
    "空气质量",
    "雾霾",
    "紫外线",
    "穿什么",
    "带伞",
    "要不要带伞",
    "weather",
    "temperature",
    "forecast",
    "rain",
    "humidity",
    "wind",
    "uv",
]


STATIC_APP_ALIASES = {
    "chrome": "Google Chrome",
    "谷歌": "Google Chrome",
    "safari": "Safari",
    "inna": "IINA",
    "iina": "IINA",
    "photo booth": "Photo Booth",
    "photobooth": "Photo Booth",
    "照片亭": "Photo Booth",
    "camera": "Camera",
    "相机": "Camera",
    "music": "Music",
    "音乐": "Music",
    "quicktime": "QuickTime Player",
    "quick time": "QuickTime Player",
    "quicktime player": "QuickTime Player",
    "reminder": "Reminders",
    "reminders": "Reminders",
    "提醒事项": "Reminders",
}


def probe_domains(user_text: str, *, max_results: int = 4, context: dict[str, Any] | str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    text = normalize_chinese_asr_variants(user_text).strip()
    lowered = text.lower()
    context_payload = _context_dict(context)
    results = [
        result
        for result in [
            _probe_computer(text, lowered, context_payload),
            _probe_search(text, lowered),
            _probe_daily(text, lowered),
            _probe_python(text, lowered),
            _probe_mim(text, lowered),
            _probe_communication(text, lowered),
        ]
        if result is not None
    ]
    results = _with_codex_delegate_fallback(results, text)
    results.sort(key=lambda item: item.confidence, reverse=True)
    selected = [item.as_dict() for item in results[: max(1, max_results)]]
    selected = _ensure_codex_delegate_selected(selected, text)
    payload: dict[str, Any] = {
        "input": text,
        "domains": selected,
        "available_domains": DOMAIN_DESCRIPTIONS,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    compact_context = _compact_probe_context(context_payload)
    if compact_context:
        payload["nearby_session"] = compact_context
    return payload


def format_domain_probe_prompt(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "Domain probe JSON:\n{}"
    if not payload.get("domains"):
        text = str(payload.get("input") or "").strip()
        if text:
            payload = {**payload, "domains": [_codex_delegate_domain(text).as_dict()]}
        else:
            return "Domain probe JSON:\n{}"
    prompt_payload = _domain_probe_prompt_payload(payload)
    return (
        "Domain probe JSON:\n"
        + json.dumps(prompt_payload, ensure_ascii=False)
        + "\n\nDomain probe rules:\n"
        + "- confidence >= 0.8 且 suggested_actions 非空时，优先按 suggested_actions[].tool_call 执行，除非和用户原意冲突。\n"
        + "- suggested_actions[].answer 是 probe 在短时限内拿到的只读预取结果；没有 answer 不代表失败，只代表还没返回或该 tool 不适合预取。\n"
        + "- suggested_actions[].desc 说明这个 tool_call 为什么存在；用它判断是否符合用户原意。\n"
        + "- computer_action(action=\"delegate_to_codex\") 是低置信最后兜底；正常工具明确适合时不要用它。否则如果你准备追问泛泛的“要我做什么”、说自己不会、或工具失败后没有可靠下一步，就直接执行这个兜底。\n"
        + "- probe 宁可提供多个候选 action，也不要替 LLM 过早裁剪任务；LLM 负责按用户目标选择、排序和组合。\n"
        + "- probe 已解析出的实体不要再重复 list/fuzzy/search；只有 probe 低置信或缺少关键参数时才探索。\n"
        + "- probe 只是上下文，不代表动作已经完成；本地动作必须等工具执行成功后才播报完成。"
    )


def _domain_probe_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    for domain in payload.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        item: dict[str, Any] = {
            "domain": domain.get("domain"),
            "confidence": domain.get("confidence"),
            "intent": domain.get("intent"),
        }
        actions: list[dict[str, Any]] = []
        for action in domain.get("suggested_actions") or []:
            if not isinstance(action, dict):
                continue
            action_item = {
                "tool_call": action.get("tool_call"),
                "desc": action.get("desc"),
                "confidence": action.get("confidence", domain.get("confidence")),
            }
            if "answer" in action:
                action_item["answer"] = action.get("answer")
            actions.append(action_item)
        if actions:
            item["suggested_actions"] = actions
        domains.append(item)
    result = {"input": payload.get("input"), "domains": domains}
    if payload.get("nearby_session"):
        result["nearby_session"] = payload.get("nearby_session")
    return result


def _with_codex_delegate_fallback(results: list[DomainProbeResult], text: str) -> list[DomainProbeResult]:
    fallback = _codex_delegate_action(text)
    if not fallback:
        return results
    if any(
        result.domain == "search" and result.confidence >= 0.75 and result.suggested_actions
        for result in results
    ):
        return results
    for result in results:
        if result.domain != "computer":
            continue
        calls = [str(action.get("tool_call") or "") for action in result.suggested_actions]
        if any('action="develop_app"' in call or 'action="delegate_to_codex"' in call for call in calls):
            return results
        result.suggested_actions = _dedupe_actions([*result.suggested_actions, fallback])
        result.context = f"{result.context} 完全不确定时可委托 Codex。"
        return results
    return [
        *results,
        _codex_delegate_domain(text),
    ]


def _ensure_codex_delegate_selected(selected: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    fallback = _codex_delegate_action(text)
    if not fallback:
        return selected
    if any(
        isinstance(domain, dict)
        and domain.get("domain") == "search"
        and float(domain.get("confidence") or 0) >= 0.75
        and domain.get("suggested_actions")
        for domain in selected
    ):
        return selected
    for domain in selected:
        if not isinstance(domain, dict) or domain.get("domain") != "computer":
            continue
        actions = domain.setdefault("suggested_actions", [])
        if not isinstance(actions, list):
            domain["suggested_actions"] = actions = []
        if not any('action="delegate_to_codex"' in str(action.get("tool_call") or "") for action in actions if isinstance(action, dict)):
            actions.append(fallback)
        return selected
    selected.append(
        _codex_delegate_domain(text).as_dict()
    )
    return selected


def _codex_delegate_domain(text: str) -> DomainProbeResult:
    fallback = _codex_delegate_action(text)
    return DomainProbeResult(
        "computer",
        0.35,
        "codex_delegate_fallback",
        "默认注入的最后兜底：其他 domain 或工具无法确定做法时，把用户原始需求委托给本机 Codex。",
        [],
        [fallback] if fallback else [],
    )


def _codex_delegate_action(text: str) -> dict[str, Any] | None:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    return _suggested(
        "delegate_to_codex",
        {"target": _coding_task_target(prompt), "args": {"prompt": prompt, "executor": "codex"}},
        0.35,
        "ultimate fallback: when no normal tool/domain knows how to complete the task, delegate the original prompt to local Codex",
        tool="computer_action",
    )[0]


def _context_dict(context: dict[str, Any] | str | None) -> dict[str, Any]:
    if not context:
        return {}
    if isinstance(context, str):
        try:
            parsed = json.loads(context)
        except json.JSONDecodeError:
            parsed = {"recent_events": [{"content": context[:300]}]}
    else:
        parsed = context
    return parsed if isinstance(parsed, dict) else {}


def _compact_probe_context(context: dict[str, Any] | str | None) -> dict[str, Any]:
    parsed = _context_dict(context)
    if not parsed:
        return {}
    recent: list[dict[str, Any]] = []
    for row in parsed.get("recent_events") or []:
        if not isinstance(row, dict):
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        recent.append(
            {
                "kind": row.get("kind"),
                "role": row.get("role"),
                "text": content[:180],
            }
        )
        if len(recent) >= 6:
            break
    compact: dict[str, Any] = {}
    summary = str(parsed.get("summary") or "").strip()
    if summary:
        compact["summary"] = summary[:400]
    front_context = str(parsed.get("front_note_context") or "").strip()
    if front_context:
        compact["front_note_context"] = front_context[:300]
    if recent:
        compact["recent_events"] = recent
    programs = _registered_program_matches("", parsed, limit=6, include_all=True)
    if programs:
        compact["registered_programs"] = [
            {"title": item.get("title"), "aliases": item.get("aliases", [])[:6], "status": item.get("status")}
            for item in programs[:6]
        ]
    return compact


def _probe_computer(text: str, lowered: str, context: dict[str, Any] | None = None) -> DomainProbeResult | None:
    if _coding_task_requested(text, lowered):
        return None
    keywords = [
        "打开", "关掉", "关闭", "退出", "切到", "前台", "窗口", "分屏", "排", "桌面", "文件", "目录", "下载",
        "download", "downloads", "截图", "相机", "camera", "app", "chrome", "safari", "inna", "iina", "reminder",
        "reminders", "提醒事项", "osascript", "shell", "bash", "按", "快捷键", "菜单", "点击", "电脑操作", "显示器",
        "外接屏", "副屏", "屏幕", "open", "launch", "start", "quit", "close", "exit", "focus", "bring", "front",
        "window", "windows", "split", "tile", "arrange", "layout", "desktop", "file", "folder", "screenshot",
        "click", "press", "hotkey", "shortcut", "menu", "display", "monitor", "screen",
    ]
    file_query = _extract_file_query(text)
    folder = _folder_from_text(text)
    display = _display_from_text(text)
    if not _has_any(text, lowered, keywords) and not file_query and not display:
        return None
    program_entities = _registered_program_matches(text, context or {})
    entities = _app_entities(text)
    searchish_open = _has_any(
        text,
        lowered,
        ["查", "搜索", "搜", "网页", "链接", "网站", "新闻", "最近", "最新", "视频", "播放", "youtube", "mv", "mtv", "search", "find", "look up", "website", "web", "link", "news", "video", "play"],
    )
    intent = "operate_computer"
    confidence = 0.72
    suggested: list[dict[str, Any]] = []
    if _has_any(text, lowered, ["关掉", "关闭", "退出", "quit", "close"]):
        intent = "close_app"
        confidence = 0.88
        args = {"target": program_entities[0]["title"]} if program_entities else ({"app": entities[0]["value"]} if entities else {})
        suggested.extend(_suggested("close_app", args, confidence, "computer probe matched app close", tool="computer_action"))
    elif _has_any(text, lowered, ["打开", "启动", "open", "launch"]):
        if searchish_open and not (program_entities or entities):
            return None
        intent = "open_app_or_path"
        confidence = 0.84
        if program_entities:
            confidence = 0.93
            suggested.extend(_suggested("open_program", {"target": program_entities[0]["title"]}, confidence, "computer probe matched registered coding app", tool="computer_action"))
        elif entities:
            args = {"app": entities[0]["value"]}
            suggested.extend(_suggested("open_app", args, confidence, "computer probe matched app open", tool="computer_action"))
        else:
            target_guess = re.sub(r"^(打开|启动|open|launch)\s*", "", text, flags=re.I).strip()
            suggested.extend(_suggested("open_program", {"target": target_guess or text}, confidence, "computer probe matched registered program open", tool="computer_action"))
        if (file_query or folder) and not entities:
            suggested.extend(_suggested("open_file", {"folder": folder or "", "query": file_query or ""}, 0.86, "computer probe matched file open", tool="computer_action"))
    elif _has_any(text, lowered, ["窗口", "分屏", "排", "陀螺", "铺开", "window", "windows", "split", "tile", "arrange", "layout"]):
        intent = "arrange_workspace"
        confidence = 0.9
        suggested.extend(_suggested("arrange_workspace", {"display": display or ""}, confidence, "computer probe matched window layout", tool="computer_action"))
    elif _has_any(text, lowered, ["截图", "抓拍", "拍照", "照一张", "screenshot", "snapshot", "photo", "take a picture"]):
        intent = "capture_or_screenshot"
        confidence = 0.82
        action = "capture_camera_snapshot" if _has_any(text, lowered, ["拍照", "抓拍", "照一张", "snapshot", "photo", "take a picture"]) else "screenshot"
        tool = "capture_camera_snapshot" if action == "capture_camera_snapshot" else "computer_action"
        suggested.extend(_suggested(action, {}, confidence, "computer probe matched capture/screenshot", tool=tool))
    elif _has_any(text, lowered, ["osascript", "apple script"]):
        intent = "run_osascript"
        confidence = 0.88
        suggested.extend(_suggested("run_osascript", {}, confidence, "computer probe matched AppleScript execution", tool="computer_action"))
    elif _has_any(text, lowered, ["shell", "bash", "命令"]):
        intent = "run_shell"
        confidence = 0.84
        suggested.extend(_suggested("run_shell", {}, confidence, "computer probe matched shell execution", tool="computer_action"))
    elif _has_any(text, lowered, ["按", "快捷键", "菜单", "点击", "电脑操作", "press", "hotkey", "shortcut", "menu", "click"]):
        intent = "computer_use"
        confidence = 0.82
        suggested.extend(_suggested("computer_use", {"app": entities[0]["value"]} if entities else {}, confidence, "computer probe matched GUI operation", tool="computer_action"))
    if display:
        intent = "open_file_then_move_window" if file_query else intent
        suggested.extend(_suggested("move_window_to_display", {"display": display, "target": entities[0]["value"] if entities else ""}, 0.84, "computer probe matched display move", tool="computer_action"))
    if file_query and display:
        intent = "open_file_then_move_window"
        suggested.insert(0, _suggested(
            "open_file_and_move_to_display",
            {"folder": folder or "", "query": file_query, "display": display, "open_with": ""},
            0.9,
            "computer probe matched file open plus display move",
            tool="computer_action",
        )[0])
        confidence = max(confidence, 0.9)
    matched_entities = [*program_entities[:3], *entities[:3]]
    if folder:
        matched_entities.append({"type": "folder", "value": folder, "source": "text", "confidence": 0.9})
    if file_query:
        matched_entities.append({"type": "file_query", "value": file_query, "source": "text", "confidence": 0.86})
    if display:
        matched_entities.append({"type": "display", "value": display, "source": "text", "confidence": 0.86})
    return DomainProbeResult("computer", confidence, intent, "用户输入像本机电脑/App/窗口/文件/shell/GUI 操作。", matched_entities, _dedupe_actions(suggested))


def _probe_search(text: str, lowered: str) -> DomainProbeResult | None:
    keywords = [
        "查", "搜索", "搜", "最近", "最新", "新闻", "结果", "网页", "链接", "youtube", "视频", "mtv", "ymca", "价格",
        "current", "latest", "search", "find", "look up", "google", "news", "result", "results", "website", "web",
        "link", "video", "price", "score", "scores",
    ]
    if not _has_any(text, lowered, keywords):
        return None
    if _note_capture_requested(text, lowered) and _deictic_only_subject(text, lowered):
        return None
    query = _strip_command_words(text)
    confidence = 0.78
    intent = "web_search"
    if _has_any(text, lowered, ["最近", "最新", "新闻", "current", "latest", "recent", "news", "today"]):
        confidence = 0.9
        intent = "current_info_search"
    if _has_any(text, lowered, ["网页", "链接", "网站", "打开", "website", "web", "link", "open"]):
        confidence = max(confidence, 0.84)
    if _has_any(text, lowered, ["youtube", "视频", "video", "music video", "mv"]):
        confidence = max(confidence, 0.88)
        intent = "video_search"
    return DomainProbeResult(
        "search",
        confidence,
        intent,
        "用户输入需要外部公开信息或网页结果。",
        [{"type": "query", "value": query, "source": "user_text", "confidence": confidence}],
        _suggested("web_search", {"query": query}, confidence, "search probe extracted query") if query else [],
    )


def _probe_daily(text: str, lowered: str) -> DomainProbeResult | None:
    text = _strip_app_open_prefix_for_daily(text)
    if not text.strip():
        return None
    lowered = text.lower()
    if _coding_task_requested(text, lowered):
        return None
    keywords = [
        *WEATHER_INTENT_KEYWORDS,
        "几点", "时间", "地址", "在哪", "哪里", "日历", "提醒", "闹钟", "地图", "路线", "怎么走", "去", "note",
        "便签", "记一下", "记下", "写一下", "记录", "记住", "记得", "待办", "todo", "calendar", "reminder",
        "reminders", "map", "route", "directions", "address", "where is", "where am i", "what time", "time in",
        "set a reminder", "remind me", "write down", "take a note", "remember",
    ]
    slot_results = parse_daily_actions(text)
    if slot_results:
        parsed_result = _daily_result_from_slot_parser(text, slot_results)
        if parsed_result is not None:
            return parsed_result
    if not _has_any(text, lowered, keywords):
        return None
    intent = "daily_info"
    action = ""
    args: dict[str, Any] = {}
    confidence = 0.82
    if _weather_intent_requested(text, lowered):
        intent = "weather"
        action = "weather"
        location = weather_location_from_text(text)
        if location and not plausible_weather_location(location):
            location = ""
        location_source = ""
        if not location:
            current = _current_address_for_probe()
            if current.get("ok"):
                location = str(current.get("address") or "").strip()
                location_source = "current_address"
            elif current.get("error"):
                args["args"] = {"location_error": str(current.get("error") or "")[:120]}
        args["target"] = location
        if location_source:
            args["args"] = {"location_source": location_source}
        confidence = 0.9
    elif _has_any(text, lowered, ["几点", "时间", "现在时间", "当前时间", "what time", "current time", "time in"]):
        intent = "time"
        action = "time"
        confidence = 0.88
    elif _has_any(text, lowered, ["地图", "路线", "怎么走", "去", "地址", "在哪", "哪里", "map", "route", "directions", "where is", "address"]):
        intent = "map"
        action = "map"
        target = _map_target_from_text(text)
        mode = "route" if _has_any(text, lowered, ["路线", "怎么走", "去", "route", "directions"]) else ("current_address" if _has_any(text, lowered, ["当前地址", "我在哪", "我在哪里", "where am i", "current address"]) else "query")
        args = {"target": target, "args": {"mode": mode}}
        confidence = 0.86
    elif _has_any(text, lowered, ["note", "便签"]) or (_note_capture_requested(text, lowered) and not _long_term_memory_requested(text, lowered)):
        intent = "front_note"
        action = "note_live"
        args = {"target": _note_capture_content(text)}
        confidence = 0.9
    elif _long_term_memory_requested(text, lowered):
        intent = "long_term_memory"
        action = "memory"
        args = {"target": _note_capture_content(text)}
        confidence = 0.88
    elif _has_any(text, lowered, ["提醒", "闹钟", "reminder", "remind me", "set a reminder"]):
        intent = "reminder"
        action = "reminder_create"
        args = {"target": _reminder_content(text)}
        confidence = 0.86
    elif _has_any(text, lowered, ["日历", "calendar", "schedule", "agenda"]):
        intent = "calendar"
        action = "calendar_list"
        confidence = 0.84
    return DomainProbeResult("daily", confidence, intent, "用户输入像日常信息、note、提醒、地图或个人上下文管理。", [], _suggested(action, args, confidence, "daily probe matched personal info task", tool="daily_action") if action else [])


def _strip_app_open_prefix_for_daily(text: str) -> str:
    value = str(text or "").strip()
    aliases = sorted((re.escape(alias) for alias in STATIC_APP_ALIASES), key=len, reverse=True)
    if not aliases:
        return value
    pattern = rf"^\s*(?:打开|启动|open|launch)\s*(?:{'|'.join(aliases)})\s*[,，、;；]?\s*"
    return re.sub(pattern, "", value, count=1, flags=re.IGNORECASE).strip()


def _daily_result_from_slot_parser(text: str, slot_result: dict[str, Any] | list[dict[str, Any]]) -> DomainProbeResult | None:
    if isinstance(slot_result, list):
        return _daily_result_from_slot_results(text, slot_result)
    action = str(slot_result.get("action") or "").strip()
    if action not in {"weather", "time", "map", "calendar_list", "reminder_list", "reminder_create", "note_live", "note_context", "memory"}:
        return None
    confidence = 0.93
    matched_entities = [
        {"type": "semantic_spans", "value": slot_result.get("spans") or [], "source": "semantic_slot_parser", "confidence": confidence},
        {"type": "semantic_resolved", "value": slot_result.get("resolved") or {}, "source": "semantic_slot_parser", "confidence": confidence},
    ]
    if bool(slot_result.get("cancelled")):
        return DomainProbeResult("daily", confidence, "cancelled", "semantic_slot_parser 解析到取消意图，不应执行 daily_action。", matched_entities, [])
    target = str(slot_result.get("target") or "").strip()
    extra = slot_result.get("args") if isinstance(slot_result.get("args"), dict) else {}
    if action == "weather" and target and not plausible_weather_location(target):
        target = ""
    if action == "weather" and not target:
        current = _current_address_for_probe()
        if current.get("ok"):
            target = str(current.get("address") or "").strip()
            extra = {**extra, "location_source": "current_address"}
        elif current.get("error"):
            extra = {**extra, "location_error": str(current.get("error") or "")[:120]}
    if action == "map":
        compact = _compact_key(text)
        if any(token in compact for token in ["怎么走", "路线", "去"]):
            extra = {**extra, "mode": "route"}
        elif any(token in compact for token in ["我在哪", "我在哪里", "当前地址"]):
            extra = {**extra, "mode": "current_address"}
        else:
            extra = {**extra, "mode": extra.get("mode") or "query"}
    if action == "reminder_create":
        extracted = _reminder_content(text)
        if extracted:
            target = extracted
            extra = {**extra, "content": extracted}
    args = {"target": target, "args": extra}
    suggested = _suggested(action, args, confidence, "daily semantic slot parser resolved fat tool call", tool="daily_action")
    return DomainProbeResult("daily", confidence, action, "semantic_slot_parser 解析出 daily fat tool 调用。", matched_entities, suggested)


def _daily_result_from_slot_results(text: str, slot_results: list[dict[str, Any]]) -> DomainProbeResult | None:
    usable = [item for item in slot_results if str(item.get("action") or "").strip() in {"weather", "time", "map", "calendar_list", "reminder_list", "reminder_create", "note_live", "note_context", "memory"}]
    if not usable:
        return None
    confidence = 0.93
    matched_entities = [
        {
            "type": "semantic_segments",
            "value": [
                {
                    "segment": item.get("segment") or {},
                    "spans": item.get("spans") or [],
                    "resolved": item.get("resolved") or {},
                    "cancelled": bool(item.get("cancelled")),
                }
                for item in slot_results
            ],
            "source": "semantic_slot_parser",
            "confidence": confidence,
        }
    ]
    suggested: list[dict[str, Any]] = []
    intents: list[str] = []
    for item in usable:
        if bool(item.get("cancelled")):
            continue
        action = str(item.get("action") or "").strip()
        target = str(item.get("target") or "").strip()
        extra = item.get("args") if isinstance(item.get("args"), dict) else {}
        segment_text = str((item.get("segment") or {}).get("text") or text)
        if action == "weather" and not target:
            current = _current_address_for_probe()
            if current.get("ok"):
                target = str(current.get("address") or "").strip()
                extra = {**extra, "location_source": "current_address"}
            elif current.get("error"):
                extra = {**extra, "location_error": str(current.get("error") or "")[:120]}
        if action == "map":
            compact = _compact_key(segment_text)
            if any(token in compact for token in ["怎么走", "路线", "去"]):
                extra = {**extra, "mode": "route"}
            elif any(token in compact for token in ["我在哪", "我在哪里", "当前地址"]):
                extra = {**extra, "mode": "current_address"}
            else:
                extra = {**extra, "mode": extra.get("mode") or "query"}
        if action == "reminder_create":
            extracted = _reminder_content(segment_text)
            if extracted:
                target = extracted
                extra = {**extra, "content": extracted}
        suggested.extend(_suggested(action, {"target": target, "args": extra}, confidence, "daily semantic slot parser resolved fat tool segment", tool="daily_action"))
        intents.append(action)
    if not suggested and any(bool(item.get("cancelled")) for item in usable):
        return DomainProbeResult("daily", confidence, "cancelled", "semantic_slot_parser 解析到取消意图，不应执行 daily_action。", matched_entities, [])
    if not suggested:
        return None
    intent = "+".join(dict.fromkeys(intents))
    return DomainProbeResult("daily", confidence, intent, "semantic_slot_parser 分段解析出多个 daily fat tool 调用。", matched_entities, _dedupe_actions(suggested))


def _weather_intent_requested(text: str, lowered: str) -> bool:
    keyword_hits = [item for item in WEATHER_INTENT_KEYWORDS if item != "uv"]
    if _has_any(text, lowered, keyword_hits):
        return True
    if re.search(r"(?<![a-z0-9])uv(?![a-z0-9])", lowered):
        return True
    compact = _compact_key(text)
    return bool(re.search(r"(今天|今日|现在|当前|目前|明天|周末|当地|这边|这里).{0,8}(冷|热|雨|风|晒|闷|潮|干)", compact))


def _probe_python(text: str, lowered: str) -> DomainProbeResult | None:
    if _coding_task_requested(text, lowered):
        target = _coding_task_target(text)
        prompt = _coding_task_prompt(text)
        args: dict[str, Any] = {"prompt": prompt}
        executor = _explicit_coding_executor(text, lowered)
        if executor:
            args["executor"] = executor
        confidence = 0.94
        return DomainProbeResult(
            "computer",
            confidence,
            "develop_app",
            "用户要求开发、实现、修复或生成可运行应用，默认交给 computer domain 提交本机开发执行器任务。",
            [{"type": "development_prompt", "value": prompt, "source": "user_text", "confidence": confidence}],
            _suggested(
                "develop_app",
                {"target": target, "args": args},
                confidence,
                "computer probe matched local coding/development request",
                tool="computer_action",
            ),
        )
    keywords = ["python", "代码", "脚本", "运行", "执行", "画图", "csv", "json", "数据", "计算", "parse", "script"]
    if not _has_any(text, lowered, keywords):
        return None
    return None


def _coding_task_requested(text: str, lowered: str) -> bool:
    compact = _compact_key(text)
    explicit_executor = bool(_explicit_coding_executor(text, lowered))
    stack_hints = ["pywebview", "uv", "typer", "agno", "inquirerpy", "inqueryerpy"]
    dev_terms = [
        "开发",
        "实现",
        "修",
        "修复",
        "改",
        "修改",
        "写",
        "加",
        "新增",
        "优化",
        "调试",
        "排查",
        "bug",
        "代码",
        "项目",
        "游戏",
        "应用",
        "工具",
        "窗口",
        "屏幕",
        "界面",
        "网页",
        "dashboard",
        "agent",
        "动画",
        "悬浮",
        "弹跳",
        "乱碰",
        "develop",
        "debug",
        "implement",
        "fix",
        "refactor",
        "build",
        "create",
        "make",
        "write",
        "code",
    ]
    if explicit_executor:
        return True
    if _has_any(text, lowered, stack_hints):
        return True
    if re.search(r"(开发|实现|新增|修复|重构|优化)(?!者|商|区|阶段|环境).{1,60}", text):
        return True
    if re.search(r"(?i)\b(?:debug|dbug|deubg|degub)\b.{0,60}", text) or re.search(r"(调试|排查).{1,60}", text):
        return True

    write_or_build = _has_any(
        text,
        lowered,
        [
            "写个",
            "写一个",
            "写一下",
            "给我写",
            "做个",
            "做一个",
            "做一下",
            "搭个",
            "搭一个",
            "开发",
            "实现",
            "新增",
            "加一个",
            "重构",
            "修复",
            "改一下",
            "优化",
            "调试",
            "排查",
            "build",
            "create",
            "make",
            "write",
            "code",
            "debug",
            "implement",
            "refactor",
            "fix",
        ],
    )
    code_object = _has_any(
        text,
        lowered,
        [
            "代码",
            "项目",
            "脚本",
            "程序",
            "工具",
            "应用",
            "app",
            "游戏",
            "网页",
            "页面",
            "窗口",
            "屏幕",
            "界面",
            "动画",
            "悬浮",
            "弹跳",
            "乱碰",
            "cli",
            "命令行",
            "dashboard",
            "agent",
            "机器人",
            "服务",
            "接口",
            "api",
            "python",
        ],
    )
    if write_or_build and code_object:
        return True
    if write_or_build and _has_any(text, lowered, ["想玩", "玩一下", "玩玩", "小游戏", "game"]):
        return True

    bug_or_project_change = _has_any(text, lowered, ["debug", "dbug", "deubg", "degub", "调试", "排查", "修 bug", "修bug", "修复 bug", "改代码", "改项目", "优化代码", "重构代码"])
    return bug_or_project_change and _has_any(text, lowered, dev_terms)


def _coding_task_prompt(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"(用|让|叫|请)?\s*(本地\s*)?(codex(?:-?cli)?|antigravity|antigrav|agr)\s*(帮我|来|去|执行|开发|处理)?", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" ，。,.")
    return value or str(text or "").strip()


def _coding_task_target(text: str) -> str:
    prompt = _coding_task_prompt(text)
    prompt = re.sub(r"(这个|一下|吧|帮我|请)", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip(" ，。,.")
    return prompt[:40] or "开发任务"


def _explicit_coding_executor(text: str, lowered: str) -> str:
    compact = _compact_key(text)
    if "codex" in lowered or "codexcli" in compact or "本地codex" in compact:
        return "codex"
    return ""


def _probe_mim(text: str, lowered: str) -> DomainProbeResult | None:
    keywords = ["mim", "mimir", "nomad", "ratatoskr", "egress", "集群", "节点", "部署", "job", "logs"]
    if not _has_any(text, lowered, keywords):
        return None
    return DomainProbeResult(
        "mim",
        0.9,
        "mim_ops",
        "用户输入像 Mimir/Nomad/Ratatoskr 集群运维任务。",
        [],
        _suggested("mim_command", {}, 0.86, "mim probe matched cluster ops task"),
    )


def _probe_communication(text: str, lowered: str) -> DomainProbeResult | None:
    keywords = ["邮件", "邮箱", "微信", "telegram", "消息", "联系人", "发给", "email", "wechat", "mail"]
    if not _has_any(text, lowered, keywords):
        return None
    return DomainProbeResult(
        "communication",
        0.86,
        "message_or_mail",
        "用户输入像邮件、微信、Telegram 或联系人任务。",
        [],
        _suggested("communication_action", {}, 0.82, "communication probe matched messaging task"),
    )


def _has_any(text: str, lowered: str, needles: list[str]) -> bool:
    return any((needle in text) or (needle.lower() in lowered) for needle in needles)


def _long_term_memory_requested(text: str, lowered: str) -> bool:
    return _has_any(text, lowered, ["记住", "记得", "以后记得", "帮我记一下", "帮我记住", "remember this", "remember that", "remember that i", "remember i"])


def _note_capture_requested(text: str, lowered: str) -> bool:
    return _has_any(text, lowered, ["记一下", "记下", "记录一下", "记录", "写一下", "写下", "记到", "写到", "note", "便签", "便签纸", "take a note", "write down", "jot down"])


def _deictic_only_subject(text: str, lowered: str) -> bool:
    deictic_tokens = ["这个", "这件", "这个事", "这件事", "这条", "这条新闻", "这件新闻", "this", "that", "it"]
    if not _has_any(text, lowered, deictic_tokens):
        return False
    stripped = _compact_key(_strip_command_words(text))
    for token in ["这个", "这件", "这个事", "这件事", "这条", "这条新闻", "这件新闻", "this", "that", "it", "新闻", "事"]:
        stripped = stripped.replace(_compact_key(token), "")
    return len(stripped) <= 1


def _note_capture_content(text: str) -> str:
    raw = str(text or "").strip()
    patterns = [
        r".*?(?:记一下|记下|记录一下|记录|写一下|写下|记到|写到)(.+)$",
        r".*?(?:note|便签|便签纸)(?:[:：，, ]*)(.+)$",
        r".*?(?:take a note|write down|jot down)(?:[:：，, ]*)(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ：:，,。")
            if value:
                return value
    return raw


def _reminder_content(text: str) -> str:
    raw = str(text or "").strip()
    value = re.sub(
        r"^(?:帮我|给我|请|记得|记得帮我|记得给我)?(?:提醒我|提醒|设个提醒|加个提醒|闹钟|"
        r"remind me to|remind me|set a reminder to|set a reminder|add a reminder to|add a reminder|reminder)\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    value = value.strip(" ：:，,。")
    return value or raw


def _map_target_from_text(text: str) -> str:
    raw = str(text or "").strip()
    patterns = [
        r".*?(?:去|到|搜索|查一下|看看|地图里看|地图)(.+?)(?:怎么走|路线|在哪|哪里|地址|$)",
        r".*?(?:route to|directions to|how do i get to|get me to)\s+(.+)$",
        r".*?(?:map|where is|address of|show me)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ：:，,。?")
            if value and not _looks_like_generic_target(value):
                return value[:80]
    value = _strip_command_words(raw)
    value = re.sub(r"(怎么走|路线|地图|地址|在哪|哪里|我在哪|我在哪里|当前地址|route|directions|map|where is|address)", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" ：:，,。?")[:80]


def _current_address_for_probe() -> dict[str, Any]:
    return current_address(timeout_seconds=2.0)


def _registered_program_matches(
    text: str,
    context: dict[str, Any],
    *,
    limit: int = 5,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    programs = context.get("registered_programs") if isinstance(context, dict) else []
    if not isinstance(programs, list):
        return []
    query_key = _app_query_key(text)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, program in enumerate(programs):
        if not isinstance(program, dict):
            continue
        program_info = program.get("program") if isinstance(program.get("program"), dict) else {}
        title = str(program_info.get("name") or program.get("title") or "").strip()
        aliases = [
            title,
            *(program.get("aliases") or [] if isinstance(program.get("aliases"), list) else []),
            *(program_info.get("aliases") or [] if isinstance(program_info.get("aliases"), list) else []),
        ]
        aliases = [str(alias or "").strip() for alias in aliases if str(alias or "").strip()]
        if include_all:
            score = 1.0
        else:
            score = max((_program_alias_score(query_key, alias) for alias in aliases), default=0.0)
        if score < 0.78 and not include_all:
            continue
        scored.append(
            (
                score,
                -index,
                {
                    "type": "program",
                    "value": title,
                    "title": title,
                    "aliases": aliases,
                    "source": "registered_program",
                    "workspace_id": program.get("workspace_id"),
                    "path": program.get("path"),
                    "status": program_info.get("status") or program.get("status"),
                    "confidence": round(score, 2),
                },
            )
        )
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item for _, _, item in scored[:limit]]


def _program_alias_score(query_key: str, alias: str) -> float:
    alias_key = _compact_key(alias)
    if not query_key or not alias_key:
        return 0.0
    if alias_key == query_key:
        return 1.0
    if alias_key in query_key or query_key in alias_key:
        return 0.94
    return difflib.SequenceMatcher(None, query_key, alias_key).ratio()


def _app_entities(text: str) -> list[dict[str, Any]]:
    compact_text = _compact_key(text)
    query_key = _app_query_key(text)
    lowered = text.lower()
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alias, app in STATIC_APP_ALIASES.items():
        if alias in lowered or _compact_key(alias) in compact_text:
            _append_app_entity(entities, seen, app, "static_alias", alias, 0.94)
    for item in _local_app_alias_catalog():
        alias_key = str(item.get("alias_key") or "")
        if not alias_key:
            continue
        confidence = 0.0
        if alias_key in compact_text:
            confidence = 0.91 if item.get("running") else 0.84
        elif len(alias_key) >= 4:
            confidence = difflib.SequenceMatcher(None, query_key or compact_text, alias_key).ratio()
            if item.get("running") and confidence >= 0.62:
                confidence = max(confidence, 0.78)
        if confidence >= 0.78:
            _append_app_entity(
                entities,
                seen,
                str(item.get("app") or ""),
                str(item.get("source") or "local_app"),
                str(item.get("alias") or ""),
                confidence,
                bundle_id=str(item.get("bundle_id") or ""),
            )
    entities.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    return entities[:5]


def _app_query_key(text: str) -> str:
    value = _strip_command_words(str(text or ""))
    value = re.sub(r"(打开|启动|关掉|关闭|退出|切到|前台|应用|程序|app|open|launch|start|close|quit|exit|focus|bring|front)", "", value, flags=re.IGNORECASE)
    return _compact_key(value) or _compact_key(text)


def _append_app_entity(
    entities: list[dict[str, Any]],
    seen: set[str],
    app: str,
    source: str,
    alias: str,
    confidence: float,
    *,
    bundle_id: str = "",
) -> None:
    if not app or app in seen:
        return
    seen.add(app)
    item: dict[str, Any] = {"type": "app", "value": app, "source": source, "alias": alias, "confidence": round(confidence, 2)}
    if bundle_id:
        item["bundle_id"] = bundle_id
    entities.append(item)


@lru_cache(maxsize=1)
def _local_app_alias_catalog() -> tuple[dict[str, Any], ...]:
    catalog: list[dict[str, Any]] = []
    seen_aliases: set[tuple[str, str]] = set()

    def add(app: str, alias: str, source: str, *, bundle_id: str = "", running: bool = False) -> None:
        app = str(app or "").strip()
        alias = str(alias or "").strip()
        alias_key = _compact_key(alias)
        if not app or not alias_key:
            return
        key = (app, alias_key)
        if key in seen_aliases:
            return
        seen_aliases.add(key)
        catalog.append({"app": app, "alias": alias, "alias_key": alias_key, "source": source, "bundle_id": bundle_id, "running": running})

    for app, bundle_id in _running_app_rows():
        for alias in _aliases_for_app(app, bundle_id):
            add(app, alias, "running_app", bundle_id=bundle_id, running=True)

    for app_path in _installed_app_paths():
        stem = app_path.stem
        bundle_id = _bundle_id_for_app(app_path)
        for alias in _aliases_for_app(stem, bundle_id):
            add(stem, alias, "installed_app", bundle_id=bundle_id, running=False)
    return tuple(catalog)


def _running_app_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        proc = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, timeout=0.25)
        for line in (proc.stdout or "").splitlines():
            command_path = line.strip()
            app_match = re.search(r"/([^/]+)\.app/", command_path)
            if not app_match:
                continue
            name = app_match.group(1)
            if name and name not in seen:
                rows.append((name, ""))
                seen.add(name)
    except Exception:
        pass
    script = '''
    set output to ""
    tell application "System Events"
      repeat with p in (application processes whose background only is false)
        set appName to name of p as text
        set bundleId to ""
        try
          set bundleId to bundle identifier of p as text
        end try
        set output to output & appName & tab & bundleId & linefeed
      end repeat
    end tell
    return output
    '''
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=0.08)
    except Exception:
        return rows
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t", 1)
        name = parts[0].strip()
        bundle_id = parts[1].strip() if len(parts) > 1 else ""
        if name and name not in seen:
            rows.append((name, bundle_id))
            seen.add(name)
    return rows


def _installed_app_paths() -> list[Path]:
    roots = [Path("/Applications"), Path.home() / "Applications"]
    apps: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            apps.extend(path for path in root.glob("*.app") if path.is_dir())
            apps.extend(path for path in root.glob("*/*.app") if path.is_dir())
        except OSError:
            continue
    return apps[:400]


def _bundle_id_for_app(path: Path) -> str:
    plist = path / "Contents" / "Info.plist"
    if not plist.exists():
        return ""
    try:
        with plist.open("rb") as handle:
            payload = plistlib.load(handle)
    except Exception:
        return ""
    return str(payload.get("CFBundleIdentifier") or "").strip()


def _aliases_for_app(app: str, bundle_id: str = "") -> list[str]:
    aliases = [app]
    aliases.extend(re.split(r"[^A-Za-z0-9\u4e00-\u9fff]+", app))
    if bundle_id:
        pieces = [piece for piece in re.split(r"[^A-Za-z0-9\u4e00-\u9fff]+", bundle_id) if piece and piece.lower() not in {"com", "app", "mac", "global"}]
        aliases.extend(pieces)
        for idx in range(len(pieces) - 1):
            aliases.append(" ".join(pieces[idx: idx + 2]))
    return [alias for alias in aliases if alias and len(_compact_key(alias)) >= 2]


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _suggested(action: str, args: dict[str, Any], confidence: float, reason: str, *, tool: str = "") -> list[dict[str, Any]]:
    if not action:
        return []
    tool_call = _tool_call_few_shot(tool, action, args) if tool else f"{action}({json.dumps(args, ensure_ascii=False)})"
    item: dict[str, Any] = {
        "tool_call": tool_call,
        "desc": reason,
        "confidence": round(confidence, 2),
    }
    answer = _probe_answer(tool, action, args, tool_call)
    if answer is not None and answer.get("ok") is not False:
        item["answer"] = answer
    return [item]


def _tool_call_few_shot(tool: str, action: str, args: dict[str, Any]) -> str:
    if tool == "computer_action":
        target = str(args.get("app") or args.get("target") or "")
        extra = args.get("args") if isinstance(args.get("args"), dict) else {}
        if target:
            return f'computer_action(action="{action}", target="{target}", args={json.dumps(extra, ensure_ascii=False)})'
        return f'computer_action(action="{action}", target="", args={json.dumps(extra or args, ensure_ascii=False)})'
    if tool == "daily_action":
        target = str(args.get("target") or "")
        extra = args.get("args") if isinstance(args.get("args"), dict) else {}
        return f'daily_action(action="{action}", target="{target}", args={json.dumps(extra, ensure_ascii=False)})'
    return f'{tool}({json.dumps(args, ensure_ascii=False)})'


def _probe_answer(tool: str, action: str, args: dict[str, Any], tool_call: str) -> dict[str, Any] | None:
    if not _probe_answer_supported(tool, action):
        return None
    new_future: Future[dict[str, Any] | None] | None = None
    with _ANSWER_LOCK:
        cached = _ANSWER_CACHE.get(tool_call)
        if cached is not None:
            return cached
        future = _ANSWER_INFLIGHT.get(tool_call)
        if future is None:
            future = _ANSWER_EXECUTOR.submit(_compute_probe_answer, tool, action, args)
            _ANSWER_INFLIGHT[tool_call] = future
            new_future = future
    if new_future is not None:
        new_future.add_done_callback(lambda done, key=tool_call: _cache_probe_answer(key, done))
    try:
        result = future.result(timeout=PROBE_ANSWER_TIMEOUT_BY_ACTION.get(action, PROBE_ANSWER_TIMEOUT_SECONDS))
    except TimeoutError:
        return None
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:240]}
    if result is None:
        return None
    if isinstance(result, dict) and result.get("ok") is False:
        with _ANSWER_LOCK:
            _ANSWER_INFLIGHT.pop(tool_call, None)
        return None
    with _ANSWER_LOCK:
        _ANSWER_CACHE[tool_call] = result
    return result


def _cache_probe_answer(tool_call: str, future: Future[dict[str, Any] | None]) -> None:
    try:
        result = future.result()
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:240]}
    with _ANSWER_LOCK:
        _ANSWER_INFLIGHT.pop(tool_call, None)
        if result is not None and not (isinstance(result, dict) and result.get("ok") is False):
            _ANSWER_CACHE[tool_call] = result


def _probe_answer_supported(tool: str, action: str) -> bool:
    if tool == "daily_action" and action in {"weather", "time", "map"}:
        return True
    return False


def _weather_query_candidates(location: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(location or "")).strip(" ：:，,。.!！?？")
    if not value:
        return []
    candidates = [value]
    parts = [part.strip() for part in re.split(r"[,，]", value) if part.strip()]
    if len(parts) >= 3:
        city_index = -3 if len(parts) >= 4 else 1
        candidates.append(", ".join(parts[city_index:]))
        candidates.append(", ".join([parts[city_index], parts[-1]]))
        candidates.append(parts[city_index])
    if len(parts) >= 2:
        candidates.append(parts[-2])
    deduped: list[str] = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _compute_probe_answer(tool: str, action: str, args: dict[str, Any]) -> dict[str, Any] | None:
    if tool != "daily_action":
        return None
    target = str(args.get("target") or "").strip()
    extra = args.get("args") if isinstance(args.get("args"), dict) else {}
    if action == "time":
        return {
            "ok": True,
            "local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
    if action == "map":
        mode = str(extra.get("mode") or "query")
        if mode == "route":
            query = {"daddr": target}
        elif mode == "current_address":
            current = current_address(timeout_seconds=2.0)
            return {"ok": bool(current.get("ok")), "mode": mode, **current}
        else:
            query = {"q": target}
        encoded = parse.urlencode({key: value for key, value in query.items() if value})
        return {"ok": bool(encoded), "mode": mode, "target": target, "maps_url": f"maps://?{encoded}", "web_url": f"https://maps.apple.com/?{encoded}"}
    if action == "weather":
        if not target:
            return {"ok": False, "error": "location is required"}
        if not plausible_weather_location(target):
            return {"ok": False, "location": target, "error": "invalid weather location"}
        errors: list[dict[str, str]] = []
        for candidate in _weather_query_candidates(target):
            try:
                url = f"https://wttr.in/{parse.quote(candidate)}?format=j1"
                req = request.Request(url, headers={"User-Agent": "GjallarhornVoiceProbe/0.1"})
                with request.urlopen(req, timeout=4.0) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
                current = (payload.get("current_condition") or [{}])[0]
                nearest = (payload.get("nearest_area") or [{}])[0]
                area = ((nearest.get("areaName") or [{}])[0].get("value") if nearest else "") or candidate
                country = ((nearest.get("country") or [{}])[0].get("value") if nearest else "") or ""
                result = {
                    "ok": True,
                    "location": candidate,
                    "resolved_location": ", ".join(part for part in [area, country] if part),
                    "temperature_c": current.get("temp_C"),
                    "feels_like_c": current.get("FeelsLikeC"),
                    "humidity_percent": current.get("humidity"),
                    "wind_kmph": current.get("windspeedKmph"),
                    "description": ((current.get("weatherDesc") or [{}])[0].get("value") if current else ""),
                }
                if candidate != target:
                    result["location_input"] = target
                    result["location_fallback"] = candidate
                return result
            except Exception as exc:
                errors.append({"location": candidate, "error": str(exc)[:160]})
        return {"ok": False, "location": target, "errors": errors, "error": errors[-1]["error"] if errors else "location not found"}
    return None


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for action in actions:
        key = (
            str(action.get("tool_call") or ""),
            str(action.get("desc") or ""),
            json.dumps(action.get("answer") or {}, ensure_ascii=False, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped[:8]


def _folder_from_text(text: str) -> str:
    compact = _compact_key(text)
    lowered = text.lower()
    if "download" in lowered or "downloads" in lowered or "下载" in compact:
        return "~/Downloads"
    if "desktop" in lowered or "桌面" in compact:
        return "~/Desktop"
    return ""


def _display_from_text(text: str) -> str:
    compact = _compact_key(text)
    lowered = text.lower()
    if any(token in compact for token in ["外接屏幕", "外接屏", "副屏", "第二屏", "第二个屏幕"]) or "external display" in lowered:
        return "external"
    if any(token in compact for token in ["主屏", "主显示器"]):
        return "primary"
    match = re.search(r"(?:display|screen|显示器|屏幕)\s*([0-9]+)", lowered)
    if match:
        return match.group(1)
    return ""


def _extract_file_query(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = [
        r"(?:downloads?|下载|download\s*文件|下载文件|下载目录|download\s*目录)(?:\s*文件)?(?:里|中|里面|文件里|目录里)?(?:\s*(?:打开|播放|找))?\s*(.+?)(?:，|,|然后|并且|再|$)",
        r"(?:打开|播放)\s*(.+?)(?:，|,|然后|并且|再|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ：:，,。 ")
            value = re.sub(r"^(?:文件里|目录里|里|中|里面)?(?:打开|播放|找)\s*", "", value, flags=re.IGNORECASE).strip(" ：:，,。 ")
            value = re.sub(r"(?:移到|移动到|放到|挪到).*$", "", value).strip(" ：:，,。 ")
            if value and not _looks_like_generic_target(value):
                return value
    return ""


def _looks_like_generic_target(value: str) -> bool:
    compact = _compact_key(value)
    return compact in {"app", "应用", "文件", "目录", "文件夹", "窗口", "屏幕"}


def _strip_command_words(text: str) -> str:
    value = re.sub(
        r"(帮我|给我|查查|查一下|搜索|搜一下|看看|打开|播放|记一下|记下|记录一下|记录|写一下|写下|写到便签|记到便签|放到note|放到 note|"
        r"please|look up|search for|search|find|google|open|launch|start|play|take a note|write down|jot down)",
        " ",
        text,
        flags=re.I,
    )
    value = re.sub(r"(?:不是|不对|不要|别)\s*.+?(?:随便一个|任意一个|哪个都行|都可以|就行|即可|$)", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" ，。,.")


def weather_location_from_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    english_location = _english_weather_location_from_text(value)
    if english_location:
        return _normalize_weather_location(english_location)
    value = re.split(
        r"(?:今天|明天|后天|今晚|明早|明天下午|明天上午|明天晚上|上午|下午|晚上|早上|中午)?(?:记得)?(?:提醒我|提醒|设个提醒|加个提醒|闹钟|写到|写下|记住|记得)",
        value,
        maxsplit=1,
    )[0]
    value = re.sub(
        r"(weather|forecast|temperature|today|current|now|recent|latest|"
        r"天气|气温|温度|预报|几度|多少度|多少|体感|冷不冷|热不热|会冷|会热|"
        r"下雨|降雨|雨|湿度|风速|刮风|风大|空气质量|空气|雾霾|紫外线|穿什么|带伞|要不要带伞|"
        r"怎么样|如何|查查|查一下|查询|看看|看一下|帮我|给我|"
        r"今天|今日|明天|后天|周末|现在|当前|目前|最近|最新|当地|这边|这里|那里|一下|吗|呢)",
        " ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip(" ：:，,。.!！?？")
    return _normalize_weather_location(value)


def _english_weather_location_from_text(text: str) -> str:
    raw = str(text or "").strip()
    patterns = [
        r"\b(?:weather|forecast|temperature|rain|humidity|wind|uv)\s+(?:in|for|at|near)\s+(.+)$",
        r"\b(?:what(?:'s| is)?|how(?:'s| is)?)\s+(?:the\s+)?(?:weather|temperature|forecast)\s+(?:like\s+)?(?:in|for|at|near)\s+(.+)$",
        r"\b(?:is it|will it)\s+(?:rain(?:ing)?|cold|hot|windy)\s+(?:in|at|near)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        value = re.sub(r"\b(?:today|tomorrow|now|currently|this week|this weekend|please|thanks)\b", " ", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip(" ：:，,。.!！?？")
        if value and not _looks_like_generic_target(value):
            return value[:80]
    return ""


def _normalize_weather_location(location: str) -> str:
    value = re.sub(r"\s+", " ", str(location or "")).strip(" ：:，,。.!！?？")
    if not value:
        return ""
    compact = _compact_key(value)
    aliases = {
        "上保楼": "圣保罗",
        "上保罗": "圣保罗",
        "圣宝罗": "圣保罗",
        "秦宝洛": "圣保罗",
        "秦保罗": "圣保罗",
        "saopaulo": "São Paulo",
        "sãopaulo": "São Paulo",
    }
    return aliases.get(compact.lower(), aliases.get(compact, value))[:40]
