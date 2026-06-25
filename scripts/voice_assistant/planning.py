from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Any
from urllib import parse

from voice_assistant.json_utils import coerce_json_object, coerce_json_value
from voice_assistant.domain_probe import weather_location_from_text
from voice_assistant.local_actions import (
    camera_app_open_requested,
    camera_capture_requested,
    front_note_requested,
    normalize_front_note_call_args,
    photo_booth_app_open_requested,
    planned_arrange_workspace_arguments,
    workspace_arrange_requested,
    workspace_terms_are_pronouns,
)
from voice_assistant.speech_text import compact_speech_text, strip_think_blocks
from voice_assistant.tool_speech import DEFAULT_TOOL_TASK_LABELS

CONCURRENT_PREFETCH_TOOLS = {
    "web_search",
    "search_news",
    "fetch_url",
    "daily_action",
    "get_weather",
    "current_datetime",
    "system_status",
}

DAILY_ACTION_DONE_CONDITIONS = {
    "weather": "查到天气",
    "time": "查到时间",
    "map": "拿到地图信息",
    "calendar_list": "查到日程",
    "reminder_list": "查到提醒",
    "reminder_create": "创建提醒",
    "note_live": "写入便签",
    "note_context": "写入上下文",
    "memory": "记住信息",
}


def execution_plan_from_domain_probe(domain_probe: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(domain_probe, dict):
        return None
    steps: list[dict[str, Any]] = []
    source_text = str(domain_probe.get("input") or "")
    for domain in domain_probe.get("domains") or []:
        if not isinstance(domain, dict) or float(domain.get("confidence") or 0.0) < 0.8:
            continue
        for suggested in domain.get("suggested_actions") or []:
            if not isinstance(suggested, dict):
                continue
            if float(suggested.get("confidence") or domain.get("confidence") or 0.0) < 0.8:
                continue
            parsed = _parse_probe_tool_call(str(suggested.get("tool_call") or ""))
            if not parsed:
                continue
            tool_name, arguments = parsed
            action = str(arguments.get("action") or "").strip()
            target = str(arguments.get("target") or "").strip()
            intent = _probe_plan_intent(tool_name, arguments)
            steps.append(
                {
                    "order": len(steps) + 1,
                    "intent": intent,
                    "kind": "tool",
                    "user_visible": False,
                    "depends_on": [],
                    "suggested_tools": [tool_name],
                    "arguments": {tool_name: arguments},
                    "done_condition": DAILY_ACTION_DONE_CONDITIONS.get(action, "完成这一步") if tool_name == "daily_action" else "完成这一步",
                    "source_tool_call": suggested.get("tool_call"),
                    "source_desc": suggested.get("desc") or "",
                    "target": target,
                }
            )
            if tool_name == "web_search" and _source_requests_open_url(source_text):
                steps.append(
                    {
                        "order": len(steps) + 1,
                        "intent": "打开网页",
                        "kind": "tool",
                        "user_visible": False,
                        "depends_on": [len(steps)],
                        "suggested_tools": ["open_url_in_browser"],
                        "arguments": {"open_url_in_browser": {"fullscreen": False, "video_fullscreen": False}},
                        "done_condition": "打开搜索结果",
                        "source_tool_call": "open_url_in_browser",
                        "source_desc": "open URL returned by current-turn search result",
                        "target": "",
                    }
                )
    if not steps:
        return None
    if source_text:
        steps.sort(key=lambda step: _probe_step_source_position(step, source_text))
        for idx, step in enumerate(steps, start=1):
            step["order"] = idx
    return {"steps": steps, "source": "domain_probe"}


def _probe_step_source_position(step: dict[str, Any], source_text: str) -> int:
    source = str(source_text or "").lower()
    args_by_tool = step.get("arguments") if isinstance(step.get("arguments"), dict) else {}
    tool_name = (step.get("suggested_tools") or [""])[0]
    args = args_by_tool.get(tool_name) if isinstance(args_by_tool, dict) else {}
    if not isinstance(args, dict):
        return 10_000
    action = str(args.get("action") or "").lower()
    target = str(args.get("target") or "").lower()
    query = str(args.get("query") or "").lower()
    candidates: list[str] = []
    if tool_name in {"web_search", "search_news"}:
        candidates.extend(["搜索", "搜", "查", "视频", "网页", "链接", "search", "find", "look up", "video", "web", "link"])
        if query:
            candidates.append(query)
    if target:
        candidates.append(target)
        if target.endswith("s"):
            candidates.append(target[:-1])
    if action == "weather":
        candidates.extend(["天气", "weather"])
    elif action == "reminder_create":
        candidates.extend(["提醒我", "提醒", "reminder"])
    elif action == "open_app":
        candidates.extend(["打开", "open", "launch"])
    positions = [source.find(candidate) for candidate in candidates if candidate and source.find(candidate) >= 0]
    return min(positions) if positions else 10_000


def _parse_probe_tool_call(tool_call: str) -> tuple[str, dict[str, Any]] | None:
    text = str(tool_call or "").strip()
    if text.startswith("web_search(") or text.startswith("search_news("):
        payload_match = re.search(r"\((\{.*\})\)\s*$", text)
        if not payload_match:
            return None
        try:
            payload = json.loads(payload_match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        tool_name = "web_search" if text.startswith("web_search(") else "search_news"
        return tool_name, payload
    if not (text.startswith("daily_action(") or text.startswith("computer_action(")):
        return None
    action_match = re.search(r'action="([^"]*)"', text)
    target_match = re.search(r'target="([^"]*)"', text)
    args_match = re.search(r"args=(\{.*\})\s*\)$", text)
    if not action_match:
        return None
    args: dict[str, Any] = {}
    if args_match:
        try:
            parsed = json.loads(args_match.group(1))
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            args = {}
    return (
        "daily_action" if text.startswith("daily_action(") else "computer_action",
        {
            "action": action_match.group(1),
            "target": target_match.group(1) if target_match else "",
            "args": args,
        },
    )


def _probe_plan_intent(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name in {"web_search", "search_news"}:
        query = compact_speech_text(str(arguments.get("query") or ""))
        return f"搜索{query[:12]}" if query else "搜索"
    if tool_name == "open_url_in_browser":
        return "打开网页"
    action = str(arguments.get("action") or "").strip()
    target = compact_speech_text(str(arguments.get("target") or ""))
    if tool_name == "computer_action":
        label = {
            "open_app": "打开应用",
            "close_app": "关闭应用",
            "focus_app": "切到应用",
            "computer_use": "操作电脑",
            "open_file": "打开文件",
            "open_file_and_move_to_display": "打开文件",
            "move_window_to_display": "移动窗口",
        }.get(action, "电脑操作")
        return f"{label}{target[:12]}" if target else label
    if tool_name != "daily_action":
        return tool_name
    label = {
        "weather": "查天气",
        "time": "查时间",
        "map": "看地图",
        "calendar_list": "看日程",
        "reminder_list": "看提醒",
        "reminder_create": "设提醒",
        "note_live": "写便签",
        "note_context": "写上下文",
        "memory": "记住",
    }.get(action, "处理日常")
    return f"{label}{target[:12]}" if target else label


def _source_requests_open_url(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", str(source_text or "").lower())
    return any(token in compact for token in ["打开", "播放", "我看一下", "放", "open", "play", "网页", "链接", "视频", "video", "mv", "mtv"])

def tool_action_subject(tool_name: str, arguments: dict[str, Any], result: Any, user_text: str = "") -> str:
    name = tool_name.split(":", 1)[-1]
    value = coerce_json_value(result)
    args = arguments or {}
    if name in {"web_search", "search_news"}:
        return search_action_subject(str(args.get("query") or ""), user_text)
    if name == "get_weather":
        location = compact_speech_text(str(args.get("location") or ""))
        return f"{location[:8]}天气" if location else "天气"
    if name == "daily_action":
        action = str(args.get("action") or "").strip()
        target = compact_speech_text(str(args.get("target") or ""))
        if action == "weather":
            return f"{target[:8]}天气" if target else "天气"
        return {"time": "时间", "map": target[:8] or "地图", "calendar_list": "日程", "reminder_list": "提醒", "reminder_create": "提醒", "note_live": "便签", "note_context": "上下文", "memory": "记忆"}.get(action, "日常")
    if name == "current_datetime":
        return "时间"
    if name == "system_status":
        return "机器状态"
    if name == "open_url_in_browser":
        url = str(args.get("url") or (value.get("url") if isinstance(value, dict) else "") or "")
        lower = url.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg")):
            return "图片"
        parsed = parse.urlparse(url)
        host = parsed.netloc.replace("www.", "")
        if "youtube" in host or "youtu.be" in host:
            return "YouTube"
        if host:
            return host.split(".")[0][:10]
        return "网页"
    if name == "fetch_url":
        url = str(args.get("url") or "")
        host = parse.urlparse(url).netloc.replace("www.", "")
        return host.split(".")[0][:10] if host else "网页"
    if name == "capture_camera_snapshot":
        return "相机"
    if name == "calendar_events":
        return "日程"
    if name == "reminders_list":
        return "提醒"
    if name == "mail_message":
        return compact_speech_text(str(args.get("to") or ""))[:10] or "邮件"
    if name in {"update_task_status", "mark_task_in_progress", "mark_task_blocked", "mark_task_done"}:
        return extract_search_subject(str(args.get("title") or user_text or ""))[:10]
    if name == "write_text_file":
        path = str(args.get("path") or "")
        return Path(path).name[:10] if path else "文件"
    if name in {"read_path", "list_path"}:
        path = str(args.get("path") or "")
        return Path(path).name[:10] if path else "文件"
    if name == "run_osascript":
        return extract_search_subject(str(args.get("purpose") or user_text or ""))[:10]
    if name == "arrange_workspace":
        return "窗口"
    if name == "front_note":
        return "便签"
    if name == "launch_python_script":
        path = str(args.get("path") or "")
        return Path(path).name[:10] if path else "脚本"
    return extract_search_subject(user_text)[:10]


def extract_search_subject(text: str) -> str:
    text = html.unescape(str(text or "")).strip()
    text = re.sub(r"https?://\S+", " ", text)
    parts = [p for p in re.split(r"[\s，,。.!！?？、；;：:《》<>「」『』\"'“”‘’()（）]+", text) if p]
    stop = {
        "最近", "最新", "今天", "现在", "当前", "目前", "recent", "latest", "current", "today",
        "查", "查查", "查询", "搜索", "搜", "搜一下", "看", "看看", "看一下", "帮我", "请",
        "干了什么", "怎么样", "怎么了", "是什么", "为什么", "新闻", "网页", "官网", "链接", "比分", "结果",
    }
    for part in parts:
        compact = cleanup_search_subject(compact_speech_text(part))
        if not compact or compact.lower() in stop:
            continue
        cleaned = cleanup_search_subject(compact)
        for token in ["最近", "最新", "今天", "现在", "当前", "目前", "干了什么", "怎么样", "新闻", "网页", "官网", "比分", "结果"]:
            cleaned = cleaned.replace(token, "")
        cleaned = cleanup_search_subject(cleaned)
        if cleaned and cleaned.lower() not in stop:
            return cleaned[:10]
    compact = cleanup_search_subject(compact_speech_text(text))
    for token in stop:
        compact = compact.replace(token, "")
    return cleanup_search_subject(compact)[:10]


def cleanup_search_subject(text: str) -> str:
    text = compact_speech_text(text)
    if not text:
        return ""
    if "特朗普" in text or "川普" in text or re.search(r"trump", text, flags=re.IGNORECASE):
        return "特朗普"
    text = re.sub(r"^(帮我|请|麻烦|查查|查一下|查询|搜索|搜一下|搜|查|看看|看一下|看)", "", text)
    text = re.sub(r"(最近|最新|今天|现在|当前|目前)$", "", text)
    text = re.sub(r"(最近|最新|今天|现在|当前|目前)", "", text)
    text = re.sub(r"(干了什么|干了啥|在干嘛|怎么样|怎么了|是什么|有什么新闻|新闻)$", "", text)
    return text.strip()


def search_action_subject(query: str, user_text: str = "") -> str:
    query_subject = extract_search_subject(query)
    if query_subject:
        return query_subject
    user_subject = extract_search_subject(user_text)
    return user_subject


def planned_search_query(text: str, kind: str = "web", user_text: str = "") -> str:
    raw = str(text or "").strip()
    fallback = str(user_text or "").strip()
    source = raw or fallback
    if not source:
        return ""
    compact = compact_speech_text(source).lower()
    now = dt.datetime.now().astimezone()
    month_year = f"{now.strftime('%B')} {now.year}"
    already_structured = bool(
        re.search(r"\b(latest|recent|current|today|news|score|price|weather)\b", raw, flags=re.IGNORECASE)
        and re.search(r"\b20\d{2}\b", raw)
        and not re.search(r"[\u4e00-\u9fff]", raw)
    )
    if already_structured:
        return raw
    subject = extract_search_subject(source) or extract_search_subject(fallback)
    if not subject:
        return raw or fallback
    alias_map = {
        "特朗普": "Trump",
        "川普": "Trump",
    }
    query_subject = alias_map.get(subject, subject)
    wants_news = kind == "news" or any(token in compact for token in ["最近", "最新", "今天", "现在", "当前", "目前", "新闻", "recent", "latest", "current", "today", "news"])
    if wants_news:
        if re.search(r"[\u4e00-\u9fff]", query_subject):
            return f"{query_subject} 最新消息 {now.year}"
        return f"{query_subject} latest news {month_year}"
    if any(token in compact for token in ["比分", "结果", "score", "scores", "result"]):
        return f"{query_subject} latest results {month_year}"
    return query_subject


def news_search_retry_query(query: str) -> str:
    normalized = planned_search_query(query, kind="news")
    return normalized if normalized and normalized != str(query or "").strip() else ""


def tool_prefetch_key(tool_name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    name = tool_name.split(":", 1)[-1]
    normalized = coerce_json_object(arguments)
    return name, json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def format_plan_summary_for_voice(plan_payload: dict[str, Any] | None, *, max_steps: int = 3, max_chars: int = 42) -> str:
    if not isinstance(plan_payload, dict):
        return ""
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return ""
    intents: list[str] = []
    max_steps = max(1, int(max_steps or 3))
    max_chars = max(16, int(max_chars or 42))
    for step in sorted((s for s in steps if isinstance(s, dict)), key=lambda s: int(s.get("order") or 0)):
        kind = str(step.get("kind") or "").strip()
        if kind not in {"tool", "speak"}:
            continue
        intent = summarize_plan_intent_for_voice(str(step.get("intent") or "").strip(), step)
        if not intent:
            tools = step.get("suggested_tools")
            if isinstance(tools, list) and tools:
                intent = tool_plan_intent_label(str(tools[0]))
            elif isinstance(tools, str):
                intent = tool_plan_intent_label(tools)
        if intent and intent not in intents:
            intents.append(intent)
        if len(intents) >= max_steps:
            break
    if not intents:
        return ""
    markers = ["先", "再", "最后"]
    parts = [f"{markers[min(index, len(markers) - 1)]}{intent}" for index, intent in enumerate(intents)]
    text = "我先" + "，".join(parts).removeprefix("先") + "。"
    if len(text) <= max_chars:
        return text
    compact = "我先按顺序来。"
    if len(intents) <= 2:
        compact = "我先" + "，再".join(intents) + "。"
    return compact[:max_chars].rstrip("，,。") + "。"


def summarize_plan_intent_for_voice(intent: str, step: dict[str, Any]) -> str:
    tools = plan_step_tools(step)
    first_tool = tools[0] if tools else ""
    raw = sanitize_plan_intent(intent)
    text = compact_speech_text(raw).lower()
    if any(token in text for token in ["youtube", "视频", "音乐视频", "mtv"]):
        return "找视频"
    if any(token in text for token in ["photobooth", "camera", "相机", "摄像头"]):
        return "开相机"
    if any(token in text for token in ["并排", "focusscreen", "focus", "前台", "窗口", "分屏"]):
        return "排窗口"
    if any(token in text for token in ["搜索", "查询", "查查", "查一下", "最近", "最新"]):
        return "查资料"
    if first_tool:
        mapped = tool_plan_intent_label(first_tool)
        if mapped and mapped != first_tool:
            return mapped
    return raw[:8] if raw else ""


def heuristic_execution_plan(user_text: str) -> dict[str, Any]:
    text = str(user_text or "").strip()
    if not text:
        return {"steps": []}
    normalized = re.sub(r"\s+", "", text)
    raw_parts = [part for part in re.split(r"(?:先|然后|接下来|接着|再|最后|同时|并且|，|,|。|；|;)", normalized) if part]
    parts = raw_parts or [normalized]
    steps: list[dict[str, Any]] = []
    for part in parts[:6]:
        step = heuristic_plan_step(part, len(steps) + 1)
        if step:
            steps.append(step)
    full_arrange_args = planned_arrange_workspace_arguments(text) if workspace_arrange_requested(text) else {}
    if full_arrange_args:
        full_terms = full_arrange_args.get("app_names") or []
        has_arrange_step = False
        for step in steps:
            if "arrange_workspace" not in plan_step_tools(step):
                continue
            has_arrange_step = True
            args = step.get("arguments")
            if not isinstance(args, dict):
                args = {}
                step["arguments"] = args
            existing = args.get("arrange_workspace")
            if not isinstance(existing, dict):
                args["arrange_workspace"] = full_arrange_args
            elif (not existing.get("app_names") or workspace_terms_are_pronouns(existing.get("app_names"))) and full_terms:
                existing["app_names"] = full_terms
                existing["query"] = existing.get("query") or text
                existing["mode"] = existing.get("mode") or full_arrange_args.get("mode") or "auto"
        if not has_arrange_step:
            steps.append({
                "order": len(steps) + 1,
                "intent": "排窗口",
                "kind": "tool",
                "user_visible": False,
                "depends_on": [],
                "suggested_tools": ["arrange_workspace"],
                "arguments": {"arrange_workspace": full_arrange_args},
                "done_condition": "排好窗口",
            })
        steps = [
            step for step in steps
            if plan_step_tools(step) or str(step.get("kind") or "") != "tool" or str(step.get("done_condition") or "") != "完成这一步"
        ]
    if not steps:
        steps.append({
            "order": 1,
            "intent": sanitize_plan_intent(text) or "处理问题",
            "kind": "tool",
            "user_visible": False,
            "depends_on": [],
            "suggested_tools": [],
            "arguments": {},
            "done_condition": "完成用户请求",
        })
    return {"steps": steps, "source": "heuristic"}


def heuristic_plan_is_confident(plan_payload: dict[str, Any] | None) -> bool:
    if not isinstance(plan_payload, dict):
        return False
    steps = plan_payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    useful_steps = 0
    for step in steps:
        if not isinstance(step, dict):
            return False
        tools = plan_step_tools(step)
        kind = str(step.get("kind") or "").strip()
        if kind == "tool" and tools:
            useful_steps += 1
            continue
        if kind == "speak":
            useful_steps += 1
            continue
        if not tools:
            return False
    return useful_steps > 0


def heuristic_plan_step(text: str, order: int) -> dict[str, Any] | None:
    intent = sanitize_plan_intent(text)
    if not intent:
        return None
    compact = compact_speech_text(text).lower()
    tools: list[str] = []
    args: dict[str, Any] = {}
    kind = "tool"
    done = "完成这一步"
    is_camera_capture = camera_capture_requested(compact)
    is_camera_open = camera_app_open_requested(compact)
    is_photo_booth_open = photo_booth_app_open_requested(compact)
    if front_note_requested(compact):
        tools.append("front_note")
        args["front_note"] = normalize_front_note_call_args({}, text)
        kind = "tool"
        done = "更新前端便签"
        return {
            "order": order,
            "intent": intent,
            "kind": kind,
            "user_visible": True,
            "depends_on": [],
            "suggested_tools": tools,
            "arguments": args,
            "done_condition": done,
        }
    if any(token in compact for token in ["总结", "简述", "说一下", "告诉我", "回答"]):
        kind = "speak"
        done = "说出阶段性结果"
    if any(token in compact for token in ["查", "搜索", "搜", "最近", "最新", "新闻", "资料"]):
        tools.extend(["web_search", "search_news"])
        args["web_search"] = {"query": planned_search_query(text, kind="web"), "max_results": 5}
        args["search_news"] = {"query": planned_search_query(text, kind="news"), "max_results": 5}
        kind = "tool"
        done = "查到可用信息"
    wants_video_or_web = not is_camera_capture and any(token in compact for token in ["打开", "播放", "网页", "链接", "youtube", "视频", "全屏", "mtv", "ymca"])
    if workspace_arrange_requested(compact):
        tools.append("arrange_workspace")
        args["arrange_workspace"] = planned_arrange_workspace_arguments(text)
        kind = "tool"
        done = "排好窗口"
    if (is_camera_open or is_photo_booth_open) and not is_camera_capture:
        app_name = "Photo Booth" if is_photo_booth_open else "Camera"
        tools.append("run_osascript")
        args["run_osascript"] = {"script": f'tell application "{app_name}" to activate', "timeout_seconds": 10, "purpose": f"打开 {app_name} 应用"}
        kind = "tool"
        done = "打开相机应用"
    if wants_video_or_web and not (is_camera_open and not any(token in compact for token in ["youtube", "视频", "video", "musicvideo", "mtv", "ymca", "网页", "链接"])):
        if not re.search(r"https?://", text) and any(token in compact for token in ["youtube", "视频", "video", "musicvideo", "mtv", "ymca"]):
            tools.append("web_search")
            args["web_search"] = {"query": video_search_query_from_text(text), "max_results": 5}
        tools.append("open_url_in_browser")
        kind = "tool"
        done = "打开目标内容"
    if is_camera_capture:
        tools.append("capture_camera_snapshot")
        kind = "tool"
        done = "拍到一张照片"
    if any(token in compact for token in ["天气", "温度"]):
        tools.append("daily_action")
        location = extract_weather_location(text)
        args["daily_action"] = {"action": "weather", "target": location, "args": {}}
        kind = "tool"
        done = "查到天气"
    seen: set[str] = set()
    tools = [tool for tool in tools if not (tool in seen or seen.add(tool))]
    return {
        "order": order,
        "intent": intent,
        "kind": kind,
        "user_visible": kind == "speak",
        "depends_on": [],
        "suggested_tools": tools,
        "arguments": args,
        "done_condition": done,
    }


def sanitize_plan_intent(text: str) -> str:
    text = strip_think_blocks(text)
    text = re.sub(r"https?://\S+", "链接", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^(我会|需要|计划|步骤|先|然后|接着|再|最后)", "", text)
    text = re.sub(r"(。|，|,|；|;|\.|!|！|\?|？)+$", "", text)
    replacements = {
        "使用web_search": "查网页",
        "使用search_news": "查新闻",
        "调用web_search": "查网页",
        "调用search_news": "查新闻",
        "trigger_fast_followup": "说结果",
        "open_url_in_browser": "打开网页",
        "capture_camera_snapshot": "拍照",
        "run_osascript": "操作电脑",
        "arrange_workspace": "排窗口",
        "front_note": "贴便签",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text[:18]


def tool_plan_intent_label(tool_name: str) -> str:
    name = tool_name.split(":", 1)[-1]
    labels = {
        "web_search": "查网页",
        "search_news": "查新闻",
        "fetch_url": "看网页",
        "get_weather": "看天气",
        "daily_action": "看日常",
        "open_url_in_browser": "打开网页",
        "run_osascript": "操作电脑",
        "capture_camera_snapshot": "拍照",
        "trigger_fast_followup": "说结果",
    }
    return labels.get(name, DEFAULT_TOOL_TASK_LABELS.get(name, "处理"))


def planned_prefetch_calls(plan_payload: dict[str, Any] | None, user_text: str) -> list[dict[str, Any]]:
    if not isinstance(plan_payload, dict):
        return []
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []
    calls: list[dict[str, Any]] = []
    for step in sorted((s for s in steps if isinstance(s, dict)), key=lambda s: int(s.get("order") or 0)):
        if str(step.get("kind") or "").strip() != "tool":
            continue
        suggested = step.get("suggested_tools")
        if isinstance(suggested, str):
            tool_names = [suggested]
        elif isinstance(suggested, list):
            tool_names = [str(name) for name in suggested if str(name or "").strip()]
        else:
            tool_names = []
        for tool_name in tool_names:
            name = tool_name.split(":", 1)[-1]
            if name not in CONCURRENT_PREFETCH_TOOLS:
                continue
            args = planned_tool_arguments(name, step, user_text)
            if args is None:
                continue
            calls.append({"tool_name": name, "arguments": args})
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        key = tool_prefetch_key(call["tool_name"], call["arguments"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def planned_tool_arguments(tool_name: str, step: dict[str, Any], user_text: str) -> dict[str, Any] | None:
    raw_arguments = step.get("arguments")
    if isinstance(raw_arguments, dict):
        direct = raw_arguments.get(tool_name)
        if isinstance(direct, dict):
            if tool_name == "front_note":
                return dict(direct)
            if tool_name == "daily_action":
                return sanitize_prefetch_arguments(tool_name, direct, user_text=user_text)
            return sanitize_prefetch_arguments(tool_name, direct, user_text=user_text)
        if all(key in raw_arguments for key in ["query"]) or all(key in raw_arguments for key in ["url"]) or all(key in raw_arguments for key in ["location"]):
            return sanitize_prefetch_arguments(tool_name, raw_arguments, user_text=user_text)
    intent = str(step.get("intent") or "").strip()
    source_text = intent or user_text
    if tool_name in {"web_search", "search_news"}:
        query = planned_search_query(source_text, kind="news" if tool_name == "search_news" else "web", user_text=user_text)
        return {"query": query, "max_results": 5} if query else None
    if tool_name == "fetch_url":
        match = re.search(r"https?://[^\s，,。!！?？、；;]+", source_text or user_text)
        return {"url": match.group(0), "max_chars": 12000} if match else None
    if tool_name in {"current_datetime", "system_status"}:
        return {}
    if tool_name == "daily_action":
        return daily_action_arguments_from_text(source_text or user_text)
    if tool_name == "get_weather":
        location = extract_weather_location(source_text or user_text)
        return {"location": location} if location else None
    if tool_name == "front_note":
        return normalize_front_note_call_args({}, source_text or user_text)
    return None


def sanitize_prefetch_arguments(tool_name: str, arguments: dict[str, Any], user_text: str = "") -> dict[str, Any] | None:
    if tool_name in {"web_search", "search_news"}:
        raw_query = str(arguments.get("query") or "").strip()
        query = planned_search_query(raw_query, kind="news" if tool_name == "search_news" else "web", user_text=user_text)
        if not query:
            return None
        return {"query": query, "max_results": max(1, min(int(arguments.get("max_results") or 5), 10))}
    if tool_name == "fetch_url":
        url = str(arguments.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return None
        return {"url": url, "max_chars": max(1, min(int(arguments.get("max_chars") or 12000), 60000))}
    if tool_name == "get_weather":
        location = weather_location_from_text(str(arguments.get("location") or "") or user_text)
        return {"location": location} if location else None
    if tool_name == "daily_action":
        action = str(arguments.get("action") or "").strip()
        if action not in {"weather", "time", "map", "calendar_list", "reminder_list"}:
            return None
        target = str(arguments.get("target") or "").strip()
        args = arguments.get("args") if isinstance(arguments.get("args"), dict) else {}
        if action == "weather":
            target = weather_location_from_text(target or user_text)
            return {"action": "weather", "target": target, "args": {}} if target else None
        if action == "map":
            mode = str(args.get("mode") or "query")
            if mode not in {"query", "route", "current_address"}:
                mode = "query"
            return {"action": "map", "target": target, "args": {"mode": mode}} if target or mode == "current_address" else None
        return {"action": action, "target": target, "args": args}
    if tool_name in {"current_datetime", "system_status"}:
        return {}
    return None


def daily_action_arguments_from_text(text: str) -> dict[str, Any] | None:
    compact = compact_speech_text(str(text or "")).lower()
    if any(token in compact for token in ["天气", "温度", "weather"]):
        location = extract_weather_location(text)
        return {"action": "weather", "target": location, "args": {}} if location else None
    if any(token in compact for token in ["几点", "时间", "当前时间", "现在时间"]):
        return {"action": "time", "target": "", "args": {}}
    if any(token in compact for token in ["地图", "路线", "怎么走", "地址", "在哪", "哪里", "map", "route"]):
        mode = "route" if any(token in compact for token in ["路线", "怎么走", "route"]) else ("current_address" if any(token in compact for token in ["当前地址", "我在哪", "我在哪里"]) else "query")
        return {"action": "map", "target": "", "args": {"mode": mode}}
    if any(token in compact for token in ["日历", "calendar"]):
        return {"action": "calendar_list", "target": "", "args": {"days": 7}}
    if any(token in compact for token in ["提醒", "reminder"]):
        return {"action": "reminder_list", "target": "", "args": {}} if any(token in compact for token in ["看看", "列出", "有哪些", "list"]) else None
    return None


def extract_weather_location(text: str) -> str:
    text = str(text or "").strip()
    if not text or ("天气" not in text and not re.search(r"\bweather\b", text, flags=re.IGNORECASE)):
        return ""
    return weather_location_from_text(compact_speech_text(text))


def video_search_query_from_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "YouTube video"
    constraint = correction_constraint_from_text(raw)
    cleaned = re.sub(r"photo\s*booth", " ", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:不是|不对|不要|别)\s*.+?(?:随便一个|任意一个|哪个都行|都可以|就行|即可|$)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(打开|播放|然后|然后再|再|接着|把|他们|它们|都|排到前台|调到前台|前台|排窗口|并排|分屏|应用|app)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,，。")
    cleaned = re.sub(r"(?i)(ymca)", " YMCA ", cleaned)
    cleaned = re.sub(r"(?i)(musicvideo)", " Music Video ", cleaned)
    cleaned = re.sub(r"(?i)(youtube)", " YouTube ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,，。")
    if "ymca" in cleaned.lower() and not re.search(r"music\s*video|mtv|视频", cleaned, flags=re.IGNORECASE):
        cleaned = f"{cleaned} music video"
    cleaned = apply_search_negative_constraint(cleaned, constraint)
    if re.search(r"youtube|视频|video|music\s*video|mtv", cleaned, flags=re.IGNORECASE):
        return cleaned
    return (cleaned + " YouTube video").strip() if cleaned else "YouTube video"


def correction_constraint_from_text(text: str) -> dict[str, str]:
    raw = str(text or "")
    match = re.search(r"(?:不是|不对|不要|别)\s*(.+?)(?:，|,|。|\.|；|;|随便一个|任意一个|哪个都行|都可以|就行|即可|$)", raw, flags=re.IGNORECASE)
    if not match:
        return {}
    rejected = re.sub(r"^(打开|播放|放|找|搜索|搜|查)\s*", "", match.group(1).strip(), flags=re.IGNORECASE)
    rejected = re.sub(r"(的|那个|这个|版本|视频|链接|网页|mtv|music\s*video|youtube)+$", "", rejected, flags=re.IGNORECASE).strip(" ：:，,。 ")
    if not rejected:
        return {}
    return {"rejected": rejected[:40]}


def apply_search_negative_constraint(query: str, constraint: dict[str, str] | None) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    rejected = str((constraint or {}).get("rejected") or "").strip()
    if not text or not rejected:
        return text
    compact_rejected = compact_speech_text(rejected).lower()
    negative_terms: list[str] = []
    if any(token in compact_rejected for token in ["官方", "official"]):
        negative_terms.append("official")
    if rejected and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", rejected):
        negative_terms.append(rejected)
    suffix = " ".join(f"-{term}" for term in dict.fromkeys(negative_terms) if term and term.lower() not in text.lower())
    return f"{text} {suffix}".strip()


def tool_task_label(tool_name: str, arguments: dict[str, Any], result: Any, user_text: str = "") -> str:
    name = tool_name.split(":", 1)[-1]
    value = coerce_json_value(result)
    args = arguments or {}
    if name == "open_url_in_browser":
        url = str(args.get("url") or (value.get("url") if isinstance(value, dict) else "") or "")
        lower = url.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg")):
            return "图片"
        if any(host in lower for host in ["youtube", "bilibili", "vimeo", "twitch"]):
            return "视频"
        return "网页"
    if name in {"web_search", "search_news"}:
        query = str(args.get("query") or "").strip()
        compact = compact_speech_text(query)
        return compact[:6] if compact else DEFAULT_TOOL_TASK_LABELS.get(name, "搜索")
    if name == "fetch_url":
        return "网页"
    if name == "get_weather":
        location = compact_speech_text(str(args.get("location") or ""))
        return f"{location[:4]}天气" if location else "天气"
    if name == "daily_action":
        action = str(args.get("action") or "")
        target = compact_speech_text(str(args.get("target") or ""))
        if action == "weather":
            return f"{target[:4]}天气" if target else "天气"
        return {"time": "时间", "map": "地图", "calendar_list": "日程", "reminder_list": "提醒", "reminder_create": "提醒", "note_live": "便签", "memory": "记忆"}.get(action, "日常")
    if name == "write_text_file":
        path = str(args.get("path") or "")
        suffix = Path(path).suffix.lower()
        if suffix in {".svg", ".png", ".jpg", ".jpeg", ".webp"}:
            return "图片"
        return "写入"
    if name == "capture_camera_snapshot":
        return "拍照"
    if name == "arrange_workspace":
        return "窗口"
    if name == "run_osascript":
        purpose = compact_speech_text(str(args.get("purpose") or ""))
        if purpose:
            return purpose[:6]
    text = compact_speech_text(user_text)
    for token, label in [
        ("天气", "天气"),
        ("拍照", "拍照"),
        ("抓拍", "拍照"),
        ("相机", "拍照"),
        ("摄像头", "拍照"),
        ("分屏", "窗口"),
        ("并排", "窗口"),
        ("陀螺", "窗口"),
        ("排窗口", "窗口"),
        ("视频", "视频"),
        ("网页", "网页"),
        ("文件", "文件"),
        ("邮件", "邮件"),
        ("日程", "日程"),
        ("提醒", "提醒"),
    ]:
        if token in text:
            return label
    return DEFAULT_TOOL_TASK_LABELS.get(name, name[:6] or "工具")


def user_request_needs_local_action(text: str) -> bool:
    text = str(text or "").lower()
    action_tokens = [
        "打开",
        "播放",
        "关闭",
        "关掉",
        "全屏",
        "退出全屏",
        "按",
        "点击",
        "运行",
        "启动",
        "执行",
        "写入",
        "保存",
        "创建",
        "画",
        "窗口",
        "分屏",
        "并排",
        "陀螺",
        "排窗口",
        "顺序切",
        "浏览器",
        "网页",
        "链接",
        "视频",
        "记事本",
        "桌面",
        "拍照",
        "抓拍",
        "照相",
        "相机",
        "摄像头",
        "open",
        "play",
        "close",
        "fullscreen",
        "click",
        "press",
        "run",
        "launch",
        "save",
        "write",
    ]
    return any(token in text for token in action_tokens)


def plan_missing_before_followup(plan_payload: dict[str, Any] | None, completed_ok_tools: set[str]) -> list[str]:
    if not isinstance(plan_payload, dict):
        return []
    raw_steps = plan_payload.get("steps")
    if not isinstance(raw_steps, list):
        return []
    steps = sorted((step for step in raw_steps if isinstance(step, dict)), key=plan_step_order)
    for step in steps:
        kind = str(step.get("kind") or "").strip().lower()
        if kind == "speak":
            return []
        if kind != "tool":
            continue
        missing = [
            tool
            for tool in plan_step_tools(step)
            if tool and tool not in completed_ok_tools
        ]
        if missing:
            return missing
    return []


def plan_missing_before_tool(tool_name: str, plan_payload: dict[str, Any] | None, completed_ok_tools: set[str]) -> list[str]:
    if not isinstance(plan_payload, dict):
        return []
    raw_steps = plan_payload.get("steps")
    if not isinstance(raw_steps, list):
        return []
    target = str(tool_name or "").split(":", 1)[-1]
    prerequisite_tools = {"web_search", "search_news", "fetch_url"}
    pending_prerequisites: list[str] = []
    for step in sorted((step for step in raw_steps if isinstance(step, dict)), key=plan_step_order):
        if str(step.get("kind") or "").strip().lower() != "tool":
            continue
        tools = plan_step_tools(step)
        for tool in tools:
            if tool == target:
                return [name for name in pending_prerequisites if name not in completed_ok_tools]
            if tool in prerequisite_tools and tool not in pending_prerequisites and tool not in completed_ok_tools:
                pending_prerequisites.append(tool)
    return []


def plan_step_order(step: dict[str, Any]) -> int:
    try:
        return int(step.get("order"))
    except (TypeError, ValueError):
        return 999


def plan_step_tools(step: dict[str, Any]) -> list[str]:
    tools = step.get("suggested_tools")
    if isinstance(tools, str):
        tools = [tools]
    if not isinstance(tools, list):
        return []
    out = []
    for tool in tools:
        name = str(tool or "").strip().split(":", 1)[-1]
        if name:
            out.append(name)
    return out


SIDE_EFFECT_TOOL_NAMES = {
    "shell_command",
    "write_text_file",
    "open_url_in_browser",
    "run_osascript",
    "arrange_workspace",
    "front_note",
    "daily_action",
    "capture_camera_snapshot",
    "launch_python_script",
    "mail_message",
}


def tool_attempts_for(tool_name: str, configured_retries: int) -> int:
    normalized = str(tool_name or "").strip().split(":", 1)[-1]
    if normalized in SIDE_EFFECT_TOOL_NAMES:
        return 1
    return max(1, int(configured_retries) + 1)


def tool_timeout_for(tool_name: str, arguments: dict[str, Any], configured_timeout: float) -> float:
    tool_name = str(tool_name or "").strip().split(":", 1)[-1]
    base = max(1.0, float(configured_timeout))
    if tool_name == "shell_command":
        try:
            requested = float(arguments.get("timeout_seconds") or base)
        except (TypeError, ValueError):
            requested = base
        return max(base, requested + 5.0)
    return base
