from __future__ import annotations

import json
import re
from typing import Any

from voice_assistant.json_utils import parse_jsonish_value
from voice_assistant.tool_speech import short_tool_error_reason, tool_log_label

def summarize_tool_context_for_voice(tool_context: list[dict[str, Any]], max_chars: int = 2400) -> str:
    lines: list[str] = []
    for item in tool_context:
        tool = str(item.get("tool") or "")
        ok = bool(item.get("ok"))
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if not ok:
            error_text = str(result.get("error") or "")[:180]
            if error_text:
                lines.append(f"- {tool} 失败：{error_text}")
            continue
        query = str(arguments.get("query") or result.get("query") or "").strip()
        if tool in {"web_search", "search_news"}:
            if query:
                lines.append(f"- {tool} 查询：{query}")
            results = result.get("results")
            if isinstance(results, list):
                for row in results[:5]:
                    if not isinstance(row, dict):
                        continue
                    title = str(row.get("title") or "").strip()
                    snippet = str(row.get("snippet") or row.get("body") or "").strip()
                    date = str(row.get("date") or "").strip()
                    source = str(row.get("source") or "").strip()
                    url = str(row.get("url") or row.get("href") or "").strip()
                    parts = []
                    if title:
                        parts.append(title)
                    if date:
                        parts.append(f"日期/时间：{date}")
                    if source:
                        parts.append(f"来源：{source}")
                    if snippet:
                        parts.append(f"摘要：{snippet}")
                    if url:
                        parts.append(f"URL：{url}")
                    if parts:
                        lines.append("  - " + "；".join(parts))
            continue
        if tool == "fetch_url":
            url = str(arguments.get("url") or result.get("url") or "").strip()
            text = str(result.get("text") or result.get("content") or "").strip()
            if url or text:
                lines.append(f"- fetch_url {url}: {text[:600]}")
            continue
        lines.append(f"- {tool}: " + json.dumps(result, ensure_ascii=False)[:600])
    summary = "\n".join(lines).strip()
    if not summary:
        return "没有可用工具事实。"
    return summary[:max_chars]


def tool_result_has_voice_facts(tool_name: str, result: Any) -> bool:
    if tool_name not in {"web_search", "search_news", "fetch_url", "get_weather", "daily_action"}:
        return False
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {"content": result}
    elif isinstance(result, dict):
        payload = result
    else:
        return False
    if tool_name in {"web_search", "search_news"}:
        rows = payload.get("results")
        if not isinstance(rows, list):
            return False
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            if str(row.get("snippet") or row.get("body") or row.get("title") or "").strip():
                return True
        return False
    if tool_name == "fetch_url":
        return bool(str(payload.get("content") or payload.get("text") or "").strip())
    if tool_name == "get_weather":
        return bool(payload.get("ok") or payload.get("temperature") or payload.get("current"))
    if tool_name == "daily_action":
        action = str(payload.get("action") or "")
        if action in {"weather", "time", "calendar_list", "reminder_list", "map"}:
            return bool(payload.get("ok", True))
    return False


def tool_result_ok(result: Any) -> bool:
    value = parse_jsonish_value(result)
    if isinstance(value, dict):
        if value.get("ok") is False:
            return False
        if value.get("opened") is False or value.get("launched") is False:
            return False
        if value.get("error") and not value.get("ok"):
            return False
        inner = value.get("result")
        if inner is not None and inner is not value:
            return tool_result_ok(inner)
        return True
    text = str(value or "").strip().lower()
    if text.startswith("error:") or text.startswith("error running python code:"):
        return False
    if "outside the allowed base directory" in text:
        return False
    if "no module named" in text:
        return False
    return True


def tool_call_signature(tool_name: str, arguments: Any) -> str:
    try:
        payload = json.dumps(parse_jsonish_value(arguments), ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = str(arguments)
    return f"{str(tool_name or '').strip().split(':', 1)[-1]}:{payload}"


def tool_voice_summary(tool_name: str, ok: bool, result: Any) -> str:
    label = tool_log_label(tool_name, "success" if ok else "failure").strip()
    if ok:
        return label
    reason = short_tool_error_reason(result)
    return f"{label}：{reason}" if reason else label


def tool_timeout_error_message(tool_name: str, timeout_seconds: float, attempt: int, attempts: int) -> str:
    return f"tool {tool_name} timed out after {timeout_seconds:.1f}s on attempt {attempt}/{attempts}"


def tool_retry_backoff_seconds(attempt: int) -> float:
    return min(0.8 * max(1, int(attempt)), 2.0)


def format_tool_start_spoken(start_phrase: str, subject: str) -> str:
    start_phrase = str(start_phrase or "").strip()
    subject = re.sub(r"\s+", "", str(subject or ""))[:10]
    if not start_phrase:
        return subject
    if not subject or start_phrase.endswith(subject):
        return start_phrase
    return f"{start_phrase} {subject}"


def format_tool_spoken_summary(task_label: str, phrase: str) -> str:
    task_label = re.sub(r"\s+", "", str(task_label or ""))[:8]
    phrase = str(phrase or "").strip()
    if not phrase:
        return ""
    if not task_label or phrase.startswith(task_label):
        return phrase
    return f"{task_label} {phrase}"


def callable_tool_map(tools: list[Any]) -> dict[str, Any]:
    return {
        getattr(tool, "__name__", ""): tool
        for tool in tools
        if callable(tool) and getattr(tool, "__name__", "")
    }
