from __future__ import annotations

import json
from typing import Any

from voice_assistant.json_utils import parse_jsonish_value
from voice_assistant.tool_runtime import summarize_tool_context_for_voice


def recent_tool_context_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for item in rows:
        try:
            arguments = json.loads(str(item.get("arguments_json") or "{}"))
        except json.JSONDecodeError:
            arguments = {}
        try:
            result = json.loads(str(item.get("result_json") or "{}"))
        except json.JSONDecodeError:
            result = {}
        context.append(
            {
                "tool": item.get("tool_name"),
                "ok": bool(item.get("ok")),
                "arguments": arguments,
                "result": result,
            }
        )
    return context


def no_followup_fallback_prompt(user_text: str, tool_context: list[dict[str, Any]]) -> tuple[str, str]:
    tool_facts = summarize_tool_context_for_voice(tool_context)
    prompt = (
        "用户刚问："
        + user_text
        + "\n\n后台没有主动产出可播报 followup，但已有工具结果。工具事实候选：\n"
        + tool_facts
        + "\n\n原始工具结果 JSON：\n"
        + json.dumps(tool_context[:5], ensure_ascii=False)
        + "\n\n先在内部锁定【用户问题】：只回答这个问题。只能基于上面的已有事实回答，不要编造、不要扩展、不要补用户没问的完整报告。"
        + "如果已有事实足够，直接给最短答案；是/否型问题直接答“是/否/下/不下/能/不能”。例如用户问“圣保罗下雨么”，天气事实不是雨，就只答“不下。”或“不下，现在是晴天。”不要自动播完整天气报告。"
        + "如果事实不足以回答，但工具事实里有可继续用的候选，就说还缺哪一个关键事实；如果缺少地点、对象、时间、URL 等必要参数，问一个最短澄清问题。"
        + "重要：web_search/search_news/fetch_url 的 title、snippet、date、url 都是可用事实候选；如果 snippet 里有日期、赛程、比分、天气、价格、名单等信息，必须基于它直接给一句简短结论。"
        + "如果用户要求打开/播放网页或视频，只有工具事实里存在成功的 open_url_in_browser 时，才可以说“已打开/正在播放”；只有搜索结果时只能说“我找到了候选，还没打开”。"
        + "如果没有本轮工具事实，但共享上下文里有最近提到的对象、人物或上一轮答案，可以基于共享上下文简短回答；缺细节时说“我只知道到这里，更多细节还要再查”。只有完全没有相关事实和上下文时，才说现在查不了。"
    )
    return prompt, tool_facts


def direct_fallback_response_from_tools(user_text: str, tool_context: list[dict[str, Any]]) -> str:
    """Build a short spoken answer from tool facts without another model call."""
    daily_texts: list[str] = []
    seen_daily: set[str] = set()
    for item in reversed(tool_context):
        if not bool(item.get("ok")):
            continue
        tool = str(item.get("tool") or "").split(":", 1)[-1]
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        parsed_result = parse_jsonish_value(result.get("result")) if "result" in result else result
        if isinstance(parsed_result, dict):
            result = parsed_result
        if tool == "computer_action" and str(result.get("action") or "") in {"delegate_to_codex", "develop_app"}:
            summary = str(result.get("_summary") or "").strip()
            if summary:
                return summary
            label = "Codex"
            executor = str(result.get("executor") or "").strip().lower()
            if executor and executor != "codex":
                label = executor
            return f"{label} 开工了。"
        if tool != "daily_action":
            continue
        if not isinstance(result, dict):
            continue
        text = _daily_action_spoken_result(result, user_text=user_text)
        key = f"{result.get('action')}:{text}"
        if text and key not in seen_daily:
            daily_texts.append(text)
            seen_daily.add(key)
    if daily_texts:
        return "".join(daily_texts[:3])
    for item in tool_context:
        if bool(item.get("ok")):
            continue
        tool = str(item.get("tool") or "").split(":", 1)[-1]
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        parsed_result = parse_jsonish_value(result.get("result")) if "result" in result else result
        if isinstance(parsed_result, dict):
            result = parsed_result
        if tool == "daily_action" and str(result.get("action") or "") == "weather":
            weather = result.get("weather") if isinstance(result.get("weather"), dict) else {}
            target = str(result.get("target") or weather.get("location") or "").strip()
            return f"地点没识别对，我听成了{target}。" if target else "地点没识别对。"
        if tool == "get_weather":
            target = str(result.get("location") or "").strip()
            return f"地点没识别对，我听成了{target}。" if target else "地点没识别对。"
    for item in tool_context:
        if not bool(item.get("ok")):
            continue
        tool = str(item.get("tool") or "").split(":", 1)[-1]
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        parsed_result = parse_jsonish_value(result.get("result")) if "result" in result else result
        if isinstance(parsed_result, dict):
            result = parsed_result
        if tool == "daily_action":
            text = _daily_action_spoken_result(result, user_text=user_text)
            if text:
                return text
        if tool == "get_weather":
            text = _weather_spoken_result(result)
            if text:
                return text
        if tool in {"web_search", "search_news"}:
            text = _search_spoken_result(result)
            if text:
                return text
        if tool == "fetch_url":
            text = str(result.get("text") or result.get("content") or "").strip()
            if text:
                return text[:160]
    return "这个我只处理到这里，还没有更明确的结果。"


def direct_response_from_domain_probe(payload: dict[str, Any] | None) -> str:
    """Probe answers are hints for the back model, not user-facing speech."""
    return ""


def _daily_action_spoken_result(result: dict[str, Any], *, user_text: str = "") -> str:
    action = str(result.get("action") or "").strip()
    if action == "weather":
        weather = result.get("weather") if isinstance(result.get("weather"), dict) else result
        return _weather_spoken_result(weather, user_text=user_text)
    if action == "time":
        payload = result.get("time") if isinstance(result.get("time"), dict) else {}
        local_time = str(payload.get("local_time") or payload.get("iso") or payload.get("now") or "").strip()
        if local_time:
            return f"现在是{local_time}。"
    if action == "reminder_create":
        title = str(result.get("title") or result.get("target") or "").strip()
        return f"提醒已设好：{title}。" if title else "提醒已设好。"
    if action in {"note_live", "note_context", "memory"}:
        summary = str(result.get("_summary") or "").strip()
        return summary or "弄好了。"
    return str(result.get("_summary") or "").strip()


def _weather_spoken_result(result: dict[str, Any], *, user_text: str = "") -> str:
    if not isinstance(result, dict) or result.get("ok") is False:
        return ""
    location = str(result.get("resolved_location") or result.get("location") or "").strip()
    temp = str(result.get("temperature_c") or result.get("temp_C") or result.get("temperature") or "").strip()
    feels = str(result.get("feels_like_c") or "").strip()
    desc = str(result.get("description") or "").strip()
    humidity = str(result.get("humidity_percent") or "").strip()
    wind = str(result.get("wind_kmph") or "").strip()
    concise = _concise_weather_answer(user_text, desc)
    if concise:
        return concise
    parts: list[str] = []
    if location:
        parts.append(f"{location}现在")
    if temp:
        parts.append(f"{temp}度")
    if feels and feels != temp:
        parts.append(f"体感{feels}度")
    if desc:
        parts.append(desc)
    if humidity:
        parts.append(f"湿度{humidity}%")
    if wind:
        parts.append(f"风速{wind}公里每小时")
    return "，".join(parts).strip("，") + ("。" if parts else "")


def _concise_weather_answer(user_text: str, description: str) -> str:
    text = str(user_text or "")
    desc = str(description or "").lower()
    if not text or not desc:
        return ""
    rainy = any(token in desc for token in ["rain", "shower", "drizzle", "thunder", "storm", "雨", "雷"])
    if any(token in text for token in ["下雨", "有雨", "降雨"]):
        return "下。" if rainy else "不下。"
    if "带伞" in text:
        return "要带。" if rainy else "不用带。"
    return ""


def _search_spoken_result(result: dict[str, Any]) -> str:
    rows = result.get("results") if isinstance(result.get("results"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or row.get("body") or "").strip()
        if title and snippet:
            return f"{title}：{snippet[:140]}"
        if snippet:
            return snippet[:160]
        if title:
            return title[:120]
    return ""
