from __future__ import annotations

from typing import Any


def tool_started_event_metadata(
    *,
    tool_name: str,
    log_word: str,
    action_subject: str,
    spoken_text: str,
    spoke_start: bool,
    speech_enabled: bool,
    cooldown_remaining: float,
    arguments: dict[str, Any],
    turn_id: str = "",
) -> dict[str, Any]:
    metadata = {
        "tool_name": tool_name,
        "log_word": log_word,
        "log_language": "zh",
        "action_subject": action_subject,
        "spoken_text": spoken_text,
        "spoke_start": spoke_start,
        "speech_enabled": speech_enabled,
        "speech_suppressed_reason": "cooldown" if speech_enabled and not spoke_start and cooldown_remaining > 0 else "",
        "speech_cooldown_remaining": round(cooldown_remaining, 3) if cooldown_remaining > 0 else 0,
        "arguments": arguments,
    }
    if turn_id:
        metadata["turn_id"] = turn_id
    return metadata


def tool_voice_summary_event_metadata(
    *,
    tool_name: str,
    ok: bool,
    phrase: str,
    spoken: bool,
    speech_enabled: bool,
    cooldown_remaining: float = 0,
    cooldown_bypassed: bool = False,
    turn_id: str = "",
) -> dict[str, Any]:
    metadata = {
        "tool_name": tool_name,
        "ok": ok,
        "phrase": phrase,
        "spoken": spoken,
        "speech_enabled": speech_enabled,
    }
    if speech_enabled:
        metadata.update(
            {
                "speech_suppressed_reason": "cooldown" if not spoken and cooldown_remaining > 0 and not cooldown_bypassed else "",
                "speech_cooldown_remaining": round(cooldown_remaining, 3) if cooldown_remaining > 0 else 0,
                "speech_cooldown_bypassed": bool(cooldown_bypassed),
            }
        )
    if turn_id:
        metadata["turn_id"] = turn_id
    return metadata


def tool_retry_event_metadata(
    *,
    attempt: int,
    attempts: int,
    reason: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "attempt": attempt,
        "attempts": attempts,
        "reason": str(reason)[:1000],
    }
    if timeout_seconds is not None:
        metadata["timeout_seconds"] = timeout_seconds
    return metadata


def plan_prefetch_started_metadata(prefetch_states: list[dict[str, Any]], turn_id: str = "") -> dict[str, Any]:
    return {
        "count": len(prefetch_states),
        "tools": [state.get("tool_name") for state in prefetch_states],
        "turn_id": turn_id,
    }


def plan_prefetch_completed_metadata(state: dict[str, Any], turn_id: str = "") -> dict[str, Any]:
    return {
        "tool_name": state.get("tool_name"),
        "arguments": state.get("arguments"),
        "ok": bool(state.get("ok")),
        "error": state.get("error"),
        "elapsed_seconds": state.get("elapsed_seconds"),
        "turn_id": turn_id,
    }


def plan_prefetch_miss_metadata(
    *,
    reason: str,
    arguments: dict[str, Any],
    wait_seconds: float | None = None,
    error: Any = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"reason": reason, "arguments": arguments}
    if wait_seconds is not None:
        metadata["wait_seconds"] = wait_seconds
    if error is not None:
        metadata["error"] = error
    return metadata


def plan_prefetch_hit_metadata(state: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {"arguments": arguments, "elapsed_seconds": state.get("elapsed_seconds")}


def agno_tool_output_metadata(
    *,
    tool_name: str,
    log_word: str,
    ok: bool,
    arguments: Any,
    result: Any,
    summary: str,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "log_word": log_word,
        "log_language": "zh",
        "ok": ok,
        "arguments": arguments,
        "result": result,
        "summary": summary,
    }
