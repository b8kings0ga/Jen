from __future__ import annotations

from typing import Any

from voice_assistant.planning import plan_missing_before_followup, plan_missing_before_tool, user_request_needs_local_action
from voice_assistant.voice_text import is_error_like_followup, is_status_only_followup


INITIAL_ANSWER_ALLOWED_TOOLS = {
    "trigger_fast_followup",
    "update_task_status",
    "add_context_note",
    "mark_task_done",
    "mark_task_in_progress",
    "mark_task_blocked",
    "open_url_in_browser",
    "run_osascript",
}

DEDUPED_ACTION_TOOLS = {
    "open_url_in_browser",
    "run_osascript",
    "arrange_workspace",
    "front_note",
    "capture_camera_snapshot",
}


def short_tool_name(name: str) -> str:
    return str(name or "").split(":", 1)[-1]


def should_stop_for_initial_answer(
    *,
    has_voice_facts: bool,
    user_text: str,
    elapsed_seconds: float,
    budget_seconds: float,
    tool_name: str,
) -> bool:
    if not has_voice_facts:
        return False
    if user_request_needs_local_action(user_text):
        return False
    if elapsed_seconds < budget_seconds:
        return False
    return short_tool_name(tool_name) not in INITIAL_ANSWER_ALLOWED_TOOLS


def tool_missing_requirements(
    tool_name: str,
    plan_payload: dict[str, Any] | None,
    completed_ok_tools: set[str],
) -> list[str]:
    name = short_tool_name(tool_name)
    if name == "trigger_fast_followup":
        return plan_missing_before_followup(plan_payload, completed_ok_tools)
    if name == "open_url_in_browser":
        return plan_missing_before_tool(name, plan_payload, completed_ok_tools)
    return []


def should_dedupe_completed_action(tool_name: str, tool_signature: str, completed_action_signatures: set[str]) -> bool:
    return short_tool_name(tool_name) in DEDUPED_ACTION_TOOLS and tool_signature in completed_action_signatures


def classify_followup_action(prompt: str, priority: int, *, interrupt_threshold: int, speak_threshold: int) -> str:
    if priority < interrupt_threshold:
        return "defer"
    if priority < speak_threshold:
        return "context_only"
    if is_status_only_followup(prompt):
        return "suppress_status"
    if is_error_like_followup(prompt):
        return "error_fallback"
    return "speak"
