from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voice_assistant.local_actions import (
    camera_tool_intent_error,
    normalize_arrange_workspace_call_args,
    osascript_looks_like_window_layout,
    planned_arrange_workspace_arguments,
    workspace_arrange_requested,
)
from voice_assistant.pro_guards import should_dedupe_completed_action, tool_missing_requirements
from voice_assistant.tool_runtime import tool_call_signature
from voice_assistant.tool_state import ToolTurnState
from voice_assistant.url_utils import duplicate_video_open_url, open_url_verification_error


@dataclass(frozen=True)
class RuntimeToolDecision:
    arguments: dict[str, Any]
    blocked_payload: dict[str, Any] | None = None
    blocked_reason: str = ""


@dataclass(frozen=True)
class ToolPreflightDecision:
    tool_signature: str
    action: str = "allow"
    event_kind: str = ""
    ok: bool = False
    error: str = ""
    message: str = ""
    instruction: str = ""
    extra: dict[str, Any] | None = None
    blocked_payload: dict[str, Any] | None = None
    blocked_reason: str = ""
    queue_voice_summary: bool = False


def prepare_runtime_tool_call(
    short_name: str,
    arguments: dict[str, Any] | None,
    user_text: str,
    plan_payload: dict[str, Any] | None,
) -> RuntimeToolDecision:
    safe_args = dict(arguments or {})
    if short_name == "open_url_in_browser" and workspace_arrange_requested(user_text):
        safe_args["fullscreen"] = False
        safe_args["video_fullscreen"] = False
        return RuntimeToolDecision(arguments=safe_args)
    if short_name == "arrange_workspace" and workspace_arrange_requested(user_text):
        return RuntimeToolDecision(arguments=normalize_arrange_workspace_call_args(safe_args, plan_payload, user_text))
    if short_name == "run_osascript" and workspace_arrange_requested(user_text) and osascript_looks_like_window_layout(safe_args):
        return RuntimeToolDecision(
            arguments=safe_args,
            blocked_reason="manual window layout blocked",
            blocked_payload={
                "ok": False,
                "error": "window layout must use arrange_workspace",
                "instruction": "Do not manually move or resize windows with run_osascript for this request. Call arrange_workspace with the target apps from the user request.",
                "suggested_arguments": planned_arrange_workspace_arguments(user_text),
            },
        )
    return RuntimeToolDecision(arguments=safe_args)


def evaluate_runtime_tool_preflight(
    short_name: str,
    safe_args: dict[str, Any],
    user_text: str,
    plan_payload: dict[str, Any] | None,
    state: ToolTurnState,
) -> ToolPreflightDecision:
    if short_name == "trigger_fast_followup":
        followup_text = str(safe_args.get("prompt") or safe_args.get("text") or "")
        if state.register_followup_text(followup_text):
            return ToolPreflightDecision(
                tool_signature="",
                action="suppress",
                event_kind="tool_duplicate_followup_suppressed",
                ok=True,
                message="duplicate followup suppressed",
                instruction="Do not call trigger_fast_followup again with the same user-facing message in this turn.",
            )
    signature = tool_call_signature(short_name, safe_args)
    if state.has_failed_signature(signature):
        return ToolPreflightDecision(
            tool_signature=signature,
            action="suppress",
            event_kind="tool_duplicate_failure_suppressed",
            ok=False,
            error="duplicate failed tool call suppressed",
            instruction="Do not call this tool again with the same arguments in this turn. Use a different tool or explain the current result briefly.",
        )
    if should_dedupe_completed_action(short_name, signature, state.completed_action_signatures):
        return ToolPreflightDecision(
            tool_signature=signature,
            action="suppress",
            event_kind="tool_duplicate_action_suppressed",
            ok=True,
            message="duplicate local action suppressed",
            instruction="This exact local action already completed in the current user turn. Continue with the next planned step.",
        )
    missing_requirements = tool_missing_requirements(short_name, plan_payload, state.completed_ok_tools)
    if missing_requirements:
        return ToolPreflightDecision(
            tool_signature=signature,
            action="block",
            blocked_reason="local action not completed",
            blocked_payload={
                "ok": False,
                "error": "local action not completed",
                "missing_tools": missing_requirements,
                "instruction": "Do not tell the user the action is complete. Call the missing local action tools first, then call trigger_fast_followup.",
            },
            extra={"missing_requirements": missing_requirements},
        )
    duplicate_video_url = duplicate_video_open_url(user_text, str(safe_args.get("url") or ""), state.opened_video_urls) if short_name == "open_url_in_browser" else ""
    if duplicate_video_url:
        return ToolPreflightDecision(
            tool_signature=signature,
            action="suppress",
            event_kind="tool_duplicate_open_suppressed",
            ok=True,
            message="duplicate video open suppressed",
            instruction="A video URL was already opened for this user turn. Do not open another video unless the user explicitly asked for multiple videos.",
            extra={"opened_url": duplicate_video_url, "opened_video_urls": sorted(state.opened_video_urls)},
        )
    camera_intent_error = camera_tool_intent_error(short_name, user_text)
    if camera_intent_error is not None:
        return ToolPreflightDecision(
            tool_signature=signature,
            action="block",
            blocked_reason=str(camera_intent_error.get("error") or "camera intent blocked"),
            blocked_payload=camera_intent_error,
            queue_voice_summary=True,
        )
    verification_error = open_url_verification_error(safe_args, user_text, state.verified_urls) if short_name == "open_url_in_browser" else None
    if verification_error is not None:
        return ToolPreflightDecision(
            tool_signature=signature,
            action="block",
            blocked_reason=str(verification_error.get("error") or "url verification failed"),
            blocked_payload=verification_error,
        )
    return ToolPreflightDecision(tool_signature=signature)
