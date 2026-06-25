from __future__ import annotations

from typing import Any

from voice_assistant.local_actions import (
    normalize_arrange_workspace_call_args,
    normalize_front_note_call_args,
    osascript_looks_like_window_layout,
    workspace_arrange_requested,
)
from voice_assistant.planning import planned_tool_arguments
from voice_assistant.url_utils import first_openable_url


PLAN_RECOVERY_ALLOWED_TOOLS = {
    "web_search",
    "search_news",
    "open_url_in_browser",
    "run_osascript",
    "arrange_workspace",
    "front_note",
}


def raw_plan_tool_args(name: str, step: dict[str, Any]) -> dict[str, Any] | None:
    raw_arguments = step.get("arguments")
    if not isinstance(raw_arguments, dict):
        return None
    direct = raw_arguments.get(name)
    if isinstance(direct, dict):
        return dict(direct)
    if name in {"open_url_in_browser", "run_osascript", "arrange_workspace", "front_note"}:
        return dict(raw_arguments)
    return None


def plan_recovery_tool_args(
    name: str,
    step: dict[str, Any],
    user_text: str,
    plan_payload: dict[str, Any] | None,
    verified_urls: list[str],
) -> dict[str, Any] | None:
    args = planned_tool_arguments(name, step, user_text)
    if args is None:
        args = raw_plan_tool_args(name, step)
    args = args or {}
    if name == "open_url_in_browser":
        args = dict(args)
        if workspace_arrange_requested(user_text):
            args["fullscreen"] = False
            args["video_fullscreen"] = False
        if not str(args.get("url") or "").startswith(("http://", "https://")):
            url = first_openable_url(verified_urls)
            if not url:
                return None
            args["url"] = url
        return args
    if name == "arrange_workspace":
        return normalize_arrange_workspace_call_args(args, plan_payload, user_text)
    if name == "front_note":
        return normalize_front_note_call_args(args, user_text)
    if name == "run_osascript" and osascript_looks_like_window_layout(args):
        return None
    return args
