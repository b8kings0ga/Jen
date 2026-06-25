from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voice_assistant.speech_text import compact_speech_text, normalize_assistant_text
from voice_assistant.tool_runtime import tool_result_has_voice_facts
from voice_assistant.url_utils import extract_verified_urls_from_tool_result, looks_like_video_url


ACTION_SIGNATURE_TOOLS = {
    "open_url_in_browser",
    "run_osascript",
    "arrange_workspace",
    "front_note",
    "capture_camera_snapshot",
}


@dataclass
class ToolTurnState:
    tool_count: int = 0
    has_voice_facts: bool = False
    completed_ok_tools: set[str] = field(default_factory=set)
    verified_urls: set[str] = field(default_factory=set)
    opened_video_urls: set[str] = field(default_factory=set)
    failed_tool_signatures: set[str] = field(default_factory=set)
    completed_action_signatures: set[str] = field(default_factory=set)
    followup_signatures: set[str] = field(default_factory=set)

    def mark_tool_started(self) -> None:
        self.tool_count += 1

    def register_followup_text(self, text: str) -> bool:
        signature = compact_speech_text(normalize_assistant_text(text))[:220]
        if not signature:
            return False
        if signature in self.followup_signatures:
            return True
        self.followup_signatures.add(signature)
        return False

    def has_failed_signature(self, signature: str) -> bool:
        return signature in self.failed_tool_signatures

    def mark_failed_signature(self, signature: str) -> None:
        self.failed_tool_signatures.add(signature)

    def mark_success(self, short_name: str, tool_signature: str, arguments: dict[str, Any], result: Any) -> None:
        if tool_result_has_voice_facts(short_name, result):
            self.has_voice_facts = True
        self.completed_ok_tools.add(short_name)
        if short_name in ACTION_SIGNATURE_TOOLS:
            self.completed_action_signatures.add(tool_signature)
        self.verified_urls.update(extract_verified_urls_from_tool_result(short_name, arguments, result))
        opened_url = str(arguments.get("url") or "").strip()
        if short_name == "open_url_in_browser" and looks_like_video_url(opened_url):
            self.opened_video_urls.add(opened_url)
