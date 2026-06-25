from __future__ import annotations

import argparse
import datetime as dt
import threading
import time
from typing import Any

from voice_assistant.json_utils import parse_json_object
from voice_assistant.prompts import COMPRESS_SYSTEM_PROMPT, FAST_SYSTEM_PROMPT
from voice_assistant.speech_text import normalize_assistant_text
from voice_assistant.store import VoiceSessionStore


AGNO_AGENT_CACHE: dict[tuple[str, str, str, str, str], object] = {}


def current_context_prompt() -> str:
    now = dt.datetime.now().astimezone()
    return (
        "当前时间上下文:\n"
        f"- 当前本地时间: {now.isoformat(timespec='seconds')}\n"
        f"- 当前日期: {now.date().isoformat()}\n"
        f"- 星期: {now.strftime('%A')}\n"
        f"- 时区: {now.tzname() or ''}"
    )


class AgentFactory:
    def __init__(self, args: argparse.Namespace, store: VoiceSessionStore) -> None:
        self.args = args
        self.store = store

    def agent(
        self,
        lane: str,
        instructions: str,
        tools: list[Any] | None = None,
        tool_hooks: list[Any] | None = None,
        *,
        reasoning_effort: str | None = None,
    ):
        tool_names = ",".join(getattr(t, "__name__", t.__class__.__name__) for t in tools or [])
        effort = (reasoning_effort or "").strip() or None
        key = (self.args.gjallarhorn_base_url.rstrip("/"), self.args.api_key, lane, effort or "", instructions, tool_names)
        if not tool_hooks and key in AGNO_AGENT_CACHE:
            return AGNO_AGENT_CACHE[key]
        from agno.agent import Agent
        from agno.models.openai.like import OpenAILike

        model_kwargs: dict[str, Any] = {
            "id": lane,
            "api_key": self.args.api_key,
            "base_url": self.args.gjallarhorn_base_url.rstrip("/"),
        }
        if effort:
            model_kwargs["reasoning_effort"] = effort
        kwargs = {
            "model": OpenAILike(**model_kwargs),
            "instructions": instructions,
            "tools": tools or [],
            "markdown": False,
        }
        if tool_hooks:
            kwargs["tool_hooks"] = tool_hooks
        agent = Agent(**kwargs)
        if not tool_hooks:
            AGNO_AGENT_CACHE[key] = agent
        return agent


class FastLaneAgent:
    def __init__(self, args: argparse.Namespace, store: VoiceSessionStore, factory: AgentFactory) -> None:
        self.args = args
        self.store = store
        self.factory = factory

    def respond(self, user_text: str, reason: str = "user_turn", turn_id: str = "") -> str:
        followup_instruction = ""
        if reason == "pro_followup":
            followup_instruction = (
                "\n\n这是后台工具/上下文更新后触发的新信息。"
                "如果触发内容包含明确事实结果，可以在开头加一个非常短、自然的中文过渡语，例如“我查到了，”或“有结果了，”。"
                "如果触发内容表示无法确认、查不了、没有足够信息或只是澄清问题，不要说“我查到了”或“有结果了”。"
                "然后只说这条触发内容本身的结论。不要扩展用户没问的内容，不要追加建议或反问。不要说你是 front lane，不要提后台 lane。"
            )
        elif reason == "waiting_filler":
            followup_instruction = (
                "\n\n用户正在等待后台处理。请只生成一句很短的中文等待语，像真人自然衔接，"
                "不要承诺已经查到结果，不要超过 12 个字。"
            )
        prompt = (
            current_context_prompt() +
            "\n\n共享上下文 JSON:\n" + self.store.context_bundle() +
            "\n\n用户最新输入/后台触发内容:\n" + user_text +
            "\n\n注意：如果这是用户刚录音后的输入，文本已经是 ASR 成功转写结果，不要再说正在转写或要求用户提供同一段音频。"
            "\n请生成适合语音播报的简短回答。只回答用户问了什么；没有问的问题不要回答。"
            + followup_instruction
        )
        started = time.monotonic()
        self.store.add_event(
            "llm_call",
            role="system",
            lane=self.args.front,
            content=f"front {self.args.front} started",
            metadata={"turn_id": turn_id, "phase": reason, "model": self.args.front, "status": "started"},
        )
        try:
            run = self.factory.agent(self.args.front, FAST_SYSTEM_PROMPT).run(prompt)
            self.store.add_event(
                "llm_call",
                role="system",
                lane=self.args.front,
                content=f"front {self.args.front} ok",
                metadata={
                    "turn_id": turn_id,
                    "phase": reason,
                    "model": self.args.front,
                    "status": "ok",
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
            )
        except Exception as exc:
            self.store.add_event(
                "llm_call",
                role="system",
                lane=self.args.front,
                content=f"front {self.args.front} error",
                metadata={
                    "turn_id": turn_id,
                    "phase": reason,
                    "model": self.args.front,
                    "status": "error",
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "error": str(exc)[:1000],
                },
            )
            raise
        content = normalize_assistant_text(str(getattr(run, "content", "") or ""))
        self.store.add_event("assistant_reply", role="assistant", lane=self.args.front, content=content, metadata={"reason": reason, "turn_id": turn_id})
        return content


class SessionCompressor:
    def __init__(self, args: argparse.Namespace, store: VoiceSessionStore, factory: AgentFactory) -> None:
        self.args = args
        self.store = store
        self.factory = factory
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._last_failed_at = 0.0

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def maybe_compress_async(self) -> None:
        if self._should_start():
            threading.Thread(target=self.compress, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.wait(30):
            if self._should_start():
                self.compress()

    def _should_start(self) -> bool:
        if not self.store.compression_due(self.args.compress_every_turns, self.args.compress_every_seconds):
            return False
        now = time.time()
        with self._lock:
            if self._running:
                return False
            retry_cooldown = max(60.0, float(self.args.compress_every_seconds))
            if self._last_failed_at and now - self._last_failed_at < retry_cooldown:
                return False
            self._running = True
            return True

    def compress(self) -> None:
        prompt = current_context_prompt() + "\n\n请压缩这个 voice session。共享上下文 JSON:\n" + self.store.context_bundle(recent_limit=50)
        ok = False
        try:
            run = self.factory.agent(self.args.compact, COMPRESS_SYSTEM_PROMPT).run(prompt)
            raw = normalize_assistant_text(str(getattr(run, "content", "") or ""))
            payload = parse_json_object(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("summary"), str):
                raise RuntimeError(f"invalid compressor response: {raw[:500]}")
            self.store.update_compression(payload)
            print(f"Session compressed via {self.args.compact}.", flush=True)
            ok = True
        except Exception as exc:
            self.store.add_event("compress_error", role="system", lane="compressor", content="compression failed", metadata={"error": str(exc)[:1000]})
        finally:
            with self._lock:
                self._running = False
                self._last_failed_at = 0.0 if ok else time.time()
