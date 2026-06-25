from __future__ import annotations

import argparse
import inspect
import json
import queue
import re
import threading
import time
from typing import Any, Callable

from voice_assistant.agents import AgentFactory, FastLaneAgent, current_context_prompt
from voice_assistant.domain_probe import format_domain_probe_prompt, probe_domains
from voice_assistant.event_metadata import (
    agno_tool_output_metadata,
    plan_prefetch_completed_metadata,
    plan_prefetch_hit_metadata,
    plan_prefetch_miss_metadata,
    plan_prefetch_started_metadata,
    tool_retry_event_metadata,
    tool_started_event_metadata,
    tool_voice_summary_event_metadata,
)
from voice_assistant.fallbacks import (
    direct_fallback_response_from_tools,
    no_followup_fallback_prompt,
    recent_tool_context_from_rows,
)
from voice_assistant.json_utils import coerce_json_object, coerce_json_value, parse_json_object
from voice_assistant.local_actions import (
    long_term_memory_requested,
)
from voice_assistant.planning import (
    format_plan_summary_for_voice,
    execution_plan_from_domain_probe,
    heuristic_execution_plan,
    heuristic_plan_is_confident,
    plan_step_order,
    plan_step_tools,
    planned_prefetch_calls,
    tool_action_subject,
    tool_attempts_for,
    tool_prefetch_key,
    tool_timeout_for,
    video_search_query_from_text,
)
from voice_assistant.plan_recovery import PLAN_RECOVERY_ALLOWED_TOOLS, plan_recovery_tool_args
from voice_assistant.prompts import PLAN_SYSTEM_PROMPT, PRO_JSON_FALLBACK_PROMPT, PRO_SYSTEM_PROMPT
from voice_assistant.pro_guards import (
    classify_followup_action,
    should_stop_for_initial_answer,
    short_tool_name,
)
from voice_assistant.speech import SpeechQueue
from voice_assistant.speech_text import normalize_assistant_text
from voice_assistant.store import VoiceSessionStore
from voice_assistant.tool_registry import build_voice_tools
from voice_assistant.tool_runtime import (
    callable_tool_map,
    format_tool_spoken_summary,
    format_tool_start_spoken,
    tool_result_ok,
    tool_retry_backoff_seconds,
    tool_timeout_error_message,
    tool_voice_summary,
)
from voice_assistant.tool_policy import evaluate_runtime_tool_preflight, prepare_runtime_tool_call
from voice_assistant.tool_state import ToolTurnState
from voice_assistant.url_utils import (
    duplicate_video_open_url,
    extract_urls_from_value,
    extract_verified_urls_from_tool_result,
    looks_like_video_url,
    normalized_url_key,
    url_explicitly_in_user_text,
    urls_match,
    user_requested_multiple_video_opens,
)
from voice_assistant.voice_text import (
    is_tool_route_unavailable,
    text_similarity,
)

def coding_or_debug_task_requested(text: str) -> bool:
    value = str(text or "")
    lowered = value.lower()
    if re.search(r"(?i)(debug|dbug|deubg|degub|develop|implement|refactor|build)", value):
        return True
    if re.search(r"(开发|实现|新增|修复|重构|优化|调试|排查)(?!者|商|区|阶段|环境).{1,80}", value):
        return True
    if any(token in lowered for token in ["修 bug", "修bug", "改代码", "改项目", "优化代码", "重构代码"]):
        return True
    return False


def tool_names_for_domain_probe(domain_probe: dict[str, Any] | None) -> set[str]:
    """Return the small LLM-visible tool set selected by the current domain probe."""
    if not isinstance(domain_probe, dict):
        return set()
    probe_input = str(domain_probe.get("input") or "").lower()
    high_confidence: set[str] = set()
    fallback: set[str] = set()
    for domain in domain_probe.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        domain_confidence = float(domain.get("confidence") or 0.0)
        for suggested in domain.get("suggested_actions") or []:
            if not isinstance(suggested, dict):
                continue
            tool_name = _tool_name_from_probe_call(str(suggested.get("tool_call") or ""))
            if not tool_name:
                continue
            confidence = float(suggested.get("confidence") or domain_confidence)
            if confidence >= 0.8:
                high_confidence.add(tool_name)
            elif tool_name == "computer_action" and 'action="delegate_to_codex"' in str(suggested.get("tool_call") or ""):
                fallback.add(tool_name)
    selected = high_confidence or fallback
    if "web_search" in selected and any(token in probe_input for token in ("打开", "open", "网页", "链接", "website", "web", "link")):
        selected.add("open_url_in_browser")
    if selected:
        selected.add("trigger_fast_followup")
    return selected


def _tool_name_from_probe_call(tool_call: str) -> str:
    match = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", str(tool_call or ""))
    return match.group(1) if match else ""


class ProLaneWorker:
    def __init__(
        self,
        args: argparse.Namespace,
        store: VoiceSessionStore,
        factory: AgentFactory,
        fast_agent: FastLaneAgent,
        speech: SpeechQueue,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self.args = args
        self.store = store
        self.factory = factory
        self.fast_agent = fast_agent
        self.speech = speech
        self.on_complete = on_complete
        self._threads: list[threading.Thread] = []

    def submit(self, user_text: str, generation: int, turn_id: str = "") -> None:
        thread = threading.Thread(target=self._run, args=(user_text, generation, turn_id), daemon=True)
        thread.start()
        self._threads.append(thread)

    def wait(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in list(self._threads):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return
            thread.join(timeout=remaining)

    def _run(self, user_text: str, generation: int, turn_id: str = "") -> None:
        started_at = time.monotonic()
        turn_started_at = time.time()
        pro_timing_id = self.store.start_turn_timing(turn_id, "pro_run", "后台", metadata={"model": self.args.back, "plan_mode": self.args.plan_mode})
        self.store.add_event(
            "task_log",
            role="assistant",
            lane=self.args.back,
            content=f"任务开始 · {user_text[:120]}",
            metadata={
                "turn_id": turn_id,
                "status": "started",
                "generation": generation,
                "plan_mode": bool(self.args.plan_mode),
                "model": self.args.back,
            },
        )
        all_tools = self._tools()
        domain_probe = self._probe_domains(user_text, turn_id=turn_id)
        tools = self._tools_for_domain_probe(domain_probe, all_tools)
        domain_probe_prompt = format_domain_probe_prompt(domain_probe)
        turn_state = ToolTurnState()
        if self.args.plan_mode:
            self._queue_plan_voice(None, generation, turn_id=turn_id)
        if self.args.plan_mode:
            plan_payload = self._make_execution_plan(user_text, tools, turn_id=turn_id, domain_probe=domain_probe)
            try:
                self._queue_plan_summary_voice(plan_payload, generation, turn_id=turn_id)
            except Exception as exc:
                self.store.add_event(
                    "execution_plan_summary_voice_error",
                    role="system",
                    lane="speech",
                    content="plan summary voice failed",
                    metadata={"error": str(exc)[:1000], "turn_id": turn_id},
                )
                print(f"Plan summary voice failed: {exc}", flush=True)
        else:
            plan_payload = execution_plan_from_domain_probe(domain_probe)
            if plan_payload:
                self.store.add_event(
                    "execution_plan",
                    role="system",
                    lane="local",
                    content=json.dumps(plan_payload, ensure_ascii=False),
                    metadata={"transcript": user_text[:500], "tool_count": len(tools), "turn_id": turn_id, "planner": "domain_probe_simple"},
                )
        if generation == self.speech.current_generation():
            self.speech.start_filler_loop(
                "working",
                initial_delay=max(1.0, float(self.args.filler_min_interval)),
                interval_range=(self.args.filler_min_interval, self.args.filler_max_interval),
            )
            self.store.add_event(
                "filler_loop_started",
                role="assistant",
                lane="speech",
                content="working",
                metadata={"source": "cached_step_tts_mini", "stage": "working", "after": "plan_summary", "turn_id": turn_id},
            )
        prefetches = (
            {}
            if self._is_local_domain_probe_plan(plan_payload)
            else self._start_plan_prefetch(plan_payload, tools, user_text, generation, turn_id=turn_id)
        )
        plan_text = json.dumps(plan_payload or {"steps": []}, ensure_ascii=False)
        timed_out = threading.Event()
        done = threading.Event()
        fallback_sent = threading.Event()
        speech_queued = threading.Event()
        tool_activity = threading.Event()
        prompt = (
            current_context_prompt() +
            "\n\n共享上下文 JSON:\n" + self.store.context_bundle() +
            "\n\n" + domain_probe_prompt +
            "\n\n执行计划 JSON:\n" + plan_text +
            "\n\n用户最新输入:\n" + user_text +
            "\n\n注意：这段用户输入已经来自录音 ASR 的成功转写，不要把它当成缺少音频文件的任务。"
            "\n请严格按执行计划 JSON 的 steps 顺序处理后台任务；必要时执行工具、维护状态、补充上下文；如果计划中当前步骤是 speak 或需要主动告诉用户，调用 trigger_fast_followup。"
            "\n不要跳过计划中更早的用户可见步骤；不要把尚未执行的本地动作说成已完成。"
        )
        handled = False

        def queue_fallback_once(since: float | None = turn_started_at) -> bool:
            if fallback_sent.is_set():
                return False
            fallback_sent.set()
            self._queue_no_followup_fallback(user_text, generation, since=since, turn_id=turn_id)
            return True

        def timeout_watchdog() -> None:
            if done.wait(self.args.pro_turn_timeout):
                return
            timed_out.set()
            if fallback_sent.is_set():
                return
            fallback_sent.set()
            self.store.add_event(
                "pro_timeout_fallback",
                role="system",
                lane=self.args.back,
                content="back lane turn exceeded time budget; front fallback requested",
                metadata={"timeout_seconds": self.args.pro_turn_timeout, "transcript": user_text[:500], "turn_id": turn_id},
            )
            self._queue_no_followup_fallback(user_text, generation, since=turn_started_at, turn_id=turn_id)

        threading.Thread(target=timeout_watchdog, daemon=True).start()

        def initial_model_watchdog() -> None:
            if done.wait(self.args.back_model_first_answer_timeout):
                return
            if fallback_sent.is_set() or speech_queued.is_set() or tool_activity.is_set():
                return
            fallback_sent.set()
            timed_out.set()
            self.store.add_event(
                "back_model_first_answer_timeout",
                role="system",
                lane=self.args.back,
                content="back model did not produce first answer in time; front fallback requested",
                metadata={"timeout_seconds": self.args.back_model_first_answer_timeout, "transcript": user_text[:500], "turn_id": turn_id},
            )
            if self._execute_plan_recovery(user_text, plan_payload, tools, generation, turn_id=turn_id):
                speech_queued.set()
                return
            self._queue_no_followup_fallback(user_text, generation, since=turn_started_at, turn_id=turn_id)

        threading.Thread(target=initial_model_watchdog, daemon=True).start()
        try:
            try:
                hook = self._tool_event_hook(
                    started_at=started_at,
                    timed_out=timed_out,
                    user_text=user_text,
                    generation=generation,
                    fallback_sent=fallback_sent,
                    speech_queued=speech_queued,
                    tool_activity=tool_activity,
                    plan_payload=plan_payload,
                    prefetches=prefetches,
                    tools=tools,
                    turn_id=turn_id,
                    state=turn_state,
                )
                local_probe_executed = self._execute_domain_probe_plan_locally(
                    user_text=user_text,
                    plan_payload=plan_payload,
                    tools=tools,
                    generation=generation,
                    hook=hook,
                    speech_queued=speech_queued,
                    tool_activity=tool_activity,
                    since=turn_started_at,
                    turn_id=turn_id,
                )
                if local_probe_executed:
                    tool_context = recent_tool_context_from_rows(
                        self.store.recent_tool_events(within_seconds=120.0, limit=12, ok_only=False, since=turn_started_at)
                    )
                    prompt += (
                        "\n\n本轮本地工具已按执行计划运行，下面是工具结果 JSON；不要重复调用这些已经完成的工具。"
                        "\n请先锁定【用户真正问的问题】，只基于这些工具结果直接回答这个问题；如果结果已经足够，不要再调用重复工具。"
                        "\n是/否型问题要短答，例如问“下雨么”时只说“下/不下”并可补一个最短依据；不要自动播完整天气报告。"
                        "\n如果这些工具结果仍不足以回答，才调用缺失信息对应的已注册工具；如果缺少必要参数，调用 trigger_fast_followup 提一个最短澄清问题。"
                        "\n如果需要播报，调用 trigger_fast_followup。"
                        "\n本轮本地工具结果 JSON:\n"
                        + json.dumps(tool_context, ensure_ascii=False)
                    )
                llm_started = time.monotonic()
                self.store.add_event(
                    "llm_call",
                    role="system",
                    lane=self.args.back,
                    content=f"back {self.args.back} started",
                    metadata={"turn_id": turn_id, "phase": "back", "model": self.args.back, "status": "started"},
                )
                try:
                    run = self.factory.agent(
                        self.args.back,
                        PRO_SYSTEM_PROMPT,
                        tools=tools,
                        tool_hooks=[hook],
                        reasoning_effort=getattr(self.args, "reasoning_effort", None),
                    ).run(prompt)
                    self.store.add_event(
                        "llm_call",
                        role="system",
                        lane=self.args.back,
                        content=f"back {self.args.back} ok",
                        metadata={
                            "turn_id": turn_id,
                            "phase": "back",
                            "model": self.args.back,
                            "status": "ok",
                            "duration_seconds": round(time.monotonic() - llm_started, 3),
                        },
                    )
                except Exception as exc:
                    self.store.add_event(
                        "llm_call",
                        role="system",
                        lane=self.args.back,
                        content=f"back {self.args.back} error",
                        metadata={
                            "turn_id": turn_id,
                            "phase": "back",
                            "model": self.args.back,
                            "status": "error",
                            "duration_seconds": round(time.monotonic() - llm_started, 3),
                            "error": str(exc)[:1000],
                        },
                    )
                    raise
                if timed_out.is_set():
                    handled = True
                    return
                self._record_run_output_tools(run)
                content = normalize_assistant_text(str(getattr(run, "content", "") or ""))
                if is_tool_route_unavailable(content):
                    self.store.add_event("pro_error", role="system", lane=self.args.back, content="back route unavailable", metadata={"error": content[:1000], "turn_id": turn_id})
                    queue_fallback_once()
                    handled = True
                    return
                if content:
                    self.store.add_event(
                        "pro_internal_note",
                        role="system",
                        lane=self.args.back,
                        content="back lane completed background reasoning",
                        metadata={"model_note": content[:2000], "turn_id": turn_id},
                    )
                    if self._queue_followup_from_model_note(content, turn_id=turn_id):
                        speech_queued.set()
            except Exception as exc:
                if timed_out.is_set() or "back lane turn budget exceeded" in str(exc):
                    queue_fallback_once()
                    handled = True
                    return
                if is_tool_route_unavailable(str(exc)):
                    self.store.add_event("pro_error", role="system", lane=self.args.back, content="back route unavailable", metadata={"error": str(exc)[:1000], "turn_id": turn_id})
                    queue_fallback_once()
                    handled = True
                    return
                self.store.add_event("pro_error", role="system", lane=self.args.back, content="background worker failed", metadata={"error": str(exc)[:1000], "turn_id": turn_id})
                if not speech_queued.is_set() and queue_fallback_once():
                    handled = True
                return
            if timed_out.is_set():
                handled = True
                return
            handled = self._drain_followups(generation, turn_id=turn_id)
            if not handled and speech_queued.is_set():
                handled = True
            if not handled:
                queue_fallback_once()
                handled = True
        finally:
            done.set()
            self.speech.stop_filler_loop()
            status = "timeout" if timed_out.is_set() else ("handled" if handled else "finished")
            counts = self.store.turn_call_counts(turn_id)
            self.store.add_event(
                "task_log",
                role="assistant",
                lane=self.args.back,
                content=f"任务结束 · {status}",
                metadata={
                    "turn_id": turn_id,
                    "status": status,
                    "handled": handled,
                    "tool_count": counts["tc"],
                    "registered_tool_count": counts["tr"],
                    "model_call_count": counts["mc"],
                    "tts_call_count": counts["tts"],
                    "duration_seconds": round(time.monotonic() - started_at, 3),
                    "model": self.args.back,
                },
            )
            self.store.add_event(
                "task_counts",
                role="assistant",
                lane=self.args.back,
                content=f"counts · ps={counts['tr']} tc={counts['tc']} mc={counts['mc']} tts={counts['tts']}",
                metadata={"turn_id": turn_id, "status": status, **counts},
            )
            self.store.end_turn_timing(pro_timing_id, status=status, metadata={"handled": handled, **counts})
            self.store.finish_open_turn_timings(turn_id, "turn_total", status=status, metadata={"handled": handled})
            if self.on_complete is not None:
                self.on_complete()

    def _queue_followup_from_model_note(self, content: str, *, turn_id: str = "") -> bool:
        payload = parse_json_object(content)
        if not payload:
            return False
        queued = False
        direct_prompt = str(payload.get("prompt") or "").strip()
        if direct_prompt:
            priority = int(payload.get("priority") or 1)
            self.store.trigger_fast_followup(direct_prompt, priority)
            self.store.add_event(
                "pro_model_note_followup",
                role="system",
                lane=self.args.back,
                content=direct_prompt[:300],
                metadata={"turn_id": turn_id, "priority": priority, "source": "prompt"},
            )
            queued = True
        for item in payload.get("fast_followups") or []:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt") or "").strip()
            if not prompt:
                continue
            priority = int(item.get("priority") or 1)
            self.store.trigger_fast_followup(prompt, priority)
            self.store.add_event(
                "pro_model_note_followup",
                role="system",
                lane=self.args.back,
                content=prompt[:300],
                metadata={"turn_id": turn_id, "priority": priority, "source": "fast_followups"},
            )
            queued = True
        return queued

    def _tool_event_hook(
        self,
        *,
        started_at: float,
        timed_out: threading.Event,
        user_text: str,
        generation: int,
        fallback_sent: threading.Event,
        speech_queued: threading.Event,
        tool_activity: threading.Event,
        plan_payload: dict[str, Any] | None = None,
        prefetches: dict[tuple[str, str], dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
        turn_id: str = "",
        state: ToolTurnState | None = None,
    ):
        state = state or ToolTurnState()

        def turn_metadata(**metadata: Any) -> dict[str, Any]:
            if turn_id:
                metadata.setdefault("turn_id", turn_id)
            return metadata

        def suppress_result(
            name: str,
            safe_args: dict[str, Any],
            event_kind: str,
            *,
            ok: bool,
            error: str = "",
            message: str = "",
            instruction: str = "",
            extra: dict[str, Any] | None = None,
        ) -> str:
            payload: dict[str, Any] = {"ok": ok}
            if error:
                payload["error"] = error
            if message:
                payload["message"] = message
                payload["duplicate"] = True
            if instruction:
                payload["instruction"] = instruction
            if extra:
                payload.update(extra)
            self.store.add_event(
                event_kind,
                role="system",
                lane=self.args.back,
                content=name,
                metadata=turn_metadata(tool_name=name, arguments=safe_args, **(extra or {})),
            )
            return json.dumps(payload, ensure_ascii=False)

        def add_tool_completed(name: str, status: str, result: Any, *, ok: bool, extra: dict[str, Any] | None = None) -> None:
            self._record_tool_completed_event(name, status, result, ok=ok, turn_id=turn_id, extra=extra)

        def block_tool_call(name: str, safe_args: dict[str, Any], result_payload: dict[str, Any], *, blocked_reason: str = "", extra: dict[str, Any] | None = None) -> str:
            result = json.dumps(result_payload, ensure_ascii=False)
            metadata = {"arguments": safe_args}
            if blocked_reason:
                metadata["blocked_reason"] = blocked_reason
            if extra:
                metadata.update(extra)
            add_tool_completed(name, "failure", result, ok=False, extra=metadata)
            print(f"back lane tool: {self.store.tool_log_label(name, 'failure')} ({name}): {blocked_reason or result_payload.get('error')}", flush=True)
            return result

        def hook(name: str, function, arguments: dict[str, Any]):
            elapsed = time.monotonic() - started_at
            if (
                should_stop_for_initial_answer(
                    has_voice_facts=state.has_voice_facts,
                    user_text=user_text,
                    elapsed_seconds=elapsed,
                    budget_seconds=self.args.initial_answer_budget,
                    tool_name=name,
                )
            ):
                timed_out.set()
                if not fallback_sent.is_set():
                    fallback_sent.set()
                    self.store.add_event(
                        "initial_answer_budget_exhausted",
                        role="system",
                        lane=self.args.back,
                        content=name,
                        metadata={
                            "elapsed_seconds": round(elapsed, 3),
                            "budget_seconds": self.args.initial_answer_budget,
                            "tool_count": state.tool_count,
                            "reason": "voice facts already available; stop more research and answer",
                        },
                    )
                    self._queue_no_followup_fallback(user_text, generation, turn_id=turn_id)
                return json.dumps(
                    {
                        "ok": False,
                        "error": "initial answer budget exhausted",
                        "instruction": "stop calling tools; the voice assistant is already answering from available tool facts",
                    },
                    ensure_ascii=False,
                )
            if timed_out.is_set() or elapsed >= self.args.pro_turn_timeout or state.tool_count >= self.args.pro_tool_call_limit:
                timed_out.set()
                self.store.add_event(
                    "pro_tool_budget_exhausted",
                    role="system",
                    lane=self.args.back,
                    content=name,
                    metadata={"tool_count": state.tool_count, "limit": self.args.pro_tool_call_limit, "elapsed_seconds": round(elapsed, 3)},
                )
                return json.dumps(
                    {
                        "ok": False,
                        "error": "back lane turn budget exceeded",
                        "instruction": "stop calling tools and let the voice assistant give the best short answer from available context",
                    },
                    ensure_ascii=False,
                )
            state.mark_tool_started()
            tool_activity.set()
            short_name = short_tool_name(name)
            tool_decision = prepare_runtime_tool_call(short_name, arguments, user_text, plan_payload)
            safe_args = tool_decision.arguments
            if tool_decision.blocked_payload is not None:
                return block_tool_call(
                    name,
                    safe_args,
                    tool_decision.blocked_payload,
                    blocked_reason=tool_decision.blocked_reason,
                )
            preflight = evaluate_runtime_tool_preflight(short_name, safe_args, user_text, plan_payload, state)
            tool_signature = preflight.tool_signature
            if preflight.action == "suppress":
                result = suppress_result(
                    name,
                    safe_args,
                    preflight.event_kind,
                    ok=preflight.ok,
                    error=preflight.error,
                    message=preflight.message,
                    instruction=preflight.instruction,
                    extra=preflight.extra,
                )
                print(f"back lane tool: {preflight.event_kind} ({name})", flush=True)
                return result
            if preflight.action == "block":
                result = block_tool_call(name, safe_args, preflight.blocked_payload or {"ok": False}, blocked_reason=preflight.blocked_reason, extra=preflight.extra)
                if preflight.queue_voice_summary:
                    self._queue_tool_voice_summary(
                        name,
                        False,
                        result,
                        generation,
                        speech_queued,
                        arguments=safe_args,
                        user_text=user_text,
                        bypass_cooldown=True,
                        turn_id=turn_id,
                    )
                return result
            started_label = self.store.tool_log_label(name, "start")
            started_subject = tool_action_subject(name, safe_args, {}, user_text)
            started_spoken_text = format_tool_start_spoken(started_label, started_subject)
            speech_enabled = self.store.tool_speech_enabled(name)
            spoke_start = False
            cooldown_remaining = self.speech.tool_speech_cooldown_remaining(name)
            if speech_enabled:
                spoke_start = self.speech.speak_tool_summary_or_filler(
                    tool_name=name.split(":", 1)[-1],
                    status="start",
                    task_label=started_subject,
                    phrase=started_label,
                    spoken_text=started_spoken_text,
                    generation=generation,
                    start_wait_timeout=self.args.tool_start_speech_wait,
                    cooldown_seconds=self.args.tool_speech_cooldown,
                    turn_id=turn_id,
                )
            print(f"back lane tool: {started_spoken_text} ({name}) {json.dumps(safe_args, ensure_ascii=False)[:500]}", flush=True)
            tool_timing_id = self.store.start_turn_timing(
                turn_id,
                "tool_call",
                short_name,
                metadata={"tool_name": name, "arguments": safe_args, "spoken_start": spoke_start},
            )
            self.store.add_event(
                "tool_started",
                role="tool",
                lane=self.args.back,
                content=started_spoken_text,
                metadata=tool_started_event_metadata(
                    tool_name=name,
                    log_word=started_label,
                    action_subject=started_subject,
                    spoken_text=started_spoken_text,
                    spoke_start=spoke_start,
                    speech_enabled=speech_enabled,
                    cooldown_remaining=cooldown_remaining,
                    arguments=safe_args,
                    turn_id=turn_id,
                ),
            )
            try:
                result = self._prefetched_tool_result(name, safe_args, prefetches)
                if result is None:
                    result = self._call_tool_with_retries(name, function, safe_args)
                if inspect.isawaitable(result):
                    raise RuntimeError(f"async tool {name} cannot run through sync voice worker")
                ok = tool_result_ok(result)
                if ok:
                    state.mark_success(short_name, tool_signature, safe_args, result)
                success_label = self.store.tool_log_label(name, "success" if ok else "failure")
                if not ok:
                    state.mark_failed_signature(tool_signature)
                add_tool_completed(name, "success" if ok else "failure", result, ok=ok)
                print(f"back lane tool: {success_label} ({name})", flush=True)
                self._queue_tool_voice_summary(
                    name,
                    ok,
                    result,
                    generation,
                    speech_queued,
                    arguments=safe_args,
                    user_text=user_text,
                    bypass_cooldown=spoke_start or not ok,
                    turn_id=turn_id,
                )
                self.store.end_turn_timing(tool_timing_id, status="ok" if ok else "failed", metadata={"ok": ok})
                return result
            except Exception as exc:
                state.mark_failed_signature(tool_signature)
                failure_label = self.store.tool_log_label(name, "failure")
                add_tool_completed(name, "failure", {"error": str(exc)[:1000]}, ok=False, extra={"error": str(exc)[:1000]})
                print(f"back lane tool: {failure_label} ({name}): {exc}", flush=True)
                self._queue_tool_voice_summary(
                    name,
                    False,
                    {"error": str(exc)[:1000]},
                    generation,
                    speech_queued,
                    arguments=safe_args,
                    user_text=user_text,
                    bypass_cooldown=True,
                    turn_id=turn_id,
                )
                self.store.end_turn_timing(tool_timing_id, status="error", metadata={"error": str(exc)[:1000]})
                raise

        return hook

    def _queue_plan_voice(self, plan_payload: dict[str, Any] | None, generation: int, turn_id: str = "") -> None:
        if generation != self.speech.current_generation():
            return
        timing_id = self.store.start_turn_timing(turn_id, "plan_voice", "计划提示")
        steps = plan_payload.get("steps") if isinstance(plan_payload, dict) else []
        step_count = len(steps) if isinstance(steps, list) else 0
        text = "我先排一下顺序。"
        if step_count == 1:
            text = "我先理一下这一步。"
        elif step_count >= 3:
            text = "我先排一下顺序。"
        self.speech.stop_filler_loop()
        spoken = self.speech.speak_text_and_wait(
            text,
            generation=generation,
            timeout=self.args.plan_speech_wait,
            turn_id=turn_id,
        )
        self.store.add_event(
            "execution_plan_voice",
            role="assistant",
            lane="speech",
            content=text,
            metadata={"spoken": spoken, "step_count": step_count, "wait_seconds": self.args.plan_speech_wait, "turn_id": turn_id},
        )
        self.store.end_turn_timing(timing_id, status="spoken" if spoken else "timeout", metadata={"spoken": spoken, "step_count": step_count})

    def _queue_plan_summary_voice(self, plan_payload: dict[str, Any] | None, generation: int, turn_id: str = "") -> None:
        if generation != self.speech.current_generation():
            return
        timing_id = self.store.start_turn_timing(turn_id, "plan_summary_voice", "计划复述")
        text = format_plan_summary_for_voice(
            plan_payload,
            max_steps=self.args.plan_summary_max_steps,
            max_chars=self.args.plan_summary_max_chars,
        )
        if not text:
            self.store.end_turn_timing(timing_id, status="skipped", metadata={"reason": "empty"})
            return
        self.speech.stop_filler_loop()
        spoken = self.speech.speak_text_and_wait(
            text,
            generation=generation,
            timeout=self.args.plan_summary_speech_wait,
            turn_id=turn_id,
        )
        self.store.add_event(
            "execution_plan_summary_voice",
            role="assistant",
            lane="speech",
            content=text,
            metadata={"spoken": spoken, "wait_seconds": self.args.plan_summary_speech_wait, "max_steps": self.args.plan_summary_max_steps, "max_chars": self.args.plan_summary_max_chars, "turn_id": turn_id},
        )
        self.store.end_turn_timing(timing_id, status="spoken" if spoken else "timeout", metadata={"spoken": spoken, "chars": len(text)})

    def _queue_tool_voice_summary(
        self,
        name: str,
        ok: bool,
        result: Any,
        generation: int,
        speech_queued: threading.Event,
        *,
        arguments: dict[str, Any] | None = None,
        user_text: str = "",
        bypass_cooldown: bool = False,
        turn_id: str = "",
    ) -> None:
        if generation != self.speech.current_generation():
            return
        if short_tool_name(name) == "trigger_fast_followup" and ok:
            self.speech.stop_filler_loop()
            if self._drain_followups(generation, turn_id=turn_id):
                speech_queued.set()
            return
        if short_tool_name(name) == "daily_action" and ok:
            phrase = self.store.tool_voice_summary(name, ok, result)
            metadata = tool_voice_summary_event_metadata(
                tool_name=name,
                ok=ok,
                phrase=phrase,
                spoken=False,
                speech_enabled=self.store.tool_speech_enabled(name),
                cooldown_remaining=0.0,
                cooldown_bypassed=False,
                turn_id=turn_id,
            )
            metadata["speech_suppressed_reason"] = "fat_tool_success"
            self.store.add_event(
                "tool_voice_summary",
                role="assistant",
                lane="speech",
                content=phrase,
                metadata=metadata,
            )
            return
        if not self.store.tool_speech_enabled(name):
            phrase = self.store.tool_voice_summary(name, ok, result)
            self.store.add_event(
                "tool_voice_summary",
                role="assistant",
                lane="speech",
                content=phrase,
                metadata=tool_voice_summary_event_metadata(
                    tool_name=name,
                    ok=ok,
                    phrase=phrase,
                    spoken=False,
                    speech_enabled=False,
                    turn_id=turn_id,
                ),
            )
            return
        self.speech.stop_filler_loop()
        phrase = self.store.tool_voice_summary(name, ok, result)
        summary = phrase
        if not summary:
            return
        status = "success" if ok else "failure"
        cooldown_remaining = self.speech.tool_speech_cooldown_remaining(name)
        spoke_summary = self.speech.speak_tool_summary_or_filler(
            tool_name=name.split(":", 1)[-1],
            status=status,
            task_label="",
            phrase=phrase,
            spoken_text=summary,
            generation=generation,
            cooldown_seconds=self.args.tool_speech_cooldown,
            ignore_cooldown=bypass_cooldown or not ok,
            turn_id=turn_id,
        )
        self.store.add_event(
            "tool_voice_summary",
            role="assistant",
            lane="speech",
            content=summary,
            metadata=tool_voice_summary_event_metadata(
                tool_name=name,
                ok=ok,
                phrase=phrase,
                spoken=spoke_summary,
                speech_enabled=True,
                cooldown_remaining=cooldown_remaining,
                cooldown_bypassed=bypass_cooldown or not ok,
                turn_id=turn_id,
            ),
        )
        if spoke_summary:
            self.store.add_event(
                "tool_progress_voice_only",
                role="system",
                lane="speech",
                content=summary,
                metadata={"tool_name": name, "turn_id": turn_id},
            )

    def _call_tool_with_retries(self, name: str, function, arguments: dict[str, Any]) -> Any:
        attempts = tool_attempts_for(name, self.args.pro_tool_retries)
        timeout = tool_timeout_for(name, arguments, self.args.pro_tool_timeout)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

            def target() -> None:
                try:
                    result_queue.put(("ok", function(**arguments)))
                except Exception as exc:
                    result_queue.put(("error", exc))

            threading.Thread(target=target, daemon=True).start()
            try:
                status, value = result_queue.get(timeout=timeout)
            except queue.Empty:
                last_error = TimeoutError(tool_timeout_error_message(name, timeout, attempt, attempts))
                self.store.add_event(
                    "tool_retry",
                    role="system",
                    lane=self.args.back,
                    content=name,
                    metadata=tool_retry_event_metadata(attempt=attempt, attempts=attempts, reason="timeout", timeout_seconds=timeout),
                )
            else:
                if status == "ok":
                    if attempt > 1:
                        self.store.add_event(
                            "tool_retry_recovered",
                            role="system",
                            lane=self.args.back,
                            content=name,
                            metadata={"attempt": attempt, "attempts": attempts},
                        )
                    return value
                last_error = value if isinstance(value, Exception) else RuntimeError(str(value))
                self.store.add_event(
                    "tool_retry" if attempt < attempts else "tool_failed_attempt",
                    role="system",
                    lane=self.args.back,
                    content=name,
                    metadata=tool_retry_event_metadata(attempt=attempt, attempts=attempts, reason=str(last_error)),
                )
            if attempt < attempts:
                time.sleep(tool_retry_backoff_seconds(attempt))
        raise RuntimeError(str(last_error) if last_error else f"tool {name} failed")

    def _start_plan_prefetch(
        self,
        plan_payload: dict[str, Any] | None,
        tools: list[Any],
        user_text: str,
        generation: int,
        turn_id: str = "",
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not self.args.plan_prefetch:
            return {}
        function_map = callable_tool_map(tools)
        planned = planned_prefetch_calls(plan_payload, user_text)
        if not planned:
            return {}
        prefetches: dict[tuple[str, str], dict[str, Any]] = {}
        semaphore = threading.Semaphore(max(1, int(self.args.plan_prefetch_max_concurrency)))
        max_calls = max(0, int(self.args.plan_prefetch_max_tools))
        for item in planned[:max_calls]:
            tool_name = item["tool_name"]
            function = function_map.get(tool_name)
            if function is None:
                continue
            arguments = item["arguments"]
            key = tool_prefetch_key(tool_name, arguments)
            if key in prefetches:
                continue
            done = threading.Event()
            state: dict[str, Any] = {
                "done": done,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": None,
                "error": None,
                "started_at": time.perf_counter(),
                "timing_id": 0,
            }
            prefetches[key] = state

            def worker(state: dict[str, Any] = state, function=function, tool_name: str = tool_name, arguments: dict[str, Any] = arguments) -> None:
                if generation != self.speech.current_generation():
                    state["error"] = "stale generation"
                    state["done"].set()
                    return
                acquired = semaphore.acquire(timeout=0.1)
                if not acquired:
                    state["error"] = "prefetch concurrency full"
                    state["done"].set()
                    return
                try:
                    print(f"Plan prefetch start: {tool_name} {json.dumps(arguments, ensure_ascii=False)[:500]}", flush=True)
                    state["timing_id"] = self.store.start_turn_timing(
                        turn_id,
                        "plan_prefetch",
                        tool_name,
                        metadata={"tool_name": tool_name, "arguments": arguments},
                    )
                    state["result"] = self._call_tool_with_retries(tool_name, function, arguments)
                    state["ok"] = tool_result_ok(state["result"])
                except Exception as exc:
                    state["error"] = str(exc)[:1000]
                    state["ok"] = False
                finally:
                    state["elapsed_seconds"] = round(time.perf_counter() - float(state["started_at"]), 3)
                    semaphore.release()
                    state["done"].set()
                    self.store.end_turn_timing(
                        int(state.get("timing_id") or 0),
                        status="ok" if bool(state.get("ok")) else "failed",
                        metadata={"ok": bool(state.get("ok")), "error": str(state.get("error") or "")[:1000]},
                    )
                    self.store.add_event(
                        "plan_prefetch_completed",
                        role="tool",
                        lane=self.args.back,
                        content=tool_name,
                        metadata=plan_prefetch_completed_metadata(state, turn_id=turn_id),
                    )
                    print(f"Plan prefetch done: {tool_name} ok={bool(state.get('ok'))} elapsed={state.get('elapsed_seconds')}", flush=True)

            threading.Thread(target=worker, daemon=True).start()
        if prefetches:
            self.store.add_event(
                "plan_prefetch_started",
                role="system",
                lane=self.args.back,
                content=f"{len(prefetches)} prefetches",
                metadata=plan_prefetch_started_metadata(list(prefetches.values()), turn_id=turn_id),
            )
        return prefetches

    def _prefetched_tool_result(
        self,
        name: str,
        arguments: dict[str, Any],
        prefetches: dict[tuple[str, str], dict[str, Any]] | None,
    ) -> Any | None:
        if not prefetches:
            return None
        key = tool_prefetch_key(name, arguments)
        state = prefetches.get(key)
        if state is None:
            return None
        done = state.get("done")
        if not isinstance(done, threading.Event):
            return None
        waited = done.wait(max(0.0, float(self.args.plan_prefetch_wait)))
        if not waited:
            self.store.add_event(
                "plan_prefetch_miss",
                role="system",
                lane=self.args.back,
                content=name,
                metadata=plan_prefetch_miss_metadata(
                    reason="prefetch still running",
                    wait_seconds=self.args.plan_prefetch_wait,
                    arguments=arguments,
                ),
            )
            return None
        if state.get("error"):
            self.store.add_event(
                "plan_prefetch_miss",
                role="system",
                lane=self.args.back,
                content=name,
                metadata=plan_prefetch_miss_metadata(
                    reason="prefetch failed",
                    error=state.get("error"),
                    arguments=arguments,
                ),
            )
            return None
        self.store.add_event(
            "plan_prefetch_hit",
            role="system",
            lane=self.args.back,
            content=name,
            metadata=plan_prefetch_hit_metadata(state, arguments),
        )
        print(f"Plan prefetch hit: {name}", flush=True)
        return state.get("result")

    def _execute_domain_probe_plan_locally(
        self,
        *,
        user_text: str,
        plan_payload: dict[str, Any] | None,
        tools: list[Any],
        generation: int,
        hook,
        speech_queued: threading.Event,
        tool_activity: threading.Event,
        since: float | None,
        turn_id: str = "",
    ) -> bool:
        if generation != self.speech.current_generation() or not self._is_local_domain_probe_plan(plan_payload):
            return False
        raw_steps = plan_payload.get("steps")
        steps = sorted((item for item in raw_steps if isinstance(item, dict)), key=plan_step_order)
        tool_map = callable_tool_map(tools)
        local_tools = {"daily_action", "computer_action"}
        if not any(name in tool_map for name in local_tools):
            return False
        self.store.add_event(
            "domain_probe_local_execution",
            role="system",
            lane="local",
            content=f"{len(steps)} domain probe steps",
            metadata={"turn_id": turn_id, "step_count": len(steps)},
        )
        ran_any = False
        failures = 0
        for step in steps:
            args_by_tool = step.get("arguments")
            step_tools = plan_step_tools(step)
            tool_name = step_tools[0] if step_tools else ""
            if tool_name not in local_tools:
                continue
            function = tool_map.get(tool_name)
            if function is None:
                continue
            args = args_by_tool.get(tool_name) if isinstance(args_by_tool, dict) else None
            if not isinstance(args, dict):
                continue
            if generation != self.speech.current_generation():
                return ran_any
            ran_any = True
            try:
                result = hook(tool_name, function, args)
                ok = tool_result_ok(result)
                if not ok:
                    failures += 1
                self.store.add_event(
                    "domain_probe_local_execution_result",
                    role="system",
                    lane="local",
                    content=tool_name,
                    metadata={"turn_id": turn_id, "tool_name": tool_name, "arguments": args, "ok": ok, "result": coerce_json_value(result)},
                )
                tool_activity.set()
            except Exception as exc:
                failures += 1
                self.store.add_event(
                    "domain_probe_local_execution_error",
                    role="system",
                    lane="local",
                    content=tool_name,
                    metadata={"turn_id": turn_id, "tool_name": tool_name, "arguments": args, "error": str(exc)[:1000]},
                )
        if not ran_any:
            return False
        self.store.add_event(
            "domain_probe_local_execution_done",
            role="system",
            lane="local",
            content="domain probe local steps completed; back summary required",
            metadata={"turn_id": turn_id, "step_count": len(steps), "failures": failures, "best_effort": True},
        )
        return True

    def _is_local_domain_probe_plan(self, plan_payload: dict[str, Any] | None) -> bool:
        if not isinstance(plan_payload, dict) or str(plan_payload.get("source") or "") != "domain_probe":
            return False
        raw_steps = plan_payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return False
        steps = [item for item in raw_steps if isinstance(item, dict)]
        local_tools = {"daily_action", "computer_action"}
        return bool(steps) and all(len(plan_step_tools(step)) == 1 and plan_step_tools(step)[0] in local_tools for step in steps)

    def _execute_plan_recovery(self, user_text: str, plan_payload: dict[str, Any] | None, tools: list[Any], generation: int, turn_id: str = "") -> bool:
        if generation != self.speech.current_generation() or not isinstance(plan_payload, dict):
            return False
        raw_steps = plan_payload.get("steps")
        if not isinstance(raw_steps, list):
            return False
        function_map = callable_tool_map(tools)
        verified_urls: list[str] = []
        opened_video_urls: set[str] = set()
        did_action = False
        for step in sorted((item for item in raw_steps if isinstance(item, dict)), key=plan_step_order):
            for tool_name in plan_step_tools(step):
                name = str(tool_name or "").split(":", 1)[-1]
                if name not in PLAN_RECOVERY_ALLOWED_TOOLS:
                    continue
                function = function_map.get(name)
                if function is None:
                    continue
                args = plan_recovery_tool_args(name, step, user_text, plan_payload, verified_urls)
                if args is None:
                    continue
                if name == "open_url_in_browser":
                    duplicate_video_url = duplicate_video_open_url(user_text, str(args.get("url") or ""), opened_video_urls)
                    if duplicate_video_url:
                        self.store.add_event(
                            "tool_duplicate_open_suppressed",
                            role="system",
                            lane=self.args.back,
                            content=name,
                            metadata={"tool_name": name, "arguments": args, "mode": "plan_recovery", "opened_url": duplicate_video_url, "opened_video_urls": sorted(opened_video_urls), "turn_id": turn_id},
                        )
                        continue
                try:
                    result = self._call_tool_with_retries(name, function, args)
                    ok = tool_result_ok(result)
                    self.store.record_tool_event(name, args, result, ok=ok, turn_id=turn_id)
                    label = self._record_tool_completed_event(
                        name,
                        "success" if ok else "failure",
                        result,
                        ok=ok,
                        turn_id=turn_id,
                        extra={"mode": "plan_recovery"},
                    )
                    print(f"Plan recovery tool: {label} ({name})", flush=True)
                    if ok:
                        did_action = True
                        verified_urls.extend(sorted(extract_verified_urls_from_tool_result(name, args, result)))
                        opened_url = str(args.get("url") or "").strip()
                        if name == "open_url_in_browser" and looks_like_video_url(opened_url):
                            opened_video_urls.add(opened_url)
                except Exception as exc:
                    self._record_tool_completed_event(
                        name,
                        "failure",
                        {"error": str(exc)[:1000]},
                        ok=False,
                        turn_id=turn_id,
                        extra={"mode": "plan_recovery", "error": str(exc)[:1000]},
                    )
        if did_action:
            self.speech.stop_filler_loop()
            self.speech.speak("我处理了一下，你看现在的窗口。", interrupt=False, generation=generation, turn_id=turn_id)
        return did_action

    def _record_tool_completed_event(
        self,
        name: str,
        status: str,
        result: Any,
        *,
        ok: bool,
        turn_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> str:
        label = self.store.tool_log_label(name, status)
        metadata = {
            "tool_name": name,
            "log_word": label,
            "log_language": "zh",
            "ok": ok,
            "summary": self.store.tool_voice_summary(name, ok, result),
        }
        if turn_id:
            metadata["turn_id"] = turn_id
        if extra:
            metadata.update(extra)
        self.store.add_event(
            "tool_completed",
            role="tool",
            lane=self.args.back,
            content=label,
            metadata=metadata,
        )
        return label

    def _record_run_output_tools(self, run_output: Any) -> None:
        for tool in getattr(run_output, "tools", None) or []:
            tool_name = getattr(tool, "tool_name", "") or ""
            if not tool_name:
                continue
            tool_args = getattr(tool, "tool_args", {}) or {}
            result = getattr(tool, "result", None)
            ok = (not bool(getattr(tool, "tool_call_error", False))) and tool_result_ok(result)
            label = self.store.tool_log_label(tool_name, "success" if ok else "failure")
            summary = self.store.tool_voice_summary(tool_name, ok, result)
            self.store.record_tool_event(
                f"agno:{tool_name}",
                coerce_json_object(tool_args),
                {"result": coerce_json_value(result), "source": "RunOutput.tools"},
                ok=ok,
            )
            self.store.add_event(
                "agno_tool_output",
                role="tool",
                lane=self.args.back,
                content=label,
                metadata=agno_tool_output_metadata(
                    tool_name=tool_name,
                    log_word=label,
                    ok=ok,
                    arguments=coerce_json_value(tool_args),
                    result=coerce_json_value(result),
                    summary=summary,
                ),
            )

    def _probe_domains(self, user_text: str, turn_id: str = "") -> dict[str, Any]:
        timing_id = self.store.start_turn_timing(turn_id, "domain_probe", "Domain Probe")
        try:
            payload = probe_domains(user_text, context=self.store.domain_probe_context(recent_limit=6))
            domains = payload.get("domains") if isinstance(payload, dict) else []
            registered_tool_count = self.store._registered_tool_count_from_domain_probe(json.dumps(payload, ensure_ascii=False), {})
            self.store.add_event(
                "domain_probe",
                role="system",
                lane="local",
                content=json.dumps(payload, ensure_ascii=False),
                metadata={
                    "turn_id": turn_id,
                    "domain_count": len(domains) if isinstance(domains, list) else 0,
                    "registered_tool_count": registered_tool_count,
                    "duration_ms": payload.get("duration_ms") if isinstance(payload, dict) else None,
                },
            )
            self.store.end_turn_timing(
                timing_id,
                status="ok",
                metadata={
                    "domain_count": len(domains) if isinstance(domains, list) else 0,
                    "registered_tool_count": registered_tool_count,
                },
            )
            return payload
        except Exception as exc:
            self.store.add_event(
                "domain_probe_error",
                role="system",
                lane="local",
                content="domain probe failed",
                metadata={"turn_id": turn_id, "error": str(exc)[:1000]},
            )
            self.store.end_turn_timing(timing_id, status="error", metadata={"error": str(exc)[:1000]})
            return {}

    def _make_execution_plan(self, user_text: str, tools: list[Any], turn_id: str = "", domain_probe: dict[str, Any] | None = None) -> dict[str, Any] | None:
        timing_id = self.store.start_turn_timing(turn_id, "plan", "计划", metadata={"model": self.args.plan})
        tool_names = [getattr(t, "__name__", t.__class__.__name__) for t in tools]
        probe_plan = execution_plan_from_domain_probe(domain_probe)
        if probe_plan:
            self.store.add_event(
                "execution_plan",
                role="system",
                lane="local",
                content=json.dumps(probe_plan, ensure_ascii=False),
                metadata={"transcript": user_text[:500], "tool_count": len(tool_names), "turn_id": turn_id, "planner": "domain_probe"},
            )
            self.store.end_turn_timing(timing_id, status="domain_probe", metadata={"tool_count": len(tool_names), "step_count": len(probe_plan.get("steps") or [])})
            return probe_plan
        heuristic = heuristic_execution_plan(user_text)
        if heuristic_plan_is_confident(heuristic):
            heuristic["source"] = "heuristic_confident"
            self.store.add_event(
                "execution_plan",
                role="system",
                lane="local",
                content=json.dumps(heuristic, ensure_ascii=False),
                metadata={"transcript": user_text[:500], "tool_count": len(tool_names), "turn_id": turn_id, "planner": "heuristic_confident"},
            )
            self.store.end_turn_timing(timing_id, status="heuristic", metadata={"tool_count": len(tool_names)})
            return heuristic
        prompt = (
            current_context_prompt()
            + "\n\n共享上下文 JSON:\n"
            + self.store.context_bundle(recent_limit=8)
            + "\n\n"
            + format_domain_probe_prompt(domain_probe)
            + "\n\n可用工具名:\n"
            + json.dumps(tool_names, ensure_ascii=False)
            + "\n\n用户最新输入:\n"
            + user_text
        )
        try:
            result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

            def target() -> None:
                try:
                    llm_started = time.monotonic()
                    self.store.add_event(
                        "llm_call",
                        role="system",
                        lane=self.args.plan,
                        content=f"plan {self.args.plan} started",
                        metadata={"turn_id": turn_id, "phase": "plan", "model": self.args.plan, "status": "started"},
                    )
                    run = self.factory.agent(
                        self.args.plan,
                        PLAN_SYSTEM_PROMPT,
                        reasoning_effort=getattr(self.args, "reasoning_effort", None),
                    ).run(prompt)
                    self.store.add_event(
                        "llm_call",
                        role="system",
                        lane=self.args.plan,
                        content=f"plan {self.args.plan} ok",
                        metadata={
                            "turn_id": turn_id,
                            "phase": "plan",
                            "model": self.args.plan,
                            "status": "ok",
                            "duration_seconds": round(time.monotonic() - llm_started, 3),
                        },
                    )
                    result_queue.put(("ok", run))
                except Exception as exc:
                    self.store.add_event(
                        "llm_call",
                        role="system",
                        lane=self.args.plan,
                        content=f"plan {self.args.plan} error",
                        metadata={
                            "turn_id": turn_id,
                            "phase": "plan",
                            "model": self.args.plan,
                            "status": "error",
                            "error": str(exc)[:1000],
                        },
                    )
                    result_queue.put(("error", exc))

            threading.Thread(target=target, daemon=True).start()
            try:
                status, value = result_queue.get(timeout=self.args.plan_timeout)
            except queue.Empty:
                fallback = heuristic_execution_plan(user_text)
                self.store.add_event(
                    "execution_plan_timeout",
                    role="system",
                    lane=self.args.plan,
                    content=json.dumps(fallback, ensure_ascii=False),
                    metadata={
                        "timeout_seconds": self.args.plan_timeout,
                        "transcript": user_text[:500],
                        "tool_count": len(tool_names),
                        "fallback": "heuristic",
                        "turn_id": turn_id,
                    },
                )
                print(f"Plan timeout after {self.args.plan_timeout:.1f}s; using heuristic plan.", flush=True)
                self.store.end_turn_timing(timing_id, status="timeout", metadata={"timeout_seconds": self.args.plan_timeout, "fallback": "heuristic"})
                return fallback
            if status == "error":
                raise value if isinstance(value, Exception) else RuntimeError(str(value))
            run = value
            raw = normalize_assistant_text(str(getattr(run, "content", "") or ""))
            payload = parse_json_object(raw)
            if not isinstance(payload, dict):
                payload = {"steps": [], "raw": raw[:2000]}
            steps = payload.get("steps")
            if not isinstance(steps, list):
                payload["steps"] = []
            self.store.add_event(
                "execution_plan",
                role="system",
                lane=self.args.plan,
                content=json.dumps(payload, ensure_ascii=False),
                metadata={"transcript": user_text[:500], "tool_count": len(tool_names), "turn_id": turn_id},
            )
            self.store.end_turn_timing(timing_id, status="ok", metadata={"tool_count": len(tool_names), "step_count": len(payload.get("steps") or [])})
            return payload
        except Exception as exc:
            fallback = heuristic_execution_plan(user_text)
            self.store.add_event(
                "execution_plan_error",
                role="system",
                lane=self.args.plan,
                content=json.dumps(fallback, ensure_ascii=False),
                metadata={"error": str(exc)[:1000], "transcript": user_text[:500], "fallback": "heuristic", "turn_id": turn_id},
            )
            self.store.end_turn_timing(timing_id, status="error", metadata={"error": str(exc)[:1000], "fallback": "heuristic"})
            return fallback

    def _run_json_fallback(self, user_text: str, reason: str, generation: int | None = None) -> bool:
        self.store.record_tool_event(
            "back_tool_calling_unavailable",
            {"lane": self.args.back},
            {"error": reason[:1000], "fallback": "json_actions"},
            ok=False,
        )
        self.store.add_event(
            "pro_error",
            role="system",
            lane=self.args.back,
            content="tool calling unavailable; using json fallback",
            metadata={"error": reason[:1000]},
        )
        prompt = (
            current_context_prompt() +
            "\n\n共享上下文 JSON:\n" + self.store.context_bundle() +
            "\n\n用户最新输入:\n" + user_text +
            "\n\n注意：这段用户输入已经来自录音 ASR 的成功转写，不要要求用户提供同一段音频。"
            "\n请输出 JSON action plan。"
        )
        try:
            llm_started = time.monotonic()
            self.store.add_event(
                "llm_call",
                role="system",
                lane=self.args.back,
                content=f"json_fallback {self.args.back} started",
                metadata={"phase": "json_fallback", "model": self.args.back, "status": "started"},
            )
            run = self.factory.agent(
                self.args.back,
                PRO_JSON_FALLBACK_PROMPT,
                reasoning_effort=getattr(self.args, "reasoning_effort", None),
            ).run(prompt)
            self.store.add_event(
                "llm_call",
                role="system",
                lane=self.args.back,
                content=f"json_fallback {self.args.back} ok",
                metadata={
                    "phase": "json_fallback",
                    "model": self.args.back,
                    "status": "ok",
                    "duration_seconds": round(time.monotonic() - llm_started, 3),
                },
            )
            raw = normalize_assistant_text(str(getattr(run, "content", "") or ""))
            payload = parse_json_object(raw)
            if not payload:
                if raw:
                    self.store.add_event(
                        "pro_internal_note",
                        role="system",
                        lane=self.args.back,
                        content="json fallback produced non-json note",
                        metadata={"mode": "json_fallback", "model_note": raw[:2000]},
                    )
                return False
            self._apply_json_fallback_payload(payload)
        except Exception as exc:
            self.store.add_event(
                "llm_call",
                role="system",
                lane=self.args.back,
                content=f"json_fallback {self.args.back} error",
                metadata={"phase": "json_fallback", "model": self.args.back, "status": "error", "error": str(exc)[:1000]},
            )
            self.store.add_event("pro_error", role="system", lane=self.args.back, content="json fallback failed", metadata={"error": str(exc)[:1000]})
            return False
        handled = self._drain_followups(generation)
        if not handled:
            self._queue_no_followup_fallback(user_text, generation if generation is not None else self.speech.current_generation())
            return True
        return handled

    def _apply_json_fallback_payload(self, payload: dict[str, Any]) -> None:
        for note in payload.get("context_notes") or []:
            if isinstance(note, str) and note.strip():
                if not long_term_memory_requested(self.store.latest_user_transcript()):
                    self.store.record_tool_event(
                        "add_context_note",
                        {"note": note},
                        {"result": "context note skipped; user did not explicitly ask to remember", "mode": "json_fallback"},
                        ok=False,
                    )
                    continue
                result = self.store.add_context_note(note)
                self.store.record_tool_event("add_context_note", {"note": note}, {"result": result, "mode": "json_fallback"})
        for task in payload.get("task_updates") or []:
            if not isinstance(task, dict):
                continue
            title = str(task.get("title", "")).strip()
            if not title:
                continue
            status = str(task.get("status", "pending")).strip() or "pending"
            summary = str(task.get("summary", "")).strip()
            self.store.update_task_status(title, status, summary)
        for followup in payload.get("fast_followups") or []:
            if not isinstance(followup, dict):
                continue
            prompt = str(followup.get("prompt", "")).strip()
            if not prompt:
                continue
            priority = int(followup.get("priority", 10) or 10)
            self.store.trigger_fast_followup(prompt, priority)

    def _drain_followups(self, generation: int, turn_id: str = "") -> bool:
        if generation != self.speech.current_generation():
            self.speech.stop_filler_loop()
            return True
        while True:
            item = self.store.pop_pending_fast_prompt()
            if item is None:
                return False
            prompt = str(item["prompt"])
            priority = int(item["priority"] or 0)
            followup_action = classify_followup_action(
                prompt,
                priority,
                interrupt_threshold=self.args.followup_interrupt_priority,
                speak_threshold=self.args.followup_speak_priority,
            )
            if followup_action == "defer":
                self.store.add_event(
                    "followup_deferred",
                    role="system",
                    lane=self.args.back,
                    content=prompt,
                    metadata={"priority": priority, "threshold": self.args.followup_interrupt_priority, "turn_id": turn_id},
                )
                continue
            if followup_action == "context_only":
                self.speech.stop_filler_loop()
                cleared = self.store.clear_pending_fast_prompts("context_only")
                self.store.add_event(
                    "followup_context_only",
                    role="system",
                    lane=self.args.back,
                    content=prompt,
                    metadata={"priority": priority, "speak_threshold": self.args.followup_speak_priority, "cleared_pending": cleared, "turn_id": turn_id},
                )
                return True
            if followup_action == "suppress_status":
                self.store.add_event(
                    "followup_suppressed",
                    role="system",
                    lane=self.args.back,
                    content=prompt,
                    metadata={"reason": "status_only_followup", "turn_id": turn_id},
                )
                continue
            if followup_action == "error_fallback":
                force_say = True
                self.store.add_event(
                    "followup_suppressed",
                    role="system",
                    lane=self.args.back,
                    content=prompt,
                    metadata={"reason": "error_like_followup", "turn_id": turn_id},
                )
                self.store.clear_pending_fast_prompts("suppressed")
                filler_stop = self.speech.start_filler_loop(
                    "blocked",
                    initial_delay=0.0,
                    interval_range=(self.args.filler_min_interval, self.args.filler_max_interval),
                )
                response = "这个我还在处理，先不播内部错误。"
            else:
                force_say = False
                duplicate = self._duplicate_recent_reply(prompt)
                if duplicate is not None:
                    self.speech.stop_filler_loop()
                    self.store.add_event(
                        "followup_deduped",
                        role="system",
                        lane=self.args.back,
                        content=prompt,
                        metadata={
                            "reason": "prompt_similar_to_recent_fast_reply",
                            "similarity": duplicate["similarity"],
                            "recent_reply": duplicate["content"][:500],
                            "turn_id": turn_id,
                        },
                    )
                    self.store.clear_pending_fast_prompts("deduped_prompt")
                    return True
                filler_stop = self.speech.start_filler_loop(
                    "transition",
                    initial_delay=0.0,
                    interval_range=(self.args.filler_min_interval, self.args.filler_max_interval),
                )
                response = prompt
                duplicate = self._duplicate_recent_reply(response, skip_current_pro_followup=True)
                if duplicate is not None:
                    filler_stop.set()
                    self.speech.stop_filler_loop()
                    self.store.add_event(
                        "followup_deduped",
                        role="system",
                        lane=self.args.back,
                        content=response,
                        metadata={
                            "reason": "response_similar_to_recent_fast_reply",
                            "similarity": duplicate["similarity"],
                            "recent_reply": duplicate["content"][:500],
                            "turn_id": turn_id,
                        },
                    )
                    self.store.clear_pending_fast_prompts("deduped_response")
                    return True
                self.store.add_event(
                    "assistant_reply",
                    role="assistant",
                    lane=self.args.back,
                    content=response,
                    metadata={"reason": "pro_followup_direct", "turn_id": turn_id},
                )
            filler_stop.set()
            self.speech.stop_filler_loop()
            self.speech.speak(
                response,
                interrupt=False,
                generation=generation,
                force_say=force_say,
                quick_say_fallback=not force_say,
                turn_id=turn_id,
            )
            self.store.add_event(
                "followup_spoken",
                role="assistant",
                lane="speech",
                content=response,
                metadata={"priority": priority, "force_say": force_say, "turn_id": turn_id},
            )
            self.store.clear_pending_fast_prompts("spoken")
            return True

    def _queue_no_followup_fallback(self, user_text: str, generation: int, since: float | None = None, turn_id: str = "") -> None:
        if generation != self.speech.current_generation():
            self.speech.stop_filler_loop()
            return
        recent_tools = self.store.recent_tool_events(within_seconds=90.0, limit=8, ok_only=False, since=since)
        tool_context = recent_tool_context_from_rows(recent_tools)
        if self._recover_missing_video_open(user_text, tool_context, generation, turn_id=turn_id):
            return
        prompt, tool_facts = no_followup_fallback_prompt(user_text, tool_context)
        self.store.add_event(
            "pro_no_followup_fallback",
            role="system",
            lane=self.args.back,
            content="back lane finished without trigger_fast_followup; front fallback requested",
            metadata={"transcript": user_text[:500], "recent_tools": tool_context[:5], "tool_facts": tool_facts[:2000], "turn_id": turn_id},
        )
        if not tool_context:
            self.speech.stop_filler_loop()
            self.speech.speak(
                "这个还没执行成功，我再处理一下。",
                interrupt=False,
                generation=generation,
                turn_id=turn_id,
            )
            return
        response = direct_fallback_response_from_tools(user_text, tool_context)
        self.store.add_event(
            "assistant_reply",
            role="assistant",
            lane=self.args.back,
            content=response,
            metadata={"reason": "pro_fallback_direct", "turn_id": turn_id},
        )
        self.speech.stop_filler_loop()
        self.speech.speak(response, interrupt=False, generation=generation, turn_id=turn_id)

    def _recover_missing_video_open(self, user_text: str, tool_context: list[dict[str, Any]], generation: int, turn_id: str = "") -> bool:
        compact = re.sub(r"\s+", "", str(user_text or "").lower())
        if not any(token in compact for token in ["播放", "打开", "放", "youtube", "视频", "mv", "mtv", "video"]):
            return False
        if any(str(item.get("tool") or "").split(":", 1)[-1] == "open_url_in_browser" and item.get("ok") for item in tool_context):
            return False
        candidates: list[str] = []
        for item in tool_context:
            if not item.get("ok"):
                continue
            for url in sorted(extract_urls_from_value(item)):
                if looks_like_video_url(url):
                    candidates.append(url)
        if not candidates:
            return False
        function = callable_tool_map(self._tools()).get("open_url_in_browser")
        if function is None:
            return False
        url = candidates[0]
        args = {"url": url, "browser": "Google Chrome", "fullscreen": False, "video_fullscreen": False}
        try:
            result = self._call_tool_with_retries("open_url_in_browser", function, args)
            ok = tool_result_ok(result)
            self.store.record_tool_event("open_url_in_browser", args, result, ok=ok, turn_id=turn_id)
            self._record_tool_completed_event(
                "open_url_in_browser",
                "success" if ok else "failure",
                result,
                ok=ok,
                turn_id=turn_id,
                extra={"mode": "no_followup_video_recovery"},
            )
            if ok:
                self.speech.stop_filler_loop()
                self.speech.speak("已经打开了。", interrupt=False, generation=generation, turn_id=turn_id)
                return True
        except Exception as exc:
            self._record_tool_completed_event(
                "open_url_in_browser",
                "failure",
                {"error": str(exc)[:1000], "url": url},
                ok=False,
                turn_id=turn_id,
                extra={"mode": "no_followup_video_recovery"},
            )
        return False

    def _duplicate_recent_reply(self, text: str, *, skip_current_pro_followup: bool = False) -> dict[str, Any] | None:
        recent = self.store.recent_assistant_replies(lane=self.args.front, within_seconds=self.args.followup_dedupe_seconds)
        filtered = []
        skipped_current = False
        for item in recent:
            try:
                metadata = json.loads(str(item.get("metadata_json") or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            if (
                skip_current_pro_followup
                and not skipped_current
                and metadata.get("reason") in {"pro_followup", "pro_followup_direct"}
                and text_similarity(text, str(item.get("content") or "")) >= 0.98
            ):
                skipped_current = True
                continue
            filtered.append(item)
        best: dict[str, Any] | None = None
        for item in filtered:
            similarity = text_similarity(text, str(item.get("content") or ""))
            if best is None or similarity > best["similarity"]:
                best = {"similarity": similarity, "content": str(item.get("content") or "")}
        if best is not None and best["similarity"] >= self.args.followup_dedupe_similarity:
            return best
        return None

    def _tools_for_domain_probe(self, domain_probe: dict[str, Any] | None, tools: list[Any]) -> list[Any]:
        selected_names = tool_names_for_domain_probe(domain_probe)
        if not selected_names:
            return []
        selected: list[Any] = []
        for tool in tools:
            name = getattr(tool, "__name__", tool.__class__.__name__)
            if name in selected_names:
                selected.append(tool)
        self.store.add_event(
            "tool_payload_selected",
            role="system",
            lane=self.args.back,
            content=", ".join(getattr(tool, "__name__", tool.__class__.__name__) for tool in selected) or "none",
            metadata={"selected_tools": [getattr(tool, "__name__", tool.__class__.__name__) for tool in selected], "source": "domain_probe"},
        )
        return selected

    def _tools(self) -> list[Any]:
        return build_voice_tools(self.args, self.store)
