from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from voice_assistant.agents import AgentFactory, FastLaneAgent, SessionCompressor
from voice_assistant.asr import transcribe_with_gjallarhorn
from voice_assistant.coding_monitor import CodingTaskMonitor
from voice_assistant.pro_lane import ProLaneWorker
from voice_assistant.recorder import Recorder
from voice_assistant.speech import SpeechQueue
from voice_assistant.store import VoiceSessionStore
from voice_assistant.voice_text import (
    classify_filler_stage,
    is_low_information_transcript,
    is_start_sound_echo,
    normalize_asr_transcript,
    strip_start_sound_echo_prefix,
    suppress_asr_hallucination,
    suppress_incomplete_fragment,
)

VOICE_ENTRYPOINT = Path(__file__).resolve().parents[1] / "jen_voice.py"


class VoiceBot:
    def __init__(self, args: argparse.Namespace, *, interactive: bool = True) -> None:
        self.args = args
        self.store = VoiceSessionStore(args.session_db, args.session_id)
        seeded_front_note_context = self.store.ensure_front_note_context_seeded()
        if seeded_front_note_context:
            print("Front note context seeded from voice session context.", flush=True)
        self.factory = AgentFactory(args, self.store)
        filler_dir = None if args.no_cached_filler else args.filler_audio_dir
        self.store.seed_filler_catalog_from_manifest(filler_dir)
        self.speech = SpeechQueue(
            args.say_voice,
            args.say_rate,
            disabled=args.no_say,
            filler_dir=filler_dir,
            speech_backend=args.speech_backend,
            gjallarhorn_base_url=args.gjallarhorn_base_url,
            api_key=args.api_key,
            tts_model=args.tts_model,
            tts_voice=args.tts_voice,
            tts_cache_dir=args.tts_cache_dir,
            tts_timeout=args.tts_timeout,
            tts_retries=args.tts_retries,
            tts_fallback_say=args.tts_fallback_say,
            start_sound_dir=args.start_sound_dir,
            verify_tls=args.verify_tls,
            store=self.store,
            speech_chunk_chars=args.speech_chunk_chars,
            speech_wait_chunk_chars=args.speech_wait_chunk_chars,
        )
        self.fast_agent = FastLaneAgent(args, self.store, self.factory)
        self.compressor = SessionCompressor(args, self.store, self.factory)
        self.coding_monitor = CodingTaskMonitor(self.store, self.speech)
        self.pro_worker = ProLaneWorker(
            args,
            self.store,
            self.factory,
            self.fast_agent,
            self.speech,
            on_complete=self.compressor.maybe_compress_async,
        )
        self.recorder = Recorder(args.recordings_dir)
        self._busy = threading.Lock()
        self._recording_press_lock = threading.Lock()
        self._recording_pressed = False
        self._starting_recording = False
        self._recording_mode = "quality"
        self._recording_session_id = 0
        self._pending_turn_lock = threading.Lock()
        self._pending_turn: dict[str, Any] | None = None
        self._pending_turn_id = 0
        self._rec_indicator_proc: subprocess.Popen | None = None
        self._front_note_proc: subprocess.Popen | None = None
        self._text_input_proc: subprocess.Popen | None = None
        if interactive:
            self.recorder.prepare()
            if not args.no_rec_indicator:
                self.ensure_recording_indicator()
            if not args.no_front_note:
                self.ensure_front_note()
            threading.Thread(target=self._prewarm_runtime, daemon=True).start()
            self.compressor.start()
            self.coding_monitor.start()

    def _prewarm_runtime(self) -> None:
        started = time.perf_counter()
        steps: list[str] = []
        try:
            from voice_assistant.daily_slot_parser import parse_daily_actions, parser_status

            parse_daily_actions("今天北京天气怎么样，然后提醒我明天喝水")
            status = parser_status()
            steps.append(f"daily_slot_parser={'ok' if status.get('available') else 'off'}")
        except Exception as exc:
            steps.append(f"daily_slot_parser_error={str(exc)[:120]}")
        try:
            from voice_assistant.tool_registry import warm_daily_runtime

            daily = warm_daily_runtime()
            steps.append(f"daily_runtime={'ok' if daily.get('ok') else 'off'}:{daily.get('seconds')}")
        except Exception as exc:
            steps.append(f"daily_runtime_error={str(exc)[:120]}")
        for phrase in [
            "我看一下 天气",
            "我看一下 提醒",
            "弄好了",
            "提醒已设好。",
        ]:
            try:
                self.speech.warm_tts_audio_async(phrase)
            except Exception as exc:
                steps.append(f"tts_warm_error={str(exc)[:80]}")
                break
        print(f"Runtime prewarm started in {time.perf_counter() - started:.2f}s: {', '.join(steps)}", flush=True)

    def start_recording(self, mode: str = "quality") -> None:
        if self.recorder.is_recording:
            return
        self.speech.stop()
        if self._busy.locked():
            print("Busy; ignoring press.", flush=True)
            return
        with self._recording_press_lock:
            if self._starting_recording:
                return
            self._recording_pressed = True
            self._recording_mode = "simple" if mode == "simple" else "quality"
            self._starting_recording = True
        try:
            self.show_recording_indicator()
            with self._recording_press_lock:
                if not self._recording_pressed:
                    self._starting_recording = False
                    self.hide_recording_indicator()
                    print("Recording cancelled before microphone start.", flush=True)
                    return
            self.speech.play_start_sound(blocking=True)
            self.recorder.start(include_preroll=False)
            with self._recording_press_lock:
                self._starting_recording = False
                self._recording_session_id += 1
                recording_session_id = self._recording_session_id
            key_label = "Right Command" if self._recording_mode == "simple" else "Right Option"
            print(f"Recording started ({self._recording_mode}). Release {key_label} to stop.", flush=True)
            self._schedule_recording_watchdog(recording_session_id, self._recording_mode)
        except Exception as exc:
            with self._recording_press_lock:
                self._starting_recording = False
                self._recording_pressed = False
            self.hide_recording_indicator()
            print(f"Record failed: {exc}", flush=True)
            self.speech.speak_error("录音启动出错了。")

    def _schedule_recording_watchdog(self, session_id: int, mode: str) -> None:
        max_seconds = max(3.0, float(getattr(self.args, "max_record_seconds", 45.0) or 45.0))

        def watchdog() -> None:
            with self._recording_press_lock:
                if session_id != self._recording_session_id or not self.recorder.is_recording:
                    return
                self._recording_pressed = False
            elapsed = time.monotonic() - self.recorder.started_at if self.recorder.started_at else max_seconds
            self.store.add_event(
                "recording_auto_stopped",
                role="system",
                lane="input",
                content="max_record_seconds",
                metadata={"mode": mode, "max_record_seconds": max_seconds, "elapsed_seconds": round(elapsed, 3)},
            )
            print(f"Recording auto-stopped after {elapsed:.2f}s; release event was probably missed.", flush=True)
            self.hide_recording_indicator()
            threading.Thread(target=self._finish_turn, daemon=True).start()

        timer = threading.Timer(max_seconds, watchdog)
        timer.daemon = True
        timer.start()

    def stop_recording(self, mode: str = "quality") -> None:
        with self._recording_press_lock:
            self._recording_pressed = False
        if not self.recorder.is_recording or self._busy.locked():
            self.hide_recording_indicator()
            return
        self.hide_recording_indicator()
        threading.Thread(target=self._finish_turn, daemon=True).start()

    def _finish_turn(self) -> None:
        if not self._busy.acquire(blocking=False):
            return
        try:
            recorded_seconds = time.monotonic() - self.recorder.started_at if self.recorder.started_at else 0.0
            stopped = self.recorder.stop()
            if stopped is None:
                print("No audio captured.", flush=True)
                self.alert_recording_indicator("没听清")
                return
            audio_path, audio_metrics = stopped
            print(f"Saved recording: {audio_path} ({recorded_seconds:.2f}s + preroll)", flush=True)
            if recorded_seconds < self.args.min_record_seconds:
                print(f"Recording ignored: too short ({recorded_seconds:.2f}s < {self.args.min_record_seconds:.2f}s)", flush=True)
                if self._short_recording_looks_like_cancel(audio_metrics):
                    self._cancel_short_recording(
                        audio_path,
                        recorded_seconds=recorded_seconds,
                        audio_metrics=audio_metrics,
                        mode=self._recording_mode,
                    )
                    return
                self._recording_ignored(audio_path, "too_short", "太短了", recorded_seconds=recorded_seconds, audio_metrics=audio_metrics)
                return
            if audio_metrics["rms"] < self.args.min_record_rms or audio_metrics["peak"] < self.args.min_record_peak:
                print(f"Recording ignored: too quiet (rms={audio_metrics['rms']:.5f}, peak={audio_metrics['peak']:.5f})", flush=True)
                self._recording_ignored(audio_path, "too_quiet", "没听清", recorded_seconds=recorded_seconds, audio_metrics=audio_metrics)
                return
            self._queue_pending_turn(
                audio_path,
                recorded_seconds=recorded_seconds,
                audio_metrics=audio_metrics,
                mode=self._recording_mode,
            )
        except Exception as exc:
            print(f"Turn failed: {exc}", flush=True)
            self.speech.speak_error("刚才处理录音出错了。", generation=self.speech.current_generation())
        finally:
            self._busy.release()

    def _queue_pending_turn(
        self,
        audio_path: Path,
        *,
        recorded_seconds: float | None,
        audio_metrics: dict[str, float],
        mode: str,
    ) -> None:
        trace_turn_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        delay = max(0.0, float(getattr(self.args, "cancel_tap_window", 0.45)))
        self.store.record_turn_timing(
            trace_turn_id,
            "recording",
            "录音",
            duration_seconds=max(0.0, float(recorded_seconds or 0.0)),
            metadata={"audio_path": str(audio_path), "interaction_mode": mode, **(audio_metrics or {})},
        )
        cancel_timing_id = self.store.start_turn_timing(
            trace_turn_id,
            "cancel_wait",
            "取消等待",
            metadata={"window_seconds": delay, "interaction_mode": mode},
        )
        with self._pending_turn_lock:
            self._pending_turn_id += 1
            turn_id = self._pending_turn_id
            self._pending_turn = {
                "id": turn_id,
                "turn_id": trace_turn_id,
                "cancel_timing_id": cancel_timing_id,
                "audio_path": audio_path,
                "recorded_seconds": recorded_seconds,
                "audio_metrics": audio_metrics,
                "mode": "simple" if mode == "simple" else "quality",
            }
        print(f"Recording pending for {delay:.2f}s cancel window ({mode}).", flush=True)
        timer = threading.Timer(delay, self._process_pending_turn, args=(turn_id,))
        timer.daemon = True
        timer.start()

    def cancel_pending_recording(self, mode: str = "quality") -> bool:
        cancelled = False
        with self._pending_turn_lock:
            pending = self._pending_turn
            if pending is not None and pending.get("mode") == ("simple" if mode == "simple" else "quality"):
                self._pending_turn = None
                cancelled = True
        if cancelled:
            turn_id = str((pending or {}).get("turn_id") or "")
            self.store.end_turn_timing(int((pending or {}).get("cancel_timing_id") or 0), status="cancelled")
            self.store.add_event("recording_cancelled", role="system", lane="input", content="quick_tap_cancel", metadata={"mode": mode, "turn_id": turn_id})
            self.speech.stop_filler_loop()
            self.speech.speak_cancel_immediate("回见。")
            self.store.record_turn_timing(turn_id, "cancel_sound", "取消音", metadata={"text": "回见。"})
            print(f"Recording cancelled by quick {mode} tap.", flush=True)
        return cancelled

    def _short_recording_looks_like_cancel(self, audio_metrics: dict[str, float] | None) -> bool:
        metrics = audio_metrics or {}
        return (
            float(metrics.get("rms") or 0.0) >= float(self.args.min_record_rms)
            and float(metrics.get("peak") or 0.0) >= float(self.args.min_record_peak)
        )

    def _cancel_short_recording(
        self,
        audio_path: Path,
        *,
        recorded_seconds: float | None,
        audio_metrics: dict[str, float] | None,
        mode: str,
    ) -> None:
        turn_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        metadata = {
            "mode": "simple" if mode == "simple" else "quality",
            "turn_id": turn_id,
            "audio_path": str(audio_path),
            "recorded_seconds": recorded_seconds,
            **(audio_metrics or {}),
        }
        self.store.add_event("recording_cancelled", role="system", lane="input", content="short_tap_cancel", metadata=metadata)
        self.speech.stop_filler_loop()
        self.speech.speak_cancel_immediate("回见。")
        self.store.record_turn_timing(turn_id, "cancel_sound", "取消音", metadata={"text": "回见。", "source": "short_tap"})
        print(f"Recording cancelled by short {mode} tap.", flush=True)

    def _process_pending_turn(self, turn_id: int) -> None:
        with self._pending_turn_lock:
            pending = self._pending_turn
            if pending is None or pending.get("id") != turn_id:
                return
            self._pending_turn = None
        if not self._busy.acquire(blocking=False):
            print("Busy; pending recording skipped.", flush=True)
            return
        try:
            self.store.end_turn_timing(int(pending.get("cancel_timing_id") or 0), status="ok")
            self.process_audio(
                pending["audio_path"],
                recorded_seconds=pending.get("recorded_seconds"),
                audio_metrics=pending.get("audio_metrics"),
                interaction_mode=str(pending.get("mode") or "quality"),
                turn_id=str(pending.get("turn_id") or ""),
            )
        except Exception as exc:
            print(f"Pending turn failed: {exc}", flush=True)
            self.speech.speak_error("刚才处理录音出错了。", generation=self.speech.current_generation())
        finally:
            self._busy.release()

    def process_audio(
        self,
        audio_path: Path,
        *,
        recorded_seconds: float | None = None,
        audio_metrics: dict[str, float] | None = None,
        interaction_mode: str = "quality",
        turn_id: str = "",
    ) -> str:
        turn_args = self._turn_args(interaction_mode)
        generation = self.speech.current_generation()
        turn_id = str(turn_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}")
        total_timing_id = self.store.start_turn_timing(turn_id, "turn_total", "整轮", metadata={"interaction_mode": interaction_mode})
        asr_timing_id = self.store.start_turn_timing(
            turn_id,
            "asr",
            "ASR",
            metadata={
                "audio_path": str(audio_path),
                "model": getattr(turn_args, "asr_model", ""),
                "language": getattr(turn_args, "language", ""),
                "prompted": bool(str(getattr(turn_args, "asr_prompt", "") or "").strip()),
            },
        )
        started = time.perf_counter()
        transcript = strip_start_sound_echo_prefix(
            suppress_incomplete_fragment(suppress_asr_hallucination(transcribe_with_gjallarhorn(audio_path, turn_args)))
        )
        raw_transcript = transcript
        transcript = normalize_asr_transcript(transcript)
        asr_seconds = time.perf_counter() - started
        self.store.end_turn_timing(asr_timing_id, status="ok" if transcript else "empty", metadata={"chars": len(transcript), "raw_chars": len(raw_transcript), "normalized": transcript != raw_transcript})
        print(f"Transcribed in {asr_seconds:.2f}s ({interaction_mode}): {transcript}", flush=True)
        if not transcript:
            print("No speech detected.", flush=True)
            self.store.end_turn_timing(total_timing_id, status="ignored", metadata={"reason": "empty_transcript"})
            self.alert_recording_indicator("没听清")
            return ""
        if is_start_sound_echo(transcript):
            print(f"Recording ignored: start sound echo ({transcript})", flush=True)
            self.store.end_turn_timing(total_timing_id, status="ignored", metadata={"reason": "start_sound_echo"})
            self._recording_ignored(
                audio_path,
                "start_sound_echo",
                "没听清",
                recorded_seconds=recorded_seconds,
                audio_metrics=audio_metrics,
                transcript=transcript,
            )
            return ""
        if is_low_information_transcript(transcript):
            print(f"Recording ignored: low information transcript ({transcript})", flush=True)
            self.store.end_turn_timing(total_timing_id, status="ignored", metadata={"reason": "low_information_transcript"})
            self._recording_ignored(
                audio_path,
                "low_information_transcript",
                "没听清",
                recorded_seconds=recorded_seconds,
                audio_metrics=audio_metrics,
                transcript=transcript,
            )
            return ""
        self.store.add_event(
            "transcript",
            role="user",
            lane="asr",
            content=transcript,
            metadata={
                "audio_path": str(audio_path),
                "raw_transcript": raw_transcript,
                "normalized": transcript != raw_transcript,
                "turn_id": turn_id,
                "interaction_mode": interaction_mode,
                "generation": generation,
                "recorded_seconds": recorded_seconds,
                **(audio_metrics or {}),
            },
        )
        self.store.record_turn_timing(turn_id, "transcript_saved", "转写入库", metadata={"chars": len(transcript), "interaction_mode": interaction_mode})
        self._handle_transcript(
            transcript,
            turn_args=turn_args,
            interaction_mode=interaction_mode,
            turn_id=turn_id,
            generation=generation,
            total_timing_id=total_timing_id,
            metadata={
                "audio_path": str(audio_path),
                "recorded_seconds": recorded_seconds,
                **(audio_metrics or {}),
            },
            lane="asr",
        )
        return ""

    def submit_text_input(self, text: str, mode: str = "quality") -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        threading.Thread(target=self._process_text_turn, args=(text, mode), daemon=True).start()
        return True

    def _process_text_turn(self, text: str, mode: str) -> None:
        if not self._busy.acquire(blocking=False):
            self.store.add_event(
                "text_input_rejected",
                role="system",
                lane="input",
                content="busy",
                metadata={"text": text[:500], "interaction_mode": mode},
            )
            self.speech.speak_error("我现在还在处理上一件事。")
            return
        try:
            self.process_text(text, interaction_mode=mode)
        finally:
            self._busy.release()

    def process_text(self, text: str, *, interaction_mode: str = "quality", turn_id: str = "") -> str:
        transcript = str(text or "").strip()
        if not transcript:
            return ""
        interaction_mode = "simple" if interaction_mode == "simple" else "quality"
        turn_args = self._turn_args(interaction_mode)
        generation = self.speech.current_generation()
        turn_id = str(turn_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}")
        total_timing_id = self.store.start_turn_timing(turn_id, "turn_total", "整轮", metadata={"interaction_mode": interaction_mode, "input": "text"})
        self.store.record_turn_timing(turn_id, "text_input", "文本输入", metadata={"chars": len(transcript), "interaction_mode": interaction_mode})
        self.store.add_event(
            "transcript",
            role="user",
            lane="text_input",
            content=transcript,
            metadata={
                "turn_id": turn_id,
                "interaction_mode": interaction_mode,
                "generation": generation,
                "input": "text",
            },
        )
        self.store.record_turn_timing(turn_id, "transcript_saved", "转写入库", metadata={"chars": len(transcript), "interaction_mode": interaction_mode, "input": "text"})
        self._handle_transcript(
            transcript,
            turn_args=turn_args,
            interaction_mode=interaction_mode,
            turn_id=turn_id,
            generation=generation,
            total_timing_id=total_timing_id,
            metadata={"input": "text"},
            lane="text_input",
        )
        return ""

    def _handle_transcript(
        self,
        transcript: str,
        *,
        turn_args: argparse.Namespace,
        interaction_mode: str,
        turn_id: str,
        generation: int,
        total_timing_id: int,
        metadata: dict[str, Any],
        lane: str,
    ) -> None:
        filler_stop: threading.Event | None = None
        filler_stage = classify_filler_stage(transcript, self.store.context_bundle(recent_limit=4))
        if self.args.model_waiting_filler:
            filler_timing_id = self.store.start_turn_timing(turn_id, "waiting_filler", "等待语")
            waiting_text = self.fast_agent.respond(transcript, reason="waiting_filler", turn_id=turn_id)
            self.speech.speak(waiting_text, interrupt=False, generation=generation, turn_id=turn_id)
            self.store.end_turn_timing(filler_timing_id, status="queued", metadata={"chars": len(waiting_text)})
        else:
            filler_stop = self.speech.start_filler_loop(
                filler_stage,
                initial_delay=0.0,
                interval_range=(self.args.filler_min_interval, self.args.filler_max_interval),
            )
            self.store.record_turn_timing(turn_id, "filler_loop", "filler", metadata={"stage": filler_stage})
            self.store.add_event(
                "filler_loop_started",
                role="assistant",
                lane="speech",
                content=filler_stage,
                metadata={"source": "cached_step_tts_mini", "stage": filler_stage, "turn_id": turn_id},
            )
        self.store.add_event(
            "fast_deferred",
            role="system",
            lane=turn_args.front,
            content="front lane deferred until back followup",
            metadata={
                "transcript": transcript,
                "interaction_mode": interaction_mode,
                "input_lane": lane,
                "back": turn_args.back,
                "plan": turn_args.plan,
                "plan_mode": turn_args.plan_mode,
                "reasoning_effort": getattr(turn_args, "reasoning_effort", None),
                "turn_id": turn_id,
                **metadata,
            },
        )
        self.pro_worker = ProLaneWorker(
            turn_args,
            self.store,
            self.factory,
            self.fast_agent,
            self.speech,
            on_complete=self.compressor.maybe_compress_async,
        )
        self.pro_worker.submit(transcript, generation, turn_id=turn_id)
        self.store.record_turn_timing(turn_id, "pro_submitted", "后台提交", metadata={"back": turn_args.back, "plan_mode": turn_args.plan_mode})

    def _recording_ignored(
        self,
        audio_path: Path,
        reason: str,
        alert_text: str,
        *,
        recorded_seconds: float | None,
        audio_metrics: dict[str, float] | None,
        transcript: str | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "audio_path": str(audio_path),
            "recorded_seconds": recorded_seconds,
            **(audio_metrics or {}),
        }
        if transcript is not None:
            metadata["transcript"] = transcript
        self.store.add_event(
            "recording_ignored",
            role="system",
            lane="asr",
            content=reason,
            metadata=metadata,
        )
        self.alert_recording_indicator(alert_text)

    def _turn_args(self, interaction_mode: str) -> argparse.Namespace:
        mode = "simple" if interaction_mode == "simple" else "quality"
        values = vars(self.args).copy()
        values["interaction_mode"] = mode
        if mode == "simple":
            values["plan_mode"] = False
            values["back"] = self.args.simple_back
            values["plan"] = self.args.simple_plan
            values["reasoning_effort"] = self.args.simple_reasoning_effort
        else:
            values["plan_mode"] = True
            values["back"] = self.args.quality_back
            values["plan"] = self.args.quality_plan
            values["reasoning_effort"] = self.args.quality_reasoning_effort
        return argparse.Namespace(**values)

    def show_recording_indicator(self) -> None:
        if self.args.no_rec_indicator:
            return
        proc = self.ensure_recording_indicator()
        if proc and proc.stdin:
            try:
                proc.stdin.write("show\n")
                proc.stdin.flush()
            except Exception as exc:
                print(f"Recording indicator failed: {exc}", flush=True)

    def hide_recording_indicator(self) -> None:
        proc = self._rec_indicator_proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._rec_indicator_proc = None
            return
        try:
            if proc.stdin:
                proc.stdin.write("hide\n")
                proc.stdin.flush()
        except Exception:
            self.stop_recording_indicator()

    def alert_recording_indicator(self, text: str) -> None:
        if self.args.no_rec_indicator:
            return
        proc = self.ensure_recording_indicator()
        if proc and proc.stdin:
            try:
                proc.stdin.write(f"notice:{text}\n")
                proc.stdin.flush()
            except Exception as exc:
                print(f"Recording indicator alert failed: {exc}", flush=True)

    def show_text_input(self, mode: str = "quality") -> None:
        if self.args.no_text_input:
            return
        proc = self.ensure_text_input()
        if proc and proc.stdin:
            try:
                proc.stdin.write(f"show:{'simple' if mode == 'simple' else 'quality'}\n")
                proc.stdin.flush()
            except Exception as exc:
                print(f"Text input failed: {exc}", flush=True)
                self.stop_text_input()

    def ensure_recording_indicator(self) -> subprocess.Popen | None:
        if self._rec_indicator_proc is not None and self._rec_indicator_proc.poll() is None:
            return self._rec_indicator_proc
        try:
            self._rec_indicator_proc = subprocess.Popen(
                [sys.executable, str(VOICE_ENTRYPOINT), "--overlay-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            return self._rec_indicator_proc
        except Exception as exc:
            print(f"Recording indicator failed: {exc}", flush=True)
            return None

    def stop_recording_indicator(self) -> None:
        proc = self._rec_indicator_proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.terminate()
        self._rec_indicator_proc = None

    def ensure_front_note(self) -> subprocess.Popen | None:
        if self._front_note_proc is not None and self._front_note_proc.poll() is None:
            if self._front_note_proc.stdin:
                try:
                    self._front_note_proc.stdin.write("show\n")
                    self._front_note_proc.stdin.flush()
                except Exception:
                    pass
            return self._front_note_proc
        api_url = f"http://{self.args.tool_dashboard_host}:{self.args.tool_dashboard_port}/api/front-note"
        try:
            self.store.update_front_note(action="show", visible=True, position="right")
        except Exception:
            pass
        try:
            self._front_note_proc = subprocess.Popen(
                [sys.executable, str(VOICE_ENTRYPOINT), "--front-note-server", "--front-note-api-url", api_url],
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            if self._front_note_proc.stdin:
                self._front_note_proc.stdin.write("show\n")
                self._front_note_proc.stdin.flush()
            return self._front_note_proc
        except Exception as exc:
            print(f"Front note failed: {exc}", flush=True)
            return None

    def ensure_text_input(self) -> subprocess.Popen | None:
        if self._text_input_proc is not None and self._text_input_proc.poll() is None:
            return self._text_input_proc
        api_url = f"http://{self.args.tool_dashboard_host}:{self.args.tool_dashboard_port}/api/text-input"
        try:
            self._text_input_proc = subprocess.Popen(
                [sys.executable, str(VOICE_ENTRYPOINT), "--text-input-server", "--text-input-api-url", api_url],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            return self._text_input_proc
        except Exception as exc:
            print(f"Text input failed: {exc}", flush=True)
            return None

    def stop_text_input(self) -> None:
        proc = self._text_input_proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                proc.terminate()
        self._text_input_proc = None

    def stop_front_note(self) -> None:
        proc = self._front_note_proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                proc.terminate()
        self._front_note_proc = None

    def shutdown(self) -> None:
        self.speech.stop()
        self.stop_recording_indicator()
        self.stop_front_note()
        self.stop_text_input()
        self.compressor.stop()

    def wait_background(self, timeout: float | None = None) -> None:
        self.pro_worker.wait(timeout=timeout)
