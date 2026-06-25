from __future__ import annotations

import base64
import json
import queue
import random
import shutil
import ssl
import subprocess
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import error, request

from voice_assistant.filler import filler_stage_for_phrase
from voice_assistant.http_client import urlopen_bytes
from voice_assistant.speech_text import (
    DEFAULT_TTS_INSTRUCTIONS,
    filler_tts_instructions,
    normalize_assistant_text,
    split_speech_text,
)

SAY_FALLBACK_PREFIX = "它出去找资料挂树上了，我来总结一下。"


class MemoryAudioPlayer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: queue.Queue[tuple[Any, int, threading.Event]] = queue.Queue()
        self._stream: Any = None
        self._sample_rate = 0
        self._current: Any = None
        self._current_done: threading.Event | None = None
        self._position = 0

    def play(self, audio: Any, sample_rate: int, *, blocking: bool = False) -> bool:
        try:
            import numpy as np
        except Exception:
            return False
        data = np.asarray(audio, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if data.size <= 0:
            return False
        if not self._ensure_stream(int(sample_rate)):
            return False
        done = threading.Event()
        self._pending.put((data, int(sample_rate), done))
        if blocking:
            done.wait(max(0.1, float(data.shape[0]) / max(1, int(sample_rate)) + 1.0))
        return True

    def stop(self) -> None:
        with self._lock:
            while True:
                try:
                    _audio, _rate, done = self._pending.get_nowait()
                    done.set()
                except queue.Empty:
                    break
            if self._current_done is not None:
                self._current_done.set()
            self._current = None
            self._current_done = None
            self._position = 0

    def _ensure_stream(self, sample_rate: int) -> bool:
        with self._lock:
            if self._stream is not None and self._sample_rate == sample_rate:
                return True
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            try:
                import sounddevice as sd

                self._stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=1024,
                    callback=self._callback,
                )
                self._stream.start()
                self._sample_rate = sample_rate
                return True
            except Exception as exc:
                print(f"Memory audio stream failed: {exc}", flush=True)
                self._stream = None
                self._sample_rate = 0
                return False

    def _callback(self, outdata: Any, frames: int, _time_info: Any, _status: Any) -> None:
        outdata.fill(0)
        offset = 0
        with self._lock:
            while offset < frames:
                if self._current is None:
                    try:
                        self._current, _rate, self._current_done = self._pending.get_nowait()
                        self._position = 0
                    except queue.Empty:
                        return
                remaining = int(self._current.shape[0]) - self._position
                if remaining <= 0:
                    if self._current_done is not None:
                        self._current_done.set()
                    self._current = None
                    self._current_done = None
                    self._position = 0
                    continue
                count = min(frames - offset, remaining)
                outdata[offset:offset + count, 0] = self._current[self._position:self._position + count]
                self._position += count
                offset += count
                if self._position >= int(self._current.shape[0]):
                    if self._current_done is not None:
                        self._current_done.set()
                    self._current = None
                    self._current_done = None
                    self._position = 0


class SpeechQueue:
    def __init__(
        self,
        voice: str,
        rate: int,
        disabled: bool = False,
        filler_dir: Path | None = None,
        speech_backend: str = "step-tts-mini",
        gjallarhorn_base_url: str = "http://localhost:4000/v1",
        api_key: str = "fake-key",
        tts_model: str = "step-tts-mini",
        tts_voice: str = "elegantgentle-female",
        tts_cache_dir: Path | None = None,
        tts_timeout: float = 90.0,
        tts_retries: int = 1,
        tts_fallback_say: bool = True,
        start_sound_dir: Path | None = None,
        verify_tls: bool = False,
        store: VoiceSessionStore | None = None,
        speech_chunk_chars: int = 36,
        speech_wait_chunk_chars: int = 28,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.disabled = disabled
        self.speech_backend = speech_backend
        self.gjallarhorn_base_url = gjallarhorn_base_url.rstrip("/")
        self.api_key = api_key
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_cache_dir = tts_cache_dir
        self.tts_timeout = tts_timeout
        self.tts_retries = max(0, tts_retries)
        self.tts_fallback_say = tts_fallback_say
        self.verify_tls = verify_tls
        self.store = store
        self.speech_chunk_chars = max(12, int(speech_chunk_chars or 36))
        self.speech_wait_chunk_chars = max(12, int(speech_wait_chunk_chars or 28))
        self.filler_dir = filler_dir
        self._filler_items = self._load_filler_items(filler_dir)
        self._start_sound_items = self._load_start_sound_items(start_sound_dir)
        self._cancel_sound_item = self._load_cancel_sound_item("回见。")
        self._last_filler_key: tuple[int, str] | None = None
        self._last_start_sound_key: int | None = None
        self._tool_cache_warming: set[str] = set()
        self._tool_speech_cooldowns: dict[str, float] = {}
        self._completed_tool_speech_keys: set[str] = set()
        self._tts_cache_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._memory_audio = MemoryAudioPlayer()
        self._proc: subprocess.Popen | None = None
        self._active_filler_stop: threading.Event | None = None
        self._speech_active = False
        self._generation = 0
        self._last_speech_triggered_at = 0.0
        self._say_only_generations: set[int] = set()
        self._say_fallback_prefix_generations: set[int] = set()
        self._speech_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()
        if self._cancel_sound_item is None:
            self.warm_tts_audio_async("回见。")

    def _enqueue_speech_item(self, item: dict[str, Any]) -> None:
        turn_id = str(item.get("turn_id") or "")
        force_say = bool(item.get("force_say"))
        cached_audio_path = str(item.get("cached_audio_path") or "")
        item_kind = str(item.get("kind") or "speech")
        if self.store is not None and turn_id and item_kind == "speech" and self.speech_backend == "step-tts-mini" and not force_say and not cached_audio_path:
            self.store.add_event(
                "tts_call",
                role="system",
                lane="speech",
                content="step-tts-mini queued",
                metadata={
                    "turn_id": turn_id,
                    "status": "queued",
                    "model": self.tts_model,
                    "voice": self.tts_voice,
                    "chars": len(str(item.get("text") or "")),
                    "kind": item_kind,
                },
            )
        with self._lock:
            self._last_speech_triggered_at = time.monotonic()
            self._speech_queue.put(item)

    def _non_llm_speech_gate_allows_locked(self, kind: str, text: str) -> bool:
        if kind not in {"filler", "tool"}:
            return True
        elapsed = time.monotonic() - self._last_speech_triggered_at if self._last_speech_triggered_at else 999.0
        threshold = random.uniform(6.0, 11.0)
        if elapsed < threshold:
            print(
                f"Drop {kind} speech by recent-speech gate: elapsed={elapsed:.1f}s threshold={threshold:.1f}s text={text}",
                flush=True,
            )
            return False
        return True

    def _mark_speech_triggered(self) -> None:
        with self._lock:
            self._last_speech_triggered_at = time.monotonic()

    def _enqueue_text_chunks(
        self,
        text: str,
        *,
        generation: int | None = None,
        force_say: bool = False,
        filler_stop: threading.Event | None = None,
        done: threading.Event | None = None,
        tool_cooldown_key: str = "",
        tool_cooldown_seconds: float = 0.0,
        max_chars: int = 60,
        turn_id: str = "",
        coalesce_tool: str = "",
        coalesce_status: str = "",
        quick_say_fallback: bool = False,
        item_kind: str = "speech",
    ) -> int:
        chunks = [text] if force_say else split_speech_text(text, max_chars=max_chars)
        if not chunks:
            if filler_stop is not None:
                filler_stop.set()
            if done is not None:
                done.set()
            return 0
        if len(chunks) > 1:
            print(f"Queue speak split into {len(chunks)} chunks: {text}", flush=True)
        for index, chunk in enumerate(chunks):
            self._enqueue_speech_item({
                "text": chunk,
                "filler_stop": filler_stop if index == 0 else None,
                "generation": generation,
                "force_say": force_say,
                "done": done if index == len(chunks) - 1 else None,
                "tool_cooldown_key": tool_cooldown_key if index == len(chunks) - 1 else "",
                "tool_cooldown_seconds": tool_cooldown_seconds,
                "turn_id": turn_id,
                "coalesce_tool": coalesce_tool,
                "coalesce_status": coalesce_status,
                "quick_say_fallback": quick_say_fallback,
                "kind": item_kind,
            })
        return len(chunks)

    def _queue_idle_locked(self) -> bool:
        proc_active = self._proc is not None and self._proc.poll() is None
        return not self._speech_active and not proc_active and self._speech_queue.empty()

    def tool_speech_cooldown_remaining(self, tool_name: str) -> float:
        key = tool_name.split(":", 1)[-1]
        with self._lock:
            until = self._tool_speech_cooldowns.get(key, 0.0)
        return max(0.0, until - time.monotonic())

    def _set_tool_speech_cooldown(self, tool_name: str, seconds: float) -> None:
        seconds = max(0.0, float(seconds or 0.0))
        if seconds <= 0:
            return
        key = tool_name.split(":", 1)[-1]
        with self._lock:
            self._tool_speech_cooldowns[key] = time.monotonic() + seconds

    def _tool_speech_key(self, tool_name: str, turn_id: str) -> str:
        if not turn_id:
            return ""
        key = tool_name.split(":", 1)[-1]
        return f"{turn_id}:{key}"

    def _mark_tool_speech_completed(self, tool_name: str, turn_id: str) -> None:
        key = self._tool_speech_key(tool_name, turn_id)
        if not key:
            return
        with self._lock:
            self._completed_tool_speech_keys.add(key)

    def _tool_start_is_obsolete(self, tool_name: str, turn_id: str) -> bool:
        key = self._tool_speech_key(tool_name, turn_id)
        if not key:
            return False
        with self._lock:
            return key in self._completed_tool_speech_keys

    def current_generation(self) -> int:
        with self._lock:
            return self._generation

    def _generation_uses_say_only(self, generation: int | None) -> bool:
        if generation is None:
            return False
        with self._lock:
            return int(generation) in self._say_only_generations

    def _mark_generation_say_only(self, generation: int | None, reason: str) -> None:
        if generation is None:
            return
        with self._lock:
            self._say_only_generations.add(int(generation))
        print(f"TTS disabled for generation={generation}; using say for rest of turn: {reason}", flush=True)

    def speak(
        self,
        text: str,
        interrupt: bool = False,
        filler_stop: threading.Event | None = None,
        generation: int | None = None,
        force_say: bool = False,
        quick_say_fallback: bool = False,
        turn_id: str = "",
    ) -> None:
        text = normalize_assistant_text(text)
        if not text:
            if filler_stop is not None:
                filler_stop.set()
            return
        if self.disabled:
            if filler_stop is not None:
                filler_stop.set()
            return
        if generation is not None and generation != self.current_generation():
            print(f"Drop stale speak generation={generation} current={self.current_generation()}: {text}", flush=True)
            if filler_stop is not None:
                filler_stop.set()
            return
        print(f"Queue speak{' via say' if force_say else ''}: {text}", flush=True)
        self._enqueue_text_chunks(
            text,
            generation=generation,
            force_say=force_say,
            filler_stop=filler_stop,
            max_chars=self.speech_chunk_chars,
            quick_say_fallback=quick_say_fallback,
            turn_id=turn_id,
            item_kind="speech",
        )

    def speak_cached_text_and_wait(
        self,
        text: str,
        *,
        generation: int,
        timeout: float,
        tool_cooldown_key: str = "",
        tool_cooldown_seconds: float = 0.0,
        turn_id: str = "",
    ) -> bool:
        text = normalize_assistant_text(text)
        if not text or self.disabled:
            return False
        if generation != self.current_generation():
            return False
        path = self.cached_tool_speech_path(text)
        if path is None:
            return False
        done = threading.Event()
        print(f"Queue cached speak and wait: {text}", flush=True)
        self._enqueue_speech_item({
            "text": text,
            "generation": generation,
            "force_say": False,
            "cached_audio_path": str(path),
            "done": done,
            "tool_cooldown_key": tool_cooldown_key,
            "tool_cooldown_seconds": tool_cooldown_seconds,
            "turn_id": turn_id,
        })
        return done.wait(max(0.1, timeout))

    def speak_text_and_wait(self, text: str, *, generation: int, timeout: float, turn_id: str = "") -> bool:
        text = normalize_assistant_text(text)
        if not text or self.disabled:
            return False
        if generation != self.current_generation():
            return False
        done = threading.Event()
        print(f"Queue speak and wait: {text}", flush=True)
        self._enqueue_text_chunks(text, generation=generation, done=done, max_chars=self.speech_wait_chunk_chars, turn_id=turn_id)
        return done.wait(max(0.1, timeout))

    def speak_error(self, text: str, filler_stop: threading.Event | None = None, generation: int | None = None) -> None:
        self.speak(text, filler_stop=filler_stop, generation=generation, force_say=True)

    def speak_cancel_immediate(self, text: str = "回见。") -> None:
        text = normalize_assistant_text(text)
        if not text or self.disabled:
            return
        self.stop()
        if text == "回见。" and self._play_preloaded_audio(self._cancel_sound_item, blocking=False):
            print(f"Speak cancel sound from memory: {text}", flush=True)
            return
        cached_audio = self.cached_tts_audio_path(text)
        if cached_audio is not None and self._play_audio(cached_audio, interrupt=True):
            print(f"Speak cancel sound from cache: {text} ({cached_audio.name})", flush=True)
            return
        self.warm_tts_audio_async(text)
        print(f"Speak cancel sound via say: {text}", flush=True)
        self._speak_with_say(text, interrupt=True)

    def _speech_worker(self) -> None:
        while True:
            item = self._speech_queue.get()
            if item is None:
                return
            text = str(item.get("text") or "")
            filler_stop = item.get("filler_stop")
            generation = item.get("generation")
            force_say = bool(item.get("force_say"))
            cached_audio_path = item.get("cached_audio_path")
            done_event = item.get("done")
            item_kind = str(item.get("kind") or "speech")
            tool_cooldown_key = str(item.get("tool_cooldown_key") or "")
            tool_cooldown_seconds = float(item.get("tool_cooldown_seconds") or 0.0)
            turn_id = str(item.get("turn_id") or "")
            coalesce_tool = str(item.get("coalesce_tool") or "")
            coalesce_status = str(item.get("coalesce_status") or "")
            quick_say_fallback = bool(item.get("quick_say_fallback"))
            played_audio = False
            if generation is not None and generation != self.current_generation():
                print(f"Skip stale queued speech generation={generation} current={self.current_generation()}: {text}", flush=True)
                if isinstance(filler_stop, threading.Event):
                    filler_stop.set()
                if isinstance(done_event, threading.Event):
                    done_event.set()
                continue
            if coalesce_status == "start" and coalesce_tool and self._tool_start_is_obsolete(coalesce_tool, turn_id):
                print(f"Skip obsolete tool start speech: {coalesce_tool} {text}", flush=True)
                if isinstance(filler_stop, threading.Event):
                    filler_stop.set()
                if isinstance(done_event, threading.Event):
                    done_event.set()
                continue
            if item_kind == "filler":
                print(f"Speak filler: {text} ({Path(str(cached_audio_path)).name if cached_audio_path else 'queue'})", flush=True)
            else:
                print(f"Speak{' via say' if force_say else ''}: {text}", flush=True)
            with self._lock:
                self._speech_active = True
            timing_id = 0
            if self.store is not None and turn_id:
                timing_id = self.store.start_turn_timing(
                    turn_id,
                    "speech_playback",
                    item_kind,
                    metadata={"text": text[:120], "cached": bool(cached_audio_path), "force_say": force_say},
                )
            try:
                force_say = force_say or self._generation_uses_say_only(generation)
                if force_say:
                    text = self._say_fallback_text(text, generation=generation)
                if cached_audio_path:
                    if isinstance(filler_stop, threading.Event):
                        filler_stop.set()
                    played_audio = self._play_audio_blocking(Path(str(cached_audio_path)))
                    continue
                if self.speech_backend == "step-tts-mini" and not force_say:
                    if self._play_tts_sse_blocking(text):
                        if isinstance(filler_stop, threading.Event):
                            filler_stop.set()
                        played_audio = True
                        continue
                    if quick_say_fallback and self.tts_fallback_say:
                        if isinstance(filler_stop, threading.Event):
                            filler_stop.set()
                        self._mark_generation_say_only(generation, "quick tts fallback")
                        print("TTS SSE failed for final followup; falling back to say immediately.", flush=True)
                        text = self._say_fallback_text(text, generation=generation)
                        force_say = True
                        if generation is not None and generation != self.current_generation():
                            print(f"Drop stale quick say fallback generation={generation} current={self.current_generation()}: {text}", flush=True)
                            continue
                        self._speak_with_say_blocking(text)
                        played_audio = True
                        continue
                    audio_path = self._tts_audio_path(text)
                    if generation is not None and generation != self.current_generation():
                        print(f"Drop stale generated speech generation={generation} current={self.current_generation()}: {text}", flush=True)
                        if isinstance(filler_stop, threading.Event):
                            filler_stop.set()
                        continue
                    if audio_path is not None:
                        if isinstance(filler_stop, threading.Event):
                            filler_stop.set()
                        if self._play_audio_blocking(audio_path):
                            played_audio = True
                            continue
                    if isinstance(filler_stop, threading.Event):
                        filler_stop.set()
                    if not self.tts_fallback_say:
                        print("TTS failed; not falling back to say.", flush=True)
                        continue
                    self._mark_generation_say_only(generation, "tts retries exhausted")
                    print("TTS failed after retries; falling back to say for this turn.", flush=True)
                    text = self._say_fallback_text(text, generation=generation)
                if generation is not None and generation != self.current_generation():
                    print(f"Drop stale say fallback generation={generation} current={self.current_generation()}: {text}", flush=True)
                    if isinstance(filler_stop, threading.Event):
                        filler_stop.set()
                    continue
                if isinstance(filler_stop, threading.Event):
                    filler_stop.set()
                self._speak_with_say_blocking(text)
                played_audio = True
            except Exception as exc:
                print(f"Speech worker failed for text={text[:80]!r}: {exc}", flush=True)
                if isinstance(filler_stop, threading.Event):
                    filler_stop.set()
                if self.tts_fallback_say and not force_say:
                    self._mark_generation_say_only(generation, f"speech worker error: {exc}")
                    fallback_text = self._say_fallback_text(text, generation=generation)
                    if generation is None or generation == self.current_generation():
                        try:
                            self._speak_with_say_blocking(fallback_text)
                            played_audio = True
                        except Exception as say_exc:
                            print(f"Speech worker say fallback failed: {say_exc}", flush=True)
            finally:
                if played_audio:
                    self._mark_speech_triggered()
                if self.store is not None and timing_id:
                    self.store.end_turn_timing(
                        timing_id,
                        status="played" if played_audio else "skipped",
                        metadata={"played": played_audio, "backend": "say" if force_say else self.speech_backend},
                    )
                if played_audio and tool_cooldown_key:
                    self._set_tool_speech_cooldown(tool_cooldown_key, tool_cooldown_seconds)
                with self._lock:
                    self._speech_active = False
                if isinstance(done_event, threading.Event):
                    done_event.set()

    def _speak_with_say_blocking(self, text: str) -> None:
        if not shutil.which("say"):
            return
        self._run_process_blocking(["say", *self._say_args(), text])

    def _say_args(self) -> list[str]:
        args: list[str] = []
        if self.voice:
            args.extend(["-v", self.voice])
        if self.rate:
            args.extend(["-r", str(self.rate)])
        return args

    def _play_audio_blocking(self, path: Path) -> bool:
        if self._play_audio_file_with_sounddevice(path, blocking=True):
            return True
        if not shutil.which("afplay"):
            return False
        self._run_process_blocking(["afplay", str(path)])
        return True

    def _choose_retry_filler_item(self) -> dict[str, Any] | None:
        candidates = [
            item for item in self._filler_items
            if item.get("stage") in {"working", "active", "transition"}
        ] or self._filler_items
        if not candidates:
            return None
        with self._lock:
            filtered = [item for item in candidates if (item["index"], item["tone"]) != self._last_filler_key]
            item = random.choice(filtered or candidates)
            self._last_filler_key = (item["index"], item["tone"])
        return dict(item)

    def _play_retry_filler_once_blocking(self) -> None:
        if self.disabled or not shutil.which("afplay"):
            return
        item = self._choose_retry_filler_item()
        if item is None:
            return
        print(f"Retry filler: {item['phrase']} ({item['path'].name})", flush=True)
        try:
            self._play_audio_blocking(Path(str(item["path"])))
        except Exception as exc:
            print(f"Retry filler failed: {exc}", flush=True)

    def _run_process_blocking(self, cmd: list[str]) -> None:
        with self._lock:
            self._stop_active_filler_loop_locked()
            self._proc = subprocess.Popen(cmd)
            proc = self._proc
        try:
            proc.wait()
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None

    def _speak_immediate(self, text: str, interrupt: bool = False, filler_stop: threading.Event | None = None) -> None:
        text = normalize_assistant_text(text)
        if not text:
            if filler_stop is not None:
                filler_stop.set()
            return
        if self.disabled:
            if filler_stop is not None:
                filler_stop.set()
            return
        if self.speech_backend == "step-tts-mini":
            if self._play_tts_sse_blocking(text, interrupt=interrupt):
                if filler_stop is not None:
                    filler_stop.set()
                return
            audio_path = self._tts_audio_path(text)
            if filler_stop is not None:
                filler_stop.set()
            if audio_path is not None and self._play_audio(audio_path, interrupt=interrupt):
                return
            if not self.tts_fallback_say:
                print("TTS failed; not falling back to say.", flush=True)
                return
            self._mark_generation_say_only(self.current_generation(), "immediate tts retries exhausted")
            print("TTS failed after retries; falling back to say for this turn.", flush=True)
            text = self._say_fallback_text(text, generation=self.current_generation())
        if filler_stop is not None:
            filler_stop.set()
        self._speak_with_say(text, interrupt=interrupt)

    def _say_fallback_text(self, text: str, generation: int | None = None) -> str:
        text = text.strip()
        if not text:
            return text
        if generation is None:
            generation = self.current_generation()
        with self._lock:
            already_prefixed = int(generation) in self._say_fallback_prefix_generations
            if not already_prefixed:
                self._say_fallback_prefix_generations.add(int(generation))
        if text.startswith(SAY_FALLBACK_PREFIX):
            return text if not already_prefixed else text[len(SAY_FALLBACK_PREFIX):].lstrip()
        if already_prefixed:
            return text
        return f"{SAY_FALLBACK_PREFIX}{text}"

    def _speak_with_say(self, text: str, interrupt: bool) -> None:
        if not shutil.which("say"):
            return
        with self._lock:
            self._stop_active_filler_loop_locked()
            if interrupt:
                self._stop_locked()
            cmd = ["say"]
            if self.voice:
                cmd.extend(["-v", self.voice])
            if self.rate:
                cmd.extend(["-r", str(self.rate)])
            cmd.append(text)
            self._proc = subprocess.Popen(cmd)

    def _play_audio(self, path: Path, interrupt: bool) -> bool:
        if interrupt:
            with self._lock:
                self._stop_active_filler_loop_locked()
                self._stop_locked()
        if self._play_audio_file_with_sounddevice(path, blocking=False):
            return True
        if not shutil.which("afplay"):
            return False
        with self._lock:
            self._stop_active_filler_loop_locked()
            if interrupt:
                self._stop_locked()
            self._proc = subprocess.Popen(["afplay", str(path)])
        return True

    def _play_preloaded_audio(self, item: dict[str, Any] | None, *, blocking: bool) -> bool:
        if not item or item.get("audio") is None or not item.get("sample_rate"):
            return False
        return self._play_audio_array_with_sounddevice(item["audio"], int(item["sample_rate"]), blocking=blocking)

    def _play_audio_file_with_sounddevice(self, path: Path, *, blocking: bool) -> bool:
        path = self._playback_wav_path(path)
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(path), dtype="float32")
        except Exception:
            return False
        return self._play_audio_array_with_sounddevice(audio, int(sample_rate), blocking=blocking)

    def _playback_wav_path(self, path: Path) -> Path:
        if path.suffix.lower() != ".mp3":
            return path
        wav_path = path.with_suffix(".wav")
        if wav_path.exists() and wav_path.stat().st_size > 0:
            return wav_path
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return path
        try:
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(path), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
                check=True,
                timeout=5,
            )
            if wav_path.exists() and wav_path.stat().st_size > 0:
                return wav_path
        except Exception as exc:
            print(f"Playback wav conversion failed for {path.name}: {exc}", flush=True)
        return path

    def _play_audio_array_with_sounddevice(self, audio: Any, sample_rate: int, *, blocking: bool) -> bool:
        return self._memory_audio.play(audio, int(sample_rate), blocking=blocking)

    def _play_tts_sse_blocking(self, text: str, interrupt: bool = False) -> bool:
        player_bin = shutil.which("ffplay") or shutil.which("mpg123")
        if not player_bin:
            return False
        player_cmd = (
            [player_bin, "-nodisp", "-autoexit", "-loglevel", "error", "-i", "pipe:0"]
            if Path(player_bin).name == "ffplay"
            else [player_bin, "-q", "-"]
        )
        body = json.dumps(
            {
                "model": self.tts_model,
                "voice": self.tts_voice,
                "input": text,
                "instructions": DEFAULT_TTS_INSTRUCTIONS,
                "response_format": "mp3",
                "stream_format": "sse",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            f"{self.gjallarhorn_base_url}/audio/speech",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        print(f"TTS SSE start: {self.tts_model} timeout={self.tts_timeout:.1f}s chars={len(text)}", flush=True)
        player: subprocess.Popen | None = None
        audio_chunks = 0
        audio_bytes = 0
        first_audio_at: float | None = None
        context = None if self.verify_tls else ssl._create_unverified_context()
        try:
            player = subprocess.Popen(player_cmd, stdin=subprocess.PIPE)
            with self._lock:
                self._stop_active_filler_loop_locked()
                if interrupt:
                    self._stop_locked()
                self._proc = player
            with request.urlopen(req, timeout=self.tts_timeout, context=context) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type.lower():
                    print(f"TTS SSE unavailable; got content-type={content_type}", flush=True)
                    return False
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    audio = str(payload.get("audio") or "")
                    if not audio:
                        continue
                    chunk = base64.b64decode(audio)
                    if not chunk:
                        continue
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter() - started
                        print(f"TTS SSE first audio in {first_audio_at:.2f}s", flush=True)
                    audio_chunks += 1
                    audio_bytes += len(chunk)
                    if player.stdin is None or player.poll() is not None:
                        return False
                    player.stdin.write(chunk)
                    player.stdin.flush()
            if player.stdin is not None:
                player.stdin.close()
            player.wait(timeout=20)
            ok = audio_chunks > 0 and player.returncode == 0
            print(
                f"TTS SSE {'played' if ok else 'failed'} in {time.perf_counter() - started:.2f}s "
                f"chunks={audio_chunks} bytes={audio_bytes}",
                flush=True,
            )
            return ok
        except TimeoutError as exc:
            print(f"TTS SSE timed out after {time.perf_counter() - started:.2f}s: {exc}", flush=True)
            return False
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"TTS SSE HTTP {exc.code}: {detail[:300]}", flush=True)
            return False
        except Exception as exc:
            print(f"TTS SSE failed after {time.perf_counter() - started:.2f}s: {exc}", flush=True)
            return False
        finally:
            if player is not None:
                if player.stdin is not None and not player.stdin.closed:
                    try:
                        player.stdin.close()
                    except Exception:
                        pass
                if player.poll() is None:
                    try:
                        player.terminate()
                        player.wait(timeout=0.8)
                    except Exception:
                        try:
                            player.kill()
                        except Exception:
                            pass
                with self._lock:
                    if self._proc is player:
                        self._proc = None

    def _tts_audio_path(self, text: str) -> Path | None:
        if self.tts_cache_dir is None:
            return None
        self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
        instructions = DEFAULT_TTS_INSTRUCTIONS
        path = self._tts_cache_path_for(text, instructions=instructions)
        if path is None:
            return None
        return self._tts_audio_path_to(text, path, instructions=instructions)

    def _tts_cache_path_for(self, text: str, *, instructions: str = DEFAULT_TTS_INSTRUCTIONS) -> Path | None:
        if self.tts_cache_dir is None:
            return None
        key = sha256(
            json.dumps(
                {"model": self.tts_model, "voice": self.tts_voice, "text": text, "instructions": instructions},
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:24]
        return self.tts_cache_dir / f"{key}.mp3"

    def cached_tts_audio_path(self, text: str, *, instructions: str = DEFAULT_TTS_INSTRUCTIONS) -> Path | None:
        if self.disabled or self.speech_backend != "step-tts-mini":
            return None
        path = self._tts_cache_path_for(text, instructions=instructions)
        if path is not None and path.exists() and path.stat().st_size > 0:
            return path
        return None

    def warm_tts_audio_async(self, text: str, *, instructions: str = DEFAULT_TTS_INSTRUCTIONS) -> None:
        text = normalize_assistant_text(text)
        if not text or self.disabled or self.speech_backend != "step-tts-mini":
            return
        if self.cached_tts_audio_path(text, instructions=instructions) is not None:
            return
        cache_key = f"{self.tts_model}:{self.tts_voice}:{instructions}:{text}"
        with self._lock:
            if cache_key in self._tool_cache_warming:
                return
            self._tool_cache_warming.add(cache_key)

        def worker() -> None:
            try:
                self._tts_audio_path(text)
            finally:
                with self._lock:
                    self._tool_cache_warming.discard(cache_key)

        threading.Thread(target=worker, daemon=True).start()

    def _tts_audio_path_to(self, text: str, path: Path, *, instructions: str) -> Path | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return path
        lock_key = str(path)
        with self._lock:
            cache_lock = self._tts_cache_locks.get(lock_key)
            if cache_lock is None:
                cache_lock = threading.Lock()
                self._tts_cache_locks[lock_key] = cache_lock
        with cache_lock:
            if path.exists() and path.stat().st_size > 0:
                return path
            return self._tts_audio_path_to_locked(text, path, instructions=instructions)

    def _tts_audio_path_to_locked(self, text: str, path: Path, *, instructions: str) -> Path | None:
        body = json.dumps(
            {
                "model": self.tts_model,
                "voice": self.tts_voice,
                "input": text,
                "instructions": instructions,
                "response_format": "mp3",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            f"{self.gjallarhorn_base_url}/audio/speech",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        print(f"TTS start: {self.tts_model} timeout={self.tts_timeout:.1f}s chars={len(text)}", flush=True)
        data = None
        last_error: Exception | None = None
        for attempt in range(1, self.tts_retries + 2):
            try:
                data = urlopen_bytes(req, timeout=self.tts_timeout, verify_tls=self.verify_tls, label=f"TTS {self.tts_model}")
                break
            except Exception as exc:
                last_error = exc
                print(f"TTS attempt {attempt} failed after {time.perf_counter() - started:.2f}s: {exc}", flush=True)
                if attempt <= self.tts_retries:
                    self._play_retry_filler_once_blocking()
                    time.sleep(0.8 * attempt)
        if data is None:
            print(f"TTS failed after {time.perf_counter() - started:.2f}s: {last_error}", flush=True)
            return None
        tmp_path = path.with_name(f".{path.name}.{threading.get_ident()}.{time.time_ns()}.tmp")
        try:
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        print(f"TTS generated in {time.perf_counter() - started:.2f}s: {path}", flush=True)
        return path

    def ensure_tool_speech_cache(self, *, tool_name: str, status: str, task_label: str, phrase: str, spoken_text: str) -> Path | None:
        if self.disabled or self.speech_backend != "step-tts-mini":
            return None
        path = self._tts_audio_path(spoken_text)
        if self.store is not None:
            self.store.record_tool_speech_cache(
                tool_name=tool_name,
                status=status,
                task_label=task_label,
                phrase=phrase,
                spoken_text=spoken_text,
                tts_model=self.tts_model,
                tts_voice=self.tts_voice,
                audio_path=str(path or ""),
                ok=path is not None,
                error="" if path is not None else "tts generation failed",
            )
        return path

    def cached_tool_speech_path(self, spoken_text: str) -> Path | None:
        return self.cached_tts_audio_path(spoken_text)

    def warm_tool_speech_cache_async(self, *, tool_name: str, status: str, task_label: str, phrase: str, spoken_text: str) -> None:
        if self.disabled or self.speech_backend != "step-tts-mini":
            return
        if self.cached_tool_speech_path(spoken_text) is not None:
            return
        cache_key = f"{self.tts_model}:{self.tts_voice}:{spoken_text}"
        with self._lock:
            if cache_key in self._tool_cache_warming:
                return
            self._tool_cache_warming.add(cache_key)

        def worker() -> None:
            try:
                self.ensure_tool_speech_cache(
                    tool_name=tool_name,
                    status=status,
                    task_label=task_label,
                    phrase=phrase,
                    spoken_text=spoken_text,
                )
            finally:
                with self._lock:
                    self._tool_cache_warming.discard(cache_key)

        threading.Thread(target=worker, daemon=True).start()

    def speak_tool_summary_or_filler(
        self,
        *,
        tool_name: str,
        status: str,
        task_label: str,
        phrase: str,
        spoken_text: str,
        generation: int,
        start_wait_timeout: float = 8.0,
        cooldown_seconds: float = 20.0,
        ignore_cooldown: bool = False,
        turn_id: str = "",
    ) -> bool:
        if self.disabled:
            return False
        if status != "start":
            self._mark_tool_speech_completed(tool_name, turn_id)
        cooldown_remaining = self.tool_speech_cooldown_remaining(tool_name)
        if cooldown_remaining > 0 and not ignore_cooldown:
            print(
                f"Tool speech cooldown skip: {tool_name} status={status} remaining={cooldown_remaining:.1f}s text={spoken_text}",
                flush=True,
            )
            return False
        if status == "start":
            path = self.cached_tool_speech_path(spoken_text)
            if path is not None:
                with self._lock:
                    if not self._non_llm_speech_gate_allows_locked("tool", spoken_text):
                        return False
                    print(f"Queue cached tool start speak: {tool_name} {spoken_text}", flush=True)
                    self._last_speech_triggered_at = time.monotonic()
                    self._speech_queue.put({
                        "text": spoken_text,
                        "generation": generation,
                        "force_say": False,
                        "cached_audio_path": str(path),
                        "turn_id": turn_id,
                        "coalesce_tool": tool_name,
                        "coalesce_status": "start",
                        "kind": "tool",
                    })
                return True
            self.warm_tool_speech_cache_async(
                tool_name=tool_name,
                status=status,
                task_label=task_label,
                phrase=phrase,
                spoken_text=spoken_text,
            )
            filler = self.speak_cached_filler(interrupt=False, stage="working")
            print(f"Tool start speech cache miss; warmed async and used filler: {tool_name} {spoken_text}", flush=True)
            return filler is not None
        path = self.cached_tool_speech_path(spoken_text)
        if path is not None:
            self.speak_tool_text(
                spoken_text,
                tool_name=tool_name,
                generation=generation,
                cooldown_seconds=cooldown_seconds,
                turn_id=turn_id,
            )
            return True
        self.warm_tool_speech_cache_async(
            tool_name=tool_name,
            status=status,
            task_label=task_label,
            phrase=phrase,
            spoken_text=spoken_text,
        )
        self.speak_tool_text(
            spoken_text,
            tool_name=tool_name,
            generation=generation,
            cooldown_seconds=cooldown_seconds,
            turn_id=turn_id,
        )
        return True

    def speak_tool_text(
        self,
        text: str,
        *,
        tool_name: str,
        generation: int,
        cooldown_seconds: float,
        turn_id: str = "",
        coalesce_status: str = "",
    ) -> None:
        text = normalize_assistant_text(text)
        if not text or self.disabled:
            return
        if generation != self.current_generation():
            return
        print(f"Queue tool speak: {tool_name} {text}", flush=True)
        with self._lock:
            allowed = self._non_llm_speech_gate_allows_locked("tool", text)
        if not allowed:
            return
        self._enqueue_text_chunks(
            text,
            generation=generation,
            tool_cooldown_key=tool_name,
            tool_cooldown_seconds=cooldown_seconds,
            max_chars=42,
            turn_id=turn_id,
            coalesce_tool=tool_name,
            coalesce_status=coalesce_status,
            item_kind="tool",
        )

    def speak_filler(self, interrupt: bool = False, text: str | None = None) -> str:
        if text is None:
            cached = self.speak_cached_filler(interrupt=interrupt)
            if cached:
                return str(cached["phrase"])
        filler = text or "让我想想。"
        self.speak(filler, interrupt=interrupt)
        return filler

    def start_filler_loop(self, stage: str, *, initial_delay: float = 0.0, interval_range: tuple[float, float] = (3.2, 5.8)) -> threading.Event:
        stop = threading.Event()
        with self._lock:
            self._stop_active_filler_loop_locked()
            self._active_filler_stop = stop

        def loop() -> None:
            if initial_delay > 0 and stop.wait(initial_delay):
                return
            while not stop.is_set():
                self.speak_cached_filler(interrupt=False, stage=stage)
                wait_for = random.uniform(*interval_range)
                stop.wait(wait_for)
            with self._lock:
                if self._active_filler_stop is stop:
                    self._active_filler_stop = None

        threading.Thread(target=loop, daemon=True).start()
        return stop

    def stop_filler_loop(self) -> None:
        with self._lock:
            self._stop_active_filler_loop_locked()

    def speak_cached_filler(self, interrupt: bool = False, tone: str | None = None, stage: str | None = None) -> dict[str, Any] | None:
        if self.disabled or not shutil.which("afplay"):
            return None
        candidates = [
            item for item in self._filler_items
            if (tone is None or item["tone"] == tone) and (stage is None or item.get("stage") == stage)
        ]
        if not candidates and stage is not None:
            candidates = [item for item in self._filler_items if tone is None or item["tone"] == tone]
        if not candidates:
            return None
        with self._lock:
            if not interrupt and not self._queue_idle_locked():
                return None
            if interrupt:
                self._stop_locked()
                self._clear_speech_queue_locked()
            filtered = [item for item in candidates if (item["index"], item["tone"]) != self._last_filler_key]
            item = random.choice(filtered or candidates)
            if not self._non_llm_speech_gate_allows_locked("filler", str(item["phrase"])):
                return None
            self._last_filler_key = (item["index"], item["tone"])
            self._last_speech_triggered_at = time.monotonic()
            self._speech_queue.put({
                "text": item["phrase"],
                "cached_audio_path": str(item["path"]),
                "force_say": False,
                "kind": "filler",
            })
        print(f"Queue filler: {item['phrase']} ({item['path'].name})", flush=True)
        return dict(item)

    def play_start_sound(self, *, blocking: bool = False) -> dict[str, Any] | None:
        if self.disabled or not self._start_sound_items:
            return None
        proc: subprocess.Popen | None = None
        with self._lock:
            if not self._queue_idle_locked():
                return None
            candidates = [item for item in self._start_sound_items if item["index"] != self._last_start_sound_key]
            item = random.choice(candidates or self._start_sound_items)
            self._last_start_sound_key = int(item["index"])
            audio = item.get("audio")
            sample_rate = item.get("sample_rate")
            if audio is not None and sample_rate:
                proc = None
            elif shutil.which("afplay"):
                self._proc = subprocess.Popen(["afplay", str(item["path"])])
                proc = self._proc
            else:
                return None
        print(f"Speak start sound: {item['phrase']} ({item['path'].name})", flush=True)
        if item.get("audio") is not None and item.get("sample_rate"):
            self._play_preloaded_audio(item, blocking=blocking)
            return {key: value for key, value in item.items() if key not in {"audio", "sample_rate"}}
        if blocking and proc is not None:
            proc.wait()
            with self._lock:
                if self._proc is proc:
                    self._proc = None
        return dict(item)

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._say_only_generations = {
                value for value in self._say_only_generations
                if value >= self._generation - 1
            }
            self._say_fallback_prefix_generations = {
                value for value in self._say_fallback_prefix_generations
                if value >= self._generation - 1
            }
            self._stop_active_filler_loop_locked()
            self._stop_locked()
            self._clear_speech_queue_locked()

    def _clear_speech_queue_locked(self) -> None:
        try:
            while True:
                self._speech_queue.get_nowait()
        except queue.Empty:
            return

    def _stop_locked(self) -> None:
        self._memory_audio.stop()
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def _stop_active_filler_loop_locked(self) -> None:
        if self._active_filler_stop is not None:
            self._active_filler_stop.set()
            self._active_filler_stop = None

    def _load_filler_items(self, filler_dir: Path | None) -> list[dict[str, Any]]:
        if filler_dir is None:
            return []
        if self.store is not None:
            rows = self.store.filler_speech_catalog()
            items = []
            for row in rows:
                path = Path(str(row.get("audio_path") or ""))
                if not path.is_absolute():
                    path = filler_dir / path
                phrase = str(row.get("phrase") or "").strip()
                tone = str(row.get("tone") or "").strip()
                if not phrase or not tone or not bool(row.get("ok")) or not path.exists():
                    continue
                items.append({
                    "id": row.get("id"),
                    "index": int(row.get("slot_index") or 0),
                    "tone": tone,
                    "phrase": phrase,
                    "path": path,
                    "seconds": row.get("seconds"),
                    "stage": str(row.get("stage") or filler_stage_for_phrase(phrase)),
                })
            if items:
                print(f"Loaded {len(items)} cached fillers from DB", flush=True)
                return self._order_filler_items(items)
        manifest_path = filler_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Filler manifest ignored: {exc}", flush=True)
            return []
        latest: dict[tuple[int, str], dict[str, Any]] = {}
        for item in manifest.get("items") or []:
            if item.get("status") != 200:
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            tone = str(item.get("tone") or "").strip()
            phrase = str(item.get("phrase") or "").strip()
            source_path = Path(str(item.get("path") or ""))
            local_path = filler_dir / source_path.name
            path = local_path if local_path.exists() else source_path
            if not tone or not phrase or not path.exists():
                continue
            latest[(index, tone)] = {
                "index": index,
                "tone": tone,
                "phrase": phrase,
                "path": path,
                "seconds": item.get("seconds"),
                "stage": filler_stage_for_phrase(phrase),
            }
        ordered = self._order_filler_items(list(latest.values()))
        if ordered:
            print(f"Loaded {len(ordered)} cached fillers from {filler_dir}", flush=True)
        return ordered

    def _order_filler_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest = {(int(item["index"]), str(item["tone"])): item for item in items}
        preferred = [
            (21, "active"),
            (21, "soft"),
            (1, "active"),
            (3, "active"),
            (4, "soft"),
            (7, "soft"),
            (17, "soft"),
            (16, "soft"),
            (22, "soft"),
            (13, "active"),
            (10, "active"),
            (11, "active"),
        ]
        ordered = [latest.pop(key) for key in preferred if key in latest]
        ordered.extend(latest[key] for key in sorted(latest))
        return ordered

    def reload_fillers(self) -> None:
        self._filler_items = self._load_filler_items(self.filler_dir)

    def warm_filler_audio(self, filler_id: int) -> dict[str, Any]:
        if self.store is None or self.filler_dir is None:
            return {"ok": False, "error": "filler store unavailable"}
        rows = [row for row in self.store.filler_speech_catalog() if int(row.get("id") or 0) == int(filler_id)]
        if not rows:
            return {"ok": False, "error": "filler not found"}
        row = rows[0]
        phrase = str(row.get("phrase") or "").strip()
        tone = str(row.get("tone") or "soft")
        if not phrase:
            return {"ok": False, "error": "phrase is required"}
        self.filler_dir.mkdir(parents=True, exist_ok=True)
        slot = int(row.get("slot_index") or filler_id)
        path = self.filler_dir / f"{slot:02d}_{tone}.mp3"
        instructions = str(row.get("instructions") or filler_tts_instructions(tone))
        started = time.perf_counter()
        audio_path = self._tts_audio_path_to(phrase, path, instructions=instructions)
        ok = audio_path is not None and audio_path.exists()
        seconds = round(time.perf_counter() - started, 3)
        self.store.update_filler_audio(filler_id, audio_path=str(path if ok else ""), ok=ok, seconds=seconds if ok else None, bytes_count=path.stat().st_size if ok else 0)
        self.reload_fillers()
        return {"ok": ok, "path": str(path), "seconds": seconds, "bytes": path.stat().st_size if ok else 0}

    def _load_start_sound_items(self, start_sound_dir: Path | None) -> list[dict[str, Any]]:
        if start_sound_dir is None:
            return []
        manifest_path = start_sound_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Start sound manifest ignored: {exc}", flush=True)
            return []
        items: list[dict[str, Any]] = []
        for raw in manifest.get("items") or []:
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            phrase = str(raw.get("phrase") or "").strip()
            source_path = Path(str(raw.get("path") or ""))
            path = start_sound_dir / source_path.name
            if not phrase or not path.exists():
                continue
            item: dict[str, Any] = {"index": index, "phrase": phrase, "path": path}
            try:
                import soundfile as sf

                audio, sample_rate = sf.read(str(path), dtype="float32")
                item["audio"] = audio
                item["sample_rate"] = int(sample_rate)
            except Exception as exc:
                print(f"Start sound will use afplay for {path.name}: {exc}", flush=True)
            items.append(item)
        items.sort(key=lambda item: int(item["index"]))
        if items:
            print(f"Loaded {len(items)} start sounds from {start_sound_dir}", flush=True)
        return items

    def _load_cancel_sound_item(self, text: str) -> dict[str, Any] | None:
        path = self.cached_tts_audio_path(text)
        if path is None:
            return None
        wav_path = path.with_name("cancel_huijian.wav")
        if not wav_path.exists() or wav_path.stat().st_size <= 0:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                try:
                    subprocess.run(
                        [ffmpeg, "-y", "-v", "error", "-i", str(path), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
                        check=True,
                        timeout=5,
                    )
                except Exception as exc:
                    print(f"Cancel sound wav conversion failed: {exc}", flush=True)
        audio_path = wav_path if wav_path.exists() and wav_path.stat().st_size > 0 else path
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(audio_path), dtype="float32")
            print(f"Loaded cancel sound from {audio_path.name}", flush=True)
            return {"text": text, "path": audio_path, "audio": audio, "sample_rate": int(sample_rate)}
        except Exception as exc:
            print(f"Cancel sound preload failed: {exc}", flush=True)
            return None
