from __future__ import annotations

import datetime as dt
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

class Recorder:
    def __init__(self, recordings_dir: Path, sample_rate: int = 16_000, block_size: int = 1600, preroll_seconds: float = 0.8) -> None:
        self.recordings_dir = recordings_dir
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._ring_chunks: deque[np.ndarray] = deque(maxlen=max(1, int(sample_rate / block_size * preroll_seconds)))
        self._lock = threading.Lock()
        self._recording = False
        self.started_at: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def prepare(self) -> None:
        if self._stream is not None:
            return

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio input status: {status}", flush=True)
            with self._lock:
                chunk = indata.copy()
                self._ring_chunks.append(chunk)
                if self._recording:
                    self._chunks.append(chunk)

        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=self.block_size, callback=callback)
        self._stream.start()

    def start(self, *, include_preroll: bool = True) -> None:
        self.prepare()
        with self._lock:
            self._chunks = list(self._ring_chunks) if include_preroll else []
            self._recording = True
            self.started_at = time.monotonic()

    def stop(self) -> tuple[Path, dict[str, float]] | None:
        if not self._recording:
            return None
        with self._lock:
            self._recording = False
            chunks = self._chunks
            self._chunks = []
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0)
        if audio.size == 0:
            return None
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio))))
        duration = float(audio.shape[0] / self.sample_rate)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        path = self.recordings_dir / f"jen-voice-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        sf.write(path, audio, self.sample_rate)
        return path, {"duration": duration, "rms": rms, "peak": peak}
