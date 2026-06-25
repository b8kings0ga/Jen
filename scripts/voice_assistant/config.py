from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "asr_model": "whisper-large-v3-turbo",
        "front": "oss",
        "back": "oss",
    },
    "step": {
        "asr_model": "whisper-large-v3-turbo",
        "front": "oss",
        "back": "step-3.5-flash-2603",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preset_probe = argparse.ArgumentParser(add_help=False)
    preset_probe.add_argument("--preset", choices=sorted(VOICE_PRESETS), default="fast")
    known, _ = preset_probe.parse_known_args(argv)
    preset_defaults = VOICE_PRESETS[known.preset]

    parser = argparse.ArgumentParser(description="Jen local voice assistant")
    parser.add_argument("--preset", choices=sorted(VOICE_PRESETS), default=known.preset)
    parser.add_argument("--overlay-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--front-note-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--front-note-api-url", default="http://127.0.0.1:8765/api/front-note", help=argparse.SUPPRESS)
    parser.add_argument("--text-input-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--text-input-api-url", default="http://127.0.0.1:8765/api/text-input", help=argparse.SUPPRESS)
    parser.add_argument("--gjallarhorn-base-url", default="http://localhost:4000/v1")
    parser.add_argument("--api-key", default="fake-key")
    parser.add_argument("--asr-model", default=preset_defaults["asr_model"])
    parser.add_argument("--language", default="zh", help="ASR language hint. Default zh biases transcription toward Simplified Chinese.")
    parser.add_argument(
        "--asr-prompt",
        default=(
            "Transcribe the user's speech as Simplified Chinese by default. Preserve English app names, "
            "commands, file names, URLs, and proper nouns only when clearly spoken in English. Do not translate "
            "those proper nouns. Never output Traditional Chinese."
        ),
    )
    parser.add_argument("--front", default=preset_defaults["front"])
    parser.add_argument("--back", default=preset_defaults["back"])
    parser.add_argument("--compact", default="free")
    parser.add_argument("--plan", default="oss")
    parser.add_argument("--simple-back", default="oss", help="Right Command back lane: no plan, low reasoning effort")
    parser.add_argument("--simple-plan", default="step-3.5-flash-2603", help=argparse.SUPPRESS)
    parser.add_argument("--simple-reasoning-effort", default="low", choices=["low", "high"])
    parser.add_argument("--quality-back", default="step-3.5-flash-2603", help="Right Option back lane: planned, higher-quality tasks")
    parser.add_argument("--quality-plan", default="step-3.5-flash-2603", help="Right Option planning lane")
    parser.add_argument("--quality-reasoning-effort", default="high", choices=["low", "high"])
    parser.add_argument("--plan-mode", dest="plan_mode", action="store_true", default=True)
    parser.add_argument("--no-plan-mode", dest="plan_mode", action="store_false")
    parser.add_argument("--plan-timeout", type=float, default=6.0)
    parser.add_argument("--plan-prefetch", dest="plan_prefetch", action="store_true", default=True)
    parser.add_argument("--no-plan-prefetch", dest="plan_prefetch", action="store_false")
    parser.add_argument("--plan-prefetch-max-concurrency", type=int, default=3)
    parser.add_argument("--plan-prefetch-max-tools", type=int, default=4)
    parser.add_argument("--plan-prefetch-wait", type=float, default=1.5)
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--session-db", type=Path, default=Path("data/voice/voice_sessions.sqlite"))
    parser.add_argument("--tool-workdir", type=Path, default=Path("data/voice/tools"))
    parser.add_argument(
        "--coding-workdir",
        type=Path,
        default=Path(os.environ.get("JEN_CODING_WORKDIR") or os.environ.get("GJALLARHORN_CODING_WORKDIR", "~/.jen")).expanduser(),
        help="Root for generated coding workspaces and executor runs.",
    )
    parser.add_argument(
        "--coding-cache-dir",
        type=Path,
        default=Path(os.environ.get("JEN_CODING_CACHE_DIR") or os.environ.get("GJALLARHORN_CODING_CACHE_DIR", "~/.jen/cache")).expanduser(),
        help="Root for coding executor uv/npm caches.",
    )
    parser.add_argument("--tool-dashboard-host", default="127.0.0.1")
    parser.add_argument("--tool-dashboard-port", type=int, default=8765)
    parser.add_argument("--no-tool-dashboard", action="store_true")
    parser.add_argument("--web-search-backend", default=os.environ.get("WEB_SEARCH_BACKEND", "auto"))
    parser.add_argument("--web-search-region", default=os.environ.get("WEB_SEARCH_REGION") or None)
    parser.add_argument("--web-search-timeout", type=int, default=int(os.environ.get("WEB_SEARCH_TIMEOUT", "20")))
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--min-record-seconds", type=float, default=0.75)
    parser.add_argument("--max-record-seconds", type=float, default=45.0)
    parser.add_argument("--min-record-rms", type=float, default=0.003)
    parser.add_argument("--min-record-peak", type=float, default=0.015)
    parser.add_argument("--cancel-tap-window", type=float, default=0.8)
    parser.add_argument("--input-wav", type=Path)
    parser.add_argument("--say-voice", default="Tingting")
    parser.add_argument("--say-rate", type=int, default=165)
    parser.add_argument("--no-say", action="store_true")
    parser.add_argument("--speech-backend", choices=["step-tts-mini", "say"], default="step-tts-mini")
    parser.add_argument("--tts-model", default="step-tts-mini")
    parser.add_argument("--tts-voice", default="elegantgentle-female")
    parser.add_argument("--tts-cache-dir", type=Path, default=Path("data/voice/tts_cache_step_tts_mini"))
    parser.add_argument("--tts-timeout", type=float, default=20.0)
    parser.add_argument("--tts-retries", type=int, default=3)
    parser.add_argument("--tts-fallback-say", dest="tts_fallback_say", action="store_true", default=True)
    parser.add_argument("--no-tts-fallback-say", dest="tts_fallback_say", action="store_false")
    parser.add_argument("--filler-audio-dir", type=Path, default=Path("data/voice/fillers_step_tts_mini"))
    parser.add_argument("--start-sound-dir", type=Path, default=Path("data/voice/start_sounds_step_tts_mini"))
    parser.add_argument("--no-cached-filler", action="store_true")
    parser.add_argument("--filler-min-interval", type=float, default=8.0)
    parser.add_argument("--filler-max-interval", type=float, default=15.0)
    parser.add_argument("--plan-speech-wait", type=float, default=4.0)
    parser.add_argument("--plan-summary-speech-wait", type=float, default=6.0)
    parser.add_argument("--plan-summary-max-steps", type=int, default=3)
    parser.add_argument("--plan-summary-max-chars", type=int, default=42)
    parser.add_argument("--speech-chunk-chars", type=int, default=36)
    parser.add_argument("--speech-wait-chunk-chars", type=int, default=28)
    parser.add_argument("--tool-start-speech-wait", type=float, default=8.0)
    parser.add_argument("--tool-speech-cooldown", type=float, default=20.0)
    parser.add_argument("--model-waiting-filler", action="store_true", help="Use the front lane to generate waiting filler instead of cached audio.")
    parser.add_argument("--no-rec-indicator", action="store_true")
    parser.add_argument("--no-front-note", action="store_true")
    parser.add_argument("--no-text-input", action="store_true")
    parser.add_argument("--double-click-window", type=float, default=0.38)
    parser.add_argument("--hold-start-delay", type=float, default=0.18)
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument("--asr-timeout", type=float, default=120.0)
    parser.add_argument("--background-timeout", type=float, default=180.0)
    parser.add_argument("--pro-turn-timeout", type=float, default=180.0)
    parser.add_argument("--back-model-first-answer-timeout", type=float, default=25.0)
    parser.add_argument("--initial-answer-budget", type=float, default=10.0)
    parser.add_argument("--pro-tool-timeout", type=float, default=20.0)
    parser.add_argument("--pro-tool-retries", type=int, default=3)
    parser.add_argument("--pro-tool-call-limit", type=int, default=20)
    parser.add_argument("--followup-interrupt-priority", type=int, default=1)
    parser.add_argument("--followup-speak-priority", type=int, default=1)
    parser.add_argument("--followup-dedupe-seconds", type=float, default=120.0)
    parser.add_argument("--followup-dedupe-similarity", type=float, default=0.72)
    parser.add_argument("--compress-every-turns", type=int, default=6)
    parser.add_argument("--compress-every-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.filler_max_interval < args.filler_min_interval:
        args.filler_min_interval, args.filler_max_interval = args.filler_max_interval, args.filler_min_interval
    return args


def check_runtime_dependencies(interactive: bool) -> None:
    if interactive and not shutil.which("say"):
        print("Warning: say command not found; use --no-say or install macOS speech tools.", flush=True)
    if interactive and not shutil.which("afplay"):
        print("Warning: afplay command not found; cached filler audio will fall back to say.", flush=True)
