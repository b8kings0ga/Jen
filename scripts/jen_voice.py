#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from voice_assistant.bot import VoiceBot
from voice_assistant.config import check_runtime_dependencies, parse_args
from voice_assistant.dashboard import run_tool_dashboard_server
from voice_assistant.input_hotkeys import HoldKeyTap
from voice_assistant.pyobjc_ui import run_front_note_server, run_recording_overlay_server, run_text_input_server

def main() -> None:
    args = parse_args()
    if args.overlay_server:
        run_recording_overlay_server()
        return
    if args.front_note_server:
        run_front_note_server(args.front_note_api_url)
        return
    if args.text_input_server:
        run_text_input_server(args.text_input_api_url)
        return
    interactive = args.input_wav is None
    check_runtime_dependencies(interactive)
    print(
        "Voice config: "
        f"preset={args.preset} front={args.front} back={args.back} compact={args.compact} plan={args.plan} plan_mode={args.plan_mode} "
        f"asr={args.asr_model} tts={args.tts_model} speech_backend={args.speech_backend} "
        f"tts_timeout={args.tts_timeout} tts_retries={args.tts_retries} "
        f"simple_back={args.simple_back} simple_reasoning_effort={args.simple_reasoning_effort} simple_plan_mode=false "
        f"quality_back={args.quality_back} quality_plan={args.quality_plan} "
        f"quality_reasoning_effort={args.quality_reasoning_effort} quality_plan_mode=true "
        f"plan_timeout={args.plan_timeout} plan_prefetch={args.plan_prefetch} "
        f"plan_speech_wait={args.plan_speech_wait} plan_summary_speech_wait={args.plan_summary_speech_wait} "
        f"tool_start_speech_wait={args.tool_start_speech_wait} "
        f"tool_speech_cooldown={args.tool_speech_cooldown}",
        flush=True,
    )
    bot = VoiceBot(args, interactive=interactive)
    dashboard_server = None
    if interactive and not args.no_tool_dashboard:
        for attempt in range(1, 11):
            try:
                dashboard_server = run_tool_dashboard_server(
                    bot.store,
                    bot.speech,
                    args.tool_dashboard_host,
                    args.tool_dashboard_port,
                    bot=bot,
                )
                break
            except OSError as exc:
                if getattr(exc, "errno", None) != 48 or attempt >= 10:
                    print(f"Tool dashboard failed to start: {exc}", flush=True)
                    break
                print(f"Tool dashboard port busy; retrying ({attempt}/10): {exc}", flush=True)
                time.sleep(0.5)
            except Exception as exc:
                print(f"Tool dashboard failed to start: {exc}", flush=True)
                break
    if args.input_wav is not None:
        try:
            bot.process_audio(args.input_wav)
            bot.wait_background(timeout=args.background_timeout)
        except Exception as exc:
            print(f"Voice bot failed: {exc}", flush=True)
            raise SystemExit(1) from exc
        finally:
            if dashboard_server is not None:
                dashboard_server.shutdown()
            bot.shutdown()
        return

    def handle_signal(signum, frame):
        if dashboard_server is not None:
            dashboard_server.shutdown()
        bot.shutdown()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    HoldKeyTap(
        bot.start_recording,
        bot.stop_recording,
        bot.cancel_pending_recording,
        bot.show_text_input,
        cancel_tap_window=args.cancel_tap_window,
        double_click_window=args.double_click_window,
        hold_start_delay=args.hold_start_delay,
    ).start()
    print(
        "Ready. Hold Right Option for planned quality mode; hold Right Command for simple fast mode. "
        "Double-tap either key for text input. Release, then tap the same key quickly to cancel.",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        bot.shutdown()
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
