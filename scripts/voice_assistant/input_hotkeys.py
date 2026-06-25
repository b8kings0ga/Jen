from __future__ import annotations

import threading
import time
from typing import Callable

from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CFRunLoopRun,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventTapCreate,
    kCFRunLoopCommonModes,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagsChanged,
    kCGHeadInsertEventTap,
    kCGKeyboardEventKeycode,
    kCGSessionEventTap,
)

RIGHT_OPTION_KEYCODE = 61
RIGHT_COMMAND_KEYCODE = 54

class HoldKeyTap:
    KEY_MODES = {
        RIGHT_OPTION_KEYCODE: ("quality", kCGEventFlagMaskAlternate, "Right Option"),
        RIGHT_COMMAND_KEYCODE: ("simple", kCGEventFlagMaskCommand, "Right Command"),
    }

    def __init__(
        self,
        on_press: Callable[[str], None],
        on_release: Callable[[str], None],
        on_cancel: Callable[[str], bool | None],
        on_double_click: Callable[[str], None] | None = None,
        *,
        cancel_tap_window: float = 0.45,
        double_click_window: float = 0.38,
        hold_start_delay: float = 0.18,
    ) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.on_cancel = on_cancel
        self.on_double_click = on_double_click
        self.cancel_tap_window = max(0.1, float(cancel_tap_window))
        self.double_click_window = max(0.1, float(double_click_window))
        self.hold_start_delay = max(0.0, float(hold_start_delay))
        self._lock = threading.Lock()
        self._key_down = False
        self._active_mode: str | None = None
        self._press_started_at = 0.0
        self._press_timer: threading.Timer | None = None
        self._press_consumed = False
        self._recording_started = False
        self._last_release_mode: str | None = None
        self._last_release_at = 0.0
        self._last_tap_mode: str | None = None
        self._last_tap_at = 0.0
        self._tap = None

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def _run(self) -> None:
        mask = 1 << kCGEventFlagsChanged

        def callback(proxy, event_type, event, refcon):
            keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
            key_info = self.KEY_MODES.get(keycode)
            if key_info is None:
                return event
            mode, flag_mask, _label = key_info
            flags = CGEventGetFlags(event)
            key_down = bool(flags & flag_mask)
            if key_down:
                self.handle_key_down(mode)
            else:
                self.handle_key_up(mode)
            return event

        self._tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap, 0, mask, callback, None)
        if self._tap is None:
            print("Cannot install hold-key event tap. Grant Accessibility permission to this terminal/Python process.", flush=True)
            return
        source = CFMachPortCreateRunLoopSource(None, self._tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
        CFRunLoopRun()

    def handle_key_down(self, mode: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        callback: Callable[[str], object] | None = None
        with self._lock:
            if self._key_down:
                return
            self._key_down = True
            self._active_mode = mode
            self._press_started_at = now
            self._press_consumed = False
            self._recording_started = False
            if self._last_release_mode == mode and now - self._last_release_at <= self.cancel_tap_window:
                self._last_release_mode = None
                self._last_release_at = 0.0
                self._last_tap_mode = None
                self._last_tap_at = 0.0
                self._press_consumed = True
                callback = self.on_cancel
            elif self._last_tap_mode == mode and now - self._last_tap_at <= self.double_click_window:
                self._last_tap_mode = None
                self._last_tap_at = 0.0
                self._press_consumed = True
                callback = self.on_double_click
            else:
                self._press_timer = threading.Timer(self.hold_start_delay, self._start_hold_if_current, args=(mode, now))
                self._press_timer.daemon = True
                self._press_timer.start()
        if callback is not None:
            threading.Thread(target=callback, args=(mode,), daemon=True).start()

    def handle_key_up(self, mode: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        callback: Callable[[str], object] | None = None
        with self._lock:
            if not self._key_down or self._active_mode != mode:
                return
            if self._press_timer is not None:
                self._press_timer.cancel()
                self._press_timer = None
            consumed = self._press_consumed
            recording_started = self._recording_started
            self._key_down = False
            self._active_mode = None
            self._press_consumed = False
            self._recording_started = False
            if consumed:
                return
            if recording_started:
                self._last_release_mode = mode
                self._last_release_at = now
                callback = self.on_release
            else:
                self._last_tap_mode = mode
                self._last_tap_at = now
                self._last_release_mode = None
                self._last_release_at = 0.0
        if callback is not None:
            threading.Thread(target=callback, args=(mode,), daemon=True).start()

    def _start_hold_if_current(self, mode: str, press_started_at: float) -> None:
        should_start = False
        with self._lock:
            if (
                self._key_down
                and self._active_mode == mode
                and self._press_started_at == press_started_at
                and not self._press_consumed
            ):
                self._recording_started = True
                self._press_timer = None
                self._last_tap_mode = None
                self._last_tap_at = 0.0
                should_start = True
        if should_start:
            threading.Thread(target=self.on_press, args=(mode,), daemon=True).start()
