from __future__ import annotations

import json
import sys
import threading
import time
from urllib import parse, request

from voice_assistant.http_client import urlopen_bytes
from voice_assistant.ui import render_front_note_editor_document

def run_recording_overlay_server() -> None:
    import math
    import objc
    from AppKit import (
        NSApplication,
        NSBackingStoreBuffered,
        NSBorderlessWindowMask,
        NSColor,
        NSFloatingWindowLevel,
        NSFont,
        NSMakeRect,
        NSScreen,
        NSTextField,
        NSWindow,
        NSView,
    )
    from Foundation import NSObject, NSTimer, NSURL
    from PyObjCTools import AppHelper

    def layer_view(frame, color, radius=0, opacity=1.0):
        view = NSView.alloc().initWithFrame_(frame)
        view.setWantsLayer_(True)
        layer = view.layer()
        layer.setBackgroundColor_(color.CGColor())
        layer.setCornerRadius_(radius)
        layer.setOpacity_(opacity)
        return view

    def text_label(frame, value, size, alpha=0.9, weight=0.35):
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(value)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setAlignment_(1)
        label.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha))
        label.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
        return label

    class OverlayController(NSObject):
        def init(self):
            self = objc.super(OverlayController, self).init()
            self.window = None
            self.root = None
            self.shell = None
            self.glow = None
            self.card = None
            self.card_label = None
            self.dot = None
            self.bars = []
            self.phase = 0.0
            self.visible_progress = 0.0
            self.target_visible = False
            return self

        def setup(self):
            screen = NSScreen.mainScreen().frame()
            width, height = 360, 152
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect((screen.size.width - width) / 2, 86, width, height),
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            self.window.setLevel_(NSFloatingWindowLevel)
            self.window.setOpaque_(False)
            self.window.setAlphaValue_(0.0)
            self.window.setBackgroundColor_(NSColor.clearColor())
            self.window.setIgnoresMouseEvents_(True)
            self.window.setHasShadow_(False)

            self.root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
            self.root.setWantsLayer_(True)
            self.window.setContentView_(self.root)

            self.glow = layer_view(
                NSMakeRect(14, 50, width - 28, 86),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.26, 0.48, 1.0, 0.28),
                31,
                0.7,
            )
            self.root.addSubview_(self.glow)
            self.shell = layer_view(
                NSMakeRect(22, 58, width - 44, 70),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.03, 0.04, 0.06, 0.82),
                26,
                1.0,
            )
            self.shell.layer().setBorderWidth_(1.0)
            self.shell.layer().setBorderColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.16).CGColor())
            self.root.addSubview_(self.shell)

            self.dot = layer_view(
                NSMakeRect(48, 85, 14, 14),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.06, 0.10, 0.95),
                7,
                1.0,
            )
            self.root.addSubview_(self.dot)

            colors = [
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.24, 0.56, 1.0, 0.96),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.70, 0.36, 1.0, 0.94),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.28, 0.58, 0.92),
            ]
            for i in range(28):
                color = colors[min(2, int(i / 10))]
                bar = layer_view(NSMakeRect(78 + i * 7.4, 88, 4, 18), color, 2, 0.9)
                self.root.addSubview_(bar)
                self.bars.append(bar)

            self.card = layer_view(
                NSMakeRect(70, 12, width - 140, 38),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.05, 0.07, 0.72),
                15,
                1.0,
            )
            self.card.layer().setBorderWidth_(1.0)
            self.card.layer().setBorderColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.12).CGColor())
            self.root.addSubview_(self.card)
            self.card_label = text_label(NSMakeRect(92, 22, width - 184, 17), "正在听", 13, 0.86, 0.35)
            self.root.addSubview_(self.card_label)

            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1 / 30, self, "animate:", None, True)

        def show(self):
            if self.window is None:
                self.setup()
            if self.card_label is not None:
                self.card_label.setStringValue_("正在听")
            self.target_visible = True
            self.window.setAlphaValue_(1.0)
            self.window.orderFrontRegardless()

        def hide(self):
            if self.window is not None:
                self.target_visible = False

        def notice_(self, text):
            if self.window is None:
                self.setup()
            if self.card_label is not None:
                self.card_label.setStringValue_(str(text))
            self.target_visible = True
            self.window.setAlphaValue_(1.0)
            self.window.orderFrontRegardless()
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.9, self, "hideAfterNotice:", None, False)

        def hideAfterNotice_(self, timer):
            self.hide()

        def animate_(self, timer):
            if self.window is None or self.root is None:
                return
            self.phase += 0.12
            if self.target_visible:
                self.visible_progress = min(1.0, self.visible_progress + 0.12)
            else:
                self.visible_progress = max(0.0, self.visible_progress - 0.16)
            progress = self.visible_progress
            self.root.setAlphaValue_(progress)

            lift = (1.0 - progress) * -14
            self.shell.setFrame_(NSMakeRect(22, 58 + lift, 316, 70))
            self.glow.setFrame_(NSMakeRect(14, 50 + lift, 332, 86))
            self.card.setFrame_(NSMakeRect(70, 12 + lift * 0.55, 220, 38))
            self.card_label.setFrame_(NSMakeRect(92, 22 + lift * 0.55, 176, 17))
            self.dot.setFrame_(NSMakeRect(48, 85 + lift, 14, 14))
            self.glow.layer().setOpacity_(0.45 + 0.22 * abs(math.sin(self.phase * 1.8)))
            self.dot.layer().setOpacity_(0.45 + 0.5 * abs(math.sin(self.phase * 2.4)))

            wave_y = 92 + lift
            for i, bar in enumerate(self.bars):
                h = 12 + 31 * (0.35 + 0.65 * abs(math.sin(self.phase + i * 0.48)))
                h += 5 * math.sin(self.phase * 1.6 + i * 0.22)
                h = max(10, min(50, h))
                bar.setFrame_(NSMakeRect(78 + i * 7.4, wave_y - h / 2, 4, h))

            if progress <= 0.01 and not self.target_visible:
                self.window.setAlphaValue_(0.0)
                self.window.orderOut_(None)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)
    controller = OverlayController.alloc().init()
    controller.setup()

    def read_commands() -> None:
        for raw in sys.stdin:
            command = raw.strip().lower()
            if command == "show":
                AppHelper.callAfter(controller.show)
            elif command == "hide":
                AppHelper.callAfter(controller.hide)
            elif command.startswith("notice:"):
                AppHelper.callAfter(controller.notice_, raw.strip()[7:])
            elif command == "quit":
                AppHelper.callAfter(app.terminate_, None)
                return

    threading.Thread(target=read_commands, daemon=True).start()
    app.run()


def run_text_input_server(api_url: str) -> None:
    import objc
    from AppKit import (
        NSApplication,
        NSBackingStoreBuffered,
        NSBorderlessWindowMask,
        NSButton,
        NSColor,
        NSFontAttributeName,
        NSFloatingWindowLevel,
        NSForegroundColorAttributeName,
        NSFont,
        NSMakeRect,
        NSMakeSize,
        NSScrollView,
        NSScreen,
        NSTextField,
        NSTextView,
        NSView,
        NSWindow,
    )
    from Foundation import NSAttributedString, NSObject
    from PyObjCTools import AppHelper

    TEXT_INPUT_MIN_HEIGHT = 96
    TEXT_INPUT_MAX_HEIGHT = 220
    SHIFT_MASK = 1 << 17
    COMMAND_MASK = 1 << 20
    KEY_A = 0
    KEY_X = 7
    KEY_C = 8
    KEY_V = 9
    KEY_W = 13
    KEY_1 = 18
    KEY_2 = 19
    KEY_RETURN = 36
    KEY_K = 40
    KEY_TAB = 48
    KEY_ESCAPE = 53
    KEY_NUMPAD_RETURN = 76

    class TextInputWindow(NSWindow):
        def canBecomeKeyWindow(self):
            return True

        def canBecomeMainWindow(self):
            return True

        def keyDown_(self, event):
            controller = getattr(self, "controller", None)
            if controller is not None and controller.handleKeyEvent_(event):
                return
            objc.super(TextInputWindow, self).keyDown_(event)

        def performKeyEquivalent_(self, event):
            controller = getattr(self, "controller", None)
            if controller is not None and controller.handleKeyEvent_(event):
                return True
            return objc.super(TextInputWindow, self).performKeyEquivalent_(event)

    def color(r: float, g: float, b: float, a: float = 1.0):
        return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)

    def label(frame, text: str, size: float, alpha: float = 0.86, weight: float = 0.35, align: int = 0):
        view = NSTextField.alloc().initWithFrame_(frame)
        view.setStringValue_(text)
        view.setBezeled_(False)
        view.setDrawsBackground_(False)
        view.setEditable_(False)
        view.setSelectable_(False)
        view.setAlignment_(align)
        view.setTextColor_(color(0.07, 0.075, 0.07, alpha))
        view.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
        return view

    class PromptTextView(NSTextView):
        def initWithController_(self, controller):
            self = objc.super(PromptTextView, self).init()
            self.controller = controller
            return self

        def keyDown_(self, event):
            if self.controller is not None and self.controller.handleKeyEvent_(event):
                return
            objc.super(PromptTextView, self).keyDown_(event)
            if self.controller is not None:
                self.controller.refreshLayout()

    class TextInputController(NSObject):
        def init(self):
            self = objc.super(TextInputController, self).init()
            self.window = None
            self.root = None
            self.scroll_view = None
            self.text_view = None
            self.placeholder = None
            self.mode_hint = None
            self.mode_segment = None
            self.mode_highlight = None
            self.mode_quality_button = None
            self.mode_simple_button = None
            self.send_button = None
            self.width = 0
            self.mode = "quality"
            return self

        def setup(self):
            screen = NSScreen.mainScreen().visibleFrame()
            width = min(720, max(520, int(screen.size.width * 0.46)))
            height = TEXT_INPUT_MIN_HEIGHT
            self.width = width
            self.window = TextInputWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(screen.origin.x + (screen.size.width - width) / 2, screen.origin.y + 76, width, height),
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            self.window.controller = self
            self.window.setLevel_(NSFloatingWindowLevel)
            self.window.setOpaque_(False)
            self.window.setAlphaValue_(1.0)
            self.window.setBackgroundColor_(NSColor.clearColor())
            self.window.setHasShadow_(True)

            self.root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
            self.root.setWantsLayer_(True)
            self.root.layer().setBackgroundColor_(color(1.0, 1.0, 0.992, 1.0).CGColor())
            self.root.layer().setCornerRadius_(22)
            self.root.layer().setBorderWidth_(1.0)
            self.root.layer().setBorderColor_(color(0.12, 0.12, 0.11, 0.14).CGColor())

            self.mode_segment = NSView.alloc().initWithFrame_(NSMakeRect(18, 10, 126, 28))
            self.mode_segment.setWantsLayer_(True)
            self.mode_segment.layer().setBackgroundColor_(color(0.07, 0.075, 0.07, 0.055).CGColor())
            self.mode_segment.layer().setCornerRadius_(14)
            self.root.addSubview_(self.mode_segment)

            self.mode_highlight = NSView.alloc().initWithFrame_(NSMakeRect(21, 13, 59, 22))
            self.mode_highlight.setWantsLayer_(True)
            self.mode_highlight.layer().setBackgroundColor_(color(0.065, 0.068, 0.064, 0.94).CGColor())
            self.mode_highlight.layer().setCornerRadius_(11)
            self.root.addSubview_(self.mode_highlight)

            self.mode_quality_button = NSButton.alloc().initWithFrame_(NSMakeRect(21, 13, 59, 22))
            self.mode_quality_button.setBordered_(False)
            self.mode_quality_button.setTarget_(self)
            self.mode_quality_button.setAction_("setQualityMode:")
            self.root.addSubview_(self.mode_quality_button)

            self.mode_simple_button = NSButton.alloc().initWithFrame_(NSMakeRect(82, 13, 59, 22))
            self.mode_simple_button.setBordered_(False)
            self.mode_simple_button.setTarget_(self)
            self.mode_simple_button.setAction_("setSimpleMode:")
            self.root.addSubview_(self.mode_simple_button)

            shortcut = label(NSMakeRect(width - 154, 17, 92, 16), "Enter 发送", 11, 0.28, 0.28, 2)
            self.root.addSubview_(shortcut)

            send_back = NSView.alloc().initWithFrame_(NSMakeRect(width - 46, 8, 34, 34))
            send_back.setWantsLayer_(True)
            send_back.layer().setBackgroundColor_(color(0.065, 0.068, 0.064, 0.96).CGColor())
            send_back.layer().setCornerRadius_(17)
            self.root.addSubview_(send_back)

            self.send_button = NSButton.alloc().initWithFrame_(NSMakeRect(width - 46, 8, 34, 34))
            self.send_button.setTitle_("↑")
            self.send_button.setAttributedTitle_(
                NSAttributedString.alloc().initWithString_attributes_(
                    "↑",
                    {
                        NSForegroundColorAttributeName: color(1.0, 1.0, 1.0, 0.96),
                        NSFontAttributeName: NSFont.systemFontOfSize_weight_(21, 0.52),
                    },
                )
            )
            self.send_button.setBordered_(False)
            self.send_button.setTarget_(self)
            self.send_button.setAction_("submit:")
            self.root.addSubview_(self.send_button)

            self.scroll_view = NSScrollView.alloc().initWithFrame_(NSMakeRect(22, 46, width - 44, 34))
            self.scroll_view.setDrawsBackground_(False)
            self.scroll_view.setBorderType_(0)
            self.scroll_view.setHasVerticalScroller_(True)
            self.scroll_view.setAutohidesScrollers_(True)
            self.text_view = PromptTextView.alloc().initWithController_(self)
            self.text_view.setFrame_(NSMakeRect(0, 0, width - 54, 31))
            self.text_view.setFont_(NSFont.systemFontOfSize_(17))
            self.text_view.setTextColor_(color(0.055, 0.055, 0.048, 0.98))
            self.text_view.setDrawsBackground_(False)
            self.text_view.setRichText_(False)
            self.text_view.setHorizontallyResizable_(False)
            self.text_view.setVerticallyResizable_(True)
            self.text_view.setMinSize_(NSMakeSize(0, 31))
            self.text_view.setMaxSize_(NSMakeSize(width - 54, 10000))
            self.text_view.textContainer().setWidthTracksTextView_(True)
            self.text_view.textContainer().setContainerSize_(NSMakeSize(width - 54, 10000))
            self.text_view.setAutomaticQuoteSubstitutionEnabled_(False)
            self.text_view.setAutomaticDashSubstitutionEnabled_(False)
            self.scroll_view.setDocumentView_(self.text_view)
            self.root.addSubview_(self.scroll_view)

            self.placeholder = label(NSMakeRect(25, 56, width - 70, 22), "问一句，或输入一个任务", 17, 0.34, 0.30)
            self.root.addSubview_(self.placeholder)

            self.window.setContentView_(self.root)
            self.setMode_(self.mode)

        def show_(self, mode):
            if self.window is None:
                self.setup()
            self.setMode_(mode)
            app = NSApplication.sharedApplication()
            app.activateIgnoringOtherApps_(True)
            self.window.makeKeyAndOrderFront_(None)
            self.window.orderFrontRegardless()
            if self.text_view is not None:
                self.window.makeFirstResponder_(self.text_view)
            self.refreshLayout()

        def hide(self):
            if self.window is not None:
                self.window.orderOut_(None)

        def refreshPlaceholder(self):
            if self.text_view is None or self.placeholder is None:
                return
            text = str(self.text_view.string() or "")
            self.placeholder.setHidden_(bool(text.strip()))

        def _desiredHeight(self):
            if self.text_view is None:
                return TEXT_INPUT_MIN_HEIGHT
            text_height = self._contentHeight()
            return max(TEXT_INPUT_MIN_HEIGHT, min(TEXT_INPUT_MAX_HEIGHT, text_height + 62))

        def _contentHeight(self):
            if self.text_view is None:
                return 31
            text_width = max(200, int(self.width or 520) - 54)
            try:
                self.text_view.setFrame_(NSMakeRect(0, 0, text_width, 10000))
                self.text_view.textContainer().setContainerSize_(NSMakeSize(text_width, 10000))
                self.text_view.textContainer().setWidthTracksTextView_(True)
                layout = self.text_view.layoutManager()
                container = self.text_view.textContainer()
                layout.ensureLayoutForTextContainer_(container)
                used = layout.usedRectForTextContainer_(container)
                return max(31, min(158, int(used.size.height) + 10))
            except Exception:
                text = str(self.text_view.string() or "")
                wrap_chars = max(26, int((max(self.width, 520) - 54) / 9))
                logical_lines = 1
                if text:
                    logical_lines = 0
                    for line in text.splitlines() or [""]:
                        logical_lines += max(1, int((len(line) + wrap_chars - 1) / wrap_chars))
                return max(31, min(158, logical_lines * 22 + 8))

        def refreshLayout(self):
            self.refreshPlaceholder()
            if self.window is None or self.root is None or self.scroll_view is None or self.text_view is None or self.placeholder is None:
                return
            target_height = self._desiredHeight()
            frame = self.window.frame()
            width = frame.size.width
            if abs(frame.size.height - target_height) > 0.5:
                self.window.setFrame_display_(NSMakeRect(frame.origin.x, frame.origin.y, width, target_height), True)
            self.root.setFrame_(NSMakeRect(0, 0, width, target_height))
            scroll_height = max(31, target_height - 62)
            self.scroll_view.setFrame_(NSMakeRect(22, 46, width - 44, scroll_height))
            text_width = max(200, width - 54)
            content_height = self._contentHeight()
            self.text_view.setMaxSize_(NSMakeSize(text_width, 10000))
            self.text_view.textContainer().setContainerSize_(NSMakeSize(text_width, 10000))
            self.text_view.textContainer().setWidthTracksTextView_(True)
            self.text_view.setFrame_(NSMakeRect(0, 0, text_width, max(scroll_height - 3, content_height)))
            self.placeholder.setFrame_(NSMakeRect(25, 46 + max(8, scroll_height - 24), width - 70, 22))

        def handleKeyEvent_(self, event):
            key = int(event.keyCode())
            flags = int(event.modifierFlags())
            has_shift = bool(flags & SHIFT_MASK)
            has_command = bool(flags & COMMAND_MASK)
            if key == KEY_ESCAPE:
                self.hide()
                return True
            if has_command and key == KEY_A:
                if self.text_view is not None:
                    self.text_view.selectAll_(None)
                return True
            if has_command and key == KEY_C:
                if self.text_view is not None:
                    self.text_view.copy_(None)
                return True
            if has_command and key == KEY_V:
                if self.text_view is not None:
                    self.text_view.paste_(None)
                self.refreshLayout()
                return True
            if has_command and key == KEY_X:
                if self.text_view is not None:
                    self.text_view.cut_(None)
                self.refreshLayout()
                return True
            if has_command and key == KEY_W:
                self.hide()
                return True
            if has_command and key == KEY_K:
                self.clear()
                return True
            if has_command and key == KEY_1:
                self.setMode_("quality")
                return True
            if has_command and key == KEY_2:
                self.setMode_("simple")
                return True
            if key == KEY_TAB:
                self.toggleMode()
                return True
            if key in {KEY_RETURN, KEY_NUMPAD_RETURN} and (has_command or not has_shift):
                self.submit_(None)
                return True
            return False

        def setMode_(self, mode):
            self.mode = "simple" if str(mode) == "simple" else "quality"
            if self.mode_highlight is not None:
                self.mode_highlight.setFrame_(NSMakeRect(82 if self.mode == "simple" else 21, 13, 59, 22))
            if self.mode_quality_button is not None:
                active = self.mode == "quality"
                self.mode_quality_button.setAttributedTitle_(
                    NSAttributedString.alloc().initWithString_attributes_(
                        "质量",
                        {
                            NSForegroundColorAttributeName: color(1.0, 1.0, 1.0, 0.96) if active else color(0.07, 0.075, 0.07, 0.50),
                            NSFontAttributeName: NSFont.systemFontOfSize_weight_(12, 0.46 if active else 0.34),
                        },
                    )
                )
            if self.mode_simple_button is not None:
                active = self.mode == "simple"
                self.mode_simple_button.setAttributedTitle_(
                    NSAttributedString.alloc().initWithString_attributes_(
                        "快速",
                        {
                            NSForegroundColorAttributeName: color(1.0, 1.0, 1.0, 0.96) if active else color(0.07, 0.075, 0.07, 0.50),
                            NSFontAttributeName: NSFont.systemFontOfSize_weight_(12, 0.46 if active else 0.34),
                        },
                    )
                )

        def toggleMode(self):
            self.setMode_("quality" if self.mode == "simple" else "simple")

        def setQualityMode_(self, sender):
            self.setMode_("quality")

        def setSimpleMode_(self, sender):
            self.setMode_("simple")

        def clear(self):
            if self.text_view is not None:
                self.text_view.setString_("")
            self.refreshLayout()

        def submit_(self, sender):
            if self.text_view is None:
                return
            text = str(self.text_view.string() or "").strip()
            if not text:
                return
            payload = json.dumps({"text": text, "mode": self.mode}, ensure_ascii=False).encode("utf-8")
            req = request.Request(api_url, data=payload, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
            try:
                request.urlopen(req, timeout=2.0).read()
            except Exception as exc:
                print(f"text input submit failed: {exc}", flush=True)
                return
            self.text_view.setString_("")
            self.refreshLayout()
            self.hide()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)
    controller = TextInputController.alloc().init()
    controller.setup()

    def read_commands() -> None:
        for raw in sys.stdin:
            command = raw.strip()
            lower = command.lower()
            if lower.startswith("show:"):
                AppHelper.callAfter(controller.show_, lower.split(":", 1)[1])
            elif lower == "show":
                AppHelper.callAfter(controller.show_, "quality")
            elif lower == "hide":
                AppHelper.callAfter(controller.hide)
            elif lower == "quit":
                AppHelper.callAfter(app.terminate_, None)
                return

    threading.Thread(target=read_commands, daemon=True).start()
    app.run()


def run_front_note_server(api_url: str) -> None:
    import objc
    from AppKit import (
        NSApplication,
        NSBackingStoreBuffered,
        NSBorderlessWindowMask,
        NSColor,
        NSFloatingWindowLevel,
        NSMakePoint,
        NSMakeRect,
        NSScreen,
        NSView,
        NSWindow,
    )
    from Foundation import NSObject, NSTimer, NSURL
    from PyObjCTools import AppHelper
    from WebKit import WKWebView, WKWebViewConfiguration

    class FrontNoteWindow(NSWindow):
        def canBecomeKeyWindow(self):
            return True

        def canBecomeMainWindow(self):
            return True

    class DragView(NSView):
        def initWithController_(self, controller):
            self = objc.super(DragView, self).init()
            self.controller = controller
            self.drag_start = None
            self.window_start = None
            return self

        def mouseDown_(self, event):
            window = self.window()
            if window is None:
                return
            if self.controller is not None:
                self.controller.dragging = True
            self.drag_start = event.locationInWindow()
            self.window_start = window.frame().origin

        def mouseDragged_(self, event):
            window = self.window()
            if window is None or self.drag_start is None or self.window_start is None:
                return
            point = event.locationInWindow()
            dx = point.x - self.drag_start.x
            dy = point.y - self.drag_start.y
            window.setFrameOrigin_(NSMakePoint(self.window_start.x + dx, self.window_start.y + dy))

        def mouseUp_(self, event):
            if self.controller is not None:
                self.controller.dragging = False
            if self.controller is not None:
                self.controller.snapToEdge()

    class FrontNoteController(NSObject):
        def init(self):
            self = objc.super(FrontNoteController, self).init()
            self.window = None
            self.root = None
            self.webview = None
            self.api_url = api_url
            self.version = -1
            self.visible = False
            self.pinned_edge = "right"
            self.width = 520
            self.height = 420
            self.expanded_origin = None
            self.hidden_origin = None
            self.target_origin = None
            self.dragging = False
            self.hide_after_at = 0.0
            return self

        def setup(self):
            screen = NSScreen.mainScreen().visibleFrame()
            self.expanded_origin = NSMakePoint(screen.origin.x + screen.size.width - self.width - 22, screen.origin.y + 120)
            self.window = FrontNoteWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(self.expanded_origin.x, self.expanded_origin.y, self.width, self.height),
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            self.window.setLevel_(NSFloatingWindowLevel)
            self.window.setOpaque_(False)
            self.window.setAlphaValue_(0.98)
            self.window.setBackgroundColor_(NSColor.clearColor())
            self.window.setHasShadow_(False)
            self.window.setIgnoresMouseEvents_(False)
            self.root = DragView.alloc().initWithController_(self)
            self.root.setFrame_(NSMakeRect(0, 0, self.width, self.height))
            self.root.setWantsLayer_(True)
            self.root.layer().setBackgroundColor_(NSColor.clearColor().CGColor())
            config = WKWebViewConfiguration.alloc().init()
            self.webview = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, self.width, self.height), config)
            if hasattr(self.webview, "setDrawsBackground_"):
                self.webview.setDrawsBackground_(False)
            if hasattr(self.webview, "setValue_forKey_"):
                try:
                    self.webview.setValue_forKey_(False, "drawsBackground")
                except Exception:
                    pass
            self.webview.setWantsLayer_(True)
            if self.webview.layer() is not None:
                self.webview.layer().setBackgroundColor_(NSColor.clearColor().CGColor())
            self.root.addSubview_(self.webview)
            self.window.setContentView_(self.root)
            parsed_api = parse.urlparse(self.api_url)
            base_url = f"{parsed_api.scheme}://{parsed_api.netloc}/" if parsed_api.scheme and parsed_api.netloc else "http://127.0.0.1:8765/"
            self.webview.loadHTMLString_baseURL_(render_front_note_editor_document(self.api_url), NSURL.URLWithString_(base_url))
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.45, self, "poll:", None, True)
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1 / 60, self, "edgeTick:", None, True)

        def poll_(self, timer):
            try:
                req = request.Request(self.api_url, headers={"Accept": "application/json"})
                raw = urlopen_bytes(req, timeout=1.5, verify_tls=True, label="front note poll")
                state = json.loads(raw.decode("utf-8"))
            except Exception:
                return
            version = int(state.get("version") or 0)
            if version == self.version:
                return
            self.version = version
            self.applyState_(state)

        def applyState_(self, state):
            self.visible = bool(state.get("visible"))
            self.pinned_edge = str(state.get("position") or "right")
            self.width = max(360, min(int(state.get("width") or 520), 980))
            self.height = max(280, min(int(state.get("height") or 420), 780))
            if self.window is None or self.webview is None or self.root is None:
                return
            frame = self.window.frame()
            self.window.setFrame_display_(NSMakeRect(frame.origin.x, frame.origin.y, self.width, self.height), True)
            self.root.setFrame_(NSMakeRect(0, 0, self.width, self.height))
            self.webview.setFrame_(NSMakeRect(0, 0, self.width, self.height))
            try:
                state_json = json.dumps(state, ensure_ascii=False)
                script = f"window.__frontNoteApplyState && window.__frontNoteApplyState({state_json});"
                self.webview.evaluateJavaScript_completionHandler_(script, None)
            except Exception:
                pass
            if self.visible:
                app = NSApplication.sharedApplication()
                app.activateIgnoringOtherApps_(True)
                self.window.makeKeyAndOrderFront_(None)
                self.window.orderFrontRegardless()
                self.snapToEdge()
            else:
                self.target_origin = self.hidden_origin
                self.window.orderOut_(None)

        def snapToEdge(self):
            if self.window is None:
                return
            screen = NSScreen.mainScreen().visibleFrame()
            frame = self.window.frame()
            left_dist = abs(frame.origin.x - screen.origin.x)
            right_dist = abs((frame.origin.x + frame.size.width) - (screen.origin.x + screen.size.width))
            if self.pinned_edge not in {"left", "right"}:
                self.pinned_edge = "left" if left_dist < right_dist else "right"
            y = min(max(frame.origin.y, screen.origin.y + 18), screen.origin.y + screen.size.height - frame.size.height - 18)
            if self.pinned_edge == "left":
                self.expanded_origin = NSMakePoint(screen.origin.x + 12, y)
                self.hidden_origin = NSMakePoint(screen.origin.x - frame.size.width + 18, y)
            else:
                self.expanded_origin = NSMakePoint(screen.origin.x + screen.size.width - frame.size.width - 12, y)
                self.hidden_origin = NSMakePoint(screen.origin.x + screen.size.width - 18, y)
            self.target_origin = self.expanded_origin

        def edgeTick_(self, timer):
            if self.window is None or not self.visible:
                return
            if self.dragging:
                return
            if self.expanded_origin is None or self.hidden_origin is None:
                self.snapToEdge()
            mouse = NSEvent.mouseLocation()
            frame = self.window.frame()
            inside_window = frame.origin.x <= mouse.x <= frame.origin.x + frame.size.width and frame.origin.y <= mouse.y <= frame.origin.y + frame.size.height
            inside_y = frame.origin.y - 42 <= mouse.y <= frame.origin.y + frame.size.height + 42
            if self.pinned_edge == "left":
                near = inside_window or (mouse.x <= NSScreen.mainScreen().visibleFrame().origin.x + 44 and inside_y)
            else:
                screen = NSScreen.mainScreen().visibleFrame()
                near = inside_window or (mouse.x >= screen.origin.x + screen.size.width - 44 and inside_y)
            now = time.time()
            if near:
                self.hide_after_at = 0.0
                target = self.expanded_origin
            else:
                if self.hide_after_at <= 0:
                    self.hide_after_at = now + 1.2
                target = self.expanded_origin if now < self.hide_after_at else self.hidden_origin
            if target is not None:
                self.target_origin = target
            if self.target_origin is None:
                return
            current = self.window.frame().origin
            next_x = current.x + (self.target_origin.x - current.x) * 0.22
            next_y = current.y + (self.target_origin.y - current.y) * 0.22
            if abs(self.target_origin.x - next_x) < 0.5:
                next_x = self.target_origin.x
            if abs(self.target_origin.y - next_y) < 0.5:
                next_y = self.target_origin.y
            self.window.setFrameOrigin_(NSMakePoint(next_x, next_y))
            hidden = abs(self.target_origin.x - self.hidden_origin.x) < 0.1 and abs(self.target_origin.y - self.hidden_origin.y) < 0.1
            self.window.setAlphaValue_(0.86 if hidden else 0.98)

    from AppKit import NSEvent

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)
    controller = FrontNoteController.alloc().init()
    controller.setup()

    def read_commands() -> None:
        for raw in sys.stdin:
            if raw.strip().lower() == "quit":
                AppHelper.callAfter(app.terminate_, None)
                return

    threading.Thread(target=read_commands, daemon=True).start()
    app.run()
