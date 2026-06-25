from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


BUNDLE_ID = "com.b8kings.jen.location"
APP_NAME = "Jen Location"


def current_address(timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Return the current reverse-geocoded address through a stable macOS app identity."""
    timeout_seconds = max(0.5, min(float(timeout_seconds or 2.0), 10.0))
    app_path = ensure_location_helper_app()
    with tempfile.TemporaryDirectory(prefix="jen-location-") as tmp:
        result_path = Path(tmp) / "result.json"
        try:
            subprocess.run(
                [
                    "open",
                    "-W",
                    "-n",
                    str(app_path),
                    "--args",
                    "--result",
                    str(result_path),
                    "--timeout",
                    str(timeout_seconds),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 4.0,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "current address lookup timed out", "app": APP_NAME}
        if not result_path.exists():
            return {"ok": False, "error": "location helper did not return a result", "app": APP_NAME}
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "error": f"invalid location helper result: {exc}", "app": APP_NAME}
        return payload if isinstance(payload, dict) else {"ok": False, "error": "invalid location helper payload", "app": APP_NAME}


def ensure_location_helper_app() -> Path:
    root = _repo_root()
    app_path = root / ".cache" / "voice_assistant" / "JenLocation.app"
    macos_dir = app_path / "Contents" / "MacOS"
    resources_dir = app_path / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleExecutable": "location-helper",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSUIElement": True,
        "NSLocationWhenInUseUsageDescription": "Jen uses your location to answer local weather and map questions.",
        "NSLocationUsageDescription": "Jen uses your location to answer local weather and map questions.",
    }
    (app_path / "Contents" / "Info.plist").write_bytes(plistlib.dumps(plist))

    executable = macos_dir / "location-helper"
    python_path = sys.executable
    scripts_path = root / "scripts"
    launcher = (
        "#!/bin/zsh\n"
        f"export PYTHONPATH={_shell_quote(str(scripts_path))}:\"${{PYTHONPATH}}\"\n"
        f"exec {_shell_quote(python_path)} -m voice_assistant.location_helper --child \"$@\"\n"
    )
    executable.write_text(launcher, encoding="utf-8")
    executable.chmod(0o755)
    return app_path


def child_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    ns = parser.parse_args(argv)
    payload = _lookup_current_address(max(0.5, min(float(ns.timeout or 2.0), 10.0)))
    Path(ns.result).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


def _lookup_current_address(timeout_seconds: float) -> dict[str, Any]:
    try:
        import objc
        from AppKit import NSApplication
        from CoreLocation import CLGeocoder, CLLocationManager, kCLLocationAccuracyKilometer
        from Foundation import NSDate, NSObject, NSRunLoop
    except Exception as exc:
        return {"ok": False, "error": f"CoreLocation unavailable: {exc}", "app": APP_NAME}

    NSApplication.sharedApplication()

    def auth_status(manager: Any) -> int:
        try:
            return int(manager.authorizationStatus())
        except Exception:
            try:
                return int(CLLocationManager.authorizationStatus())
            except Exception:
                return -1

    class Delegate(NSObject):
        def init(self):  # type: ignore[no-untyped-def]
            self = objc.super(Delegate, self).init()
            if self is None:
                return None
            self.location = None
            self.error = None
            return self

        def locationManager_didUpdateLocations_(self, manager, locations):  # type: ignore[no-untyped-def]
            if locations and len(locations):
                self.location = locations[-1]

        def locationManager_didFailWithError_(self, manager, error):  # type: ignore[no-untyped-def]
            self.error = str(error)

    manager = CLLocationManager.alloc().init()
    delegate = Delegate.alloc().init()
    manager.setDelegate_(delegate)
    try:
        manager.requestWhenInUseAuthorization()
    except Exception:
        pass
    before = auth_status(manager)
    try:
        manager.setDesiredAccuracy_(kCLLocationAccuracyKilometer)
        manager.startUpdatingLocation()
    except Exception as exc:
        return {"ok": False, "error": f"location start failed: {exc}", "authorization_status": before, "app": APP_NAME}

    deadline = time.time() + timeout_seconds
    location = None
    while time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
        location = delegate.location or manager.location()
        if location is not None:
            break
    manager.stopUpdatingLocation()
    status = auth_status(manager)
    if location is None:
        error = delegate.error or _authorization_error(status)
        return {
            "ok": False,
            "error": error or "current location unavailable",
            "authorization_status": status,
            "permission_prompted": before == 0,
            "app": APP_NAME,
        }

    state: dict[str, Any] = {"done": False, "payload": {"ok": False, "error": "reverse geocode unavailable", "app": APP_NAME}}

    def handler(placemarks, error):  # type: ignore[no-untyped-def]
        if error is not None or not placemarks:
            state["payload"] = {"ok": False, "error": str(error or "reverse geocode empty"), "authorization_status": status, "app": APP_NAME}
        else:
            placemark = placemarks[0]
            parts = [
                str(value)
                for value in [
                    placemark.name(),
                    placemark.locality(),
                    placemark.administrativeArea(),
                    placemark.country(),
                ]
                if value
            ]
            state["payload"] = {
                "ok": True,
                "address": "，".join(dict.fromkeys(parts)),
                "authorization_status": status,
                "app": APP_NAME,
            }
        state["done"] = True

    CLGeocoder.alloc().init().reverseGeocodeLocation_completionHandler_(location, handler)
    deadline = time.time() + min(2.0, timeout_seconds)
    while time.time() < deadline and not state["done"]:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    return state["payload"]


def _authorization_error(status: int) -> str:
    if status in {1, 2}:
        return "location permission denied"
    if status == 0:
        return "location permission not determined"
    return "current location unavailable"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(child_main(sys.argv[2:]))
    print(json.dumps(current_address(), ensure_ascii=False))
