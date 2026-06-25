from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from voice_assistant.coding_workspace import CodingServiceRegistry, CodingWorkspaceIndex, read_manifest, register_workspace_program


REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_MARKERS = (
    "import webview",
    "from webview",
    "webview.create_window",
    "webview.start",
    "nswindow",
    "nspanel",
    "wkwebview",
)

LOCAL_ASSET_EXTENSIONS = {
    ".css",
    ".gif",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".m4a",
    ".mp3",
    ".ogg",
    ".png",
    ".svg",
    ".wav",
    ".webp",
}


def coding_executor_label(executor: str | None) -> str:
    value = str(executor or "codex").strip().lower()
    if value == "antigravity":
        return "Antigravity"
    return "Codex"


def summarize_coding_event(event: dict[str, Any], executor: str | None = "codex") -> tuple[str, str] | None:
    label = coding_executor_label(executor)
    event_type = str(event.get("type") or "")
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    if event_type == "turn.completed":
        return "artifact_ready", f"{label} 产物完成"
    if event_type != "item.completed":
        return None
    item_type = str(item.get("type") or "")
    if item_type == "agent_message":
        text = _short_text(str(item.get("text") or ""), 64)
        if text:
            return "running", f"{label} 进展 · {text}"
    if item_type == "command_execution":
        command = _short_command(str(item.get("command") or ""))
        status = str(item.get("status") or "")
        if command:
            if status == "completed":
                return "running", f"{label} 进展 · 运行完 {command}"
            return "running", f"{label} 进展 · 正在运行 {command}"
    if item_type == "file_change":
        names: list[str] = []
        for change in item.get("changes") or []:
            if isinstance(change, dict):
                path = str(change.get("path") or "")
                if path:
                    names.append(Path(path).name)
        if names:
            return "running", f"{label} 进展 · 正在改 {'、'.join(names[:3])}"
    if item_type == "error":
        message = _short_text(str(item.get("message") or ""), 80)
        if message:
            return "running", f"{label} 进展 · {message}"
    return None


def summarize_codex_event(event: dict[str, Any]) -> tuple[str, str] | None:
    return summarize_coding_event(event, "codex")


def coding_completion_failed(text: str) -> bool:
    compact = str(text or "").lower()
    failure_markers = [
        "无法成功运行",
        "无法在当前",
        "无法启动",
        "无法创建",
        "无图形界面",
        "无界面环境",
        "不能成功运行",
        "没有成功运行",
        "modulenotfounderror",
        "no module named",
        "nsscreen.mainscreen",
        "attributeerror",
        "dns",
        "failed to fetch",
        "request failed",
        "cannot start gui",
        "cannot run",
        "could not run",
    ]
    return any(marker in compact for marker in failure_markers)


def antigravity_failure_summary(text: str, label: str = "Antigravity", *, include_transient_auth: bool = True) -> str:
    compact = str(text or "").lower()
    if not compact:
        return ""
    checks = [
        (("error: timed out waiting for response", "timed out waiting for response"), "等待模型响应超时"),
        (("resource_exhausted", "individual quota reached", "quota reached", "quota"), "Antigravity 配额不足"),
        (("model unreachable", "unsupported model", "model not found"), "模型不可用"),
        (("print-timeout", "antigravity cli"), "Antigravity 偏离开发任务"),
        (("--print-timeout", "agy"), "Antigravity 偏离开发任务"),
    ]
    if include_transient_auth:
        checks.extend([
            (("you are not logged into antigravity", "not logged into antigravity"), "Antigravity 未登录"),
            (("failed to get oauth token", "error getting token source"), "Antigravity 登录态不可用"),
        ])
    for markers, reason in checks:
        if any(marker in compact for marker in markers):
            return f"{label} 失败 · {reason}"
    return ""


def workspace_entry_needs_host_launch(path: Path) -> bool:
    if path.name == "run.sh" or path.suffix == ".sh":
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except Exception:
        return False
    return any(marker in text for marker in GUI_MARKERS)


class _HtmlAssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name.lower() in {"src", "href", "poster"} and value:
                self.paths.append(value)


def _normalize_local_asset_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value or value.startswith(("#", "data:", "mailto:", "javascript:")):
        return ""
    parsed = urlsplit(value)
    if parsed.scheme and parsed.scheme not in {"file"}:
        return ""
    if parsed.netloc:
        return ""
    path = unquote(parsed.path or value).strip()
    if path.startswith("/"):
        return ""
    suffix = Path(path).suffix.lower()
    if suffix not in LOCAL_ASSET_EXTENSIONS:
        return ""
    return path


def validate_workspace_static_assets(workspace: Path) -> dict[str, Any]:
    """Verify local runtime assets referenced by generated HTML/CSS/JS exist."""
    missing: list[str] = []
    checked: list[str] = []
    source_files = [
        path
        for pattern in ("*.html", "css/*.css", "js/*.js", "*.js", "*.css")
        for path in workspace.glob(pattern)
        if path.is_file()
    ]
    seen_refs: set[tuple[str, str]] = set()
    quoted_asset_re = re.compile(r"""["'`]([^"'`]+?\.(?:css|gif|jpe?g|js|json|m4a|mp3|ogg|png|svg|wav|webp)(?:[?#][^"'`]*)?)["'`]""", re.IGNORECASE)
    css_url_re = re.compile(r"""url\((["']?)([^"')]+)\1\)""", re.IGNORECASE)

    for source in source_files:
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        refs: list[str] = []
        if source.suffix.lower() == ".html":
            parser = _HtmlAssetParser()
            try:
                parser.feed(text)
            except Exception:
                pass
            refs.extend(parser.paths)
        refs.extend(match.group(1) for match in quoted_asset_re.finditer(text))
        refs.extend(match.group(2) for match in css_url_re.finditer(text))
        for ref in refs:
            normalized = _normalize_local_asset_path(ref)
            if not normalized:
                continue
            key = (str(source.relative_to(workspace)), normalized)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            checked.append(f"{key[0]} -> {normalized}")
            candidates = [
                (source.parent / normalized).resolve(),
                (workspace / normalized).resolve(),
            ]
            valid_candidates = [
                target
                for target in candidates
                if str(target).startswith(str(workspace.resolve()))
            ]
            if not any(target.exists() and target.is_file() for target in valid_candidates):
                missing.append(f"{key[0]} -> {normalized}")
    return {"ok": not missing, "checked": checked, "missing": missing}


def resolve_host_venv() -> Path:
    """Resolve the Jen runtime venv shared by coding task launchers."""
    candidates: list[Path] = []
    env_venv = os.environ.get("VIRTUAL_ENV")
    if env_venv:
        candidates.append(Path(env_venv).expanduser())
    candidates.append(REPO_ROOT / ".venv")
    candidates.append(Path(sys.executable).resolve().parent.parent)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "bin" / "python").exists() or (resolved / "bin" / "python3").exists():
            return resolved
    return candidates[-1].resolve()


def coding_cache_root() -> Path:
    return Path(os.environ.get("JEN_CODING_CACHE_DIR") or os.environ.get("GJALLARHORN_CODING_CACHE_DIR", "~/.jen/cache")).expanduser().resolve()


def coding_runtime_env(base_env: dict[str, str] | None = None, *, venv: Path | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    cache_root = coding_cache_root()
    uv_cache = cache_root / "uv"
    npm_cache = cache_root / "npm"
    uv_cache.mkdir(parents=True, exist_ok=True)
    npm_cache.mkdir(parents=True, exist_ok=True)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["npm_config_cache"] = str(npm_cache)
    env["NPM_CONFIG_CACHE"] = str(npm_cache)
    env.setdefault("npm_config_prefer_offline", "true")
    env.setdefault("NPM_CONFIG_PREFER_OFFLINE", "true")
    if venv is not None:
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def host_venv_has_module(venv: Path, module: str) -> bool:
    python = venv / "bin" / "python"
    if not python.exists():
        python = venv / "bin" / "python3"
    if not python.exists():
        return False
    try:
        proc = subprocess.run(
            [str(python), "-c", f"import {module}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


class CodingAppRunner:
    def __init__(self, *, venv: Path | None = None, launch_wait_seconds: float = 3.0) -> None:
        self.venv = venv or resolve_host_venv()
        self.launch_wait_seconds = max(0.1, float(launch_wait_seconds or 3.0))

    def find_entry(self, task: dict[str, Any]) -> Path | None:
        workspace_raw = str(task.get("workspace") or task.get("cwd") or "").strip()
        if not workspace_raw:
            return None
        workspace = Path(workspace_raw)
        if not workspace.exists() or not workspace.is_dir():
            return None
        run_script = workspace / "run.sh"
        if run_script.exists() and run_script.is_file():
            return run_script
        candidates = [workspace / "app.py", workspace / "main.py"]
        candidates.extend(sorted(workspace.glob("*.py")))
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists() and candidate.is_file() and workspace_entry_needs_host_launch(candidate):
                return self._ensure_run_script(candidate) or candidate
        return None

    def _ensure_run_script(self, entry: Path) -> Path | None:
        workspace = entry.parent
        run_script = workspace / "run.sh"
        if run_script.exists() and run_script.is_file():
            return run_script
        if entry.suffix != ".py":
            return None
        content = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd \"$(dirname \"$0\")\"\n"
            f"export VIRTUAL_ENV=\"${{VIRTUAL_ENV:-{self.venv}}}\"\n"
            "export UV_CACHE_DIR=\"${UV_CACHE_DIR:-$HOME/.jen/cache/uv}\"\n"
            "export npm_config_cache=\"${npm_config_cache:-$HOME/.jen/cache/npm}\"\n"
            "export NPM_CONFIG_CACHE=\"${NPM_CONFIG_CACHE:-$npm_config_cache}\"\n"
            "export PATH=\"$VIRTUAL_ENV/bin:$PATH\"\n"
            f"exec uv run --active --no-sync python {entry.name}\n"
        )
        try:
            run_script.write_text(content, encoding="utf-8")
            run_script.chmod(0o755)
            return run_script
        except Exception:
            return None

    def launch(self, entry: Path) -> dict[str, Any]:
        workspace = entry.parent
        run_log = workspace / ".voice_app_run.log"
        self._terminate_workspace_launch(workspace)
        asset_check = validate_workspace_static_assets(workspace)
        if not asset_check.get("ok"):
            missing = asset_check.get("missing") or []
            error = "missing runtime asset: " + "; ".join(str(item) for item in missing[:5])
            try:
                with run_log.open("a", encoding="utf-8") as log:
                    log.write(f"\n--- voice launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    log.write(f"cwd={workspace}\n")
                    log.write(f"VIRTUAL_ENV={self.venv}\n")
                    log.write(f"{error}\n")
            except Exception:
                pass
            return {
                "ok": False,
                "entry": str(entry),
                "run_log": str(run_log),
                "active_venv": str(self.venv),
                "error": error,
                "asset_check": asset_check,
            }
        if entry.name == "run.sh" or entry.suffix == ".sh":
            try:
                entry.chmod(entry.stat().st_mode | 0o111)
            except Exception:
                pass
            argv = ["bash", entry.name]
        else:
            argv = ["uv", "run", "--active", "--no-sync", "python", entry.name]
        env = coding_runtime_env(os.environ.copy(), venv=self.venv)
        try:
            with run_log.open("a", encoding="utf-8") as log:
                log.write(f"\n--- voice launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                log.write(f"cwd={workspace}\n")
                log.write(f"VIRTUAL_ENV={self.venv}\n")
                log.write(f"cmd={' '.join(argv)}\n")
                log.flush()
                proc = subprocess.Popen(
                    argv,
                    cwd=str(workspace),
                    stdout=log,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
            self._write_pid_file(workspace, proc.pid)
            time.sleep(self.launch_wait_seconds)
            returncode = proc.poll()
            if returncode is None:
                window_check = self._verify_gui_window(workspace)
                if window_check.get("required") and not window_check.get("ok"):
                    try:
                        self._terminate_process_tree(proc.pid)
                    except Exception:
                        pass
                    _reap_process(proc)
                    return {
                        "ok": False,
                        "pid": proc.pid,
                        "entry": str(entry),
                        "run_log": str(run_log),
                        "active_venv": str(self.venv),
                        "window_check": window_check,
                        "error": str(window_check.get("error") or "process started but no matching window appeared"),
                    }
                _reap_process(proc)
                return {
                    "ok": True,
                    "pid": proc.pid,
                    "entry": str(entry),
                    "run_log": str(run_log),
                    "active_venv": str(self.venv),
                    "window_check": window_check,
                }
            tail = _tail_text(run_log, 2000)
            return {
                "ok": False,
                "entry": str(entry),
                "run_log": str(run_log),
                "active_venv": str(self.venv),
                "returncode": returncode,
                "error": tail or f"process exited with {returncode}",
            }
        except Exception as exc:
            try:
                with run_log.open("a", encoding="utf-8") as log:
                    log.write(f"\nlaunch exception: {exc}\n")
            except Exception:
                pass
            return {
                "ok": False,
                "entry": str(entry),
                "run_log": str(run_log),
                "active_venv": str(self.venv),
                "error": str(exc),
            }

    @staticmethod
    def _pid_file(workspace: Path) -> Path:
        return workspace / ".voice_app.pid"

    def _write_pid_file(self, workspace: Path, pid: int) -> None:
        try:
            self._pid_file(workspace).write_text(f"{pid}\n", encoding="utf-8")
        except Exception:
            pass

    def _terminate_workspace_launch(self, workspace: Path) -> None:
        pid_file = self._pid_file(workspace)
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pid = 0
        if pid > 0:
            self._terminate_process_tree(pid)
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _terminate_process_tree(pid: int) -> None:
        if pid <= 0:
            return
        try:
            os.killpg(os.getpgid(pid), 15)
            return
        except Exception:
            pass
        try:
            os.kill(pid, 15)
        except Exception:
            pass

    def _verify_gui_window(self, workspace: Path) -> dict[str, Any]:
        if not _workspace_looks_like_gui_app(workspace):
            return {"required": False, "ok": True}
        manifest = read_manifest(workspace) or {}
        capabilities = {str(item).strip().lower() for item in (manifest.get("capabilities") or [])}
        program = manifest.get("program") if isinstance(manifest.get("program"), dict) else {}
        window_match = program.get("window_match") if isinstance(program.get("window_match"), dict) else {}
        app_names = window_match.get("app_names")
        if not isinstance(app_names, list) or not app_names:
            app_name = str(window_match.get("app_name") or "").strip()
            app_names = [app_name] if app_name else ["python3", "Python"]
        title_keywords = [str(item).strip() for item in (window_match.get("title_keywords") or []) if str(item).strip()]
        title_keywords.extend([str(manifest.get("title") or ""), workspace.name])
        title_keywords = _unique_texts(title_keywords)
        deadline = time.monotonic() + max(1.0, self.launch_wait_seconds)
        last_error = ""
        while time.monotonic() < deadline:
            snapshot = _visible_window_snapshot(app_names=app_names, timeout_seconds=1.5)
            if snapshot.get("ok"):
                windows = snapshot.get("windows") if isinstance(snapshot.get("windows"), list) else []
                match = _match_window_snapshot(windows, app_names, title_keywords)
                if match:
                    return {"required": True, "ok": True, "matched": match, "windows": windows[:12]}
                last_error = "process started but no matching window appeared"
            else:
                last_error = str(snapshot.get("error") or "window enumeration failed")
            time.sleep(0.25)
        result = {
            "required": True,
            "ok": False,
            "app_names": app_names,
            "title_keywords": title_keywords[:10],
            "error": last_error or "window verification timed out",
        }
        if _window_verification_can_be_soft(result, workspace, capabilities):
            result["ok"] = True
            result["unverified"] = True
            result["warning"] = "native app process is alive but window enumeration timed out"
        return result


class CodingTaskMonitor:
    def __init__(
        self,
        store: Any,
        speech: Any,
        *,
        poll_seconds: float = 3.0,
        speech_interval_seconds: float = 60.0,
    ) -> None:
        self.store = store
        self.speech = speech
        self.poll_seconds = max(0.5, float(poll_seconds or 3.0))
        self.speech_interval_seconds = max(5.0, float(speech_interval_seconds or 60.0))
        self.runner = CodingAppRunner()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="coding-task-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self.store.add_event(
                    "coding_monitor_error",
                    role="system",
                    lane="coding",
                    content="coding monitor failed",
                    metadata={"error": str(exc)[:1000]},
                )
            self._stop.wait(self.poll_seconds)

    def poll_once(self) -> None:
        for task in self.store.coding_tasks(statuses=("running", "launching"), limit=50):
            self._poll_task(task)

    def _poll_task(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("task_id") or "")
        executor = str(task.get("executor") or "codex")
        label = coding_executor_label(executor)
        stdout_log = Path(str(task.get("stdout_log") or ""))
        executor_log = Path(str(task.get("executor_log") or ""))
        offset = int(task.get("last_offset") or 0)
        last_summary = str(task.get("last_summary") or "")
        status = str(task.get("status") or "running")
        completed = False
        failed = False
        summary = last_summary
        new_offset = offset

        if stdout_log.exists():
            with stdout_log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    new_offset = handle.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        if executor == "antigravity":
                            failure_summary = antigravity_failure_summary(line, label)
                            if failure_summary:
                                failed = True
                                summary = failure_summary
                                break
                            event_summary = f"{label} 进展 · {_short_text(line, 80)}"
                            if event_summary != summary:
                                summary = event_summary
                                self.store.record_coding_task_status(task_id, "running", event_summary, {
                                    "run_id": task.get("run_id"),
                                    "stdout_log": str(stdout_log),
                                })
                        continue
                    parsed = summarize_coding_event(event, executor)
                    if not parsed:
                        continue
                    event_status, event_summary = parsed
                    if event_summary and event_summary != summary and event_status != "artifact_ready":
                        summary = event_summary
                        self.store.record_coding_task_status(task_id, event_status, event_summary, {
                            "run_id": task.get("run_id"),
                            "stdout_log": str(stdout_log),
                        })
                    elif event_status == "artifact_ready":
                        summary = event_summary
                    if event_status == "artifact_ready":
                        completed = True
                        break
        elif not summary:
            summary = f"{label} 进展 · 等待日志生成"

        pid_alive = self._pid_is_alive(int(task.get("pid") or 0))
        if not completed and executor == "antigravity":
            log_failure = self._executor_log_failure(executor_log, label, include_transient_auth=not pid_alive)
            if log_failure:
                failed = True
                summary = log_failure
                self._terminate_pid(int(task.get("pid") or 0))
        if not failed and not completed and executor == "antigravity" and not pid_alive and summary and summary != last_summary:
            completed = True

        if not failed and not completed and not pid_alive:
            failed = True
            summary = summary if summary and summary != last_summary else f"{label} 失败 · 任务已退出，没拿到完成事件"

        now = time.time()
        updates: dict[str, Any] = {"last_offset": new_offset}
        if summary:
            updates["last_summary"] = summary
        if completed:
            final_summary = self._completion_summary(task, fallback=summary)
            if coding_completion_failed(final_summary):
                status = "failed"
                if final_summary.startswith(f"{label} 产物完成"):
                    failed_summary = final_summary.replace(f"{label} 产物完成", f"{label} 失败", 1)
                elif final_summary.startswith(f"{label} 完成"):
                    failed_summary = final_summary.replace(f"{label} 完成", f"{label} 失败", 1)
                else:
                    failed_summary = final_summary
                self.store.record_coding_task_status(task_id, "failed", failed_summary, {
                    "run_id": task.get("run_id"),
                    "last_message_path": task.get("last_message_path"),
                })
                self._speak(_spoken_coding_summary(failed_summary))
                updates.update({"status": "failed", "completed_at": now, "last_summary": failed_summary})
            else:
                self.store.record_coding_task_status(task_id, "running", f"{label} 产物完成", {
                    "run_id": task.get("run_id"),
                    "last_message_path": task.get("last_message_path"),
                })
                updates["status"] = "launching"
                self.store.record_coding_task_status(task_id, "registering_app", f"{label} 进展 · 正在注册应用打开方式", {
                    "run_id": task.get("run_id"),
                    "workspace_id": task.get("workspace_id"),
                })
                launch_result = self._maybe_launch_workspace_app(task)
                if launch_result and not launch_result.get("ok"):
                    status = "failed"
                    failed_summary = f"{label} 失败 · 写完了但没启动：{_short_text(str(launch_result.get('error') or '启动失败'), 70)}"
                    self.store.record_coding_task_status(task_id, "failed", failed_summary, {
                        "run_id": task.get("run_id"),
                        "entry": launch_result.get("entry"),
                        "run_log": launch_result.get("run_log"),
                        "returncode": launch_result.get("returncode"),
                    })
                    self._speak(_spoken_coding_summary(failed_summary))
                    updates.update({"status": "failed", "completed_at": now, "last_summary": failed_summary})
                else:
                    status = "completed"
                    completed_summary = final_summary.replace(f"{label} 产物完成", f"{label} 完成", 1)
                    if completed_summary == final_summary and final_summary.startswith(f"{label} 进展"):
                        completed_summary = final_summary.replace(f"{label} 进展", f"{label} 完成", 1)
                    workspace_manifest = self._update_workspace_after_completion(task, final_summary, launch_result)
                    metadata = {
                        "run_id": task.get("run_id"),
                        "last_message_path": task.get("last_message_path"),
                        "workspace_id": task.get("workspace_id"),
                    }
                    if launch_result:
                        entry_name = Path(str(launch_result.get("entry") or "")).name
                        completed_summary = f"{label} 完成 · 已写完并启动 {entry_name or '应用'}"
                        metadata.update({
                            "entry": launch_result.get("entry"),
                            "pid": launch_result.get("pid"),
                            "run_log": launch_result.get("run_log"),
                        })
                    if workspace_manifest:
                        metadata["workspace_id"] = workspace_manifest.get("workspace_id") or metadata.get("workspace_id")
                        metadata["workspace"] = workspace_manifest.get("path")
                    self.store.record_coding_task_status(task_id, "completed", completed_summary, metadata)
                    self._speak(_spoken_coding_summary(completed_summary))
                    updates.update({"status": "completed", "completed_at": now, "last_summary": completed_summary})
        elif failed:
            status = "failed"
            self.store.record_coding_task_status(task_id, "failed", summary, {
                "run_id": task.get("run_id"),
                "stderr_log": task.get("stderr_log"),
                "executor_log": task.get("executor_log"),
            })
            self._speak(_spoken_coding_summary(summary))
            updates.update({"status": "failed", "completed_at": now})
        elif now >= float(task.get("next_speech_at") or 0):
            speech_summary = summary or f"{label} 进展 · 还在处理"
            self._speak(_spoken_coding_summary(speech_summary))
            updates["next_speech_at"] = now + self.speech_interval_seconds
        if status == "running" and "next_speech_at" not in updates and not float(task.get("next_speech_at") or 0):
            updates["next_speech_at"] = now + self.speech_interval_seconds
        self.store.update_coding_task(task_id, **updates)

    def _completion_summary(self, task: dict[str, Any], fallback: str) -> str:
        label = coding_executor_label(str(task.get("executor") or "codex"))
        path = Path(str(task.get("last_message_path") or ""))
        if path.exists() and path.is_file():
            text = _short_text(path.read_text(encoding="utf-8", errors="replace"), 90)
            if text:
                return f"{label} 产物完成 · {text}"
        return fallback or f"{label} 产物完成"

    def _maybe_launch_workspace_app(self, task: dict[str, Any]) -> dict[str, Any] | None:
        entry = self.runner.find_entry(task)
        if entry is None:
            return None
        task_id = str(task.get("task_id") or "")
        label = coding_executor_label(str(task.get("executor") or "codex"))
        self.store.record_coding_task_status(task_id, "launching_app", f"{label} 进展 · 正在启动 {entry.name}", {
            "run_id": task.get("run_id"),
            "entry": str(entry),
        })
        result = self.runner.launch(entry)
        if result.get("ok"):
            self.store.record_coding_task_status(task_id, "verifying_app", f"{label} 进展 · 已验证 {entry.name} 可启动", {
                "run_id": task.get("run_id"),
                "entry": str(entry),
                "pid": result.get("pid"),
                "run_log": result.get("run_log"),
            })
        return result

    def _update_workspace_after_completion(
        self,
        task: dict[str, Any],
        final_summary: str,
        launch_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        workspace_raw = str(task.get("workspace") or task.get("cwd") or "").strip()
        if not workspace_raw:
            return None
        workspace = Path(workspace_raw)
        if not workspace.exists() or not workspace.is_dir():
            return None
        try:
            index = CodingWorkspaceIndex(self.store, workspace.parent)
            services = None
            if launch_result and launch_result.get("ok"):
                services = CodingServiceRegistry(index).probe_services(workspace)
            manifest = index.update_from_files(
                workspace,
                title=str(task.get("target") or workspace.name),
                summary=_strip_executor_summary_prefix(final_summary),
                services=services,
            )
            entry = launch_result.get("entry") if launch_result else None
            if entry:
                manifest = register_workspace_program(
                    manifest,
                    entry=entry,
                    active_venv=launch_result.get("active_venv") or task.get("active_venv"),
                    launch_result=launch_result,
                    status="ready",
                )
                index.upsert(manifest)
            elif read_manifest(workspace):
                index.upsert(manifest)
            return manifest
        except Exception as exc:
            self.store.add_event(
                "coding_workspace_update_error",
                role="system",
                lane="coding",
                content="workspace update failed",
                metadata={"task_id": task.get("task_id"), "workspace": str(workspace), "error": str(exc)[:1000]},
            )
            return None

    def _speak(self, text: str) -> None:
        try:
            self.speech.speak(_spoken_coding_summary(text), interrupt=False, quick_say_fallback=True)
        except Exception:
            pass

    def _executor_log_failure(self, path: Path, label: str, *, include_transient_auth: bool) -> str:
        if not path.exists() or not path.is_file():
            return ""
        return antigravity_failure_summary(_tail_text(path, 12000), label, include_transient_auth=include_transient_auth)

    @staticmethod
    def _terminate_pid(pid: int) -> None:
        if pid <= 0:
            return
        try:
            os.killpg(os.getpgid(pid), 15)
            return
        except Exception:
            pass
        try:
            os.kill(pid, 15)
        except Exception:
            pass

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return True


def _short_text(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _short_command(command: str) -> str:
    value = " ".join(str(command or "").split())
    if not value:
        return ""
    if value.startswith("/bin/zsh -lc "):
        value = value[len("/bin/zsh -lc "):].strip("'\"")
    return _short_text(value, 48)


def _strip_executor_summary_prefix(text: str) -> str:
    value = str(text or "").strip()
    for label in ("Codex", "Antigravity"):
        for prefix in (f"{label} 产物完成 ·", f"{label} 完成 ·", f"{label} 失败 ·"):
            if value.startswith(prefix):
                return value[len(prefix):].strip()
    return value


def _spoken_coding_summary(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    label = "开发器"
    if value.startswith("Codex"):
        label = "Codex"
    elif value.startswith("Antigravity"):
        label = "Antigravity"
    lower = value.lower()
    if "写完了但没启动" in value or "失败" in value:
        if "no module named" in lower or "缺少" in value:
            return f"{label} 失败了，写完了但没启动，缺依赖。"
        if "timeout" in lower or "超时" in value:
            return f"{label} 失败了，写完了但没启动，启动超时。"
        return f"{label} 失败了，写完了但没启动。"
    if "已写完并启动" in value or "完成" in value:
        return f"{label} 完成了，应用已经启动。"
    if "产物完成" in value:
        return f"{label} 写完了，正在启动验证。"
    if "还在处理" in value:
        return f"{label} 还在处理。"
    if "正在启动" in value:
        return f"{label} 正在启动应用。"
    if "正在注册" in value:
        return f"{label} 正在记录打开方式。"
    cleaned = re.sub(r"(/Users|/private|/tmp|/var|data/voice)[^\\s,，。；;:：)\\]}]+", "文件路径", value)
    cleaned = re.sub(r"\\b[A-Za-z_][A-Za-z0-9_]*\\.(py|json|log|txt|sqlite|db|npz|wav|mp3|sh)\\b", "文件", cleaned)
    return _short_text(cleaned, 48)


def _tail_text(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def _workspace_looks_like_gui_app(workspace: Path) -> bool:
    for candidate in [workspace / "app.py", workspace / "main.py", *sorted(workspace.glob("*.py"))[:8]]:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
        if any(marker in text for marker in GUI_MARKERS):
            return True
    return False


def _window_verification_can_be_soft(result: dict[str, Any], workspace: Path, capabilities: set[str]) -> bool:
    """Treat macOS window-enumeration timeouts as soft for live GUI processes.

    System Events can hang or time out while pywebview/Cocoa is starting, even
    after the process has created a window. A live GUI process plus a timeout is
    a weaker signal than a traceback or process exit, so the monitor should not
    kill the app just because accessibility enumeration was slow.
    """
    error = str(result.get("error") or "").lower()
    if "timed out" not in error:
        return False
    return bool(capabilities & {"native_app", "pywebview_app"} or _workspace_looks_like_gui_app(workspace))


def _visible_window_snapshot(*, app_names: list[Any] | None = None, timeout_seconds: float = 1.5) -> dict[str, Any]:
    requested_apps = [str(item).strip() for item in (app_names or []) if str(item).strip()]
    if requested_apps:
        windows: list[dict[str, str]] = []
        errors: list[str] = []
        for app_name in requested_apps:
            snapshot = _visible_windows_for_app(app_name, timeout_seconds=timeout_seconds)
            if snapshot.get("ok"):
                windows.extend(snapshot.get("windows") or [])
            else:
                errors.append(str(snapshot.get("error") or "window enumeration failed"))
        if windows:
            return {"ok": True, "windows": windows}
        return {"ok": False, "error": "; ".join(errors)[:500] or "no windows found"}
    script = r'''
    tell application "System Events"
      set out to {}
      repeat with p in application processes whose visible is true
        set appName to name of p
        try
          repeat with w in windows of p
            try
              set windowName to name of w
            on error
              set windowName to ""
            end try
            set end of out to appName & "\t" & windowName
          end repeat
        end try
      end repeat
      return out
    end tell
    '''
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=max(0.5, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "window enumeration timed out"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "").strip()[:500] or f"osascript exited {proc.returncode}"}
    windows: list[dict[str, str]] = []
    output = (proc.stdout or "").strip()
    for item in re.split(r",\s*", output):
        item = item.strip()
        if not item or "\t" not in item:
            continue
        app_name, title = item.split("\t", 1)
        windows.append({"app_name": app_name.strip(), "title": title.strip()})
    return {"ok": True, "windows": windows}


def _visible_windows_for_app(app_name: str, *, timeout_seconds: float = 1.5) -> dict[str, Any]:
    quoted = app_name.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "System Events" to get name of every window of application process "{quoted}"'
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=max(0.5, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"window enumeration timed out for {app_name}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "").strip()[:500] or f"osascript exited {proc.returncode}"}
    windows = [{"app_name": app_name, "title": item.strip()} for item in re.split(r",\s*", (proc.stdout or "").strip()) if item.strip()]
    return {"ok": True, "windows": windows}


def _match_window_snapshot(windows: list[dict[str, Any]], app_names: list[Any], title_keywords: list[str]) -> dict[str, Any] | None:
    app_keys = {_compact_text(str(item)) for item in app_names if str(item).strip()}
    title_keys = [_compact_text(item) for item in title_keywords if _compact_text(item)]
    for window in windows:
        app = str(window.get("app_name") or "")
        title = str(window.get("title") or "")
        app_key = _compact_text(app)
        title_key = _compact_text(title)
        app_matches = not app_keys or app_key in app_keys
        title_matches = not title_keys or any(key and (key in title_key or title_key in key) for key in title_keys)
        if app_matches and title_matches:
            return {"app_name": app, "title": title}
    return None


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        key = _compact_text(text)
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").lower())


def _reap_process(proc: subprocess.Popen[Any]) -> None:
    def wait_for_exit() -> None:
        try:
            proc.wait()
        except Exception:
            pass

    threading.Thread(target=wait_for_exit, daemon=True).start()
