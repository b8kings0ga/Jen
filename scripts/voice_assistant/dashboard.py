from __future__ import annotations

import http.server
import json
import threading
from socketserver import ThreadingMixIn
from typing import Any
from urllib import parse

from voice_assistant.domain_probe import probe_domains
from voice_assistant.front_note import parse_front_note_bool
from voice_assistant.ui import render_domain_probe_debug_html, render_front_note_editor_document, render_tool_dashboard_html


def _domain_probe_debug_payload(message: str, store: VoiceSessionStore | None = None) -> dict[str, Any]:
    context = store.domain_probe_context(recent_limit=6) if store is not None else None
    payload = probe_domains(message, context=context)
    suggestions: list[dict[str, Any]] = []
    for domain in payload.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        for action in domain.get("suggested_actions") or []:
            if not isinstance(action, dict):
                continue
            tool_call = str(action.get("tool_call") or "")
            item = {
                "domain": domain.get("domain"),
                "intent": domain.get("intent"),
                "confidence": action.get("confidence", domain.get("confidence")),
                "tool_call": tool_call,
                "desc": action.get("desc") or "",
            }
            if "answer" in action:
                item["answer"] = action.get("answer")
            suggestions.append(item)
    return {"ok": True, "message": message, "probe": payload, "tool_suggestions": suggestions}

class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True






def run_tool_dashboard_server(store: VoiceSessionStore, speech: SpeechQueue | None, host: str, port: int, bot: Any | None = None) -> ThreadingHTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "JenToolDashboard/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            if self.path.startswith("/api/front-note"):
                return
            print(f"dashboard: {self.address_string()} {format % args}", flush=True)

        def _send(self, status: int, body: str | bytes, content_type: str = "text/html; charset=utf-8") -> None:
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False, indent=2), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed = parse.urlparse(self.path)
            if parsed.path == "/api/tools":
                self._json(200, {"tools": store.tool_speech_catalog()})
                return
            if parsed.path == "/api/fillers":
                self._json(200, {"fillers": store.filler_speech_catalog()})
                return
            if parsed.path == "/api/pipeline":
                query = parse.parse_qs(parsed.query or "")
                limit = int((query.get("limit") or ["180"])[-1] or 180)
                self._json(200, store.recent_pipeline(limit=limit))
                return
            if parsed.path == "/api/domain-probe":
                query = parse.parse_qs(parsed.query or "")
                message = str((query.get("message") or query.get("q") or [""])[-1] or "")
                self._json(200, _domain_probe_debug_payload(message, store))
                return
            if parsed.path == "/api/lane-log":
                query = parse.parse_qs(parsed.query or "")
                limit = int((query.get("limit") or ["300"])[-1] or 300)
                self._json(200, store.lane_call_log(limit=limit))
                return
            if parsed.path == "/api/live-sessions":
                query = parse.parse_qs(parsed.query or "")
                limit = int((query.get("limit") or ["50"])[-1] or 50)
                self._json(200, {"current_session_id": store.session_id, "sessions": store.list_sessions(limit=limit)})
                return
            if parsed.path == "/api/live-session":
                query = parse.parse_qs(parsed.query or "")
                limit = int((query.get("limit") or ["300"])[-1] or 300)
                session_id = str((query.get("session_id") or [store.session_id])[-1] or store.session_id)
                self._json(200, store.pipeline_for_session(session_id, limit=limit, live_view=True))
                return
            if parsed.path == "/api/front-note":
                state = store.front_note_state()
                self._json(200, state)
                return
            if parsed.path == "/front-note":
                self._send(200, render_front_note_editor_document(f"http://{host}:{port}/api/front-note"))
                return
            if parsed.path == "/probe":
                self._send(200, render_domain_probe_debug_html())
                return
            if parsed.path in {"", "/"}:
                self._send(200, render_tool_dashboard_html(store.tool_speech_catalog(), store.filler_speech_catalog()))
                return
            self._send(404, "not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:
            parsed = parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            if parsed.path == "/api/front-note":
                try:
                    if "application/json" in content_type:
                        payload = json.loads(raw.decode("utf-8"))
                    else:
                        payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                    state = store.update_front_note(
                        action=str(payload.get("action") or "show"),
                        tab=str(payload.get("tab") or ""),
                        content=str(payload.get("content") or ""),
                        html=str(payload.get("html") or ""),
                        media=payload.get("media"),
                        active_tab=str(payload.get("active_tab") or ""),
                        source=str(payload.get("source") or "api"),
                        allow_empty=parse_front_note_bool(payload.get("allow_empty"), False),
                        position=str(payload.get("position") or "right"),
                        visible=parse_front_note_bool(payload.get("visible"), True),
                        width=int(payload.get("width") or 520),
                        height=int(payload.get("height") or 420),
                    )
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, {"ok": True, "state": state})
                return
            if parsed.path == "/api/text-input":
                try:
                    if bot is None:
                        raise RuntimeError("voice bot unavailable")
                    if "application/json" in content_type:
                        payload = json.loads(raw.decode("utf-8"))
                    else:
                        payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                    text = str(payload.get("text") or "").strip()
                    mode = "simple" if str(payload.get("mode") or "") == "simple" else "quality"
                    if not text:
                        raise ValueError("text is required")
                    accepted = bool(bot.submit_text_input(text, mode))
                    if not accepted:
                        raise RuntimeError("text input was not accepted")
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, {"ok": True, "mode": mode})
                return
            if parsed.path == "/api/live-note":
                try:
                    if "application/json" in content_type:
                        payload = json.loads(raw.decode("utf-8"))
                    else:
                        payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                    content = str(payload.get("content") or "").strip()
                    session_id = str(payload.get("session_id") or store.session_id)
                    note_id = store.add_live_note(content, session_id=session_id)
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, {"ok": True, "id": note_id, "session_id": session_id})
                return
            if parsed.path == "/api/domain-probe":
                try:
                    if "application/json" in content_type:
                        payload = json.loads(raw.decode("utf-8"))
                    else:
                        payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                    message = str(payload.get("message") or payload.get("q") or "")
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, _domain_probe_debug_payload(message, store))
                return
            if parsed.path == "/api/fillers":
                try:
                    if "application/json" in content_type:
                        payload = json.loads(raw.decode("utf-8"))
                    else:
                        payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                    filler_id = store.add_filler_speech(
                        phrase=str(payload.get("phrase") or ""),
                        tone=str(payload.get("tone") or "soft"),
                        stage=str(payload.get("stage") or ""),
                        instructions=str(payload.get("instructions") or ""),
                    )
                    if speech is not None:
                        speech.reload_fillers()
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, {"ok": True, "filler_id": filler_id})
                return
            if parsed.path.startswith("/api/fillers/") and parsed.path.endswith("/warm"):
                try:
                    filler_id = int(parsed.path.removeprefix("/api/fillers/").removesuffix("/warm"))
                    if speech is None:
                        raise RuntimeError("speech queue unavailable")
                    result = speech.warm_filler_audio(filler_id)
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, result)
                return
            if parsed.path.startswith("/api/fillers/"):
                try:
                    filler_id = int(parsed.path.removeprefix("/api/fillers/"))
                    if "application/json" in content_type:
                        payload = json.loads(raw.decode("utf-8"))
                    else:
                        payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                    store.update_filler_speech(
                        filler_id,
                        phrase=str(payload.get("phrase") or ""),
                        tone=str(payload.get("tone") or "soft"),
                        stage=str(payload.get("stage") or ""),
                        instructions=str(payload.get("instructions") or ""),
                    )
                    if speech is not None:
                        speech.reload_fillers()
                except Exception as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                    return
                self._json(200, {"ok": True, "filler_id": filler_id})
                return
            if not parsed.path.startswith("/api/tools/"):
                self._send(404, "not found", "text/plain; charset=utf-8")
                return
            tool_name = parse.unquote(parsed.path.removeprefix("/api/tools/")).strip()
            try:
                if "application/json" in content_type:
                    payload = json.loads(raw.decode("utf-8"))
                else:
                    payload = {k: v[-1] for k, v in parse.parse_qs(raw.decode("utf-8")).items()}
                store.update_tool_speech(
                    tool_name,
                    str(payload.get("task_label") or store.tool_task_label(tool_name) or ""),
                    str(payload.get("start_phrase") or ""),
                    str(payload.get("success_phrase") or ""),
                    str(payload.get("failure_phrase") or ""),
                    bool(payload["speech_enabled"]) if "speech_enabled" in payload else store.tool_speech_enabled(tool_name),
                )
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
                return
            if "application/json" in content_type:
                self._json(200, {"ok": True, "tool": tool_name})
            else:
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_DELETE(self) -> None:
            parsed = parse.urlparse(self.path)
            if not parsed.path.startswith("/api/fillers/"):
                self._send(404, "not found", "text/plain; charset=utf-8")
                return
            try:
                filler_id = int(parsed.path.removeprefix("/api/fillers/"))
                result = store.delete_filler_speech(filler_id)
                if speech is not None:
                    speech.reload_fillers()
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
                return
            self._json(200, {"ok": True, **result})

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Tool speech dashboard: http://{host}:{port}/", flush=True)
    return server
