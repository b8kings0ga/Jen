from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from voice_assistant.filler import filler_stage_for_phrase
from voice_assistant.front_note import (
    front_note_content_to_html,
    front_note_html_to_text,
    sanitize_front_note_html,
    sanitize_front_note_media,
)
from voice_assistant.json_utils import coerce_json_object
from voice_assistant.tool_speech import (
    DEFAULT_SILENT_TOOLS,
    DEFAULT_TOOL_LOG_WORDS,
    DEFAULT_TOOL_TASK_LABELS,
    TOOL_LOG_WORDS,
    short_tool_error_reason,
    tool_log_words,
)


INTERNAL_COUNT_EXCLUDED_TOOLS = {"trigger_fast_followup"}


class VoiceSessionStore:
    def __init__(self, db_file: Path, session_id: str) -> None:
        self.db_file = db_file
        self.session_id = session_id
        self._lock = threading.RLock()
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_file, timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                  session_id TEXT PRIMARY KEY,
                  summary TEXT NOT NULL DEFAULT '',
                  active_tasks_json TEXT NOT NULL DEFAULT '[]',
                  user_preferences_json TEXT NOT NULL DEFAULT '[]',
                  open_threads_json TEXT NOT NULL DEFAULT '[]',
                  turn_count INTEGER NOT NULL DEFAULT 0,
                  last_compressed_turn INTEGER NOT NULL DEFAULT 0,
                  last_compressed_at REAL NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT '',
                  lane TEXT NOT NULL DEFAULT '',
                  content TEXT NOT NULL DEFAULT '',
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_states(
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  title TEXT NOT NULL,
                  status TEXT NOT NULL,
                  summary TEXT NOT NULL DEFAULT '',
                  updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_fast_prompts(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  prompt TEXT NOT NULL,
                  priority INTEGER NOT NULL DEFAULT 5,
                  status TEXT NOT NULL DEFAULT 'pending',
                  created_at REAL NOT NULL,
                  handled_at REAL
                );
                CREATE TABLE IF NOT EXISTS tool_events(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  tool_name TEXT NOT NULL,
                  arguments_json TEXT NOT NULL DEFAULT '{}',
                  result_json TEXT NOT NULL DEFAULT '{}',
                  ok INTEGER NOT NULL DEFAULT 1,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_kv(
                  session_id TEXT NOT NULL,
                  key TEXT NOT NULL,
                  value TEXT NOT NULL DEFAULT '',
                  updated_at REAL NOT NULL,
                  PRIMARY KEY(session_id, key)
                );
                CREATE TABLE IF NOT EXISTS tool_speech_catalog(
                  tool_name TEXT PRIMARY KEY,
                  task_label TEXT NOT NULL DEFAULT '',
                  start_phrase TEXT NOT NULL DEFAULT '',
                  success_phrase TEXT NOT NULL DEFAULT '',
                  failure_phrase TEXT NOT NULL DEFAULT '',
                  speech_enabled INTEGER NOT NULL DEFAULT 1,
                  language TEXT NOT NULL DEFAULT 'zh',
                  updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_speech_cache(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tool_name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  task_label TEXT NOT NULL DEFAULT '',
                  phrase TEXT NOT NULL,
                  spoken_text TEXT NOT NULL,
                  tts_model TEXT NOT NULL DEFAULT '',
                  tts_voice TEXT NOT NULL DEFAULT '',
                  audio_path TEXT NOT NULL DEFAULT '',
                  ok INTEGER NOT NULL DEFAULT 0,
                  error TEXT NOT NULL DEFAULT '',
                  generated_at REAL NOT NULL,
                  UNIQUE(tool_name, status, spoken_text, tts_model, tts_voice)
                );
                CREATE TABLE IF NOT EXISTS filler_speech_catalog(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  slot_index INTEGER NOT NULL,
                  tone TEXT NOT NULL DEFAULT 'soft',
                  stage TEXT NOT NULL DEFAULT 'opening',
                  phrase TEXT NOT NULL DEFAULT '',
                  instructions TEXT NOT NULL DEFAULT '',
                  audio_path TEXT NOT NULL DEFAULT '',
                  ok INTEGER NOT NULL DEFAULT 0,
                  seconds REAL,
                  bytes INTEGER NOT NULL DEFAULT 0,
                  updated_at REAL NOT NULL,
                  UNIQUE(slot_index, tone)
                );
                CREATE TABLE IF NOT EXISTS front_note_state(
                  session_id TEXT PRIMARY KEY,
                  content TEXT NOT NULL DEFAULT '',
                  media_json TEXT NOT NULL DEFAULT '[]',
                  active_tab TEXT NOT NULL DEFAULT 'live',
                  live_html TEXT NOT NULL DEFAULT '',
                  live_text TEXT NOT NULL DEFAULT '',
                  live_media_json TEXT NOT NULL DEFAULT '[]',
                  live_version INTEGER NOT NULL DEFAULT 0,
                  live_updated_at REAL NOT NULL DEFAULT 0,
                  context_html TEXT NOT NULL DEFAULT '',
                  context_text TEXT NOT NULL DEFAULT '',
                  context_media_json TEXT NOT NULL DEFAULT '[]',
                  context_version INTEGER NOT NULL DEFAULT 0,
                  context_updated_at REAL NOT NULL DEFAULT 0,
                  visible INTEGER NOT NULL DEFAULT 0,
                  position TEXT NOT NULL DEFAULT 'right',
                  width INTEGER NOT NULL DEFAULT 520,
                  height INTEGER NOT NULL DEFAULT 420,
                  version INTEGER NOT NULL DEFAULT 0,
                  updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_timings(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  turn_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  label TEXT NOT NULL DEFAULT '',
                  status TEXT NOT NULL DEFAULT 'running',
                  started_at REAL NOT NULL,
                  ended_at REAL,
                  duration_seconds REAL,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_turn_timings_session_turn ON turn_timings(session_id, turn_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_turn_timings_session_stage ON turn_timings(session_id, stage, started_at);
                CREATE TABLE IF NOT EXISTS coding_tasks(
                  task_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  run_id TEXT NOT NULL DEFAULT '',
                  pid INTEGER NOT NULL DEFAULT 0,
                  status TEXT NOT NULL DEFAULT 'running',
                  workspace_id TEXT NOT NULL DEFAULT '',
                  target TEXT NOT NULL DEFAULT '',
                  cwd TEXT NOT NULL DEFAULT '',
                  workspace TEXT NOT NULL DEFAULT '',
                  executor TEXT NOT NULL DEFAULT 'codex',
                  model TEXT NOT NULL DEFAULT '',
                  stdout_log TEXT NOT NULL DEFAULT '',
                  stderr_log TEXT NOT NULL DEFAULT '',
                  executor_log TEXT NOT NULL DEFAULT '',
                  last_message_path TEXT NOT NULL DEFAULT '',
                  prompt_path TEXT NOT NULL DEFAULT '',
                  active_venv TEXT NOT NULL DEFAULT '',
                  last_offset INTEGER NOT NULL DEFAULT 0,
                  last_summary TEXT NOT NULL DEFAULT '',
                  next_speech_at REAL NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  completed_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_coding_tasks_session_status ON coding_tasks(session_id, status, updated_at);
                CREATE TABLE IF NOT EXISTS coding_workspaces(
                  workspace_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  path TEXT NOT NULL UNIQUE,
                  title TEXT NOT NULL DEFAULT '',
                  aliases_json TEXT NOT NULL DEFAULT '[]',
                  tags_json TEXT NOT NULL DEFAULT '[]',
                  summary TEXT NOT NULL DEFAULT '',
                  capabilities_json TEXT NOT NULL DEFAULT '[]',
                  entrypoints_json TEXT NOT NULL DEFAULT '[]',
                  services_json TEXT NOT NULL DEFAULT '[]',
                  program_json TEXT NOT NULL DEFAULT '{}',
                  related_workspace_ids_json TEXT NOT NULL DEFAULT '[]',
                  status TEXT NOT NULL DEFAULT 'active',
                  last_task_at REAL NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_coding_workspaces_session_updated ON coding_workspaces(session_id, updated_at);
                """
            )
            now = time.time()
            columns = {row["name"] for row in con.execute("PRAGMA table_info(tool_speech_catalog)").fetchall()}
            added_speech_enabled = "speech_enabled" not in columns
            if added_speech_enabled:
                con.execute("ALTER TABLE tool_speech_catalog ADD COLUMN speech_enabled INTEGER NOT NULL DEFAULT 1")
            tool_event_columns = {row["name"] for row in con.execute("PRAGMA table_info(tool_events)").fetchall()}
            if "turn_id" not in tool_event_columns:
                con.execute("ALTER TABLE tool_events ADD COLUMN turn_id TEXT NOT NULL DEFAULT ''")
            coding_columns = {row["name"] for row in con.execute("PRAGMA table_info(coding_tasks)").fetchall()}
            coding_column_defs = {
                "target": "TEXT NOT NULL DEFAULT ''",
                "workspace": "TEXT NOT NULL DEFAULT ''",
                "last_offset": "INTEGER NOT NULL DEFAULT 0",
                "last_summary": "TEXT NOT NULL DEFAULT ''",
                "next_speech_at": "REAL NOT NULL DEFAULT 0",
                "completed_at": "REAL",
                "active_venv": "TEXT NOT NULL DEFAULT ''",
                "workspace_id": "TEXT NOT NULL DEFAULT ''",
                "executor": "TEXT NOT NULL DEFAULT 'codex'",
                "executor_log": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in coding_column_defs.items():
                if column not in coding_columns:
                    con.execute(f"ALTER TABLE coding_tasks ADD COLUMN {column} {definition}")
            workspace_columns = {row["name"] for row in con.execute("PRAGMA table_info(coding_workspaces)").fetchall()}
            workspace_column_defs = {
                "program_json": "TEXT NOT NULL DEFAULT '{}'",
                "related_workspace_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "last_task_at": "REAL NOT NULL DEFAULT 0",
            }
            for column, definition in workspace_column_defs.items():
                if column not in workspace_columns:
                    con.execute(f"ALTER TABLE coding_workspaces ADD COLUMN {column} {definition}")
            front_note_columns = {row["name"] for row in con.execute("PRAGMA table_info(front_note_state)").fetchall()}
            front_note_column_defs = {
                "active_tab": "TEXT NOT NULL DEFAULT 'live'",
                "live_html": "TEXT NOT NULL DEFAULT ''",
                "live_text": "TEXT NOT NULL DEFAULT ''",
                "live_media_json": "TEXT NOT NULL DEFAULT '[]'",
                "live_version": "INTEGER NOT NULL DEFAULT 0",
                "live_updated_at": "REAL NOT NULL DEFAULT 0",
                "context_html": "TEXT NOT NULL DEFAULT ''",
                "context_text": "TEXT NOT NULL DEFAULT ''",
                "context_media_json": "TEXT NOT NULL DEFAULT '[]'",
                "context_version": "INTEGER NOT NULL DEFAULT 0",
                "context_updated_at": "REAL NOT NULL DEFAULT 0",
            }
            for column, definition in front_note_column_defs.items():
                if column not in front_note_columns:
                    con.execute(f"ALTER TABLE front_note_state ADD COLUMN {column} {definition}")
            con.execute(
                "UPDATE front_note_state SET "
                "live_text=CASE WHEN live_text='' THEN content ELSE live_text END, "
                "live_html=CASE WHEN live_html='' AND content!='' THEN content ELSE live_html END, "
                "live_media_json=CASE WHEN live_media_json='[]' THEN media_json ELSE live_media_json END, "
                "live_updated_at=CASE WHEN live_updated_at=0 THEN updated_at ELSE live_updated_at END "
                "WHERE content!='' OR media_json!='[]'"
            )
            con.execute("UPDATE front_note_state SET width=520, height=420 WHERE width=360 AND height=240")
            con.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at) VALUES(?, ?, ?)",
                (self.session_id, now, now),
            )
            for tool_name, words in TOOL_LOG_WORDS.items():
                con.execute(
                    "INSERT INTO tool_speech_catalog(tool_name, task_label, start_phrase, success_phrase, failure_phrase, speech_enabled, language, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, 'zh', ?) "
                    "ON CONFLICT(tool_name) DO NOTHING",
                    (
                        tool_name,
                        DEFAULT_TOOL_TASK_LABELS.get(tool_name, ""),
                        words.get("start", ""),
                        words.get("success", ""),
                        words.get("failure", ""),
                        0 if tool_name in DEFAULT_SILENT_TOOLS else 1,
                        now,
                    ),
                )
            con.execute(
                "UPDATE tool_speech_catalog SET success_phrase=?, updated_at=? "
                "WHERE tool_name='trigger_fast_followup' AND success_phrase=?",
                ("排进语音了", now, "已经说出去了"),
            )
            con.execute(
                "UPDATE tool_speech_catalog SET start_phrase=?, success_phrase=?, failure_phrase=?, updated_at=? "
                "WHERE tool_name='coding_action' AND start_phrase=? AND success_phrase=? AND failure_phrase=?",
                ("我叫开发器写", "开发开工了", "开发没启动", now, "我叫 Codex 写", "Codex 开工了", "Codex 没启动"),
            )
            if added_speech_enabled:
                con.executemany(
                    "UPDATE tool_speech_catalog SET speech_enabled=0 WHERE tool_name=?",
                    [(tool_name,) for tool_name in DEFAULT_SILENT_TOOLS],
                )

    def add_event(self, kind: str, role: str = "", lane: str = "", content: str = "", metadata: dict[str, Any] | None = None) -> int:
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO events(session_id, kind, role, lane, content, metadata_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, kind, role, lane, content, json.dumps(metadata or {}, ensure_ascii=False), time.time()),
            )
            if kind == "transcript":
                con.execute("UPDATE sessions SET turn_count=turn_count+1, updated_at=? WHERE session_id=?", (time.time(), self.session_id))
            else:
                con.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (time.time(), self.session_id))
            return int(cur.lastrowid)

    def start_turn_timing(self, turn_id: str, stage: str, label: str = "", metadata: dict[str, Any] | None = None) -> int:
        turn_id = str(turn_id or "").strip()
        stage = str(stage or "").strip()
        if not turn_id or not stage:
            return 0
        now = time.time()
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO turn_timings(session_id, turn_id, stage, label, status, started_at, metadata_json) VALUES(?, ?, ?, ?, 'running', ?, ?)",
                (self.session_id, turn_id, stage, str(label or ""), now, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def end_turn_timing(self, timing_id: int, status: str = "ok", metadata: dict[str, Any] | None = None) -> None:
        if not timing_id:
            return
        now = time.time()
        status = str(status or "ok")
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT metadata_json, started_at FROM turn_timings WHERE session_id=? AND id=?",
                (self.session_id, int(timing_id)),
            ).fetchone()
            if row is None:
                return
            merged = coerce_json_object(row["metadata_json"])
            if metadata:
                merged.update(metadata)
            started_at = float(row["started_at"] or now)
            con.execute(
                "UPDATE turn_timings SET status=?, ended_at=?, duration_seconds=?, metadata_json=? WHERE session_id=? AND id=?",
                (status, now, max(0.0, now - started_at), json.dumps(merged, ensure_ascii=False), self.session_id, int(timing_id)),
            )

    def record_turn_timing(
        self,
        turn_id: str,
        stage: str,
        label: str = "",
        *,
        status: str = "ok",
        duration_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: float | None = None,
    ) -> int:
        turn_id = str(turn_id or "").strip()
        stage = str(stage or "").strip()
        if not turn_id or not stage:
            return 0
        ended_at = time.time()
        if duration_seconds is None:
            duration_seconds = 0.0
        duration_seconds = max(0.0, float(duration_seconds or 0.0))
        if started_at is None:
            started_at = ended_at - duration_seconds
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO turn_timings(session_id, turn_id, stage, label, status, started_at, ended_at, duration_seconds, metadata_json) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    turn_id,
                    stage,
                    str(label or ""),
                    str(status or "ok"),
                    float(started_at),
                    ended_at,
                    duration_seconds,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def finish_open_turn_timings(self, turn_id: str, stage: str = "", status: str = "ok", metadata: dict[str, Any] | None = None) -> None:
        turn_id = str(turn_id or "").strip()
        if not turn_id:
            return
        stage_clause = "AND stage=?" if stage else ""
        params: list[Any] = [self.session_id, turn_id]
        if stage:
            params.append(stage)
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT id FROM turn_timings WHERE session_id=? AND turn_id=? AND ended_at IS NULL " + stage_clause,
                params,
            ).fetchall()
        for row in rows:
            self.end_turn_timing(int(row["id"]), status=status, metadata=metadata)

    def add_context_note(self, note: str, source: str = "pro") -> str:
        self.add_event("context_note", role="system", lane=source, content=note)
        self.append_front_note_context_update(f"- [note] {note.strip()}")
        return "context note saved"

    def latest_user_transcript(self) -> str:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT content FROM events WHERE session_id=? AND kind='transcript' AND role='user' ORDER BY id DESC LIMIT 1",
                (self.session_id,),
            ).fetchone()
        return str(row["content"] or "") if row else ""

    def front_note_state(self) -> dict[str, Any]:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM front_note_state WHERE session_id=?", (self.session_id,)).fetchone()
        if row is None:
            return {
                "session_id": self.session_id,
                "active_tab": "live",
                "live": {"html": "", "text": "", "media": [], "version": 0, "updated_at": 0.0},
                "context": {"html": "", "text": "", "media": [], "version": 0, "updated_at": 0.0},
                "content": "",
                "media": [],
                "visible": False,
                "position": "right",
                "width": 520,
                "height": 420,
                "version": 0,
                "updated_at": 0.0,
            }
        def load_media(value: str) -> list[dict[str, str]]:
            try:
                parsed = json.loads(value or "[]")
            except json.JSONDecodeError:
                parsed = []
            return sanitize_front_note_media(parsed if isinstance(parsed, list) else [])

        live_html = str(row["live_html"] or "")
        live_text = str(row["live_text"] or "") or front_note_html_to_text(live_html) or str(row["content"] or "")
        live_media = load_media(row["live_media_json"] or row["media_json"] or "[]")
        context_html = str(row["context_html"] or "")
        context_text = str(row["context_text"] or "") or front_note_html_to_text(context_html)
        context_media = load_media(row["context_media_json"] or "[]")
        active_tab = str(row["active_tab"] or "live").strip().lower()
        if active_tab not in {"live", "context"}:
            active_tab = "live"
        return {
            "session_id": self.session_id,
            "active_tab": active_tab,
            "live": {
                "html": live_html,
                "text": live_text,
                "media": live_media,
                "version": int(row["live_version"] or 0),
                "updated_at": float(row["live_updated_at"] or 0.0),
            },
            "context": {
                "html": context_html,
                "text": context_text,
                "media": context_media,
                "version": int(row["context_version"] or 0),
                "updated_at": float(row["context_updated_at"] or 0.0),
            },
            "content": live_text,
            "media": live_media,
            "visible": bool(row["visible"]),
            "position": str(row["position"] or "right"),
            "width": int(row["width"] or 520),
            "height": int(row["height"] or 420),
            "version": int(row["version"] or 0),
            "updated_at": float(row["updated_at"] or 0.0),
        }

    def update_front_note(
        self,
        *,
        action: str,
        tab: str = "live",
        content: str = "",
        html: str = "",
        media: Any = None,
        active_tab: str = "",
        source: str = "agent",
        allow_empty: bool = False,
        position: str = "right",
        visible: bool = True,
        width: int = 520,
        height: int = 420,
    ) -> dict[str, Any]:
        action = str(action or "show").strip().lower()
        if action not in {"show", "hide", "update", "append", "clear", "pin_edge"}:
            raise ValueError("front_note action must be show, hide, update, append, clear, or pin_edge")
        current = self.front_note_state()
        tab = str(tab or current.get("active_tab") or "live").strip().lower()
        if tab not in {"live", "context"}:
            tab = "live"
        source = str(source or "agent").strip().lower()
        if tab == "context" and source in {"agent", "api"} and action == "update" and str((current.get("context") or {}).get("text") or "").strip():
            action = "append"
        if tab == "live" and source != "human" and action in {"show", "update", "append", "clear"}:
            visible = True
            active_tab = "live"
        active_tab = str(active_tab or "").strip().lower()
        if active_tab not in {"live", "context"}:
            active_tab = tab if action in {"show", "update", "append", "clear"} else str(current.get("active_tab") or "live")
        position = str(position or current.get("position") or "right").strip().lower()
        if position not in {"left", "right", "center"}:
            position = "right"
        width = max(360, min(int(width or current.get("width") or 520), 980))
        height = max(280, min(int(height or current.get("height") or 420), 780))
        if isinstance(media, str):
            try:
                media = json.loads(media)
            except json.JSONDecodeError:
                media = [{"type": "link", "url": media, "title": media}]
        if not isinstance(media, list):
            media = current.get(tab, {}).get("media") or []
        cleaned_media = sanitize_front_note_media(media)
        next_visible = bool(current.get("visible"))
        next_live = dict(current.get("live") or {})
        next_context = dict(current.get("context") or {})
        target = next_context if tab == "context" else next_live
        target_html = str(target.get("html") or "")
        target_text = str(target.get("text") or "")
        target_media = list(target.get("media") or [])
        supplied_html = sanitize_front_note_html(html) if html else (front_note_content_to_html(content) if content else "")
        supplied_text = front_note_html_to_text(supplied_html) if supplied_html else str(content or "")
        content_changed = False
        if action == "clear":
            target_html = ""
            target_text = ""
            target_media = []
            content_changed = True
        elif action == "append":
            if supplied_html:
                target_html = "\n".join(part for part in [target_html.strip(), supplied_html.strip()] if part)
                target_text = "\n".join(part for part in [target_text.strip(), supplied_text.strip()] if part)
            if cleaned_media:
                target_media = sanitize_front_note_media([*target_media, *cleaned_media])
            content_changed = bool(supplied_html or cleaned_media)
            next_visible = bool(visible)
        elif action == "update":
            if source == "human" and not supplied_text.strip() and str(target_text or "").strip():
                content_changed = False
            else:
                target_html = supplied_html
                target_text = supplied_text
                target_media = cleaned_media
                content_changed = True
            next_visible = bool(visible)
        elif action == "show":
            if supplied_html or cleaned_media:
                if supplied_html:
                    target_html = supplied_html
                    target_text = supplied_text
                if cleaned_media:
                    target_media = cleaned_media
                content_changed = True
            next_visible = True
        elif action == "hide":
            next_visible = False
        elif action == "pin_edge":
            next_visible = bool(visible)
        if tab == "context":
            next_context.update({"html": target_html, "text": target_text, "media": target_media})
        else:
            next_live.update({"html": target_html, "text": target_text, "media": target_media})
        now = time.time()
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM front_note_state WHERE session_id=?", (self.session_id,)).fetchone()
            version = int(row["version"] or 0) + 1 if row else 1
            live_version = int((row["live_version"] if row else 0) or 0)
            context_version = int((row["context_version"] if row else 0) or 0)
            live_updated_at = float((row["live_updated_at"] if row else 0) or 0)
            context_updated_at = float((row["context_updated_at"] if row else 0) or 0)
            if content_changed and tab == "live":
                live_version += 1
                live_updated_at = now
            if content_changed and tab == "context":
                context_version += 1
                context_updated_at = now
            con.execute(
                "INSERT INTO front_note_state("
                "session_id, content, media_json, active_tab, live_html, live_text, live_media_json, live_version, live_updated_at, "
                "context_html, context_text, context_media_json, context_version, context_updated_at, visible, position, width, height, version, updated_at"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "content=excluded.content, media_json=excluded.media_json, active_tab=excluded.active_tab, "
                "live_html=excluded.live_html, live_text=excluded.live_text, live_media_json=excluded.live_media_json, "
                "live_version=excluded.live_version, live_updated_at=excluded.live_updated_at, "
                "context_html=excluded.context_html, context_text=excluded.context_text, context_media_json=excluded.context_media_json, "
                "context_version=excluded.context_version, context_updated_at=excluded.context_updated_at, "
                "visible=excluded.visible, position=excluded.position, width=excluded.width, height=excluded.height, "
                "version=excluded.version, updated_at=excluded.updated_at",
                (
                    self.session_id,
                    str(next_live.get("text") or ""),
                    json.dumps(next_live.get("media") or [], ensure_ascii=False),
                    active_tab,
                    str(next_live.get("html") or ""),
                    str(next_live.get("text") or ""),
                    json.dumps(next_live.get("media") or [], ensure_ascii=False),
                    live_version,
                    live_updated_at,
                    str(next_context.get("html") or ""),
                    str(next_context.get("text") or ""),
                    json.dumps(next_context.get("media") or [], ensure_ascii=False),
                    context_version,
                    context_updated_at,
                    1 if next_visible else 0,
                    position,
                    width,
                    height,
                    version,
                    now,
                ),
            )
        state = self.front_note_state()
        ui_state_only = source == "api" and action in {"show", "hide", "pin_edge"} and not content_changed
        if not ui_state_only:
            self.record_tool_event(
                "front_note",
                {
                    "action": action,
                    "tab": tab,
                    "source": source,
                    "content_chars": len(str(content or "") or str(html or "")),
                    "media_count": len(cleaned_media),
                    "position": position,
                    "visible": next_visible,
                },
                {"ok": True, "version": state["version"], "visible": state["visible"], "position": state["position"]},
            )
        return state

    def append_front_note_context_update(self, line: str) -> None:
        line = str(line or "").strip()
        if not line:
            return
        state = self.front_note_state()
        block = "## Session Updates\n" + line
        try:
            self.update_front_note(
                action="append",
                tab="context",
                content=block,
                active_tab=str(state.get("active_tab") or "context"),
                source="system",
                visible=bool(state.get("visible")),
                width=int(state.get("width") or 520),
                height=int(state.get("height") or 420),
            )
        except Exception as exc:
            self.add_event("front_note_sync_error", role="system", lane="store", content="front note context sync failed", metadata={"error": str(exc)[:1000], "line": line[:500]})

    def update_task_status(self, title: str, status: str, summary: str = "") -> str:
        task_id = re.sub(r"\s+", "-", title.strip().lower())[:80] or uuid.uuid4().hex
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO task_states(id, session_id, title, status, summary, updated_at) VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status, summary=excluded.summary, updated_at=excluded.updated_at",
                (task_id, self.session_id, title.strip(), status, summary.strip(), time.time()),
            )
        self.record_tool_event("update_task_status", {"title": title, "status": status, "summary": summary}, {"task_id": task_id})
        task_line = f"- [{status}] {title.strip()}"
        if summary.strip():
            task_line += f": {summary.strip()}"
        self.append_front_note_context_update(task_line)
        return f"task {task_id} marked {status}"

    def trigger_fast_followup(self, prompt: str, priority: int = 10) -> str:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO pending_fast_prompts(session_id, prompt, priority, status, created_at) VALUES(?, ?, ?, 'pending', ?)",
                (self.session_id, prompt.strip(), int(priority), time.time()),
            )
        self.record_tool_event("trigger_fast_followup", {"prompt": prompt, "priority": priority}, {"queued": True})
        return "fast followup queued"

    def pop_pending_fast_prompt(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM pending_fast_prompts WHERE session_id=? AND status='pending' ORDER BY priority DESC, id DESC LIMIT 1",
                (self.session_id,),
            ).fetchone()
            if row is None:
                return None
            con.execute("UPDATE pending_fast_prompts SET status='handled', handled_at=? WHERE id=?", (time.time(), row["id"]))
            return dict(row)

    def clear_pending_fast_prompts(self, reason: str) -> int:
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE pending_fast_prompts SET status=?, handled_at=? WHERE session_id=? AND status='pending'",
                (f"superseded:{reason}", time.time(), self.session_id),
            )
            return int(cur.rowcount or 0)

    def recent_assistant_replies(self, lane: str = "", within_seconds: float = 120.0, limit: int = 6) -> list[dict[str, Any]]:
        cutoff = time.time() - within_seconds
        lane_clause = "AND lane=?" if lane else ""
        params: list[Any] = [self.session_id, cutoff]
        if lane:
            params.append(lane)
        params.append(limit)
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT content, lane, metadata_json, created_at FROM events "
                f"WHERE session_id=? AND kind='assistant_reply' AND created_at>=? {lane_clause} "
                "ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_tool_events(self, within_seconds: float = 60.0, limit: int = 5, ok_only: bool = False, since: float | None = None) -> list[dict[str, Any]]:
        cutoff = since if since is not None else time.time() - within_seconds
        ok_clause = "AND ok=1" if ok_only else ""
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT tool_name, arguments_json, result_json, ok, created_at FROM tool_events "
                f"WHERE session_id=? AND created_at>=? {ok_clause} "
                "ORDER BY id DESC LIMIT ?",
                (self.session_id, cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_tool_event(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any], ok: bool = True, turn_id: str = "") -> None:
        status = "success" if ok else "failure"
        logged_result = dict(result)
        logged_result.setdefault("_log_word", self.tool_log_label(tool_name, status))
        logged_result.setdefault("_log_language", "zh")
        logged_result.setdefault("_summary", self.tool_voice_summary(tool_name, ok, logged_result))
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO tool_events(session_id, tool_name, arguments_json, result_json, ok, created_at, turn_id) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, tool_name, json.dumps(arguments, ensure_ascii=False), json.dumps(logged_result, ensure_ascii=False), 1 if ok else 0, time.time(), str(turn_id or "")),
            )

    def create_coding_task(self, task: dict[str, Any]) -> str:
        task_id = str(task.get("task_id") or uuid.uuid4())
        now = time.time()
        next_speech_at = float(task["next_speech_at"]) if "next_speech_at" in task else now + 60.0
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO coding_tasks("
                "task_id, session_id, run_id, pid, status, workspace_id, target, cwd, workspace, executor, model, stdout_log, stderr_log, "
                "executor_log, last_message_path, prompt_path, active_venv, last_offset, last_summary, next_speech_at, created_at, updated_at, completed_at"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    self.session_id,
                    str(task.get("run_id") or ""),
                    int(task.get("pid") or 0),
                    str(task.get("status") or "running"),
                    str(task.get("workspace_id") or ""),
                    str(task.get("target") or ""),
                    str(task.get("cwd") or ""),
                    str(task.get("workspace") or task.get("cwd") or ""),
                    str(task.get("executor") or "codex"),
                    str(task.get("model") or ""),
                    str(task.get("stdout_log") or ""),
                    str(task.get("stderr_log") or ""),
                    str(task.get("executor_log") or ""),
                    str(task.get("last_message_path") or ""),
                    str(task.get("prompt_path") or ""),
                    str(task.get("active_venv") or ""),
                    int(task.get("last_offset") or 0),
                    str(task.get("last_summary") or ""),
                    next_speech_at,
                    now,
                    now,
                    task.get("completed_at"),
                ),
            )
            con.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, self.session_id))
        return task_id

    def coding_tasks(self, statuses: tuple[str, ...] = ("running",), limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        placeholders = ",".join("?" for _ in statuses)
        query = (
            "SELECT * FROM coding_tasks WHERE session_id=? "
            f"AND status IN ({placeholders}) ORDER BY updated_at ASC LIMIT ?"
        )
        with self._lock, self._connect() as con:
            rows = con.execute(query, (self.session_id, *statuses, limit)).fetchall()
        return [dict(row) for row in rows]

    def last_coding_executor(self) -> str:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT executor FROM coding_tasks WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (self.session_id,),
            ).fetchone()
        if row is None:
            return ""
        value = str(row["executor"] or "").strip().lower()
        return value if value in {"codex", "antigravity"} else ""

    def update_coding_task(self, task_id: str, **updates: Any) -> None:
        task_id = str(task_id or "").strip()
        if not task_id or not updates:
            return
        allowed = {"status", "pid", "last_offset", "last_summary", "next_speech_at", "completed_at"}
        fields = [key for key in updates if key in allowed]
        if not fields:
            return
        values = [updates[key] for key in fields]
        fields.append("updated_at")
        values.append(time.time())
        assignments = ", ".join(f"{field}=?" for field in fields)
        with self._lock, self._connect() as con:
            con.execute(
                f"UPDATE coding_tasks SET {assignments} WHERE session_id=? AND task_id=?",
                (*values, self.session_id, task_id),
            )
            con.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (time.time(), self.session_id))

    def record_coding_task_status(self, task_id: str, status: str, summary: str, metadata: dict[str, Any] | None = None) -> int:
        content = str(summary or "").strip()
        if not content:
            return 0
        payload = {"task_id": str(task_id or ""), "status": str(status or ""), **(metadata or {})}
        return self.add_event("coding_task_status", role="assistant", lane="coding", content=content, metadata=payload)

    def upsert_coding_workspace(self, workspace: dict[str, Any]) -> str:
        workspace_id = str(workspace.get("workspace_id") or uuid.uuid4())
        path = str(workspace.get("path") or "").strip()
        if not path:
            raise ValueError("workspace path is required")
        now = time.time()
        payload = {
            "aliases_json": json.dumps(workspace.get("aliases") or [], ensure_ascii=False),
            "tags_json": json.dumps(workspace.get("tags") or [], ensure_ascii=False),
            "capabilities_json": json.dumps(workspace.get("capabilities") or [], ensure_ascii=False),
            "entrypoints_json": json.dumps(workspace.get("entrypoints") or [], ensure_ascii=False),
            "services_json": json.dumps(workspace.get("services") or [], ensure_ascii=False),
            "program_json": json.dumps(workspace.get("program") or {}, ensure_ascii=False),
            "related_workspace_ids_json": json.dumps(workspace.get("related_workspace_ids") or [], ensure_ascii=False),
        }
        with self._lock, self._connect() as con:
            existing = con.execute(
                "SELECT workspace_id, session_id, created_at FROM coding_workspaces WHERE path=?",
                (path,),
            ).fetchone()
            if existing:
                workspace_id = str(existing["workspace_id"])
                row_session_id = str(existing["session_id"] or self.session_id)
                con.execute(
                    """
                    UPDATE coding_workspaces SET
                      title=?, aliases_json=?, tags_json=?, summary=?, capabilities_json=?, entrypoints_json=?,
                      services_json=?, program_json=?, related_workspace_ids_json=?, status=?, last_task_at=?, updated_at=?
                    WHERE session_id=? AND workspace_id=?
                    """,
                    (
                        str(workspace.get("title") or ""),
                        payload["aliases_json"],
                        payload["tags_json"],
                        str(workspace.get("summary") or ""),
                        payload["capabilities_json"],
                        payload["entrypoints_json"],
                        payload["services_json"],
                        payload["program_json"],
                        payload["related_workspace_ids_json"],
                        str(workspace.get("status") or "active"),
                        float(workspace.get("last_task_at") or now),
                        now,
                        row_session_id,
                        workspace_id,
                    ),
                )
            else:
                con.execute(
                    """
                    INSERT INTO coding_workspaces(
                      workspace_id, session_id, path, title, aliases_json, tags_json, summary,
                      capabilities_json, entrypoints_json, services_json, program_json, related_workspace_ids_json,
                      status, last_task_at, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        self.session_id,
                        path,
                        str(workspace.get("title") or ""),
                        payload["aliases_json"],
                        payload["tags_json"],
                        str(workspace.get("summary") or ""),
                        payload["capabilities_json"],
                        payload["entrypoints_json"],
                        payload["services_json"],
                        payload["program_json"],
                        payload["related_workspace_ids_json"],
                        str(workspace.get("status") or "active"),
                        float(workspace.get("last_task_at") or now),
                        now,
                        now,
                    ),
                )
            con.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, self.session_id))
        return workspace_id

    def coding_workspace(self, workspace_id: str | None = None, path: str | None = None) -> dict[str, Any] | None:
        if not workspace_id and not path:
            return None
        if workspace_id:
            where = "workspace_id=?"
            value = workspace_id
        else:
            where = "path=?"
            value = path
        with self._lock, self._connect() as con:
            row = con.execute(
                f"SELECT * FROM coding_workspaces WHERE session_id=? AND {where} LIMIT 1",
                (self.session_id, value),
            ).fetchone()
        return self._coding_workspace_from_row(row) if row else None

    def coding_workspaces(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT * FROM coding_workspaces WHERE session_id=? ORDER BY last_task_at DESC, updated_at DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
        return [self._coding_workspace_from_row(row) for row in rows]

    @staticmethod
    def _coding_workspace_from_row(row: Any) -> dict[str, Any]:
        def load_json(name: str, default: Any) -> Any:
            try:
                return json.loads(row[name] or "")
            except Exception:
                return default

        return {
            "workspace_id": row["workspace_id"],
            "path": row["path"],
            "title": row["title"],
            "aliases": load_json("aliases_json", []),
            "tags": load_json("tags_json", []),
            "summary": row["summary"],
            "capabilities": load_json("capabilities_json", []),
            "entrypoints": load_json("entrypoints_json", []),
            "services": load_json("services_json", []),
            "program": load_json("program_json", {}),
            "related_workspace_ids": load_json("related_workspace_ids_json", []),
            "status": row["status"],
            "last_task_at": row["last_task_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT session_id, summary, turn_count, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": str(row["session_id"] or ""),
                "summary": str(row["summary"] or ""),
                "turn_count": int(row["turn_count"] or 0),
                "created_at": float(row["created_at"] or 0.0),
                "updated_at": float(row["updated_at"] or 0.0),
                "current": str(row["session_id"] or "") == self.session_id,
            }
            for row in rows
        ]

    def add_live_note(self, content: str, session_id: str | None = None) -> int:
        text = str(content or "").strip()
        if not text:
            raise ValueError("live note content is required")
        target_session = str(session_id or self.session_id).strip() or self.session_id
        now = time.time()
        with self._lock, self._connect() as con:
            con.execute("INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at) VALUES(?, ?, ?)", (target_session, now, now))
            cur = con.execute(
                "INSERT INTO events(session_id, kind, role, lane, content, metadata_json, created_at) VALUES(?, 'live_note', 'user', 'front_note', ?, ?, ?)",
                (target_session, text, json.dumps({"source": "human"}, ensure_ascii=False), now),
            )
            con.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, target_session))
            return int(cur.lastrowid)

    def recent_pipeline(self, limit: int = 180) -> dict[str, Any]:
        return self.pipeline_for_session(self.session_id, limit=limit)

    def lane_call_log(self, limit: int = 300) -> dict[str, Any]:
        limit = max(20, min(int(limit or 300), 1000))
        with self._lock, self._connect() as con:
            event_rows = con.execute(
                "SELECT kind, lane, content, metadata_json FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
            timing_rows = con.execute(
                "SELECT stage FROM turn_timings WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
        lane_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        call_counts: dict[str, int] = {"tr": 0, "tc": 0, "mc": 0, "tts": 0}
        for row in event_rows:
            kind = str(row["kind"] or "")
            metadata = coerce_json_object(row["metadata_json"])
            lane = str(row["lane"] or row["kind"] or "unknown")
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            if kind == "domain_probe":
                call_counts["tr"] += self._registered_tool_count_from_domain_probe(row["content"], metadata)
                payload = coerce_json_object(row["content"])
                for item in payload.get("domains") or []:
                    if not isinstance(item, dict):
                        continue
                    domain = str(item.get("domain") or "").strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            elif kind == "tool_started":
                tool = str(metadata.get("tool_name") or row["content"] or "unknown").split(":", 1)[-1]
                if tool not in INTERNAL_COUNT_EXCLUDED_TOOLS:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
                    call_counts["tc"] += 1
            elif kind == "llm_call" and str(metadata.get("status") or "") == "started":
                call_counts["mc"] += 1
            elif kind == "tts_call":
                call_counts["tts"] += 1
        timing_counts: dict[str, int] = {}
        for row in timing_rows:
            stage = str(row["stage"] or "unknown")
            timing_counts[stage] = timing_counts.get(stage, 0) + 1

        def top(counts: dict[str, int]) -> list[dict[str, Any]]:
            return [{"name": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]

        return {
            "session_id": self.session_id,
            "limit": limit,
            "lanes": top(lane_counts),
            "domains": top(domain_counts),
            "tools": top(tool_counts),
            "calls": [{"name": key, "count": value} for key, value in call_counts.items()],
            "timings": top(timing_counts),
        }

    def turn_call_counts(self, turn_id: str) -> dict[str, int]:
        turn_id = str(turn_id or "").strip()
        counts = {"tr": 0, "tc": 0, "mc": 0, "tts": 0}
        if not turn_id:
            return counts
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT kind, content, metadata_json FROM events WHERE session_id=? ORDER BY id DESC LIMIT 1000",
                (self.session_id,),
            ).fetchall()
        for row in rows:
            metadata = coerce_json_object(row["metadata_json"])
            if str(metadata.get("turn_id") or "") != turn_id:
                continue
            kind = str(row["kind"] or "")
            if kind == "domain_probe":
                counts["tr"] += self._registered_tool_count_from_domain_probe(row["content"], metadata)
            elif kind == "tool_started":
                tool = str(metadata.get("tool_name") or row["content"] or "unknown").split(":", 1)[-1]
                if tool not in INTERNAL_COUNT_EXCLUDED_TOOLS:
                    counts["tc"] += 1
            elif kind == "llm_call" and str(metadata.get("status") or "") == "started":
                counts["mc"] += 1
            elif kind == "tts_call":
                counts["tts"] += 1
        return counts

    def _registered_tool_count_from_domain_probe(self, content: Any, metadata: dict[str, Any] | None = None) -> int:
        metadata = metadata or {}
        explicit = metadata.get("registered_tool_count")
        if explicit is not None:
            try:
                return max(0, int(explicit))
            except (TypeError, ValueError):
                pass
        payload = coerce_json_object(content)
        calls: set[str] = set()
        for domain_item in payload.get("domains") or []:
            if not isinstance(domain_item, dict):
                continue
            for action in domain_item.get("suggested_actions") or []:
                if not isinstance(action, dict):
                    continue
                tool_call = str(action.get("tool_call") or "").strip()
                tool_name = str(action.get("tool") or "").strip()
                if tool_call:
                    calls.add(tool_call)
                elif tool_name:
                    calls.add(tool_name.split(":", 1)[-1])
        return len(calls)

    def pipeline_for_session(self, session_id: str, limit: int = 180, *, live_view: bool = False) -> dict[str, Any]:
        target_session = str(session_id or self.session_id).strip() or self.session_id
        limit = max(20, min(int(limit or 180), 500))
        with self._lock, self._connect() as con:
            event_rows = con.execute(
                "SELECT id, kind, role, lane, content, metadata_json, created_at FROM events "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (target_session, limit),
            ).fetchall()
            tool_rows = con.execute(
                "SELECT id, tool_name, arguments_json, result_json, ok, created_at, turn_id FROM tool_events "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (target_session, limit),
            ).fetchall()
            timing_rows = con.execute(
                "SELECT id, turn_id, stage, label, status, started_at, ended_at, duration_seconds, metadata_json FROM turn_timings "
                "WHERE session_id=? ORDER BY started_at DESC LIMIT ?",
                (target_session, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in event_rows:
            metadata = coerce_json_object(row["metadata_json"])
            items.append({
                "source": "events",
                "id": int(row["id"]),
                "turn_id": str(metadata.get("turn_id") or ""),
                "kind": str(row["kind"] or ""),
                "role": str(row["role"] or ""),
                "lane": str(row["lane"] or ""),
                "content": str(row["content"] or ""),
                "metadata": metadata,
                "created_at": float(row["created_at"] or 0),
            })
        for row in tool_rows:
            arguments = coerce_json_object(row["arguments_json"])
            result = coerce_json_object(row["result_json"])
            if (
                str(row["tool_name"] or "") == "front_note"
                and str(arguments.get("source") or "").lower() == "api"
                and str(arguments.get("action") or "").lower() in {"show", "hide", "pin_edge"}
                and int(arguments.get("content_chars") or 0) == 0
                and int(arguments.get("media_count") or 0) == 0
            ):
                continue
            turn_id = str(row["turn_id"] or arguments.get("_turn_id") or result.get("_turn_id") or "")
            items.append({
                "source": "tool_events",
                "id": int(row["id"]),
                "turn_id": turn_id,
                "kind": "tool_event",
                "role": "tool",
                "lane": "",
                "content": str(row["tool_name"] or ""),
                "metadata": {"tool_name": row["tool_name"], "arguments": arguments, "result": result, "ok": bool(row["ok"])},
                "created_at": float(row["created_at"] or 0),
            })
        for row in timing_rows:
            metadata = coerce_json_object(row["metadata_json"])
            duration = row["duration_seconds"]
            items.append({
                "source": "turn_timings",
                "id": int(row["id"]),
                "turn_id": str(row["turn_id"] or ""),
                "kind": "timing",
                "role": "system",
                "lane": "timing",
                "content": str(row["label"] or row["stage"] or ""),
                "metadata": {
                    "stage": str(row["stage"] or ""),
                    "label": str(row["label"] or ""),
                    "status": str(row["status"] or ""),
                    "duration_seconds": None if duration is None else round(float(duration), 3),
                    "started_at": float(row["started_at"] or 0),
                    "ended_at": None if row["ended_at"] is None else float(row["ended_at"] or 0),
                    **metadata,
                },
                "created_at": float(row["started_at"] or 0),
            })
        items.sort(key=lambda item: (float(item.get("created_at") or 0), str(item.get("source") or ""), int(item.get("id") or 0)))
        stats = self._pipeline_stats(items)
        output_items = self._live_display_items(items) if live_view else items
        turns: dict[str, dict[str, Any]] = {}
        for item in output_items:
            turn_id = str(item.get("turn_id") or "")
            if not turn_id:
                continue
            turn = turns.setdefault(turn_id, {"turn_id": turn_id, "started_at": item["created_at"], "last_at": item["created_at"], "items": [], "timings": []})
            turn["started_at"] = min(float(turn["started_at"]), float(item["created_at"]))
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            ended_at = metadata.get("ended_at") if item.get("source") == "turn_timings" else None
            turn["last_at"] = max(float(turn["last_at"]), float(ended_at or item["created_at"]))
            turn["items"].append(item)
            if item.get("source") == "turn_timings":
                turn["timings"].append(item)
        for turn in turns.values():
            turn["timings"].sort(key=lambda item: float((item.get("metadata") or {}).get("started_at") or item.get("created_at") or 0))
        ordered_turns = sorted(turns.values(), key=lambda item: float(item.get("last_at") or 0), reverse=True)
        return {"session_id": target_session, "current_session_id": self.session_id, "items": output_items[-limit:], "turns": ordered_turns[:20], "stats": stats}

    def _pipeline_stats(self, items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        lane_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        call_counts: dict[str, int] = {"tr": 0, "tc": 0, "mc": 0, "tts": 0}
        for item in items:
            lane = str(item.get("lane") or "")
            kind = str(item.get("kind") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if lane and kind != "timing":
                lane_counts[lane] = lane_counts.get(lane, 0) + 1
            if kind == "tool_event":
                continue
            if kind == "tool_started":
                tool = str((metadata.get("tool_name") if isinstance(metadata, dict) else "") or item.get("content") or "").split(":", 1)[-1]
                if tool and tool not in INTERNAL_COUNT_EXCLUDED_TOOLS:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
                    call_counts["tc"] += 1
            if kind == "llm_call" and str(metadata.get("status") or "") == "started":
                call_counts["mc"] += 1
            if kind == "tts_call":
                call_counts["tts"] += 1
            if kind == "domain_probe":
                call_counts["tr"] += self._registered_tool_count_from_domain_probe(item.get("content"), metadata)
                payload = coerce_json_object(item.get("content"))
                for domain_item in payload.get("domains") or []:
                    if not isinstance(domain_item, dict):
                        continue
                    domain = str(domain_item.get("domain") or "").strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        def top(counts: dict[str, int]) -> list[dict[str, Any]]:
            return [{"name": key, "count": value} for key, value in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:8]]

        return {
            "lanes": top(lane_counts),
            "tools": top(tool_counts),
            "domains": top(domain_counts),
            "calls": [{"name": key, "count": value} for key, value in call_counts.items()],
        }

    def _live_display_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(items)

    def session_value(self, key: str, default: str = "") -> str:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT value FROM session_kv WHERE session_id=? AND key=?",
                (self.session_id, key),
            ).fetchone()
        return str(row["value"]) if row else default

    def set_session_value(self, key: str, value: str) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO session_kv(session_id, key, value, updated_at) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (self.session_id, key, value, time.time()),
            )

    def tool_speech_catalog(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT tool_name, task_label, start_phrase, success_phrase, failure_phrase, speech_enabled, language, updated_at "
                "FROM tool_speech_catalog ORDER BY tool_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_tool_speech(
        self,
        tool_name: str,
        task_label: str,
        start_phrase: str,
        success_phrase: str,
        failure_phrase: str,
        speech_enabled: bool = True,
    ) -> None:
        tool_name = tool_name.strip()
        if not tool_name:
            raise ValueError("tool_name is required")
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO tool_speech_catalog(tool_name, task_label, start_phrase, success_phrase, failure_phrase, speech_enabled, language, updated_at) "
                "VALUES(?, ?, ?, ?, ?, ?, 'zh', ?) "
                "ON CONFLICT(tool_name) DO UPDATE SET "
                "task_label=excluded.task_label, start_phrase=excluded.start_phrase, success_phrase=excluded.success_phrase, "
                "failure_phrase=excluded.failure_phrase, speech_enabled=excluded.speech_enabled, "
                "language=excluded.language, updated_at=excluded.updated_at",
                (
                    tool_name,
                    task_label.strip(),
                    start_phrase.strip(),
                    success_phrase.strip(),
                    failure_phrase.strip(),
                    1 if speech_enabled else 0,
                    time.time(),
                ),
            )

    def tool_words(self, tool_name: str) -> dict[str, Any]:
        key = tool_name.split(":", 1)[-1]
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT task_label, start_phrase, success_phrase, failure_phrase, speech_enabled FROM tool_speech_catalog WHERE tool_name=?",
                (key,),
            ).fetchone()
        if row is None:
            words = tool_log_words(key)
            return {"task_label": DEFAULT_TOOL_TASK_LABELS.get(key, ""), "speech_enabled": key not in DEFAULT_SILENT_TOOLS, **words}
        return {
            "task_label": str(row["task_label"] or ""),
            "start": str(row["start_phrase"] or DEFAULT_TOOL_LOG_WORDS["start"]),
            "success": str(row["success_phrase"] or DEFAULT_TOOL_LOG_WORDS["success"]),
            "failure": str(row["failure_phrase"] or DEFAULT_TOOL_LOG_WORDS["failure"]),
            "speech_enabled": bool(row["speech_enabled"]),
        }

    def tool_log_label(self, tool_name: str, status: str) -> str:
        words = self.tool_words(tool_name)
        return words.get(status) or DEFAULT_TOOL_LOG_WORDS.get(status, "")

    def tool_task_label(self, tool_name: str) -> str:
        return self.tool_words(tool_name).get("task_label", "")

    def tool_speech_enabled(self, tool_name: str) -> bool:
        return bool(self.tool_words(tool_name).get("speech_enabled", True))

    def tool_voice_summary(self, tool_name: str, ok: bool, result: Any) -> str:
        label = self.tool_log_label(tool_name, "success" if ok else "failure").strip()
        if ok:
            weather = self._weather_voice_summary(tool_name, result)
            if weather:
                return weather
            return label
        reason = short_tool_error_reason(result)
        return f"{label}：{reason}" if reason else label

    def _weather_voice_summary(self, tool_name: str, result: Any) -> str:
        if str(tool_name or "").split(":", 1)[-1] != "get_weather":
            return ""
        payload = coerce_json_object(result)
        if not payload or payload.get("ok") is False:
            return ""
        location = str(payload.get("resolved_location") or payload.get("location") or "").strip()
        temp = str(payload.get("temperature_c") or payload.get("temperature") or "").strip()
        feels = str(payload.get("feels_like_c") or "").strip()
        desc = str(payload.get("description") or "").strip()
        humidity = str(payload.get("humidity_percent") or "").strip()
        wind = str(payload.get("wind_kmph") or "").strip()
        parts: list[str] = []
        if location:
            parts.append(f"{location}现在")
        else:
            parts.append("现在")
        if temp:
            parts.append(f"{temp}度")
        if feels and feels != temp:
            parts.append(f"体感{feels}度")
        if desc:
            parts.append(desc)
        if humidity:
            parts.append(f"湿度{humidity}%")
        if wind:
            parts.append(f"风速{wind}公里每小时")
        return "，".join(parts) + "。" if len(parts) > 1 else ""

    def seed_filler_catalog_from_manifest(self, filler_dir: Path | None) -> None:
        if filler_dir is None:
            return
        manifest_path = filler_dir / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return
        now = time.time()
        with self._lock, self._connect() as con:
            for item in manifest.get("items") or []:
                try:
                    slot_index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                tone = str(item.get("tone") or "").strip() or "soft"
                phrase = str(item.get("phrase") or "").strip()
                if not phrase:
                    continue
                source_path = Path(str(item.get("path") or ""))
                local_path = filler_dir / source_path.name
                path = local_path if local_path.exists() else source_path
                con.execute(
                    "INSERT INTO filler_speech_catalog(slot_index, tone, stage, phrase, instructions, audio_path, ok, seconds, bytes, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(slot_index, tone) DO NOTHING",
                    (
                        slot_index,
                        tone,
                        filler_stage_for_phrase(phrase),
                        phrase,
                        str(item.get("instructions") or ""),
                        str(path),
                        1 if path.exists() and int(item.get("status") or 0) == 200 else 0,
                        item.get("seconds"),
                        int(item.get("bytes") or (path.stat().st_size if path.exists() else 0)),
                        now,
                    ),
                )

    def filler_speech_catalog(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT id, slot_index, tone, stage, phrase, instructions, audio_path, ok, seconds, bytes, updated_at "
                "FROM filler_speech_catalog ORDER BY stage, slot_index, tone"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_filler_speech(self, filler_id: int, *, phrase: str, tone: str, stage: str, instructions: str = "") -> None:
        phrase = phrase.strip()
        if not phrase:
            raise ValueError("phrase is required")
        tone = tone.strip() or "soft"
        stage = stage.strip() or filler_stage_for_phrase(phrase)
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM filler_speech_catalog WHERE id=?", (int(filler_id),)).fetchone()
            if row is None:
                raise ValueError("filler not found")
            reset_audio = phrase != row["phrase"] or tone != row["tone"] or instructions != row["instructions"]
            con.execute(
                "UPDATE filler_speech_catalog SET phrase=?, tone=?, stage=?, instructions=?, "
                "audio_path=CASE WHEN ? THEN '' ELSE audio_path END, ok=CASE WHEN ? THEN 0 ELSE ok END, "
                "seconds=CASE WHEN ? THEN NULL ELSE seconds END, bytes=CASE WHEN ? THEN 0 ELSE bytes END, updated_at=? WHERE id=?",
                (phrase, tone, stage, instructions, reset_audio, reset_audio, reset_audio, reset_audio, time.time(), int(filler_id)),
            )

    def update_filler_audio(self, filler_id: int, *, audio_path: str, ok: bool, seconds: float | None = None, bytes_count: int = 0) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE filler_speech_catalog SET audio_path=?, ok=?, seconds=?, bytes=?, updated_at=? WHERE id=?",
                (audio_path, 1 if ok else 0, seconds, int(bytes_count or 0), time.time(), int(filler_id)),
            )

    def add_filler_speech(self, *, phrase: str, tone: str = "soft", stage: str = "", instructions: str = "") -> int:
        phrase = phrase.strip()
        if not phrase:
            raise ValueError("phrase is required")
        tone = tone.strip() or "soft"
        stage = stage.strip() or filler_stage_for_phrase(phrase)
        with self._lock, self._connect() as con:
            row = con.execute("SELECT COALESCE(MAX(slot_index), 0) + 1 AS next_slot FROM filler_speech_catalog").fetchone()
            slot_index = int(row["next_slot"] or 1)
            cur = con.execute(
                "INSERT INTO filler_speech_catalog(slot_index, tone, stage, phrase, instructions, audio_path, ok, seconds, bytes, updated_at) "
                "VALUES(?, ?, ?, ?, ?, '', 0, NULL, 0, ?)",
                (slot_index, tone, stage, phrase, instructions.strip(), time.time()),
            )
            return int(cur.lastrowid)

    def delete_filler_speech(self, filler_id: int) -> dict[str, Any]:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM filler_speech_catalog WHERE id=?", (int(filler_id),)).fetchone()
            if row is None:
                raise ValueError("filler not found")
            con.execute("DELETE FROM filler_speech_catalog WHERE id=?", (int(filler_id),))
        path_text = str(row["audio_path"] or "")
        path = Path(path_text) if path_text else None
        deleted_audio = False
        try:
            if path is not None and path.is_file():
                path.unlink()
                deleted_audio = True
        except Exception:
            deleted_audio = False
        return {"id": int(filler_id), "deleted_audio": deleted_audio, "audio_path": str(path or "")}

    def record_tool_speech_cache(
        self,
        *,
        tool_name: str,
        status: str,
        task_label: str,
        phrase: str,
        spoken_text: str,
        tts_model: str,
        tts_voice: str,
        audio_path: str,
        ok: bool,
        error: str = "",
    ) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO tool_speech_cache(tool_name, status, task_label, phrase, spoken_text, tts_model, tts_voice, audio_path, ok, error, generated_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(tool_name, status, spoken_text, tts_model, tts_voice) DO UPDATE SET "
                "task_label=excluded.task_label, phrase=excluded.phrase, audio_path=excluded.audio_path, ok=excluded.ok, "
                "error=excluded.error, generated_at=excluded.generated_at",
                (
                    tool_name,
                    status,
                    task_label,
                    phrase,
                    spoken_text,
                    tts_model,
                    tts_voice,
                    audio_path,
                    1 if ok else 0,
                    error[:1000],
                    time.time(),
                ),
            )

    def stored_context_snapshot(self) -> dict[str, Any]:
        with self._lock, self._connect() as con:
            session = con.execute("SELECT * FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()
            tasks = [
                dict(row)
                for row in con.execute(
                    "SELECT title, status, summary FROM task_states WHERE session_id=? ORDER BY updated_at DESC LIMIT 24",
                    (self.session_id,),
                )
            ]
            notes = [
                str(row["content"] or "")
                for row in con.execute(
                    "SELECT content FROM events WHERE session_id=? AND kind='context_note' ORDER BY id DESC LIMIT 20",
                    (self.session_id,),
                )
            ]
        return {
            "summary": str(session["summary"] or "") if session else "",
            "active_tasks": tasks,
            "context_notes": notes,
            "user_preferences": json.loads(session["user_preferences_json"] if session else "[]"),
            "open_threads": json.loads(session["open_threads_json"] if session else "[]"),
        }

    def stored_context_snapshot_text(self) -> str:
        snapshot = self.stored_context_snapshot()
        lines: list[str] = []
        summary = str(snapshot.get("summary") or "").strip()
        if summary:
            lines.extend(["## Summary", summary])
        tasks = snapshot.get("active_tasks") or []
        if tasks:
            lines.extend(["", "## Tasks"])
            for task in tasks:
                title = str(task.get("title") or "").strip()
                status = str(task.get("status") or "").strip()
                task_summary = str(task.get("summary") or "").strip()
                line = f"- [{status}] {title}" if status else f"- {title}"
                if task_summary:
                    line += f": {task_summary}"
                lines.append(line)
        notes = snapshot.get("context_notes") or []
        if notes:
            lines.extend(["", "## Context Notes"])
            for note in notes:
                note = str(note or "").strip()
                if note:
                    lines.append(f"- {note}")
        preferences = snapshot.get("user_preferences") or []
        if preferences:
            lines.extend(["", "## User Preferences"])
            for item in preferences:
                lines.append(f"- {item}")
        threads = snapshot.get("open_threads") or []
        if threads:
            lines.extend(["", "## Open Threads"])
            for item in threads:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()

    def ensure_front_note_context_seeded(self) -> bool:
        state = self.front_note_state()
        current_text = str((state.get("context") or {}).get("text") or "").strip()
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT value FROM session_kv WHERE session_id=? AND key='front_note_context_seeded'",
                (self.session_id,),
            ).fetchone()
        if row is not None or current_text:
            return False
        snapshot_text = self.stored_context_snapshot_text()
        if not snapshot_text:
            return False
        self.update_front_note(
            action="update",
            tab="context",
            content=snapshot_text,
            active_tab=str(state.get("active_tab") or "live"),
            position=str(state.get("position") or "right"),
            visible=bool(state.get("visible")),
            width=int(state.get("width") or 520),
            height=int(state.get("height") or 420),
        )
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO session_kv(session_id, key, value, updated_at) VALUES(?, 'front_note_context_seeded', '1', ?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (self.session_id, time.time()),
            )
        return True

    def context_bundle(self, recent_limit: int = 10) -> str:
        with self._lock, self._connect() as con:
            session = con.execute("SELECT * FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()
            tasks = [dict(row) for row in con.execute("SELECT title, status, summary FROM task_states WHERE session_id=? ORDER BY updated_at DESC LIMIT 20", (self.session_id,))]
            events = [dict(row) for row in con.execute("SELECT kind, role, lane, content FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?", (self.session_id, recent_limit))]
        events.reverse()
        front_note = self.front_note_state()
        front_note_context = str((front_note.get("context") or {}).get("text") or "").strip()
        if not front_note_context:
            front_note_context = self.stored_context_snapshot_text()
        return json.dumps(
            {
                "summary": session["summary"] if session else "",
                "active_tasks": tasks,
                "user_preferences": json.loads(session["user_preferences_json"] if session else "[]"),
                "open_threads": json.loads(session["open_threads_json"] if session else "[]"),
                "front_note_context": front_note_context,
                "recent_events": events,
            },
            ensure_ascii=False,
        )

    def domain_probe_context(self, recent_limit: int = 6) -> dict[str, Any]:
        recent_limit = max(1, min(int(recent_limit or 6), 12))
        with self._lock, self._connect() as con:
            session = con.execute("SELECT summary, user_preferences_json, open_threads_json FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()
            rows = con.execute(
                "SELECT kind, role, lane, content FROM events "
                "WHERE session_id=? AND kind IN ('transcript','assistant_reply','followup_spoken','context_note','task_log') "
                "ORDER BY id DESC LIMIT ?",
                (self.session_id, recent_limit),
            ).fetchall()
            program_rows = con.execute(
                "SELECT workspace_id, path, title, aliases_json, program_json, status FROM coding_workspaces "
                "WHERE program_json IS NOT NULL AND program_json!='{}' "
                "ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
        events = [dict(row) for row in rows]
        events.reverse()
        front_note = self.front_note_state()
        front_note_context = str((front_note.get("context") or {}).get("text") or "").strip()
        payload: dict[str, Any] = {
            "summary": str(session["summary"] or "") if session else "",
            "front_note_context": front_note_context or self.stored_context_snapshot_text(),
            "recent_events": events,
        }
        if session is not None:
            payload["user_preferences"] = json.loads(session["user_preferences_json"] or "[]")
            payload["open_threads"] = json.loads(session["open_threads_json"] or "[]")
        registered_programs: list[dict[str, Any]] = []
        for row in program_rows:
            program = json.loads(row["program_json"] or "{}")
            if not isinstance(program, dict) or not program.get("open_method"):
                continue
            aliases = json.loads(row["aliases_json"] or "[]")
            registered_programs.append(
                {
                    "workspace_id": row["workspace_id"],
                    "path": row["path"],
                    "title": row["title"],
                    "aliases": aliases,
                    "program": {
                        "name": program.get("name") or row["title"],
                        "aliases": program.get("aliases") or aliases,
                        "kind": program.get("kind"),
                        "status": program.get("status") or row["status"],
                    },
                }
            )
        if registered_programs:
            payload["registered_programs"] = registered_programs
        return payload

    def compression_due(self, every_turns: int, every_seconds: int) -> bool:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT turn_count, last_compressed_turn, last_compressed_at FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()
        if row is None:
            return False
        if row["turn_count"] - row["last_compressed_turn"] >= every_turns:
            return True
        return time.time() - float(row["last_compressed_at"] or 0) >= every_seconds and row["turn_count"] > row["last_compressed_turn"]

    def update_compression(self, payload: dict[str, Any]) -> None:
        created_at = time.time()
        event_content = json.dumps(payload, ensure_ascii=False)
        with self._lock, self._connect() as con:
            turn_count = con.execute("SELECT turn_count FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()["turn_count"]
            con.execute(
                "UPDATE sessions SET summary=?, active_tasks_json=?, user_preferences_json=?, open_threads_json=?, "
                "last_compressed_turn=?, last_compressed_at=?, updated_at=? WHERE session_id=?",
                (
                    str(payload.get("summary", "")),
                    json.dumps(payload.get("active_tasks", []), ensure_ascii=False),
                    json.dumps(payload.get("user_preferences", []), ensure_ascii=False),
                    json.dumps(payload.get("open_threads", []), ensure_ascii=False),
                    turn_count,
                    time.time(),
                    time.time(),
                    self.session_id,
                ),
            )
            con.execute(
                "INSERT INTO events(session_id, kind, role, lane, content, metadata_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, "compression", "system", "compressor", event_content, "{}", created_at),
            )
