from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib import request


MANIFEST_DIR = ".gjallarhorn"
MANIFEST_NAME = "workspace.json"
REUSE_THRESHOLD = 0.35


@dataclass
class WorkspaceDecision:
    workspace_id: str
    path: Path
    title: str
    reused: bool
    score: float
    reason: str
    manifest: dict[str, Any]


def manifest_path(workspace: Path) -> Path:
    return workspace / MANIFEST_DIR / MANIFEST_NAME


def wants_new_workspace(text: str) -> bool:
    compact = _compact(text).lower()
    markers = [
        "新建",
        "新做",
        "重新做",
        "从头做",
        "另做",
        "再做一个",
        "再写一个",
        "写一个新的",
        "做一个新的",
        "新的项目",
        "新项目",
        "不要复用",
        "别复用",
    ]
    return any(marker in compact for marker in markers)


def wants_existing_workspace(text: str, args: dict[str, Any] | None = None) -> bool:
    safe_args = args or {}
    task_mode = str(safe_args.get("task_mode") or safe_args.get("mode") or "").strip().lower().replace("-", "_")
    if task_mode in {"extend_existing", "modify_existing", "continue_existing", "reuse", "reuse_existing"}:
        return True
    if task_mode in {"create_new", "new", "compose_with_existing"}:
        return False
    compact = _compact(text).lower()
    if wants_new_workspace(text):
        return False
    markers = [
        "继续",
        "接着",
        "沿着",
        "基于",
        "在原来",
        "在之前",
        "刚才那个",
        "刚才做的",
        "之前那个",
        "已有",
        "现有",
        "原项目",
        "这个项目",
        "给",
        "加",
        "加个",
        "加一个",
        "增加",
        "改",
        "改一下",
        "修改",
        "修",
        "修复",
        "debug",
        "dbug",
        "deubg",
        "degub",
        "调试",
        "排查",
        "优化",
        "重构",
        "更新",
        "完善",
    ]
    if not any(marker in compact for marker in markers):
        return False
    create_markers = ["写个", "写一个", "做个", "做一个", "开发一个", "实现一个", "创建一个"]
    if any(marker in compact for marker in create_markers) and not any(marker in compact for marker in ["继续", "基于", "已有", "现有", "之前", "刚才"]):
        return False
    return True


def title_from_text(text: str, fallback: str = "codex") -> str:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return fallback
    value = re.sub(r"^(帮我|给我|用codex|用 codex|用antigravity|用 antigravity|写一个|写个|做一个|做个|实现|开发)", "", value, flags=re.I).strip()
    return value[:48] or fallback


def slug_from_title(title: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "-", title).strip("-")
    return value[:48] or "codex"


def default_manifest(workspace_id: str, path: Path, title: str, *, aliases: list[str] | None = None) -> dict[str, Any]:
    now = time.time()
    return {
        "workspace_id": workspace_id,
        "path": str(path),
        "title": title,
        "aliases": _unique([title, *(aliases or [])]),
        "tags": _tags_from_text(title),
        "summary": "",
        "capabilities": [],
        "entrypoints": [],
        "services": [],
        "program": {},
        "related_workspace_ids": [],
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }


def read_manifest(path: Path) -> dict[str, Any] | None:
    file_path = manifest_path(path)
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    file_path = manifest_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["path"] = str(path)
    manifest["updated_at"] = time.time()
    file_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return file_path


class CodingWorkspaceIndex:
    def __init__(self, store: Any, workspace_root: Path) -> None:
        self.store = store
        self.workspace_root = workspace_root.resolve()

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        self.sync_from_disk()
        return self.store.coding_workspaces(limit=limit)

    def get(self, workspace_id: str | None = None, path: str | None = None) -> dict[str, Any] | None:
        found = self.store.coding_workspace(workspace_id=workspace_id, path=path)
        if found:
            manifest = read_manifest(Path(found["path"]))
            if manifest:
                found = merge_workspace_record(found, manifest)
        return found

    def upsert(self, manifest: dict[str, Any]) -> str:
        workspace_id = self.store.upsert_coding_workspace(manifest)
        manifest["workspace_id"] = workspace_id
        write_manifest(Path(str(manifest["path"])), manifest)
        return workspace_id

    def sync_from_disk(self) -> None:
        if not self.workspace_root.exists():
            return
        for file_path in self.workspace_root.glob(f"*/{MANIFEST_DIR}/{MANIFEST_NAME}"):
            manifest = read_manifest(file_path.parents[1])
            if manifest:
                self.store.upsert_coding_workspace(manifest)

    def update_from_files(self, workspace: Path, *, title: str = "", summary: str = "", services: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        existing = read_manifest(workspace) or {}
        workspace_id = str(existing.get("workspace_id") or uuid.uuid4())
        resolved_title = str(existing.get("title") or title or workspace.name)
        manifest = default_manifest(workspace_id, workspace, resolved_title, aliases=list(existing.get("aliases") or []))
        manifest.update(existing)
        manifest["title"] = resolved_title
        if summary:
            manifest["summary"] = summary
        manifest["tags"] = _unique([*(manifest.get("tags") or []), *_tags_from_text(" ".join([resolved_title, summary]))])
        manifest["entrypoints"] = discover_entrypoints(workspace)
        manifest["capabilities"] = _unique([*(manifest.get("capabilities") or []), *_capabilities_from_files(workspace, summary)])
        if services is not None:
            manifest["services"] = services
        manifest["last_task_at"] = time.time()
        self.upsert(manifest)
        return manifest

    def dedupe_related(self, *, threshold: float = 0.55, limit: int = 200) -> list[dict[str, Any]]:
        """Mark strongly related workspaces without deleting or moving files."""
        workspaces = self.list(limit=limit)
        updates: list[dict[str, Any]] = []
        for index, left in enumerate(workspaces):
            for right in workspaces[index + 1 :]:
                score = max(
                    workspace_score(_workspace_identity_text(left), right),
                    workspace_score(_workspace_identity_text(right), left),
                )
                if score < threshold:
                    continue
                canonical, duplicate = _canonical_workspace(left, right)
                canonical_manifest = read_manifest(Path(str(canonical["path"]))) or canonical
                duplicate_manifest = read_manifest(Path(str(duplicate["path"]))) or duplicate
                canonical_id = str(canonical_manifest.get("workspace_id") or canonical.get("workspace_id") or "")
                duplicate_id = str(duplicate_manifest.get("workspace_id") or duplicate.get("workspace_id") or "")
                if not canonical_id or not duplicate_id or canonical_id == duplicate_id:
                    continue
                canonical_related = _unique([*(canonical_manifest.get("related_workspace_ids") or []), duplicate_id, *(duplicate_manifest.get("related_workspace_ids") or [])])
                duplicate_related = _unique([*(duplicate_manifest.get("related_workspace_ids") or []), canonical_id])
                if canonical_related != canonical_manifest.get("related_workspace_ids"):
                    canonical_manifest["related_workspace_ids"] = canonical_related
                    self.upsert(canonical_manifest)
                    updates.append({"canonical": canonical_id, "related": duplicate_id, "score": round(score, 3)})
                if duplicate_related != duplicate_manifest.get("related_workspace_ids"):
                    duplicate_manifest["related_workspace_ids"] = duplicate_related
                    self.upsert(duplicate_manifest)
        return updates


class CodingWorkspaceSelector:
    def __init__(self, index: CodingWorkspaceIndex, workspace_root: Path) -> None:
        self.index = index
        self.workspace_root = workspace_root.resolve()

    def select(self, *, target: str, prompt: str, args: dict[str, Any], run_id: str) -> WorkspaceDecision:
        text = " ".join(part for part in [target, prompt] if part).strip()
        explicit_workspace_id = str(args.get("workspace_id") or "").strip()
        explicit_cwd = str(args.get("cwd") or "").strip()
        if explicit_workspace_id:
            found = self.index.get(workspace_id=explicit_workspace_id)
            if found:
                return self._decision(found, reused=True, score=1.0, reason="explicit workspace_id")
        if explicit_cwd:
            path = Path(explicit_cwd).expanduser().resolve()
            if not path.exists() or not path.is_dir():
                raise ValueError("cwd does not exist or is not a directory")
            manifest = read_manifest(path) or default_manifest(str(uuid.uuid4()), path, title_from_text(text, path.name), aliases=[target, prompt])
            self.index.upsert(manifest)
            return self._decision(manifest, reused=True, score=1.0, reason="explicit cwd")

        self.index.dedupe_related()
        explicit_existing = wants_existing_workspace(text, args)
        explicit_new = wants_new_workspace(text) or _task_mode(args) in {"create_new", "new", "compose_with_existing"}
        if not explicit_new:
            matches = rank_workspaces(text, [workspace for workspace in self.index.list(limit=100) if workspace_is_active(workspace)])
            if matches and matches[0][0] >= REUSE_THRESHOLD:
                score, found = matches[0]
                reason = "explicit existing intent fuzzy match" if explicit_existing else "aggressive fuzzy reuse"
                return self._decision(found, reused=True, score=score, reason=reason)

        title = title_from_text(target or prompt)
        path = self.workspace_root / f"{run_id}-{slug_from_title(title)}"
        path.mkdir(parents=True, exist_ok=True)
        # Do not put the full prompt into aliases. Prompts often mention dependencies
        # such as "跟已有 map app 交互", which would make future fuzzy matching
        # resolve the dependency instead of the app being created.
        manifest = default_manifest(str(uuid.uuid4()), path, title, aliases=[target])
        self.index.upsert(manifest)
        return self._decision(manifest, reused=False, score=0.0, reason="created new workspace")

    @staticmethod
    def _decision(record: dict[str, Any], *, reused: bool, score: float, reason: str) -> WorkspaceDecision:
        path = Path(str(record["path"])).expanduser().resolve()
        manifest = read_manifest(path) or record
        return WorkspaceDecision(
            workspace_id=str(manifest.get("workspace_id") or record.get("workspace_id") or ""),
            path=path,
            title=str(manifest.get("title") or record.get("title") or path.name),
            reused=reused,
            score=float(score),
            reason=reason,
            manifest=merge_workspace_record(record, manifest),
        )


class CodingServiceRegistry:
    def __init__(self, index: CodingWorkspaceIndex) -> None:
        self.index = index

    def probe_services(self, workspace: Path) -> list[dict[str, Any]]:
        manifest = read_manifest(workspace) or {}
        services = manifest.get("services") if isinstance(manifest.get("services"), list) else []
        checked: list[dict[str, Any]] = []
        for service in services:
            if not isinstance(service, dict):
                continue
            service = dict(service)
            url = str(service.get("url") or "")
            if not _is_localhost_url(url):
                service["status"] = "ignored"
            else:
                service["status"] = "up" if _probe_service(url, str(service.get("health_path") or "/health")) else "down"
                service["last_seen"] = time.time()
            checked.append(service)
        if checked:
            current = read_manifest(workspace) or {}
            current["services"] = checked
            self.index.upsert(current)
        return checked


def rank_workspaces(text: str, workspaces: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    scored = [(workspace_score(text, workspace), workspace) for workspace in workspaces]
    scored = [(score, workspace) for score, workspace in scored if score > 0]
    scored.sort(key=lambda item: (item[0], float(item[1].get("last_task_at") or item[1].get("updated_at") or 0)), reverse=True)
    return scored


def workspace_is_reusable(workspace: dict[str, Any]) -> bool:
    if not workspace_is_active(workspace):
        return False
    program = workspace.get("program") if isinstance(workspace.get("program"), dict) else {}
    if program.get("open_method"):
        return True
    entrypoints = workspace.get("entrypoints") if isinstance(workspace.get("entrypoints"), list) else []
    if entrypoints:
        return True
    path_text = str(workspace.get("path") or "")
    if not path_text:
        return False
    path = Path(path_text)
    for name in ("run.sh", "app.py", "main.py"):
        if (path / name).exists():
            return True
    try:
        for child in path.iterdir():
            if child.name in {MANIFEST_DIR, "pyproject.toml"} or child.name.startswith(".voice_"):
                continue
            return True
    except Exception:
        return False
    return False


def workspace_is_active(workspace: dict[str, Any]) -> bool:
    status = str(workspace.get("status") or "").strip().lower()
    if status in {"failed", "archived", "deleted"}:
        return False
    return True


def workspace_score(text: str, workspace: dict[str, Any]) -> float:
    normalized_text = _workspace_match_text(text)
    query_tokens = set(_tokens(normalized_text))
    if not query_tokens:
        return 0.0
    haystack_parts = [
        workspace.get("title", ""),
        workspace.get("summary", ""),
        " ".join(workspace.get("aliases") or []),
        " ".join(workspace.get("tags") or []),
        " ".join(workspace.get("capabilities") or []),
        Path(str(workspace.get("path") or "")).name,
    ]
    haystack = " ".join(str(part) for part in haystack_parts)
    hay_tokens = set(_tokens(haystack))
    if not hay_tokens:
        return 0.0
    common = len(query_tokens & hay_tokens)
    overlap = max(
        common / max(1, len(query_tokens)),
        common / max(1, len(hay_tokens)),
    )
    compact_query = _compact(normalized_text)
    compact_title = _compact(workspace.get("title", ""))
    substring_bonus = 0.25 if compact_title and (compact_title in compact_query or compact_query in compact_title) else 0.0
    if not substring_bonus and any(len(token) >= 3 and token in compact_query for token in hay_tokens):
        substring_bonus = 0.25
    hay_compacts = [_compact(part) for part in haystack_parts if _compact(part)]
    similarity = max((SequenceMatcher(None, compact_query, item).ratio() for item in hay_compacts), default=0.0)
    similarity_bonus = 0.35 * similarity if similarity >= 0.45 else 0.0
    return min(1.0, overlap + substring_bonus + similarity_bonus)


def discover_entrypoints(workspace: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    run_script = workspace / "run.sh"
    if run_script.exists():
        entries.append({"type": "script", "path": "run.sh", "role": "app"})
    for name in ["app.py", "main.py"]:
        path = workspace / name
        if path.exists():
            entries.append({"type": "python", "path": name, "role": "app"})
    for file_path in sorted(workspace.glob("*.py")):
        if file_path.name in {"app.py", "main.py"}:
            continue
        entries.append({"type": "python", "path": file_path.name, "role": "module"})
    return entries[:12]


def merge_workspace_record(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    merged = dict(record)
    for key in ["workspace_id", "path", "title", "aliases", "tags", "summary", "capabilities", "entrypoints", "services", "program", "related_workspace_ids", "status", "last_task_at"]:
        value = manifest.get(key)
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def workspace_context_for_prompt(manifest: dict[str, Any]) -> str:
    services = manifest.get("services") or []
    program = manifest.get("program") if isinstance(manifest.get("program"), dict) else {}
    open_method = program.get("open_method") if isinstance(program.get("open_method"), dict) else {}
    window_match = program.get("window_match") if isinstance(program.get("window_match"), dict) else {}
    compact_program = {
        "name": program.get("name"),
        "aliases": _limit_texts(program.get("aliases") or [], 8),
        "kind": program.get("kind"),
        "status": program.get("status"),
        "open_method": {
            "type": open_method.get("type"),
            "entrypoint": open_method.get("entrypoint"),
            "argv": _limit_texts(open_method.get("argv") or [], 8),
        }
        if open_method
        else {},
        "window_match": {
            "app_names": _limit_texts(window_match.get("app_names") or [], 5),
            "title_keywords": _limit_texts(window_match.get("title_keywords") or [], 8),
        }
        if window_match
        else {},
    }
    return json.dumps(
        {
            "workspace_id": manifest.get("workspace_id"),
            "title": manifest.get("title"),
            "aliases": _limit_texts(manifest.get("aliases") or [], 8),
            "summary": _clip_text(str(manifest.get("summary") or ""), 360),
            "tags": _limit_texts(manifest.get("tags") or [], 12),
            "capabilities": _limit_texts(manifest.get("capabilities") or [], 12),
            "entrypoints": _compact_entrypoints(manifest.get("entrypoints") or []),
            "services": _compact_services(services),
            "program": compact_program,
        },
        ensure_ascii=False,
        indent=2,
    )


def _clip_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _limit_texts(values: Any, limit: int) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = _clip_text(str(value), 80)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _compact_entrypoints(values: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for value in values or []:
        if isinstance(value, dict):
            items.append({
                "type": _clip_text(str(value.get("type") or ""), 40),
                "path": _clip_text(str(value.get("path") or ""), 120),
                "role": _clip_text(str(value.get("role") or ""), 40),
            })
        elif value:
            items.append({"path": _clip_text(str(value), 120)})
        if len(items) >= 12:
            break
    return items


def _compact_services(values: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for service in values or []:
        if not isinstance(service, dict) or service.get("status") not in {"up", "unknown", None}:
            continue
        items.append({
            "name": _clip_text(str(service.get("name") or ""), 80),
            "url": _clip_text(str(service.get("url") or ""), 160),
            "health_path": _clip_text(str(service.get("health_path") or ""), 80),
            "capabilities": _limit_texts(service.get("capabilities") or [], 8),
            "status": service.get("status") or "unknown",
        })
        if len(items) >= 8:
            break
    return items


def register_workspace_program(
    manifest: dict[str, Any],
    *,
    entry: str | Path | None = None,
    active_venv: str | Path | None = None,
    launch_result: dict[str, Any] | None = None,
    status: str = "ready",
) -> dict[str, Any]:
    """Attach a deterministic app-open contract to a workspace manifest."""
    updated = dict(manifest)
    workspace = Path(str(updated.get("path") or ".")).expanduser()
    workspace_id = str(updated.get("workspace_id") or "")
    title = str(updated.get("title") or workspace.name or "app").strip()
    aliases = _workspace_aliases(title, updated, extra=[workspace.name])
    entry_path = Path(str(entry or "app.py"))
    entry_name = entry_path.name
    venv_text = str(active_venv or "")
    if entry_name == "run.sh" or entry_path.suffix == ".sh":
        open_method = {
            "type": "script",
            "cwd": str(workspace),
            "entrypoint": entry_name,
            "argv": ["bash", entry_name],
            "env": {"VIRTUAL_ENV": venv_text} if venv_text else {},
        }
    else:
        open_method = {
            "type": "uv_python",
            "cwd": str(workspace),
            "entrypoint": entry_name,
            "argv": ["uv", "run", "--active", "--no-sync", "python", entry_name],
            "env": {"VIRTUAL_ENV": venv_text} if venv_text else {},
        }
    last_launch = dict(launch_result or {})
    existing_program = updated.get("program") if isinstance(updated.get("program"), dict) else {}
    existing_window_match = existing_program.get("window_match") if isinstance(existing_program.get("window_match"), dict) else None
    window_keywords = _unique([title, *aliases])[:10]
    program = {
        "program_id": f"coding:{workspace_id or workspace.name}",
        "workspace_id": workspace_id,
        "name": title,
        "aliases": aliases,
        "kind": "coding_app",
        "open_method": open_method,
        "window_match": existing_window_match or {
            "app_name": "python3",
            "app_names": ["python3", "Python"],
            "title_keywords": window_keywords,
        },
        "service": None,
        "capabilities": _unique([*(updated.get("capabilities") or []), "desktop_app"]),
        "last_launch": last_launch,
        "status": status,
        "updated_at": time.time(),
    }
    updated["aliases"] = aliases
    updated["program"] = program
    updated["status"] = status
    return updated


def resolve_workspace_program(query: str, workspaces: list[dict[str, Any]], *, threshold: float = 0.35) -> dict[str, Any] | None:
    ranked = rank_workspaces(query, workspaces)
    for score, workspace in ranked:
        program = workspace.get("program") if isinstance(workspace.get("program"), dict) else {}
        if score >= threshold and program.get("open_method"):
            return {**workspace, "score": round(score, 3), "program": program}
    return None


def _probe_service(url: str, health_path: str) -> bool:
    endpoint = url.rstrip("/") + "/" + health_path.lstrip("/")
    try:
        with request.urlopen(endpoint, timeout=1.0) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


def _is_localhost_url(url: str) -> bool:
    return url.startswith("http://127.0.0.1") or url.startswith("http://localhost") or url.startswith("http://[::1]")


def _capabilities_from_files(workspace: Path, summary: str) -> list[str]:
    snippets: list[str] = [summary, " ".join(path.name for path in workspace.glob("*"))]
    for name in ("app.py", "main.py", "index.html"):
        file_path = workspace / name
        if file_path.exists() and file_path.is_file():
            try:
                snippets.append(file_path.read_text(encoding="utf-8", errors="replace")[:8000])
            except Exception:
                pass
    text = " ".join(snippets).lower()
    caps: list[str] = []
    if "webview" in text or (workspace / "app.py").exists() or (workspace / "main.py").exists():
        caps.append("desktop_app")
    if "import webview" in text or "webview.create_window" in text or "pywebview" in text:
        caps.append("pywebview_app")
    if "game" in text or "游戏" in summary or "pac" in text or "tetris" in text:
        caps.append("game")
    if "fastapi" in text or "flask" in text or "http" in text:
        caps.append("local_service")
    return caps


def _task_mode(args: dict[str, Any] | None) -> str:
    safe_args = args or {}
    return str(safe_args.get("task_mode") or safe_args.get("mode") or "").strip().lower().replace("-", "_")


def _workspace_identity_text(workspace: dict[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in [
            workspace.get("title"),
            " ".join(workspace.get("aliases") or []),
            " ".join(workspace.get("tags") or []),
            workspace.get("summary"),
            Path(str(workspace.get("path") or "")).name,
        ]
    )


def _canonical_workspace(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    def priority(workspace: dict[str, Any]) -> tuple[int, float]:
        program = workspace.get("program") if isinstance(workspace.get("program"), dict) else {}
        has_open_method = 1 if program.get("open_method") else 0
        ready = 1 if str(workspace.get("status") or "") == "ready" else 0
        recent = float(workspace.get("last_task_at") or workspace.get("updated_at") or 0)
        return (has_open_method + ready, recent)

    return (left, right) if priority(left) >= priority(right) else (right, left)


def _workspace_aliases(title: str, manifest: dict[str, Any], *, extra: list[Any] | None = None) -> list[str]:
    normalized = _workspace_match_text(title)
    aliases = _unique([title, normalized, *(manifest.get("aliases") or []), *(extra or [])])
    variants: list[str] = []
    for alias in aliases:
        compact = _compact(alias)
        if compact:
            variants.append(compact)
        if "大战" in compact:
            variants.append(compact.replace("大战", "大转"))
        if "大战" in compact:
            variants.append(compact.replace("大战", "大栈"))
    return _unique([*aliases, *variants])


def _tags_from_text(text: str) -> list[str]:
    tokens = _tokens(text)
    return _unique([token for token in tokens if len(token) >= 2][:12])


def _tokens(text: Any) -> list[str]:
    compact = _compact(text).lower()
    if not compact:
        return []
    latin = re.findall(r"[a-z0-9]{2,}", compact)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", compact)
    chunks: list[str] = []
    for value in chinese:
        chunks.append(value)
        chunks.extend(value[i : i + 2] for i in range(0, max(0, len(value) - 1)))
        chunks.extend(value[i : i + 3] for i in range(0, max(0, len(value) - 2)))
    return _unique([*latin, *chunks])


def _workspace_match_text(text: Any) -> str:
    value = _compact(text)
    value = re.sub(r"(?i)(debug|dbug|deubg|degub|develop|implement|refactor|build)", "", value)
    value = re.sub(
        r"(帮我|给我|请|一下|一个|这个|那个|目前|现在|还有|没有|显示|开发|实现|新增|修复|重构|优化|调试|排查|继续|接着|沿着|基于|在原来|在之前|刚才那个|刚才做的|之前那个|已有|现有|原项目|项目|代码|应用|游戏)",
        "",
        value,
    )
    return value or _compact(text)


def _compact(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def _unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
