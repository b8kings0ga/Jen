from __future__ import annotations

import difflib
import re
import subprocess
import time
from typing import Any

from voice_assistant.json_utils import parse_jsonish_value
from voice_assistant.speech_text import compact_speech_text

DEFAULT_BROWSER_APP = "Google Chrome"


def _plan_step_order(step: dict[str, Any]) -> int:
    try:
        return int(step.get("order"))
    except (TypeError, ValueError):
        return 999

def camera_capture_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    if any(token in compact for token in ["photobooth", "photo booth"]) and not any(
        token in compact for token in ["拍照", "抓拍", "照一张", "拍一张", "snapshot", "takephoto", "takepicture", "capturephoto"]
    ):
        return False
    capture_tokens = [
        "拍照",
        "抓拍",
        "照相",
        "拍一张",
        "照一张",
        "拍张",
        "照张",
        "拍个照",
        "拍一下",
        "snapshot",
        "takephoto",
        "take photo",
        "takepicture",
        "take picture",
        "takeaphoto",
        "capturephoto",
        "capture photo",
    ]
    return any(token in compact for token in capture_tokens)


def camera_app_open_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    camera_tokens = ["camera", "相机", "摄像头", "镜头", "前置镜头", "前置摄像头"]
    open_tokens = ["打开", "启动", "开一下", "开开", "open", "launch", "activate"]
    return any(token in compact for token in camera_tokens) and any(token in compact for token in open_tokens)


def photo_booth_app_open_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    open_tokens = ["打开", "启动", "开一下", "开开", "open", "launch", "activate"]
    return any(token in compact for token in ["photobooth", "photo booth"]) and any(token in compact for token in open_tokens)


def camera_tool_intent_error(tool_name: str, user_text: str) -> dict[str, Any] | None:
    name = str(tool_name or "").split(":", 1)[-1]
    if name != "capture_camera_snapshot":
        return None
    if camera_app_open_requested(user_text) and not camera_capture_requested(user_text):
        return {
            "ok": False,
            "error": "camera app open requested, not camera snapshot",
            "instruction": 'The user asked to open Camera.app or camera preview, not take a photo. Call run_osascript with script: tell application "Camera" to activate',
        }
    return None


def workspace_arrange_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    arrange_tokens = ["排窗口", "排一下", "排列", "分屏", "并排", "并拍", "铺开", "摆出来", "顺序切", "陀螺", "切屏幕", "铺满屏幕", "窗口布局", "windowarrangement", "arrangewindow", "arrangeworkspace", "focusscreen", "focus", "聚焦", "前台", "调到前台", "置前"]
    window_tokens = ["窗口", "屏幕", "screen", "chrome", "codex", "camera", "相机", "photobooth", "youtube", "video", "视频", "music", "音乐", "浏览器", "应用"]
    return any(token in compact for token in arrange_tokens) and any(token in compact for token in window_tokens)


def front_note_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    return any(token in compact for token in ["当前note", "前端note", "frontnote", "contextnote", "上下文note", "note里", "note写", "贴个note", "贴个便签", "便签", "便签纸", "编签", "便利贴", "显示卡片", "屏幕上贴", "浮动note"])


def front_note_context_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    return any(token in compact for token in ["contextnote", "上下文note", "上下文信息", "context信息", "放进context", "写进context"])


def long_term_memory_requested(text: str) -> bool:
    compact = compact_speech_text(str(text or "")).lower()
    return any(token in compact for token in ["记住", "记得", "以后记得", "帮我记一下", "帮我记住", "rememberthis", "rememberthat"])


def extract_front_note_content(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = [
        r".*?(?:在当前\s*note\s*写下|在当前\s*note\s*写|当前\s*note\s*写下|当前\s*note\s*写|在\s*note\s*里写下|在\s*note\s*里写|写到\s*note\s*里|贴个\s*note\s*写|贴个便签写|便签写|写进便签|写进编签|编签)(.+)$",
        r".*?(?:front\s*note|前端\s*note|浮动\s*note|屏幕上贴个\s*note|屏幕上贴个便签|显示卡片)(?:[:：，, ]*)(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ：:，,。")
            if value:
                return value
    return raw


def normalize_front_note_call_args(arguments: dict[str, Any], user_text: str) -> dict[str, Any]:
    safe_args = dict(arguments or {})
    action = str(safe_args.get("action") or "").strip().lower()
    if action not in {"show", "hide", "update", "append", "clear", "pin_edge"}:
        action = "update"
    compact = compact_speech_text(user_text).lower()
    tab = str(safe_args.get("tab") or "").strip().lower()
    if tab not in {"live", "context"}:
        tab = "context" if front_note_context_requested(user_text) else "live"
    elif tab == "context" and not front_note_context_requested(user_text):
        tab = "live"
    content = str(safe_args.get("content") or "").strip()
    if not content and action in {"show", "update", "append"}:
        content = extract_front_note_content(user_text)
    html_value = str(safe_args.get("html") or "").strip()
    position = str(safe_args.get("position") or "").strip().lower()
    if position not in {"left", "right", "center"}:
        position = "left" if any(token in compact for token in ["左边", "左侧", "left"]) else "right"
    return {
        "action": action,
        "tab": tab,
        "content": content,
        "html": html_value,
        "media": safe_args.get("media") or [],
        "position": position,
        "visible": bool(safe_args.get("visible", True)),
        "width": max(360, min(int(safe_args.get("width") or 520), 980)),
        "height": max(280, min(int(safe_args.get("height") or 420), 780)),
    }


def planned_arrange_workspace_arguments(text: str) -> dict[str, Any]:
    compact = compact_speech_text(str(text or "")).lower()
    mode = "parallel" if any(token in compact for token in ["并排", "并拍", "parallel", "等宽", "focusscreen", "focus"]) else "auto"
    terms = window_query_terms(text)
    return {"query": text, "app_names": terms, "mode": mode, "open_if_missing": True, "max_windows": 4}


def normalize_arrange_workspace_call_args(arguments: dict[str, Any], plan_payload: dict[str, Any] | None, user_text: str) -> dict[str, Any]:
    safe_args = dict(arguments or {})
    planned_from_user = planned_arrange_workspace_arguments(user_text)
    planned = planned_arrange_args_from_plan(plan_payload) or {}
    current_terms = window_query_terms(str(safe_args.get("query") or ""), safe_args.get("app_names"))
    planned_terms = normalize_workspace_app_names(planned.get("app_names")) or normalize_workspace_app_names(planned_from_user.get("app_names"))
    merged_terms = merge_workspace_terms(current_terms, planned_terms, window_query_terms(user_text))
    if merged_terms:
        safe_args["app_names"] = merged_terms
    if not str(safe_args.get("query") or "").strip():
        safe_args["query"] = planned.get("query") or planned_from_user.get("query") or user_text
    raw_mode = str(safe_args.get("mode") or "").strip().lower()
    if not raw_mode or raw_mode in {"focus", "front", "foreground", "前台", "置前", "聚焦"}:
        planned_mode = str(planned.get("mode") or "").strip().lower()
        if planned_mode in {"focus", "front", "foreground", "前台", "置前", "聚焦"}:
            planned_mode = ""
        safe_args["mode"] = planned_mode or planned_from_user.get("mode") or "auto"
    if "open_if_missing" not in safe_args:
        safe_args["open_if_missing"] = bool(planned.get("open_if_missing", planned_from_user.get("open_if_missing", True)))
    if "max_windows" not in safe_args:
        planned_count = len(merged_terms) if merged_terms else int(planned.get("max_windows") or planned_from_user.get("max_windows") or 4)
        safe_args["max_windows"] = max(1, min(planned_count or 4, 4))
    return safe_args


def merge_workspace_terms(*groups: list[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for term in group or []:
            key = compact_speech_text(str(term or "")).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            terms.append(str(term))
    return terms


def planned_arrange_args_from_plan(plan_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan_payload, dict):
        return None
    raw_steps = plan_payload.get("steps")
    if not isinstance(raw_steps, list):
        return None
    for step in sorted((item for item in raw_steps if isinstance(item, dict)), key=_plan_step_order):
        raw_arguments = step.get("arguments")
        if not isinstance(raw_arguments, dict):
            continue
        direct = raw_arguments.get("arrange_workspace")
        if isinstance(direct, dict):
            return direct
    return None


def osascript_looks_like_window_layout(arguments: dict[str, Any]) -> bool:
    script = str((arguments or {}).get("script") or "")
    purpose = str((arguments or {}).get("purpose") or "")
    text = compact_speech_text(f"{purpose}\n{script}").lower()
    if not text:
        return False
    layout_tokens = [
        "setpositionofwindow",
        "setsizeofwindow",
        "setboundsofwindow",
        "setpositionoffrontwindow",
        "setsizeoffrontwindow",
        "setboundsoffrontwindow",
        "并排",
        "分屏",
        "调整chrome窗口",
        "调整photobooth窗口",
        "调整窗口",
        "移动窗口",
    ]
    return any(token in text for token in layout_tokens)


def normalize_browser_osascript(script: str) -> str:
    script = str(script or "")
    if "Safari" not in script:
        return script
    replacements = {
        'application "Safari"': f'application "{DEFAULT_BROWSER_APP}"',
        "application 'Safari'": f"application '{DEFAULT_BROWSER_APP}'",
        'process "Safari"': f'process "{DEFAULT_BROWSER_APP}"',
        "process 'Safari'": f"process '{DEFAULT_BROWSER_APP}'",
    }
    for old, new in replacements.items():
        script = script.replace(old, new)
    return script


WINDOW_APP_ALIASES: dict[str, list[str]] = {
    "chrome": ["Google Chrome"],
    "googlechrome": ["Google Chrome"],
    "browser": ["Google Chrome"],
    "web": ["Google Chrome"],
    "网页": ["Google Chrome"],
    "浏览器": ["Google Chrome"],
    "youtube": ["Google Chrome"],
    "视频": ["Google Chrome"],
    "safari": ["Google Chrome"],
    "codex": ["Codex"],
    "code": ["Code"],
    "vscode": ["Code"],
    "music": ["Music"],
    "音乐": ["Music"],
    "wechat": ["WeChat"],
    "微信": ["WeChat"],
    "photobooth": ["Photo Booth"],
    "footbooth": ["Photo Booth"],
    "footboot": ["Photo Booth"],
    "footboots": ["Photo Booth"],
    "photoboot": ["Photo Booth"],
    "photoboots": ["Photo Booth"],
    "photo booth": ["Photo Booth"],
    "camera": ["Camera", "Photo Booth"],
    "相机": ["Camera", "Photo Booth"],
    "摄像头": ["Camera", "Photo Booth"],
    "前置镜头": ["Camera", "Photo Booth"],
}

WORKSPACE_LAUNCHABLE_APPS = [
    "Google Chrome",
    "Codex",
    "Code",
    "Music",
    "WeChat",
    "Photo Booth",
    "Camera",
]


def window_query_terms(query: str, app_names: Any = None) -> list[str]:
    raw_terms: list[str] = []
    explicit_terms = normalize_workspace_app_names(app_names)
    raw_terms.extend(explicit_terms)
    query_text = str(query or "").strip()
    query_compact = compact_speech_text(query_text).lower()
    if query_text and not explicit_terms:
        query_text = re.sub(r"photo\s+booth", "photobooth", query_text, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"(把|这些|几个|窗口|屏幕|排一下|排列|排到前台|调到前台|置前|前台|分屏|显示|摆出来|铺开|陀螺|顺序切|并排|并拍|左边|右边|全屏|放到|切到|打开|应用|切一下|然后|然后再|再|接着|他们|它们)",
            " ",
            query_text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[+＋&＆/／]+", " ", cleaned)
        raw_terms.extend(part.strip() for part in re.split(r"[,，、\s和跟与]+", cleaned) if part.strip())
    terms: list[str] = []
    for term in raw_terms:
        lowered = compact_speech_text(term).lower()
        if lowered in {"", "+", "＋", "&", "＆", "/", "／", "and", "then", "然后", "然后再", "再", "接着", "它们", "他们", "them", "都", "全部", "focus", "screen", "focusscreen", "前台", "排到前台", "调到前台", "置前"} or "它们" in lowered or "他们" in lowered:
            continue
        if lowered == "music" and any(token in query_compact for token in ["musicvideo", "视频", "youtube", "video"]):
            continue
        aliases = WINDOW_APP_ALIASES.get(lowered)
        if aliases:
            terms.extend(aliases)
        elif "photobooth" in lowered:
            terms.append("Photo Booth")
        elif lowered in {"footboot", "footboots", "footbooth", "photoboot", "photoboots"}:
            terms.append("Photo Booth")
        elif lowered in {"web", "网页"}:
            terms.append("Google Chrome")
        elif any(token in lowered for token in ["youtube", "视频", "video"]) or (
            any(token in lowered for token in ["trump", "ymca", "mca", "特朗普", "川普"]) and any(token in query_compact for token in ["youtube", "视频", "video"])
        ):
            terms.append("Google Chrome")
        elif term:
            terms.append(term)
    if query_compact:
        if any(token in query_compact for token in ["web", "网页", "浏览器", "youtube", "视频", "video"]):
            terms.append("Google Chrome")
        if any(token in query_compact for token in ["photobooth", "photoboot", "photoboots", "footbooth", "footboot", "footboots"]):
            terms.append("Photo Booth")
    seen: set[str] = set()
    return [term for term in terms if not (term.lower() in seen or seen.add(term.lower()))]


def workspace_terms_are_pronouns(value: Any) -> bool:
    terms = normalize_workspace_app_names(value)
    if not terms:
        return True
    useful = 0
    for term in terms:
        lowered = compact_speech_text(term).lower()
        if lowered in {"然后", "then", "它们", "他们", "them", "focus", "screen", "focusscreen"}:
            continue
        if "它们" in lowered or "他们" in lowered:
            continue
        useful += 1
    return useful == 0


def normalize_workspace_app_names(app_names: Any = None) -> list[str]:
    value = parse_jsonish_value(app_names)
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,，、]+|(?:\s+(?:and|和|跟|与)\s+)", value, flags=re.IGNORECASE) if part.strip()]
    if isinstance(value, dict):
        for key in ("app_names", "apps", "names", "windows"):
            if key in value:
                return normalize_workspace_app_names(value.get(key))
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for part in value:
            if isinstance(part, dict):
                candidate = part.get("app_name") or part.get("name") or part.get("title")
                if candidate:
                    out.append(str(candidate).strip())
            elif str(part or "").strip():
                out.append(str(part).strip())
        return [item for item in out if item]
    return [str(value).strip()] if str(value or "").strip() else []


def normalize_workspace_mode(mode: Any = "auto") -> str:
    raw = str(mode or "auto").strip().lower()
    text = compact_speech_text(raw).lower()
    if raw in {"side-by-side", "side_by_side"} or text in {"parallel", "sidebyside", "split", "splitscreen", "columns", "column", "并排", "分屏", "左右", "左右并排"}:
        return "parallel"
    return "auto"


def visible_process_names(timeout_seconds: float = 2.0) -> list[str]:
    try:
        proc = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get name of application processes whose visible is true'],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return []
    if proc.returncode != 0:
        return []
    return [part.strip() for part in (proc.stdout or "").strip().split(",") if part.strip()]


def target_process_names(terms: list[str], visible_names: list[str], max_apps: int = 12) -> list[str]:
    if not terms:
        return visible_names[:max_apps]
    scored: list[tuple[int, int, str]] = []
    for term in terms:
        term_key = compact_speech_text(term).lower()
        aliases = WINDOW_APP_ALIASES.get(term_key, [term])
        for alias in aliases:
            alias_key = compact_speech_text(alias).lower()
            if not alias_key:
                continue
            for app_name in visible_names:
                app_key = compact_speech_text(app_name).lower()
                score = fuzzy_workspace_score(alias_key, app_key, app_key)
                if score >= 72:
                    scored.append((score, -visible_names.index(app_name), app_name))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    seen: set[str] = set()
    return [name for _, _, name in scored if not (name.lower() in seen or seen.add(name.lower()))][:max_apps]


def enumerate_app_windows(app_name: str, timeout_seconds: float = 5.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    script = f'''
    tell application "System Events"
      set out to ""
      tell application process {applescript_quote(app_name)}
        set winIndex to 0
        repeat with w in windows
          set winIndex to winIndex + 1
          set winName to name of w as text
          set winPos to position of w
          set winSize to size of w
          set out to out & name & "|||" & winIndex & "|||" & winName & "|||" & (item 1 of winPos) & "|||" & (item 2 of winPos) & "|||" & (item 1 of winSize) & "|||" & (item 2 of winSize) & linefeed
        end repeat
      end tell
      return out
    end tell
    '''
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout_seconds)
    except Exception as exc:
        return [], {"app_name": app_name, "error": str(exc)}
    if proc.returncode != 0:
        return [], {"app_name": app_name, "returncode": proc.returncode, "stderr": (proc.stderr or "").strip()[:1000]}
    windows: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("|||")
        if len(parts) != 7:
            continue
        app_name, index_text, title, x_text, y_text, width_text, height_text = parts
        try:
            windows.append({
                "app_name": app_name.strip(),
                "window_index": int(index_text),
                "title": title.strip(),
                "position": [int(float(x_text)), int(float(y_text))],
                "size": [int(float(width_text)), int(float(height_text))],
            })
        except ValueError:
            continue
    return windows, {}


def enumerate_visible_windows(terms: list[str] | None = None, max_apps: int = 12) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    visible_names = visible_process_names()
    if not visible_names:
        return [], {"error": "no visible application processes"}
    app_names = target_process_names(terms or [], visible_names, max_apps=max_apps)
    errors: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for app_name in app_names:
        app_windows, error_info = enumerate_app_windows(app_name)
        if error_info:
            errors.append(error_info)
        windows.extend(app_windows)
    return windows, {"visible_apps": visible_names, "scanned_apps": app_names, "errors": errors[:5]}


def desktop_usable_bounds(timeout_seconds: float = 3.0) -> dict[str, int]:
    fallback = {"x": 0, "y": 38, "width": 1470, "height": 918}
    try:
        proc = subprocess.run(
            ["osascript", "-e", 'tell application "Finder" to get bounds of window of desktop'],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        numbers = [int(float(part.strip())) for part in re.split(r"[, ]+", (proc.stdout or "").strip()) if part.strip()]
        if proc.returncode == 0 and len(numbers) >= 4:
            left, top, right, bottom = numbers[:4]
            usable_top = max(38, top)
            return {"x": left, "y": usable_top, "width": max(1, right - left), "height": max(1, bottom - usable_top)}
    except Exception:
        pass
    return fallback


def match_workspace_windows(windows: list[dict[str, Any]], query: str, app_names: Any, max_windows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    max_windows = max(1, min(int(max_windows or 4), 4))
    terms = window_query_terms(query, app_names)
    if not terms:
        selected = windows[:max_windows]
        return selected, windows[max_windows:max_windows + 8], terms
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for term in terms:
        term_key = compact_speech_text(term).lower()
        best: tuple[int, int, dict[str, Any]] | None = None
        for index, window in enumerate(windows):
            key = workspace_window_key(window)
            if key in seen:
                continue
            haystack = compact_speech_text(f"{window.get('app_name', '')} {window.get('title', '')}").lower()
            app_key = compact_speech_text(str(window.get("app_name") or "")).lower()
            score = fuzzy_workspace_score(term_key, app_key, haystack)
            if score >= 45 and (best is None or (score, -index) > (best[0], best[1])):
                best = (score, -index, window)
        if best is None:
            continue
        window = best[2]
        key = workspace_window_key(window)
        seen.add(key)
        selected.append(window)
        if len(selected) >= max_windows:
            break
    if len(selected) < max_windows:
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, window in enumerate(windows):
            key = workspace_window_key(window)
            if key in seen:
                continue
            haystack = compact_speech_text(f"{window.get('app_name', '')} {window.get('title', '')}").lower()
            app_key = compact_speech_text(str(window.get("app_name") or "")).lower()
            score = max(fuzzy_workspace_score(compact_speech_text(term).lower(), app_key, haystack) for term in terms)
            if score >= 45:
                scored.append((score, -index, window))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        for _, _, window in scored:
            key = workspace_window_key(window)
            if key in seen:
                continue
            seen.add(key)
            selected.append(window)
            if len(selected) >= max_windows:
                break
    selected_keys = {workspace_window_key(window) for window in selected}
    candidates = [window for window in windows if workspace_window_key(window) not in selected_keys][:8]
    return selected, candidates, terms


def workspace_window_key(window: dict[str, Any]) -> tuple[str, int]:
    return (str(window.get("app_name") or ""), int(window.get("window_index") or 0))


def fuzzy_workspace_score(term_key: str, app_key: str, haystack: str) -> int:
    term_key = str(term_key or "").strip().lower()
    app_key = str(app_key or "").strip().lower()
    haystack = str(haystack or "").strip().lower()
    if not term_key:
        return 0
    app_keys = workspace_app_key_variants(app_key)
    if term_key in app_keys:
        return 100
    if any(term_key in key for key in app_keys):
        return 86
    if any(key and key in term_key for key in app_keys):
        return 78
    if term_key in haystack:
        return 62
    app_ratio = max((difflib.SequenceMatcher(None, term_key, key).ratio() for key in app_keys if key), default=0.0)
    if app_ratio >= 0.88:
        return 76
    if app_ratio >= 0.82:
        return 72
    if app_ratio >= 0.74:
        return 64
    best_piece_ratio = 0.0
    for piece in re.split(r"[^a-z0-9\u4e00-\u9fff]+", haystack):
        if not piece:
            continue
        best_piece_ratio = max(best_piece_ratio, difflib.SequenceMatcher(None, term_key, piece).ratio())
    if best_piece_ratio >= 0.84:
        return 54
    if best_piece_ratio >= 0.72:
        return 45
    return 0


def workspace_app_key_variants(app_key: str) -> list[str]:
    key = compact_speech_text(str(app_key or "")).lower()
    variants = [key] if key else []
    for prefix in ("google", "microsoft", "apple"):
        if key.startswith(prefix) and len(key) > len(prefix) + 2:
            variants.append(key[len(prefix):])
    if key.endswith("app") and len(key) > 5:
        variants.append(key[:-3])
    seen: set[str] = set()
    return [value for value in variants if value and not (value in seen or seen.add(value))]


def expected_workspace_window_count(terms: list[str], max_windows: int) -> int:
    app_keys: set[str] = set()
    for term in terms or []:
        key = compact_speech_text(str(term or "")).lower()
        if not key:
            continue
        aliases = WINDOW_APP_ALIASES.get(key, [term])
        for alias in aliases:
            alias_key = compact_speech_text(str(alias or "")).lower()
            if alias_key:
                app_keys.add(alias_key)
    return min(max(1, int(max_windows or 4)), len(app_keys))


def workspace_layout_rects(count: int, bounds: dict[str, int], mode: str = "auto", rotation_index: int = 0) -> list[dict[str, int]]:
    count = max(0, min(int(count or 0), 4))
    if count <= 0:
        return []
    x = int(bounds.get("x", 0))
    y = int(bounds.get("y", 38))
    width = max(1, int(bounds.get("width", 1470)))
    height = max(1, int(bounds.get("height", 918)))
    mode = str(mode or "auto").strip().lower()
    if mode == "parallel":
        cell_width = max(1, width // count)
        return [
            {"x": x + cell_width * index, "y": y, "width": width - cell_width * index if index == count - 1 else cell_width, "height": height}
            for index in range(count)
        ]
    if count == 1:
        return [{"x": x, "y": y, "width": width, "height": height}]
    half_width = max(1, width // 2)
    right_width = max(1, width - half_width)
    if count == 2:
        return [
            {"x": x, "y": y, "width": half_width, "height": height},
            {"x": x + half_width, "y": y, "width": right_width, "height": height},
        ]
    if count == 3:
        half_height = max(1, height // 2)
        return [
            {"x": x, "y": y, "width": half_width, "height": height},
            {"x": x + half_width, "y": y, "width": right_width, "height": half_height},
            {"x": x + half_width, "y": y + half_height, "width": right_width, "height": height - half_height},
        ]
    third_height = max(1, height // 3)
    return [
        {"x": x, "y": y, "width": half_width, "height": height},
        {"x": x + half_width, "y": y, "width": right_width, "height": third_height},
        {"x": x + half_width, "y": y + third_height, "width": right_width, "height": third_height},
        {"x": x + half_width, "y": y + third_height * 2, "width": right_width, "height": height - third_height * 2},
    ]


def rotate_windows_for_workspace(windows: list[dict[str, Any]], rotation_index: int, mode: str) -> tuple[list[dict[str, Any]], int]:
    if len(windows) != 4 or str(mode or "auto").strip().lower() != "auto":
        return windows, rotation_index
    main_index = rotation_index % len(windows)
    ordered = [windows[main_index], *windows[:main_index], *windows[main_index + 1:]]
    return ordered, rotation_index + 1


def open_missing_workspace_apps(terms: list[str], windows: list[dict[str, Any]]) -> list[str]:
    existing = {compact_speech_text(str(window.get("app_name") or "")).lower() for window in windows}
    opened: list[str] = []
    for term in terms:
        aliases = launchable_workspace_app_candidates(term)
        for app_name in aliases:
            app_key = compact_speech_text(app_name).lower()
            if app_key in existing:
                continue
            try:
                subprocess.run(["open", "-a", app_name], check=True, timeout=5)
                opened.append(app_name)
                existing.add(app_key)
                break
            except Exception:
                continue
    if opened:
        time.sleep(1.0)
    return opened


def launchable_workspace_app_candidates(term: str) -> list[str]:
    term_key = compact_speech_text(str(term or "")).lower()
    aliases = WINDOW_APP_ALIASES.get(term_key)
    if aliases:
        return aliases
    scored: list[tuple[int, str]] = []
    for app_name in WORKSPACE_LAUNCHABLE_APPS:
        app_key = compact_speech_text(app_name).lower()
        score = fuzzy_workspace_score(term_key, app_key, app_key)
        if score >= 64:
            scored.append((score, app_name))
    scored.sort(reverse=True, key=lambda item: item[0])
    if scored:
        return [app_name for _, app_name in scored]
    return [term] if str(term or "").strip() else []


def apply_workspace_layout(actions: list[dict[str, Any]], timeout_seconds: float = 12.0) -> dict[str, Any]:
    if not actions:
        return {"returncode": 0, "stdout": "no_actions", "stderr": ""}
    lines = ['tell application "System Events"']
    for action in actions:
        app_name = str(action["app_name"])
        window_index = int(action["window_index"])
        rect = action["new_bounds"]
        lines.extend([
            f'  tell application process {applescript_quote(app_name)}',
            "    try",
            "      set frontmost to true",
            f"      set position of window {window_index} to {{{rect['x']}, {rect['y']}}}",
            f"      set size of window {window_index} to {{{rect['width']}, {rect['height']}}}",
            "    end try",
            "  end tell",
        ])
    first_app = str(actions[0]["app_name"])
    lines.extend(["end tell", f'tell application {applescript_quote(first_app)} to activate', 'return "ok"'])
    script = "\n".join(lines)
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout_seconds)
        return {"returncode": proc.returncode, "stdout": (proc.stdout or "").strip()[:12000], "stderr": (proc.stderr or "").strip()[:6000]}
    except Exception as exc:
        return {"error": str(exc)}


def applescript_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def press_video_fullscreen_shortcut(*, click_first: bool = True) -> dict[str, Any]:
    click_block = ""
    if click_first:
        click_block = """
        try
          set frontWindow to window 1
          set windowPosition to position of frontWindow
          set windowSize to size of frontWindow
          set clickX to (item 1 of windowPosition) + ((item 1 of windowSize) div 2)
          set clickY to (item 2 of windowPosition) + ((item 2 of windowSize) div 2)
          click at {clickX, clickY}
          delay 0.4
        end try
        """
    script = f'''
    set candidates to {{"{DEFAULT_BROWSER_APP}"}}
    tell application "System Events"
      repeat with appName in candidates
        if exists process appName then
          tell process appName
            set frontmost to true
            delay 0.25
            {click_block}
            keystroke "f"
            return appName & ":pressed_f"
          end tell
        end if
      end repeat
    end tell
    return "no_browser_process"
    '''
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=8)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        ok = proc.returncode == 0 and stdout.endswith(":pressed_f")
        return {"attempted": True, "ok": ok, "result": stdout, **({"stderr": stderr[:1000]} if stderr else {})}
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def set_browser_fullscreen() -> dict[str, Any]:
    script = '''
    set candidates to {"''' + DEFAULT_BROWSER_APP + '''"}
    tell application "System Events"
      repeat with appName in candidates
        if exists process appName then
          tell process appName
            set frontmost to true
            delay 0.25
            try
              click menu item "Enter Full Screen" of menu 1 of menu bar item "View" of menu bar 1
              return appName & ":entered"
            on error errText
              return appName & ":failed:" & errText
            end try
          end tell
        end if
      end repeat
    end tell
    return "no_browser_process"
    '''
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=8)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        ok = proc.returncode == 0 and not stdout.endswith(":failed") and ":failed:" not in stdout and stdout != "no_browser_process"
        return {"attempted": True, "ok": ok, "result": stdout, **({"stderr": stderr[:1000]} if stderr else {})}
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def exit_browser_fullscreen_if_needed(terms: list[str] | None) -> dict[str, Any]:
    normalized = {compact_speech_text(str(term or "")).lower() for term in (terms or [])}
    if not any(term in normalized for term in {"googlechrome", "chrome", "youtube", "视频", "browser", "浏览器", "网页"}):
        return {"attempted": False}
    script = '''
    set candidates to {"''' + DEFAULT_BROWSER_APP + '''"}
    tell application "System Events"
      repeat with appName in candidates
        if exists process appName then
          tell process appName
            set frontmost to true
            delay 0.2
            try
              key code 53
              delay 0.2
            end try
            try
              click menu item "Exit Full Screen" of menu 1 of menu bar item "View" of menu bar 1
              delay 0.4
              return appName & ":exit_menu"
            on error errText
              return appName & ":no_exit:" & errText
            end try
          end tell
        end if
      end repeat
    end tell
    return "no_browser_process"
    '''
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=6)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        ok = proc.returncode == 0 and "no_exit" not in stdout and stdout != "no_browser_process"
        return {"attempted": True, "ok": ok, "result": stdout, **({"stderr": stderr[:1000]} if stderr else {})}
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def run_osascript_tool(tool_name: str, script: str, arguments: dict[str, Any], store: VoiceSessionStore) -> dict[str, Any]:
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
        result = {"returncode": proc.returncode, "stdout": (proc.stdout or "").strip()[:12000], "stderr": (proc.stderr or "").strip()[:6000]}
        ok = proc.returncode == 0
    except Exception as exc:
        result = {"error": str(exc)}
        ok = False
    store.record_tool_event(tool_name, arguments, result, ok=ok)
    return result

