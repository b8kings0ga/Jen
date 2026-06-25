from __future__ import annotations

import datetime as dt
import html
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import parse, request

from voice_assistant.http_client import urlopen_bytes, urlopen_text
from voice_assistant.coding_monitor import CodingAppRunner, coding_runtime_env, host_venv_has_module, resolve_host_venv
from voice_assistant.coding_workspace import (
    CodingWorkspaceIndex,
    CodingWorkspaceSelector,
    CodingServiceRegistry,
    discover_entrypoints,
    register_workspace_program,
    read_manifest,
    rank_workspaces,
    resolve_workspace_program,
    workspace_context_for_prompt,
)
from voice_assistant.json_utils import parse_jsonish_value
from voice_assistant.location_helper import current_address
from voice_assistant.weather_location import plausible_weather_location
from voice_assistant.local_actions import (
    DEFAULT_BROWSER_APP,
    applescript_quote,
    apply_workspace_layout,
    desktop_usable_bounds,
    enumerate_visible_windows,
    expected_workspace_window_count,
    exit_browser_fullscreen_if_needed,
    front_note_context_requested,
    front_note_requested,
    long_term_memory_requested,
    match_workspace_windows,
    normalize_browser_osascript,
    normalize_front_note_call_args,
    normalize_workspace_app_names,
    normalize_workspace_mode,
    open_missing_workspace_apps,
    press_video_fullscreen_shortcut,
    rotate_windows_for_workspace,
    run_osascript_tool,
    set_browser_fullscreen,
    window_query_terms,
    workspace_layout_rects,
)
from voice_assistant.planning import news_search_retry_query
from voice_assistant.url_utils import looks_like_video_url, video_search_retry_query


CODING_ACTION_PROMPT_PREFIX = """Coding task defaults:

Runtime Defaults
- Use the host Jen uv environment. Run dependency-backed code with `uv run --active --no-sync ...` from the task workspace.
- Do not create an isolated workspace venv, run `uv sync`, run pip install, or rely on bare `python` when the host venv is needed.
- Package caches are opaque implementation details for uv/npm only. Do not inspect cache directories with `find`, `rg`, `ls`, `du`, or similar commands.
- Stay inside the task workspace unless the prompt explicitly names another path. Never run `find`, `rg`, `ls`, `du`, or `git status` against `/Users`, `/Users/a1234/.jen`, `/Users/a1234/.jen/cache`, parent repos, or unrelated workspaces.
- Prefer Typer for CLIs, Agno for agent workflows, and InquirerPy for terminal prompts when they fit the task.

Desktop App Contract
- Visible desktop apps, animations, toys, and games default to pywebview with local HTML/CSS/JavaScript.
- Use `app.py` or `main.py` as a tiny pywebview launcher, `index.html` for the app shell, and a root `run.sh` as the stable public open method.
- `run.sh` must `cd` to its own directory, export the Active host venv as `VIRTUAL_ENV`, prepend `$VIRTUAL_ENV/bin` to PATH, and launch with `uv run --active --no-sync ...`.
- For mini desktop apps, use a transparent pywebview window with one rounded inner app card, one custom close button, no native title bar, no browser fallback, no default fullscreen, and no grey outer frame.

Game Engine Contract
- Local desktop games default to pywebview shell + small local JavaScript. Use an engine only when it is already present in the workspace or the user explicitly asks for that engine.
- Do not use Phaser 4 unless the user explicitly asks for Phaser 4 or the existing workspace already uses Phaser 4. If Phaser 4 is explicit, use the `phaser4-game` skill and Context7 before coding Phaser APIs.
- If an engine runtime is missing, do not search cache directories. Either use `npm pack <package>@<version>` once when network/package access is acceptable, or report the missing runtime as a blocker.
- Do not silently replace a requested engine with another engine. If the requested engine is unavailable, keep the workspace runnable only if the user accepts the fallback.
- Platformer games must use the local platformer rule-engine chain: intent -> logic map -> playability validation -> autotile plan -> socket validation -> collision/events -> `level.json`. Do not let a Phaser/Pixi/canvas scene invent the level or pick raw frame ids.

Game Art Contract
- When the workspace contains `.gjallarhorn/assets/kenney_pixel-platformer`, game art must come from that asset pack. Use its tilemaps, tilesheets, character sprites, and sample maps instead of drawing new final art.
- Before choosing frames, read `.gjallarhorn/assets/kenney_pixel-platformer/gjallarhorn_asset_manifest.json`; for platformers, consume manifest v2 `tiles` and `autotile_rules` through the rule engine instead of hand-picking frames.
- Do not create final characters, terrain, hazards, collectibles, or backgrounds from scratch with canvas primitives, CSS blocks, or generated SVG/PNG when the Kenney asset pack can represent them.
- You may use simple debug overlays for collision/physics during validation, but remove or hide them for the final app.
- Copy only the needed Kenney files from the staged workspace asset pack into the public runtime folder, preserving attribution/source paths in comments or metadata.

Quality Gates
- A finished app must be usable, not just runnable. Controls respond immediately; visual hierarchy is polished; restart/pause/close work; win/lose or completion states are clear.
- Phaser is only the runtime; visual quality is still mandatory. Do not ship primitive demo art: no plain rectangles/circles as final characters, hazards, buttons, or obstacles.
- Every mini-game needs a named art direction, restrained palette, layered background, shaped characters or objects, lighting/shadow/highlight details, and a polished HUD.
- Use Phaser Graphics only for intentional illustrated shapes. If primitives start looking generic, create simple local SVG/PNG assets and load them through Phaser instead.
- Mini-games need fair starts, human-reactable hazards, reachable goals/collectibles, balanced speed, and deterministic playability checks for their genre.
- For grid games, validate map dimensions, passable spawns, reachable collectibles/goals, and enemy fairness. For physics/runner games, validate spawn, obstacle spacing, movement balance, restart, and the first few seconds of play.
- Do not use tkinter, pygame, curses, terminal-only UI, terminal printout demos, browser-opened HTML, placeholder rectangles, or text fallbacks unless explicitly requested or required by an existing project.

Validation
- Before finishing, run syntax/static checks that match the artifact: at minimum `python -m py_compile app.py` for pywebview apps and JS syntax checks for local scripts.
- For Phaser games, include a static smoke check that verifies the local Phaser asset exists and key gameplay constants or map/playability constraints are valid.
- Do not use `timeout ... run.sh`; do not use ps, osascript, browser fallback as the proof of GUI success; the host voice service will launch and verify visible apps after the coding executor finishes.
- If automatic validation cannot run, report the exact blocker and the closest safe validation that did run.
- Respect existing repo stack and scope when editing an existing project.
"""

CODING_ACTION_DEFAULT_PYPROJECT = """[project]
name = "voice-coding-workspace"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
]

[tool.uv]
package = false
"""


ANTIGRAVITY_DISABLED_REASON = "antigravity executor disabled"


def _normalize_coding_executor(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "")
    if raw in {"codex", "codexcli", "codex_cli", "localcodex", "本地codex"}:
        return "codex"
    if raw in {"antigravity", "antigrav", "agr", "ag", "anti_gravity"}:
        return "antigravity"
    return ""


def _coding_executor_label(executor: str) -> str:
    return "Antigravity" if executor == "antigravity" else "Codex"


def _coding_workdir(build_args: Any) -> Path:
    configured = getattr(build_args, "coding_workdir", None)
    if configured:
        return Path(configured).expanduser().resolve()
    env_value = (os.environ.get("JEN_CODING_WORKDIR") or os.environ.get("GJALLARHORN_CODING_WORKDIR", "")).strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (build_args.tool_workdir.resolve() / "coding_workspaces")


def _coding_cache_dir(build_args: Any) -> Path:
    configured = getattr(build_args, "coding_cache_dir", None)
    if configured:
        return Path(configured).expanduser().resolve()
    env_value = (os.environ.get("JEN_CODING_CACHE_DIR") or os.environ.get("GJALLARHORN_CODING_CACHE_DIR", "")).strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("~/.jen/cache").expanduser().resolve()


def _coding_codex_home(build_args: Any) -> Path:
    configured = getattr(build_args, "coding_codex_home", None)
    if configured:
        return Path(configured).expanduser().resolve()
    env_value = (os.environ.get("JEN_CODEX_HOME") or os.environ.get("GJALLARHORN_CODEX_HOME", "")).strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("~/.jen/codex").expanduser().resolve()


def _kenney_platformer_source(build_args: Any) -> Path | None:
    configured = getattr(build_args, "kenney_platformer_asset_dir", None)
    if configured:
        return Path(configured).expanduser().resolve()
    env_value = (os.environ.get("JEN_KENNEY_PLATFORMER_ASSET_DIR") or os.environ.get("GJALLARHORN_KENNEY_PLATFORMER_ASSET_DIR", "")).strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    default = Path("~/Downloads/kenney_pixel-platformer").expanduser().resolve()
    return default if default.exists() else None


def _kenney_socket_air() -> dict[str, list[str]]:
    return {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["air"]}


def _kenney_background_palette(frame: int) -> str:
    if frame in {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}:
        return "sky"
    if frame in {4, 5, 12, 13, 20, 21}:
        return "desert"
    return "green"


def _kenney_background_sockets(frame: int) -> dict[str, list[str]]:
    palette = _kenney_background_palette(frame)
    row = frame // 8
    col = frame % 8
    left_is_edge = col == 0 or _kenney_background_palette(frame - 1) != palette
    right_is_edge = col == 7 or _kenney_background_palette(frame + 1) != palette
    if row == 0:
        top = ["air"]
        bottom = [f"bg:{palette}:sky_to_scene:c{col}"]
    elif row == 1:
        top = [f"bg:{palette}:sky_to_scene:c{col}"]
        bottom = [f"bg:{palette}:scene_to_deep:c{col}"]
    else:
        top = [f"bg:{palette}:scene_to_deep:c{col}", f"bg:{palette}:deep:c{col}"]
        bottom = ["air", f"bg:{palette}:deep:c{col}"]
    left = ["air", f"bg:{palette}:h{row}:c{col}"] if left_is_edge else [f"bg:{palette}:h{row}:c{col}"]
    right = ["air", f"bg:{palette}:h{row}:c{col + 1}"] if right_is_edge else [f"bg:{palette}:h{row}:c{col + 1}"]
    return {"top": top, "right": right, "bottom": bottom, "left": left}


def _kenney_background_socket_blacklist(frame: int) -> dict[str, list[str]]:
    row = frame // 8
    blacklist: dict[str, list[str]] = {}
    if row > 0:
        blacklist["top"] = ["air"]
    if row < 2:
        blacklist["bottom"] = ["air"]
    return blacklist


def _kenney_water_or_pipe_sockets(frame: int) -> dict[str, list[str]]:
    groups = ((33, 34, 35), (53, 54, 55), (73, 74, 75), (93, 94, 95), (113, 114, 115), (132, 133, 134, 135))
    group = next((item for item in groups if frame in item), (frame,))
    local_col = group.index(frame)
    row_socket = f"water_pipe:{group[0]}:row"
    col_socket = f"water_pipe:c{local_col}"
    left = ["air", row_socket] if local_col == 0 else [row_socket]
    right = ["air", row_socket] if local_col == 2 else [row_socket]
    return {
        "top": ["air", col_socket] if frame in {33, 34, 35, 93, 94, 95} else [col_socket],
        "right": right,
        "bottom": ["air", col_socket] if frame in {73, 74, 75, 132, 133, 134, 135} else [col_socket],
        "left": left,
    }


def _kenney_water_or_pipe_socket_blacklist(frame: int) -> dict[str, list[str]]:
    return {} if frame in {33, 34, 35, 93, 94, 95} else {"top": ["air"]}


def _kenney_blue_pipe_sockets(role: str) -> dict[str, list[str]]:
    if role == "pipe_cap_top":
        return {"top": ["air"], "right": ["air"], "bottom": ["blue_pipe"], "left": ["air"]}
    if role == "pipe_body":
        return {"top": ["blue_pipe"], "right": ["air"], "bottom": ["blue_pipe"], "left": ["air"]}
    if role == "pipe_cap_bottom":
        return {"top": ["blue_pipe"], "right": ["air"], "bottom": ["air"], "left": ["air"]}
    return _kenney_socket_air()


def _kenney_terrain_socket_override(tile_id: str) -> dict[str, list[str]] | None:
    sockets = {
        "ground.grass.top.left": {"top": ["air"], "right": ["grass_top"], "bottom": ["dirt", "air"], "left": ["air"]},
        "ground.grass.top.middle": {"top": ["air"], "right": ["grass_top"], "bottom": ["dirt", "air"], "left": ["grass_top"]},
        "ground.grass.top.right": {"top": ["air"], "right": ["air"], "bottom": ["dirt", "air"], "left": ["grass_top"]},
        "ground.dirt.fill.left": {"top": ["dirt"], "right": ["dirt"], "bottom": ["dirt", "air"], "left": ["air", "dirt"]},
        "ground.dirt.fill": {"top": ["dirt"], "right": ["dirt", "air"], "bottom": ["dirt", "air"], "left": ["dirt", "air"]},
        "ground.dirt.fill.right": {"top": ["dirt"], "right": ["air", "dirt"], "bottom": ["dirt", "air"], "left": ["dirt"]},
        "platform.grass.left": {"top": ["air"], "right": ["grass_top"], "bottom": ["air"], "left": ["air"]},
        "platform.grass.middle": {"top": ["air"], "right": ["grass_top"], "bottom": ["air"], "left": ["grass_top"]},
        "platform.grass.right": {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["grass_top"]},
    }
    return sockets.get(tile_id)


def _kenney_apply_manual_socket_overrides(tile: dict[str, Any]) -> None:
    sheet = str(tile.get("sheet") or "")
    role = str(tile.get("role") or "")
    frame = int(tile.get("frame") or 0)
    terrain_sockets = _kenney_terrain_socket_override(str(tile.get("id") or ""))
    if terrain_sockets:
        tile["sockets"] = terrain_sockets
        tile["socket_blacklist"] = {}
        tile["condition_override"] = "terrain_semantic_socket"
    elif sheet == "backgrounds":
        tile["sockets"] = _kenney_background_sockets(frame)
        tile["socket_blacklist"] = _kenney_background_socket_blacklist(frame)
        tile["condition_override"] = "background_vertical_band"
    elif role == "water_or_pipe":
        tile["sockets"] = _kenney_water_or_pipe_sockets(frame)
        tile["socket_blacklist"] = _kenney_water_or_pipe_socket_blacklist(frame)
        tile["condition_override"] = "water_pipe_structure"
    elif role in {"pipe_cap_top", "pipe_body", "pipe_cap_bottom"}:
        tile["sockets"] = _kenney_blue_pipe_sockets(role)
        tile["socket_blacklist"] = {}
        tile["condition_override"] = "blue_pipe_stack"
    elif str(tile.get("cell_type") or "") in {"spawn", "hazard", "goal", "collectible"}:
        tile["sockets"] = _kenney_socket_air()
        tile["socket_blacklist"] = {}
        tile["condition_override"] = "actor_or_event_air"
    elif role == "wood_or_crate":
        tile["sockets"] = _kenney_socket_air()
        tile["socket_blacklist"] = {}
        tile["condition_override"] = "standalone_crate_air"
    elif str(tile.get("cell_type") or "") == "decoration" and sheet != "backgrounds":
        tile["sockets"] = _kenney_socket_air()
        tile["socket_blacklist"] = {}
        tile["condition_override"] = "standalone_decoration_air"


def _kenney_apply_auto_conditions(tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = _kenney_platformer_source(type("_Args", (), {"kenney_platformer_asset_dir": None})())
    if source is None:
        return tiles
    try:
        classifier = importlib.import_module("kenney_tile_condition_classifier")
        classified = classifier.classify_asset_dir(source)
        classifier.apply_conditions_to_tiles(tiles, classified)
    except Exception:
        return tiles
    for tile in tiles:
        _kenney_apply_manual_socket_overrides(tile)
    return tiles


def _kenney_tile_def(
    tile_id: str,
    sheet: str,
    frame: int,
    cell_type: str,
    role: str,
    tags: list[str],
    *,
    collision_type: str = "none",
    sockets: dict[str, list[str]] | None = None,
    socket_blacklist: dict[str, list[str]] | None = None,
    placement: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    animation: str | None = None,
) -> dict[str, Any]:
    collision: dict[str, Any] = {"type": collision_type}
    if collision_type in {"solid", "one_way", "platform", "hazard", "collectible", "trigger", "water", "ladder"}:
        collision.update({"shape": "rect", "rect": [0, 0, 18, 18]})
    return {
        "id": tile_id,
        "sheet": sheet,
        "frame": frame,
        "cell_type": cell_type,
        "role": role,
        "tags": tags,
        "sockets": sockets or _kenney_socket_air(),
        "socket_blacklist": socket_blacklist or {},
        "placement": placement or {},
        "collision": collision,
        "events": events or [],
        "animation": animation,
    }


def _kenney_classify_frame(sheet: str, frame: int) -> dict[str, Any]:
    if sheet == "backgrounds":
        if frame in {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}:
            return {"cell_type": "decoration", "role": "sky_background", "tags": ["background", "sky"], "collision": "none"}
        if frame in {4, 5, 12, 13, 20, 21}:
            return {"cell_type": "decoration", "role": "desert_background", "tags": ["background", "desert"], "collision": "none"}
        return {"cell_type": "decoration", "role": "green_background", "tags": ["background", "green"], "collision": "none"}

    if sheet == "characters":
        if frame in {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}:
            return {"cell_type": "spawn", "role": "player_variant", "tags": ["player", "character"], "collision": "none"}
        if frame in {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}:
            return {"cell_type": "hazard", "role": "enemy", "tags": ["enemy", "hazard"], "collision": "hazard"}
        return {"cell_type": "decoration", "role": "npc_or_marker", "tags": ["npc", "character"], "collision": "none"}

    terrain_groups: list[tuple[set[int], dict[str, Any]]] = [
        ({0, 1, 2, 3, 20, 21, 22, 23}, {"cell_type": "solid", "role": "ground", "tags": ["ground", "grass", "solid"], "collision": "solid"}),
        ({4, 5, 6, 24, 25, 40, 41, 42, 43, 60, 61, 62, 63, 120, 121, 122, 123, 140, 141, 142, 143}, {"cell_type": "solid", "role": "dirt_fill", "tags": ["ground", "dirt", "solid"], "collision": "solid"}),
        ({47, 48, 49, 50, 51, 90, 91, 92, 105, 106, 146, 147}, {"cell_type": "solid", "role": "wood_or_crate", "tags": ["wood", "crate", "solid"], "collision": "solid"}),
        ({71}, {"cell_type": "decoration", "role": "ladder", "tags": ["ladder"], "collision": "ladder"}),
        ({33, 34, 35, 53, 54, 55, 73, 74, 75, 93, 94, 95, 113, 114, 115, 132, 133, 134, 135}, {"cell_type": "water", "role": "water_or_pipe", "tags": ["water", "pipe"], "collision": "water"}),
        ({64, 65, 66, 68, 144, 145, 148, 149}, {"cell_type": "hazard", "role": "hazard", "tags": ["hazard"], "collision": "hazard"}),
        ({27, 67, 151, 152}, {"cell_type": "collectible", "role": "collectible", "tags": ["collectible"], "collision": "collectible"}),
        ({44, 45, 46, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179}, {"cell_type": "decoration", "role": "ui", "tags": ["ui"], "collision": "none"}),
        ({84, 85, 86, 87, 88, 129, 130}, {"cell_type": "decoration", "role": "sign_or_door", "tags": ["sign", "door"], "collision": "none"}),
        ({107, 108}, {"cell_type": "decoration", "role": "spring_or_button", "tags": ["trigger", "spring"], "collision": "trigger"}),
        ({111, 112}, {"cell_type": "goal", "role": "flag", "tags": ["goal", "flag"], "collision": "trigger"}),
        ({16, 17, 18, 19, 36, 37, 38, 39, 56, 57, 58, 59, 76, 77, 78, 79, 96, 97, 98, 99, 116, 117, 118, 119, 124, 125, 126, 127, 128, 136, 137, 138, 139}, {"cell_type": "decoration", "role": "foliage", "tags": ["foliage"], "collision": "none"}),
    ]
    for frames, classification in terrain_groups:
        if frame in frames:
            return classification
    return {"cell_type": "decoration", "role": "terrain_prop", "tags": ["terrain", "decoration"], "collision": "none"}


def _kenney_decoration_placement(sheet: str, frame: int, role: str, tags: list[str]) -> dict[str, Any]:
    if sheet == "backgrounds":
        return {"anchor": "background_tile", "depth": "background"}
    if sheet == "characters":
        return {"anchor": "actor", "depth": "front"} if "player" in tags or "enemy" in tags else {"anchor": "no_random", "depth": "front"}
    if role == "foliage":
        random_background_frames: set[int] = set()
        random_ground_frames = {124, 125, 126, 127, 128, 139}
        if frame in random_background_frames:
            return {"anchor": "background_only", "depth": "back"}
        if frame in random_ground_frames:
            return {"anchor": "ground_top", "depth": "front"}
        return {"anchor": "structure", "depth": "front"}
    if role in {"sign_or_door", "spring_or_button", "ladder"}:
        return {"anchor": "structure", "depth": "front"}
    if role in {"pipe_cap_top", "pipe_body", "pipe_cap_bottom"}:
        return {"anchor": "vertical_pipe_stack", "depth": "front"}
    return {"anchor": "none", "depth": "back"}


def _kenney_expand_all_tile_defs(tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = list(tiles)
    for tile in expanded:
        if tile.get("sheet") == "backgrounds":
            frame = int(tile.get("frame") or 0)
            tile["sockets"] = _kenney_background_sockets(frame)
            tile["socket_blacklist"] = _kenney_background_socket_blacklist(frame)
        elif tile.get("role") == "water_or_pipe":
            frame = int(tile.get("frame") or 0)
            tile["sockets"] = _kenney_water_or_pipe_sockets(frame)
            tile["socket_blacklist"] = _kenney_water_or_pipe_socket_blacklist(frame)
        else:
            tile.setdefault("socket_blacklist", {})
        tile.setdefault(
            "placement",
            _kenney_decoration_placement(
                str(tile.get("sheet") or ""),
                int(tile.get("frame") or 0),
                str(tile.get("role") or ""),
                [str(tag) for tag in tile.get("tags") or []],
            ),
        )
    covered = {(str(tile.get("sheet")), int(tile.get("frame"))) for tile in expanded}
    frame_counts = {"terrain": 180, "backgrounds": 24, "characters": 27}
    for sheet, count in frame_counts.items():
        for frame in range(count):
            if (sheet, frame) in covered:
                continue
            classification = _kenney_classify_frame(sheet, frame)
            cell_type = str(classification["cell_type"])
            role = str(classification["role"])
            tags = list(classification["tags"])
            tile_id = f"{cell_type}.{role}.{sheet}.{frame:03d}"
            events: list[dict[str, Any]] = []
            if cell_type == "collectible":
                events = [{"on": "player_overlap", "action": "collect", "args": {"value": 1}}]
            elif cell_type == "hazard":
                events = [{"on": "player_overlap", "action": "damage", "args": {"amount": 1}}]
            elif cell_type == "goal":
                events = [{"on": "player_overlap", "action": "finish", "args": {}}]
            expanded.append(
                _kenney_tile_def(
                    tile_id,
                    sheet,
                    frame,
                    cell_type,
                    role,
                    tags,
                    collision_type=str(classification["collision"]),
                    sockets=(
                        _kenney_background_sockets(frame)
                        if sheet == "backgrounds"
                        else _kenney_water_or_pipe_sockets(frame)
                        if role == "water_or_pipe"
                        else None
                    ),
                    socket_blacklist=(
                        _kenney_background_socket_blacklist(frame)
                        if sheet == "backgrounds"
                        else _kenney_water_or_pipe_socket_blacklist(frame)
                        if role == "water_or_pipe"
                        else None
                    ),
                    placement=_kenney_decoration_placement(sheet, frame, role, tags),
                    events=events,
                )
            )
    expanded = _kenney_apply_auto_conditions(expanded)
    expanded.sort(key=lambda tile: (str(tile["cell_type"]), str(tile["sheet"]), int(tile["frame"]), str(tile["id"])))
    return expanded


def _kenney_platformer_manifest() -> dict[str, Any]:
    manifest = {
        "version": 2,
        "name": "kenney_pixel-platformer",
        "source": "/Users/a1234/Downloads/kenney_pixel-platformer",
        "tile_size": 18,
        "rules": [
            "Use semantic frame groups before choosing art.",
            "Do not use random tile ids for final visible art.",
            "Use tilesheets when possible; avoid hundreds of individual tile files.",
            "Use the explicit animations map before inferring animated frame sequences from adjacent tiles.",
            "When a semantic group has a matching animation, use the animation name, fps, loop flag, and ordered frames.",
            "Current Flappy scope is intentionally tiny: use only sky_background, ground_grass, pink_player, flying_enemy, and blue_pipe unless the user expands the scope.",
            "Use debug primitives only for temporary collision visualization.",
        ],
        "sheets": {
            "terrain": {
                "path": "Tilemap/tilemap_packed.png",
                "tile_width": 18,
                "tile_height": 18,
                "columns": 20,
                "rows": 9,
                "frame_count": 180,
                "gid_first": 28,
            },
            "backgrounds": {
                "path": "Tilemap/tilemap-backgrounds_packed.png",
                "tile_width": 24,
                "tile_height": 24,
                "columns": 8,
                "rows": 3,
                "frame_count": 24,
            },
            "characters": {
                "path": "Tilemap/tilemap-characters_packed.png",
                "tile_width": 24,
                "tile_height": 24,
                "columns": 9,
                "rows": 3,
                "frame_count": 27,
                "gid_first": 1,
                "tiled_offset": {"x": -3, "y": 0},
            },
        },
        "semantic_groups": {
            "sky_background": {
                "sheet": "backgrounds",
                "frames": [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19],
                "notes": "Only the pale sky/cloud background tiles shown in the reference image.",
            },
            "ground_grass": {
                "sheet": "terrain",
                "frames": [1, 2, 3, 21, 22, 23],
                "notes": "Only the grass-topped dirt ground tiles shown in the reference image.",
            },
            "pink_player": {
                "sheet": "characters",
                "frames": [4, 5],
                "notes": "Pink astronaut two-frame player from the reference image.",
            },
            "flying_enemy": {
                "sheet": "characters",
                "frames": [24, 25, 26],
                "notes": "Three-frame flying creature from the reference image.",
            },
            "blue_pipe": {
                "sheet": "terrain",
                "frames": [95, 115, 135],
                "notes": "Blue vertical pipe/tube pieces from the reference image. Use frame 95/135 as caps and 115 as body.",
            },
        },
        "animations": {
            "player_pink_walk": {
                "sheet": "characters",
                "frames": [4, 5],
                "fps": 6,
                "loop": True,
                "group": "pink_player",
                "notes": "Pink astronaut two-frame walk/idle cycle.",
            },
            "flying_enemy_flap": {
                "sheet": "characters",
                "frames": [24, 25, 26],
                "fps": 8,
                "loop": True,
                "group": "flying_enemy",
                "notes": "Three-frame flying enemy flap. Good for Flappy-style player or airborne enemy.",
            },
            "blue_pipe_pulse": {
                "sheet": "terrain",
                "frames": [95, 115, 135],
                "fps": 3,
                "loop": True,
                "group": "blue_pipe",
                "notes": "Reference blue pipe pieces; usually draw as cap/body/cap instead of full animation.",
            },
        },
        "tiles": [
            {
                "id": "sky.cloud.left",
                "sheet": "backgrounds",
                "frame": 8,
                "cell_type": "decoration",
                "role": "background",
                "tags": ["sky", "cloud"],
                "sockets": {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["air"]},
                "collision": {"type": "none"},
                "events": [],
                "animation": None,
            },
            {
                "id": "ground.grass.top.left",
                "sheet": "terrain",
                "frame": 1,
                "cell_type": "solid",
                "role": "top_left",
                "tags": ["ground", "grass", "solid"],
                "sockets": {"top": ["air"], "right": ["grass_top"], "bottom": ["dirt", "air"], "left": ["air"]},
                "collision": {"type": "solid", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "ground.grass.top.middle",
                "sheet": "terrain",
                "frame": 2,
                "cell_type": "solid",
                "role": "top",
                "tags": ["ground", "grass", "solid"],
                "sockets": {"top": ["air"], "right": ["grass_top"], "bottom": ["dirt", "air"], "left": ["grass_top"]},
                "collision": {"type": "solid", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "ground.grass.top.right",
                "sheet": "terrain",
                "frame": 3,
                "cell_type": "solid",
                "role": "top_right",
                "tags": ["ground", "grass", "solid"],
                "sockets": {"top": ["air"], "right": ["air"], "bottom": ["dirt", "air"], "left": ["grass_top"]},
                "collision": {"type": "solid", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "ground.dirt.fill.left",
                "sheet": "terrain",
                "frame": 21,
                "cell_type": "solid",
                "role": "fill_left",
                "tags": ["ground", "dirt", "solid"],
                "sockets": {"top": ["dirt"], "right": ["dirt"], "bottom": ["dirt", "air"], "left": ["air", "dirt"]},
                "collision": {"type": "solid", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "ground.dirt.fill",
                "sheet": "terrain",
                "frame": 22,
                "cell_type": "solid",
                "role": "fill",
                "tags": ["ground", "dirt", "solid"],
                "sockets": {"top": ["dirt"], "right": ["dirt", "air"], "bottom": ["dirt", "air"], "left": ["dirt", "air"]},
                "collision": {"type": "solid", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "ground.dirt.fill.right",
                "sheet": "terrain",
                "frame": 23,
                "cell_type": "solid",
                "role": "fill_right",
                "tags": ["ground", "dirt", "solid"],
                "sockets": {"top": ["dirt"], "right": ["air", "dirt"], "bottom": ["dirt", "air"], "left": ["dirt"]},
                "collision": {"type": "solid", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "platform.grass.left",
                "sheet": "terrain",
                "frame": 1,
                "cell_type": "platform",
                "role": "top_left",
                "tags": ["platform", "grass", "one_way"],
                "sockets": {"top": ["air"], "right": ["grass_top"], "bottom": ["air"], "left": ["air"]},
                "collision": {"type": "one_way", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "platform.grass.middle",
                "sheet": "terrain",
                "frame": 2,
                "cell_type": "platform",
                "role": "top",
                "tags": ["platform", "grass", "one_way"],
                "sockets": {"top": ["air"], "right": ["grass_top"], "bottom": ["air"], "left": ["grass_top"]},
                "collision": {"type": "one_way", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "platform.grass.right",
                "sheet": "terrain",
                "frame": 3,
                "cell_type": "platform",
                "role": "top_right",
                "tags": ["platform", "grass", "one_way"],
                "sockets": {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["grass_top"]},
                "collision": {"type": "one_way", "shape": "rect", "rect": [0, 0, 18, 18]},
                "events": [],
                "animation": None,
            },
            {
                "id": "player.pink",
                "sheet": "characters",
                "frame": 4,
                "cell_type": "spawn",
                "role": "player",
                "tags": ["player", "pink"],
                "sockets": {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["air"]},
                "collision": {"type": "none"},
                "events": [],
                "animation": "player_pink_walk",
            },
            {
                "id": "enemy.flying",
                "sheet": "characters",
                "frame": 24,
                "cell_type": "hazard",
                "role": "enemy",
                "tags": ["enemy", "flying", "hazard"],
                "sockets": {"top": ["air"], "right": ["air"], "bottom": ["air"], "left": ["air"]},
                "collision": {"type": "hazard", "shape": "rect", "rect": [2, 2, 14, 14]},
                "events": [{"on": "player_overlap", "action": "damage", "args": {"amount": 1}}],
                "animation": "flying_enemy_flap",
            },
            {
                "id": "pipe.blue.cap.top",
                "sheet": "terrain",
                "frame": 95,
                "cell_type": "decoration",
                "role": "pipe_cap_top",
                "tags": ["pipe", "blue", "decoration"],
                "sockets": {"top": ["air"], "right": ["air"], "bottom": ["blue_pipe"], "left": ["air"]},
                "collision": {"type": "none"},
                "events": [],
                "animation": None,
            },
            {
                "id": "pipe.blue.body",
                "sheet": "terrain",
                "frame": 115,
                "cell_type": "decoration",
                "role": "pipe_body",
                "tags": ["pipe", "blue", "decoration"],
                "sockets": {"top": ["blue_pipe"], "right": ["air"], "bottom": ["blue_pipe"], "left": ["air"]},
                "collision": {"type": "none"},
                "events": [],
                "animation": None,
            },
            {
                "id": "pipe.blue.cap.bottom",
                "sheet": "terrain",
                "frame": 135,
                "cell_type": "decoration",
                "role": "pipe_cap_bottom",
                "tags": ["pipe", "blue", "decoration"],
                "sockets": {"top": ["blue_pipe"], "right": ["air"], "bottom": ["air"], "left": ["air"]},
                "collision": {"type": "none"},
                "events": [],
                "animation": None,
            },
        ],
        "autotile_rules": [
            {
                "id": "grass.solid.basic",
                "theme": "grass",
                "cell_type": "solid",
                "role_map": {
                    "top_left": "ground.grass.top.left",
                    "top": "ground.grass.top.middle",
                    "top_right": "ground.grass.top.right",
                    "fill": "ground.dirt.fill",
                    "single": "ground.grass.top.middle",
                },
            },
            {
                "id": "grass.platform.basic",
                "theme": "grass",
                "cell_type": "platform",
                "role_map": {
                    "top_left": "platform.grass.left",
                    "top": "platform.grass.middle",
                    "top_right": "platform.grass.right",
                    "fill": "platform.grass.middle",
                    "single": "platform.grass.middle",
                },
            },
            {
                "id": "grass.hazard.flying",
                "theme": "grass",
                "cell_type": "hazard",
                "role_map": {"hazard": "enemy.flying", "single": "enemy.flying"},
            },
        ],
        "example_maps": [
            "Tiled/tilemap-example-a.tmx",
            "Tiled/tilemap-example-b.tmx",
        ],
        "visual_references": ["Preview.png", "SampleA.png", "SampleB.png"],
    }
    manifest["tiles"] = _kenney_expand_all_tile_defs(manifest["tiles"])
    return manifest


def _stage_kenney_platformer_assets(cwd: Path, build_args: Any) -> Path | None:
    source = _kenney_platformer_source(build_args)
    if source is None or not source.exists() or not source.is_dir():
        return None
    destination = cwd / ".gjallarhorn" / "assets" / "kenney_pixel-platformer"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".DS_Store", "__MACOSX"),
    )
    (destination / "gjallarhorn_asset_manifest.json").write_text(
        json.dumps(_kenney_platformer_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _ensure_coding_codex_home(codex_home: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "skills").mkdir(parents=True, exist_ok=True)
    (codex_home / "tmp").mkdir(parents=True, exist_ok=True)
    (codex_home / "cache").mkdir(parents=True, exist_ok=True)

    config_path = codex_home / "config.toml"
    if not config_path.exists():
        config_path.write_text(
            "# Jen voice coding Codex home.\n"
            "# Keep this isolated from the interactive ~/.codex skill and MCP list.\n"
            "\n"
            "[mcp_servers]\n\n"
            "[mcp_servers.context7]\n"
            'command = "npx"\n'
            'args = ["-y", "@upstash/context7-mcp"]\n',
            encoding="utf-8",
        )
    else:
        config_text = config_path.read_text(encoding="utf-8")
        if "[mcp_servers.context7]" not in config_text:
            suffix = (
                "\n"
                "[mcp_servers.context7]\n"
                'command = "npx"\n'
                'args = ["-y", "@upstash/context7-mcp"]\n'
            )
            config_path.write_text(config_text.rstrip() + "\n" + suffix, encoding="utf-8")

    auth_path = codex_home / "auth.json"
    source_auth = Path("~/.codex/auth.json").expanduser()
    if source_auth.exists() and not auth_path.exists():
        try:
            auth_path.symlink_to(source_auth)
        except Exception:
            shutil.copy2(source_auth, auth_path)

    skill_dir = codex_home / "skills" / "desktop-mini-game"
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            "---\n"
            "name: desktop-mini-game\n"
            "description: Use when building or debugging local desktop mini games, pywebview game shells, PixiJS games, Phaser games, Flappy/Pac-Man style games, or game playability/visual-quality issues.\n"
            "---\n\n"
            "# Desktop Mini Game\n\n"
            "Default stack:\n"
            "- Use `pywebview + uv` for the desktop shell.\n"
            "- Prefer `PixiJS + tiny-game-core` for lightweight games.\n"
            "- Use a Phaser 3 template only for map-heavy or physics-heavy games.\n"
            "- Do not use Phaser 4 unless the user explicitly asks for Phaser 4. When they do, use the `phaser4-game` skill and check Context7 before coding Phaser 4 APIs.\n"
            "- Do not fall back to canvas-only, pygame, tkinter, browser-opened HTML, or CDN runtime.\n\n"
            "Required files:\n"
            "- `app.py`\n"
            "- `index.html`\n"
            "- `run.sh`\n"
            "- local `vendor/` runtime when an engine is used\n\n"
            "Validation:\n"
            "- `uv run --active --no-sync python -m py_compile app.py`\n"
            "- JavaScript syntax check for local scripts.\n"
            "- Verify the requested runtime is actually imported and used.\n"
            "- Use staged Kenney platformer assets for final game art when `.gjallarhorn/assets/kenney_pixel-platformer` exists.\n"
            "- Use `animations` from `gjallarhorn_asset_manifest.json` for animated characters, coins, water, and flying sprites before inventing frame sequences.\n"
            "- Add a playability smoke check for the genre.\n",
            encoding="utf-8",
        )

    kenney_skill_dir = codex_home / "skills" / "kenney-platformer-assets"
    kenney_skill_path = kenney_skill_dir / "SKILL.md"
    if not kenney_skill_path.exists():
        kenney_skill_dir.mkdir(parents=True, exist_ok=True)
        kenney_skill_path.write_text(
            "---\n"
            "name: kenney-platformer-assets\n"
            "description: Use when building or debugging 2D platformer, side-scroller, tile-based, Flappy-style, runner, arcade, or pixel-art games in the Jen coding workspace. Requires using the staged Kenney Pixel Platformer asset pack instead of drawing final game art from scratch.\n"
            "---\n\n"
            "# Kenney Platformer Assets\n\n"
            "When `.gjallarhorn/assets/kenney_pixel-platformer` exists in the task workspace, use that pack for game visuals. Do not create final characters, terrain, hazards, collectibles, or backgrounds from scratch.\n\n"
            "## Asset Pack\n\n"
            "Workspace path: `.gjallarhorn/assets/kenney_pixel-platformer`\n\n"
            "Important files:\n"
            "- `gjallarhorn_asset_manifest.json`: semantic index. Read this first before selecting frames.\n"
            "- `gjallarhorn_asset_manifest.json.tiles`: manifest v2 tile defs with stable ids, sockets, collision, events, tags, and optional animation.\n"
            "- `gjallarhorn_asset_manifest.json.autotile_rules`: role maps for choosing legal tile ids from a logic map.\n"
            "- `gjallarhorn_asset_manifest.json.animations`: explicit animation index with sheet, frames, fps, loop, group, and notes. Use this before guessing adjacent frames.\n"
            "- `Tilemap/tilemap_packed.png`: packed 18px x 18px terrain/object tiles, 20 columns x 9 rows, 180 tiles.\n"
            "- `Tilemap/tilemap-backgrounds_packed.png`: packed 24px x 24px background tiles, 8 columns x 3 rows, 24 tiles.\n"
            "- `Tilemap/tilemap-characters_packed.png`: packed 24px x 24px characters, 9 columns x 3 rows, 27 tiles, Tiled offset x=-3.\n"
            "- `Tiled/tileset-tiles.tsx` and `Tiled/tileset-characters.tsx`: tileset metadata.\n"
            "- `Tiled/tilemap-example-a.tmx` and `Tiled/tilemap-example-b.tmx`: reference maps and layer names.\n"
            "- `SampleA.png`, `SampleB.png`, `Preview.png`: visual references only, not runtime sprites.\n\n"
            "## Rules\n\n"
            "- Copy only needed files into the app public assets folder; keep the staged source pack unchanged.\n"
            "- Choose art by semantic group from `gjallarhorn_asset_manifest.json`; do not pick random tile ids.\n"
            "- For platformer levels, call the rule engine to generate `level.json`; do not directly place tile frames in game scene code.\n"
            "- Use manifest v2 `tiles` and `autotile_rules` for legal tile selection, socket validation, collision, and events.\n"
            "- For animation, choose an entry from `animations` and preserve its ordered frames, fps, and loop behavior.\n"
            "- Prefer tilesheets over hundreds of individual `Tiles/tile_*.png` files.\n"
            "- Use tile sizes from metadata: terrain/object tiles are 18x18; backgrounds and characters are 24x24.\n"
            "- Use spritesheet frame coordinates rather than redrawing sprites on canvas.\n"
            "- Simple debug rectangles are allowed only for temporary collision visualization, not final art.\n"
            "- If a requested game cannot be represented by this pack, adapt the game theme to this platformer art before inventing new art.\n\n"
            "## Validation\n\n"
            "- Confirm runtime code loads at least one file from `.gjallarhorn/assets/kenney_pixel-platformer` or copied derivatives.\n"
            "- Confirm selected frame ids are documented in `gjallarhorn_asset_manifest.json`.\n"
            "- Confirm platformer runtime loads generated `level.json` and does not generate the map inside Phaser/Pixi/canvas.\n"
            "- Confirm animated sprites use a named manifest animation rather than an undocumented frame sequence.\n"
            "- Confirm final visible player, platforms, hazards/obstacles, collectibles, and background are asset-backed.\n",
            encoding="utf-8",
        )

    phaser_skill_dir = codex_home / "skills" / "phaser4-game"
    phaser_skill_path = phaser_skill_dir / "SKILL.md"
    if not phaser_skill_path.exists():
        phaser_skill_dir.mkdir(parents=True, exist_ok=True)
        phaser_skill_path.write_text(
            "---\n"
            "name: phaser4-game\n"
            "description: Use when the user explicitly asks for Phaser 4, Phaser v4, Phaser next, or debugging a Phaser 4 browser/desktop game. This skill focuses on playable Phaser 4 games inside a local pywebview/uv desktop shell and requires checking Context7 docs before relying on Phaser API memory.\n"
            "---\n\n"
            "# Phaser 4 Game\n\n"
            "Use this skill only when Phaser 4 is explicit. For ordinary desktop mini games, prefer the lighter `desktop-mini-game` path unless the user requested Phaser 4 or an existing project already uses it.\n\n"
            "## First Step\n\n"
            "Before coding Phaser 4 API usage, query Context7 for current Phaser 4 docs/examples. Do not assume Phaser 3 APIs work unchanged in Phaser 4.\n\n"
            "Useful Context7 queries:\n"
            "- `phaser 4 scene lifecycle`\n"
            "- `phaser 4 input keyboard cursor keys`\n"
            "- `phaser 4 arcade physics overlap collider`\n"
            "- `phaser 4 game config scale`\n"
            "- `phaser 4 loader spritesheet image`\n\n"
            "If Context7 is unavailable, say that clearly in the task log and constrain the implementation to APIs verified from local package files or installed examples.\n\n"
            "## Required Stack\n\n"
            "- Desktop shell: `pywebview + uv`.\n"
            "- Runtime: local installed package files or local `vendor/` assets. Do not use CDN fallback.\n"
            "- Cache directories are opaque. Do not run `find`, `rg`, `ls`, or `du` against `/Users`, `/Users/a1234/.jen`, or `/Users/a1234/.jen/cache` to discover Phaser files.\n"
            "- If `vendor/phaser.min.js` is absent, use one explicit package-manager command such as `npm pack phaser@<version>` when package access is acceptable, or report the missing runtime as the blocker. Do not manually crawl npm cache directories.\n"
            "- Entrypoints: `run.sh`, `app.py`, `index.html`.\n"
            "- `run.sh` must start from the workspace root using the host environment: `uv run --active --no-sync python app.py`\n\n"
            "## Implementation Rules\n\n"
            "- For platformers, Phaser 4 is only a runtime adapter: preload manifest sheets, load generated `level.json`, create visual/collision/event layers from it, and never regenerate the level in the scene.\n"
            "- Use the `physics_profile` from `level.json`; default Arcade-style values are gravity_y=900, max_velocity_x=220, acceleration_x=1400, drag_x=1800, jump_velocity_y=-420, coyote_time_ms=90, jump_buffer_ms=110.\n"
            "- Solid tiles become static bodies, platform/one_way tiles become one-way platforms, and hazard/collectible/goal become overlap bodies plus events.\n"
            "- Player spawn must come from `level.json.spawn`, not from a hard-coded scene guess.\n"
            "- Keep the game loop deterministic and small. Prefer one scene, explicit constants, and clear update order.\n"
            "- Bind keyboard input directly and verify it in code. Arrow-key games must also support `WASD` when reasonable.\n"
            "- Keep collision boxes visible during development if the task is debugging playability; remove or hide them only after verification.\n"
            "- Tune difficulty for the first 20 seconds of play: slow enemies, wide gaps, forgiving collision, and progressive difficulty.\n"
            "- Never ship an attractive but unplayable game. Playability beats visual polish.\n"
            "- Avoid native fullscreen unless the user explicitly asks for fullscreen.\n"
            "- For pywebview windows, prefer frameless only when requested and provide an in-app close button.\n\n"
            "## Validation\n\n"
            "- `uv run --active --no-sync python -m py_compile app.py`\n"
            "- JavaScript syntax check, for example `node --check <script>` when scripts are split out.\n"
            "- Verify Phaser 4 is actually loaded and referenced; do not silently replace it with raw canvas or Phaser 3.\n"
            "- Verify Phaser 4 loads generated `level.json` and does not directly choose tile frame ids.\n"
            "- Verify `physics_profile` values are applied or explicitly translated into Phaser config/body settings.\n"
            "- Use staged Kenney platformer assets for final game art when `.gjallarhorn/assets/kenney_pixel-platformer` exists.\n"
            "- Smoke test movement keys, restart, frameless close button, and first-level human playability.\n\n"
            "## Failure Handling\n\n"
            "If Phaser 4 package/API behavior blocks the task, do not fake completion. Record the blocker, keep the workspace runnable, and suggest either a smaller Phaser 4 repro or switching to PixiJS/Phaser 3 for production playability.\n",
            encoding="utf-8",
        )


def _coding_executor_model(executor: str) -> str:
    if executor == "antigravity":
        return os.environ.get("JEN_ANTIGRAVITY_MODEL") or os.environ.get("GJALLARHORN_ANTIGRAVITY_MODEL", "Gemini 3.5 Flash (Low)")
    return os.environ.get("JEN_CODEX_MODEL") or os.environ.get("GJALLARHORN_CODEX_MODEL", "gpt-5.3-codex-spark")


def _coding_executor_binary(executor: str) -> str:
    configured = (os.environ.get("JEN_CODEX_BIN") or os.environ.get("GJALLARHORN_CODEX_BIN", "")).strip()
    if configured:
        return configured if Path(configured).exists() else (shutil.which(configured) or "")
    return shutil.which("codex") or ""


def _select_coding_executor(safe_args: dict[str, Any], _store: Any) -> dict[str, Any]:
    requested_raw = safe_args.get("executor") or safe_args.get("executor_type") or safe_args.get("coding_executor")
    requested = _normalize_coding_executor(requested_raw)
    if requested_raw and not requested:
        return {
            "ok": False,
            "executor": str(requested_raw),
            "executor_selection_reason": "explicit",
            "executor_fallback": False,
            "error": f"unsupported coding executor: {requested_raw}",
        }

    if requested:
        if requested == "antigravity":
            return {
                "ok": False,
                "executor": requested,
                "executor_model": _coding_executor_model(requested),
                "executor_bin": "",
                "executor_selection_reason": "explicit",
                "executor_fallback": False,
                "error": ANTIGRAVITY_DISABLED_REASON,
            }
        binary = _coding_executor_binary(requested)
        return {
            "ok": bool(binary),
            "executor": requested,
            "executor_model": _coding_executor_model(requested),
            "executor_bin": binary,
            "executor_selection_reason": "explicit",
            "executor_fallback": False,
            **({} if binary else {"error": f"{requested} executable not found in PATH"}),
        }

    selected = "codex"
    binary = _coding_executor_binary(selected)
    if binary:
        return {
            "ok": True,
            "executor": selected,
            "executor_model": _coding_executor_model(selected),
            "executor_bin": binary,
            "executor_selection_reason": "default",
            "executor_fallback": False,
        }

    return {
        "ok": False,
        "executor": selected,
        "executor_model": _coding_executor_model(selected),
        "executor_bin": "",
        "executor_selection_reason": "default",
        "executor_fallback": False,
        "error": f"{selected} executable not found in PATH",
    }


def _public_program_payload(value: Any) -> Any:
    """Return a speech/model-safe copy of program launch data."""
    if isinstance(value, list):
        return [_public_program_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key == "window_check":
            continue
        if key == "error" and "window enumeration" in str(item).lower():
            continue
        cleaned[key] = _public_program_payload(item)
    return cleaned


def _terminate_program_workspace(workspace: Path, program: dict[str, Any]) -> tuple[bool, int, str]:
    """Terminate a registered workspace app, including uv/python children."""
    launchd_removed = _bootout_workspace_launchd_jobs(workspace)
    candidates: list[int] = []
    last_launch = program.get("last_launch") if isinstance(program.get("last_launch"), dict) else {}
    try:
        candidates.append(int(last_launch.get("pid") or 0))
    except Exception:
        pass
    pid_file = workspace / ".voice_app.pid"
    try:
        candidates.append(int(pid_file.read_text(encoding="utf-8").strip() or "0"))
    except Exception:
        pass

    errors: list[str] = []
    killed_pid = 0
    for pid in [item for index, item in enumerate(candidates) if item > 0 and item not in candidates[:index]]:
        try:
            os.killpg(os.getpgid(pid), 15)
            killed_pid = pid
            break
        except ProcessLookupError:
            continue
        except Exception as exc:
            errors.append(str(exc))
        try:
            os.kill(pid, 15)
            killed_pid = pid
            break
        except ProcessLookupError:
            continue
        except Exception as exc:
            errors.append(str(exc))
    if killed_pid:
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        return True, killed_pid, ""
    if launchd_removed:
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        return True, 0, ""
    return False, 0, "; ".join(errors) or "没有可关闭进程"


def _bootout_workspace_launchd_jobs(workspace: Path) -> bool:
    workspace_text = str(workspace)
    try:
        uid = str(os.getuid())
        listing = subprocess.run(
            ["launchctl", "print", f"gui/{uid}"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except Exception:
        return False
    labels: list[str] = []
    for line in listing.splitlines():
        match = re.search(r"\b(gjallarhorn\.[A-Za-z0-9_.-]+)\b", line)
        if match:
            labels.append(match.group(1))
    removed = False
    for label in sorted(set(labels)):
        try:
            detail = subprocess.run(
                ["launchctl", "print", f"gui/{uid}/{label}"],
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout
        except Exception:
            continue
        if workspace_text not in detail:
            continue
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        removed = True
    return removed


def _codex_executor_argv(
    binary: str,
    model: str,
    cwd: Path,
    last_message_path: Path,
    prompt: str,
) -> list[str]:
    return [
        binary,
        "--ask-for-approval",
        "never",
        "exec",
        "--model",
        model,
        "--skip-git-repo-check",
        "--cd",
        str(cwd),
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-last-message",
        str(last_message_path),
        prompt,
    ]


def _reap_background_process(proc: subprocess.Popen[Any]) -> None:
    def wait_for_exit() -> None:
        try:
            proc.wait()
        except Exception:
            pass

    threading.Thread(target=wait_for_exit, daemon=True, name=f"voice-tool-proc-{proc.pid}").start()


def _osascript_modifier_list(raw_modifiers: Any) -> str:
    if raw_modifiers is None:
        return ""
    if isinstance(raw_modifiers, str):
        values = [part.strip().lower() for part in re.split(r"[,+ ]+", raw_modifiers) if part.strip()]
    elif isinstance(raw_modifiers, list):
        values = [str(part).strip().lower() for part in raw_modifiers if str(part).strip()]
    else:
        values = []
    mapping = {
        "cmd": "command down",
        "command": "command down",
        "⌘": "command down",
        "shift": "shift down",
        "option": "option down",
        "alt": "option down",
        "control": "control down",
        "ctrl": "control down",
    }
    modifiers = []
    seen = set()
    for value in values:
        mapped = mapping.get(value)
        if mapped and mapped not in seen:
            modifiers.append(mapped)
            seen.add(mapped)
    if not modifiers:
        return ""
    return "{" + ", ".join(modifiers) + "}"


def _normalize_daily_action(action: str) -> str:
    value = str(action or "").strip().lower().replace("-", "_")
    return {
        "date": "time",
        "datetime": "time",
        "now": "time",
        "weather_current": "weather",
        "calendar": "calendar_list",
        "calendar_events": "calendar_list",
        "reminders": "reminder_list",
        "reminders_list": "reminder_list",
        "list_reminders": "reminder_list",
        "create_reminder": "reminder_create",
        "add_reminder": "reminder_create",
        "reminder": "reminder_create",
        "front_note": "note_live",
        "note": "note_live",
        "live_note": "note_live",
        "context_note": "note_context",
        "add_context_note": "memory",
        "remember": "memory",
        "location": "map",
        "gps": "map",
        "maps": "map",
        "route": "map",
        "directions": "map",
        "map_query": "map",
        "current_address": "map",
    }.get(value, value)


def _daily_map(target: str, args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode") or args.get("map_mode") or "").strip().lower()
    if not mode:
        mode = "route" if any(args.get(key) for key in ("from", "to", "origin", "destination")) else "query"
    if mode in {"current", "current_address", "address", "where_am_i"}:
        return _current_address_result(float(args.get("timeout_seconds") or 2.0))
    origin = str(args.get("from") or args.get("origin") or "").strip()
    destination = str(args.get("to") or args.get("destination") or target or "").strip()
    if mode == "route":
        if origin.lower() in {"current", "当前位置", "当前地址", "这里"}:
            current = _current_address_result(float(args.get("timeout_seconds") or 2.0))
            if not current.get("ok"):
                return {"ok": False, "mode": "route", "error": "需要当前位置权限或明确出发地", "detail": current.get("error")}
            origin = str(current.get("address") or "")
        if not destination:
            return {"ok": False, "mode": "route", "error": "destination is required"}
        query = {"daddr": destination}
        if origin:
            query["saddr"] = origin
        return _map_url_result("route", destination, query)
    query_text = target or destination or str(args.get("query") or "").strip()
    if not query_text:
        return {"ok": False, "mode": "query", "error": "place is required"}
    return _map_url_result("query", query_text, {"q": query_text})


def _map_url_result(mode: str, place: str, query: dict[str, str]) -> dict[str, Any]:
    encoded = parse.urlencode({key: value for key, value in query.items() if value})
    return {
        "ok": True,
        "mode": mode,
        "place": place,
        "maps_url": f"maps://?{encoded}",
        "web_url": f"https://maps.apple.com/?{encoded}",
    }


def _current_address_result(timeout_seconds: float = 2.0) -> dict[str, Any]:
    return current_address(timeout_seconds=timeout_seconds)


def _weather_query_candidates(location: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(location or "")).strip(" ：:，,。.!！?？")
    if not value:
        return []
    candidates = [value]
    parts = [part.strip() for part in re.split(r"[,，]", value) if part.strip()]
    if len(parts) >= 3:
        city_index = -3 if len(parts) >= 4 else 1
        candidates.append(", ".join(parts[city_index:]))
        candidates.append(", ".join([parts[city_index], parts[-1]]))
        candidates.append(parts[city_index])
    if len(parts) >= 2:
        candidates.append(parts[-2])
    deduped: list[str] = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _create_reminder(title: str, args: dict[str, Any]) -> dict[str, Any]:
    title = str(title or "").strip()
    if not title:
        return {"ok": False, "error": "reminder title is required"}
    list_name = str(args.get("list") or args.get("reminder_list") or "Reminders").strip() or "Reminders"
    due_text = str(args.get("due_at") or args.get("due") or args.get("remind_at") or args.get("time") or "").strip()
    due_dt = _reminder_due_datetime(due_text)
    due_at = due_dt.isoformat(timespec="minutes") if due_dt else due_text
    eventkit = _create_reminder_eventkit(title, list_name=list_name, due_at=due_at, due_dt=due_dt)
    if eventkit.get("ok"):
        return eventkit
    due_line = ""
    if due_dt:
        due_line = _reminder_applescript_due_lines(due_dt)
    elif due_at:
        due_line = f"set due date of newReminder to date {applescript_quote(due_at)}"
    script = f'''
    with timeout of 5 seconds
      tell application "Reminders"
        set targetList to list {applescript_quote(list_name)}
        set newReminder to make new reminder at end of reminders of targetList with properties {{name:{applescript_quote(title)}}}
        {due_line}
      end tell
      return "ok"
    end timeout
    '''
    proc = _run_reminders_osascript_with_recovery(script)
    if proc.returncode != 0:
        return {
            "ok": False,
            "title": title,
            "list": list_name,
            "error": (proc.stderr or proc.stdout or eventkit.get("error") or "").strip()[:500],
            "eventkit_error": str(eventkit.get("error") or "")[:500],
        }
    return {"ok": True, "title": title, "list": list_name, "due_at": due_at, "backend": "applescript"}


def _reminder_due_datetime(value: str, *, now: dt.datetime | None = None) -> dt.datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    base = now or dt.datetime.now().astimezone()
    lowered = raw.lower()
    relative = re.search(r"([0-9零一二两三四五六七八九十半个]+)(?:个)?(分钟|小时|钟头)(?:后|之后)", raw)
    if relative:
        raw_amount = relative.group(1)
        unit = relative.group(2)
        if "半" in raw_amount and unit in {"小时", "钟头"}:
            return (base + dt.timedelta(minutes=30)).replace(second=0, microsecond=0)
        amount = _chinese_number_to_int(raw_amount)
        if amount is None and "半" in raw_amount:
            amount = 30 if unit == "分钟" else None
        if amount is None:
            return None
        if unit == "分钟":
            return (base + dt.timedelta(minutes=amount)).replace(second=0, microsecond=0)
        if unit in {"小时", "钟头"}:
            return (base + dt.timedelta(hours=amount)).replace(second=0, microsecond=0)
    iso_match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})(?:[ tT](\d{1,2})(?::(\d{1,2}))?)?", raw)
    if iso_match:
        year, month, day = [int(part) for part in iso_match.group(1).split("-")]
        hour = int(iso_match.group(2) or 9)
        minute = int(iso_match.group(3) or 0)
        return base.replace(year=year, month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
    days = 0
    if "后天" in raw:
        days = 2
    elif "明天" in raw or "明早" in raw or "明晚" in raw:
        days = 1
    elif "今天" in raw or "今晚" in raw:
        days = 0
    elif "tomorrow" in lowered:
        days = 1
    elif "today" in lowered:
        days = 0
    else:
        return None
    hour = 9
    minute = 0
    time_match = re.search(r"(?:(\d{1,2})|([零一二两三四五六七八九十]{1,3}))(?:[:：点](?:(\d{1,2})|([零一二两三四五六七八九十]{1,3}))?)", raw)
    if time_match:
        hour_value = int(time_match.group(1)) if time_match.group(1) else _chinese_number_to_int(str(time_match.group(2) or ""))
        minute_value = int(time_match.group(3)) if time_match.group(3) else _chinese_number_to_int(str(time_match.group(4) or ""))
        if hour_value is not None:
            hour = hour_value
        minute = minute_value if minute_value is not None else 0
    if any(token in raw for token in ["明早", "早上", "上午"]):
        hour = hour if time_match and hour != 9 else 9
    elif "中午" in raw:
        hour = hour if time_match and hour != 9 else 12
    elif any(token in raw for token in ["下午", "明晚", "今晚", "晚上"]):
        if time_match and 1 <= hour <= 11:
            hour += 12
        elif not time_match:
            hour = 20 if any(token in raw for token in ["明晚", "今晚", "晚上"]) else 15
    due = (base + dt.timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= base and days == 0:
        due += dt.timedelta(days=1)
    return due


def _chinese_number_to_int(value: str) -> int | None:
    raw = str(value or "").strip()
    if raw.endswith("个"):
        raw = raw[:-1]
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in {"半", "半个"}:
        return None
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if raw in digits:
        return digits[raw]
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _reminder_applescript_due_lines(due_dt: dt.datetime) -> str:
    return f'''
        set reminderDueDate to current date
        set year of reminderDueDate to {due_dt.year}
        set month of reminderDueDate to {due_dt.month}
        set day of reminderDueDate to {due_dt.day}
        set hours of reminderDueDate to {due_dt.hour}
        set minutes of reminderDueDate to {due_dt.minute}
        set seconds of reminderDueDate to 0
        set due date of newReminder to reminderDueDate
        tell newReminder to make new alarm at end of alarms with properties {{trigger date:reminderDueDate}}
        '''


def _create_reminder_eventkit(title: str, *, list_name: str, due_at: str = "", due_dt: dt.datetime | None = None) -> dict[str, Any]:
    try:
        import EventKit
        import Foundation
    except Exception as exc:
        return {"ok": False, "error": f"EventKit unavailable: {exc}"}
    try:
        status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeReminder)
        allowed = {
            getattr(EventKit, "EKAuthorizationStatusAuthorized", 3),
            getattr(EventKit, "EKAuthorizationStatusFullAccess", 3),
            getattr(EventKit, "EKAuthorizationStatusWriteOnly", 4),
        }
        if status not in allowed:
            return {"ok": False, "error": f"Reminders permission not granted: {status}"}
        store = EventKit.EKEventStore.alloc().init()
        calendar = None
        for item in store.calendarsForEntityType_(EventKit.EKEntityTypeReminder) or []:
            if str(item.title() or "") == list_name:
                calendar = item
                break
        if calendar is None:
            calendar = store.defaultCalendarForNewReminders()
        if calendar is None:
            return {"ok": False, "error": "no reminders calendar available"}
        reminder = EventKit.EKReminder.reminderWithEventStore_(store)
        reminder.setTitle_(title)
        reminder.setCalendar_(calendar)
        if due_dt is not None:
            components = Foundation.NSDateComponents.alloc().init()
            components.setCalendar_(Foundation.NSCalendar.currentCalendar())
            components.setTimeZone_(Foundation.NSTimeZone.localTimeZone())
            components.setYear_(due_dt.year)
            components.setMonth_(due_dt.month)
            components.setDay_(due_dt.day)
            components.setHour_(due_dt.hour)
            components.setMinute_(due_dt.minute)
            components.setSecond_(0)
            reminder.setDueDateComponents_(components)
            nsdate = Foundation.NSDate.dateWithTimeIntervalSince1970_(due_dt.timestamp())
            reminder.addAlarm_(EventKit.EKAlarm.alarmWithAbsoluteDate_(nsdate))
        ok, err = store.saveReminder_commit_error_(reminder, True, None)
        if not ok:
            return {"ok": False, "title": title, "list": str(calendar.title() or list_name), "error": str(err or "save reminder failed")[:500]}
        return {"ok": True, "title": title, "list": str(calendar.title() or list_name), "due_at": due_at, "backend": "eventkit"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:500]}


def warm_daily_runtime() -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"ok": True}
    try:
        import EventKit

        status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeReminder)
        store = EventKit.EKEventStore.alloc().init()
        calendar = store.defaultCalendarForNewReminders()
        result.update(
            {
                "eventkit": True,
                "reminders_status": int(status),
                "default_reminders_list": str(calendar.title() or "") if calendar is not None else "",
            }
        )
    except Exception as exc:
        result.update({"ok": False, "eventkit": False, "error": str(exc)[:500]})
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def _run_reminders_osascript_with_recovery(script: str) -> subprocess.CompletedProcess[str]:
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(2):
        try:
            return subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=6)
        except subprocess.TimeoutExpired as exc:
            last = subprocess.CompletedProcess(
                exc.cmd,
                124,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=f"Reminders AppleScript timed out after {exc.timeout}s",
            )
            if attempt == 0:
                subprocess.run(["killall", "Reminders"], capture_output=True, text=True, timeout=2)
                time.sleep(0.6)
    return last or subprocess.CompletedProcess(["osascript"], 1, stdout="", stderr="Reminders AppleScript failed")


def build_voice_tools(args: Any, store: Any) -> list[Any]:
    build_args = args

    def add_context_note(note: str) -> str:
        """Save a long-lived memory only when the user explicitly says remember/记住/记得/以后记得/帮我记一下."""
        latest = store.latest_user_transcript()
        if not long_term_memory_requested(latest):
            result = {"result": "context note skipped; user did not explicitly ask to remember", "latest_transcript": latest[:200]}
            store.record_tool_event("add_context_note", {"note": note}, result, ok=False)
            return result["result"]
        result = store.add_context_note(note)
        store.record_tool_event("add_context_note", {"note": note}, {"result": result})
        return result

    def front_note(
        action: str,
        tab: str = "live",
        content: str = "",
        html: str = "",
        media: list[dict[str, Any]] | str | None = None,
        source: str = "agent",
        position: str = "right",
        visible: bool = True,
        width: int = 520,
        height: int = 420,
    ) -> str:
        """Control the floating rich front note. Default to tab=live for sticky notes/cards. Use tab=context only when the user explicitly says context/上下文 note."""
        try:
            latest = store.latest_user_transcript()
            normalized = normalize_front_note_call_args(
                {"action": action, "tab": tab, "content": content, "html": html, "media": media, "position": position, "visible": visible, "width": width, "height": height},
                latest,
            )
            if normalized["tab"] == "live" and normalized["action"] in {"show", "update", "append"}:
                note_text = str(normalized.get("content") or "") or str(normalized.get("html") or "")
                if note_text.strip():
                    store.add_live_note(note_text, session_id=store.session_id)
                state = store.update_front_note(
                    action="show",
                    tab="live",
                    active_tab="live",
                    source=source,
                    allow_empty=True,
                    position=normalized["position"],
                    visible=normalized["visible"],
                    width=normalized["width"],
                    height=normalized["height"],
                )
                result = {"ok": True, "state": state}
                return json.dumps(result, ensure_ascii=False)
            state = store.update_front_note(
                action=normalized["action"],
                tab=normalized["tab"],
                content=normalized["content"],
                html=normalized["html"],
                media=normalized["media"],
                source=source,
                allow_empty=True,
                position=normalized["position"],
                visible=normalized["visible"],
                width=normalized["width"],
                height=normalized["height"],
            )
            result = {"ok": True, "state": state}
            ok = True
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
            ok = False
            store.record_tool_event(
                "front_note",
                {"action": action, "tab": tab, "source": source, "content_chars": len(str(content or "") or str(html or "")), "position": position, "visible": visible},
                result,
                ok=False,
            )
        return json.dumps(result, ensure_ascii=False)

    def update_task_status(title: str, status: str, summary: str = "") -> str:
        """Update a task status. Status should be pending, in_progress, blocked, done, or failed."""
        return store.update_task_status(title, status, summary)

    def mark_task_in_progress(title: str, summary: str = "") -> str:
        """Mark a task as in progress."""
        return store.update_task_status(title, "in_progress", summary)

    def mark_task_blocked(title: str, summary: str = "") -> str:
        """Mark a task as blocked without exposing raw errors to the user."""
        return store.update_task_status(title, "blocked", summary)

    def mark_task_done(title: str, summary: str = "") -> str:
        """Mark a task as done."""
        return store.update_task_status(title, "done", summary)

    def trigger_fast_followup(prompt: str, priority: int = 10) -> str:
        """Ask the fast voice assistant to proactively speak a new user-facing followup."""
        return store.trigger_fast_followup(prompt, priority)

    def fetch_weather_once(location: str) -> dict[str, Any]:
        url = f"https://wttr.in/{parse.quote(location)}?format=j1"
        req = request.Request(url, headers={"User-Agent": "JenVoice/0.1"})
        payload = json.loads(urlopen_text(req, timeout=12.0, verify_tls=True, label="weather"))
        current = (payload.get("current_condition") or [{}])[0]
        nearest = (payload.get("nearest_area") or [{}])[0]
        area = ((nearest.get("areaName") or [{}])[0].get("value") if nearest else "") or location
        country = ((nearest.get("country") or [{}])[0].get("value") if nearest else "") or ""
        return {
            "ok": True,
            "location": location,
            "resolved_location": ", ".join(part for part in [area, country] if part),
            "temperature_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "humidity_percent": current.get("humidity"),
            "wind_kmph": current.get("windspeedKmph"),
            "description": ((current.get("weatherDesc") or [{}])[0].get("value") if current else ""),
        }

    def get_weather(location: str) -> str:
        """Fetch current weather for a city or region. Requires an explicit location from the user."""
        location = location.strip()
        if not location:
            result = {"ok": False, "error": "location is required"}
            store.record_tool_event("get_weather", {"location": location}, result, ok=False)
            return json.dumps(result, ensure_ascii=False)
        if not plausible_weather_location(location):
            result = {"ok": False, "location": location, "error": "invalid weather location"}
            store.record_tool_event("get_weather", {"location": location}, result, ok=False)
            return json.dumps(result, ensure_ascii=False)
        try:
            errors: list[dict[str, str]] = []
            for candidate in _weather_query_candidates(location):
                try:
                    result = fetch_weather_once(candidate)
                    if candidate != location:
                        result["location_input"] = location
                        result["location_fallback"] = candidate
                    store.record_tool_event("get_weather", {"location": location, "candidates": _weather_query_candidates(location)}, result)
                    return json.dumps(result, ensure_ascii=False)
                except Exception as exc:
                    errors.append({"location": candidate, "error": str(exc)[:240]})
            message = errors[-1]["error"] if errors else "location not found"
            result = {"ok": False, "location": location, "errors": errors, "error": message}
            store.record_tool_event("get_weather", {"location": location, "candidates": _weather_query_candidates(location)}, result, ok=False)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            result = {"ok": False, "location": location, "error": str(exc)[:1000]}
            store.record_tool_event("get_weather", {"location": location}, result, ok=False)
            return json.dumps(result, ensure_ascii=False)

    def current_datetime() -> str:
        """Return the current local datetime, UTC datetime, and timezone."""
        result = {
            "local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "timezone": time.tzname,
        }
        store.record_tool_event("current_datetime", {}, result)
        return json.dumps(result, ensure_ascii=False)

    def daily_action(action: str, target: str = "", args: dict[str, Any] | str | None = None) -> str:
        """Fat daily domain tool. Actions: weather, time, calendar_list, reminder_list, reminder_create, note_live, note_context, memory, map."""
        safe_args = parse_jsonish_value(args) if isinstance(args, str) else (args or {})
        if not isinstance(safe_args, dict):
            safe_args = {}
        normalized = _normalize_daily_action(action)
        target_text = str(target or safe_args.get("target") or "").strip()

        def done(result: dict[str, Any]) -> str:
            result.setdefault("action", normalized)
            result.setdefault("target", target_text)
            ok = bool(result.get("ok", True))
            store.record_tool_event("daily_action", {"action": normalized, "target": target_text, "args": safe_args}, result, ok=ok)
            return json.dumps(result, ensure_ascii=False)

        try:
            if normalized == "weather":
                location = target_text or str(safe_args.get("location") or "").strip()
                if location and not plausible_weather_location(location):
                    location = ""
                if not location and bool(safe_args.get("use_current_location")):
                    current = _current_address_result(float(safe_args.get("timeout_seconds") or 2.0))
                    if current.get("ok"):
                        location = str(current.get("address") or "").strip()
                    else:
                        return done({"ok": False, "error": "需要位置权限或明确地点", "location_probe": current})
                if not location:
                    current = _current_address_result(float(safe_args.get("timeout_seconds") or 2.0))
                    if current.get("ok"):
                        location = str(current.get("address") or "").strip()
                        safe_args = {**safe_args, "location_source": "current_address"}
                    else:
                        return done({"ok": False, "error": "地点没识别对，需要位置权限或明确地点", "location_probe": current})
                payload = parse_jsonish_value(get_weather(location))
                return done({"ok": bool(isinstance(payload, dict) and payload.get("ok", True)), "weather": payload})
            if normalized == "time":
                return done({"ok": True, "time": parse_jsonish_value(current_datetime())})
            if normalized == "calendar_list":
                days = safe_args.get("days", 7)
                return done({"ok": True, "calendar": parse_jsonish_value(calendar_events(days))})
            if normalized == "reminder_list":
                return done({"ok": True, "reminders": parse_jsonish_value(reminders_list())})
            if normalized == "reminder_create":
                title = target_text or str(safe_args.get("title") or safe_args.get("content") or "").strip()
                return done(_create_reminder(title, safe_args))
            if normalized == "note_live":
                content = target_text or str(safe_args.get("content") or "").strip()
                return done({"ok": True, "note": parse_jsonish_value(front_note("append", tab="live", content=content))})
            if normalized == "note_context":
                latest = store.latest_user_transcript()
                if not front_note_context_requested(latest):
                    return done({"ok": False, "error": "context note requires explicit context/上下文 request"})
                content = target_text or str(safe_args.get("content") or "").strip()
                return done({"ok": True, "note": parse_jsonish_value(front_note("append", tab="context", content=content))})
            if normalized == "memory":
                note = target_text or str(safe_args.get("note") or safe_args.get("content") or "").strip()
                payload = add_context_note(note)
                return done({"ok": not str(payload).startswith("context note skipped"), "memory": payload})
            if normalized == "map":
                return done(_daily_map(target_text, safe_args))
            return done({"ok": False, "error": f"unsupported daily action: {action}"})
        except Exception as exc:
            return done({"ok": False, "error": str(exc)[:1000]})

    def system_status() -> str:
        """Return a concise macOS system status: uptime, load, memory pressure, disk, and top CPU processes."""
        commands = {
            "uptime": ["uptime"],
            "memory_pressure": ["memory_pressure"],
            "disk": ["df", "-h", "/"],
            "top_cpu": ["ps", "-arcwwwxo", "pid,pcpu,pmem,comm", "-r"],
        }
        result: dict[str, Any] = {}
        ok = True
        for name, cmd in commands.items():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                text = (out.stdout or out.stderr).strip()
                if name == "top_cpu":
                    text = "\n".join(text.splitlines()[:8])
                result[name] = text[:4000]
            except Exception as exc:
                ok = False
                result[name] = f"error: {exc}"
        store.record_tool_event("system_status", {}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def computer_action(action: str, target: str = "", args: dict[str, Any] | str | None = None) -> str:
        """Fat computer domain tool for local computer-use. Actions: open_app, close_app, focus_app, arrange_workspace, run_shell, run_osascript, computer_use, screenshot, develop_app, delegate_to_codex."""
        raw_args = parse_jsonish_value(args) if isinstance(args, str) else (args or {})
        safe_args = raw_args if isinstance(raw_args, dict) else {}
        normalized = str(action or safe_args.get("action") or "").strip().lower()
        delegate_to_codex = normalized in {"delegate_to_codex", "delegate", "codex_fallback", "fallback_to_codex", "ask_codex"}
        normalized = {
            "open": "open_app",
            "launch": "open_app",
            "activate": "focus_app",
            "focus": "focus_app",
            "front": "focus_app",
            "close": "close_app",
            "quit": "close_app",
            "arrange": "arrange_workspace",
            "layout": "arrange_workspace",
            "shell": "run_shell",
            "bash": "run_shell",
            "osascript": "run_osascript",
            "apple_script": "run_osascript",
            "gui": "computer_use",
            "computer-use": "computer_use",
            "computeruse": "computer_use",
            "develop": "develop_app",
            "develop_app": "develop_app",
            "coding": "develop_app",
            "coding_submit": "develop_app",
            "coding_submit_task": "develop_app",
            "submit_task": "develop_app",
            "delegate_to_codex": "develop_app",
            "delegate": "develop_app",
            "codex_fallback": "develop_app",
            "fallback_to_codex": "develop_app",
            "ask_codex": "develop_app",
            "open_program": "open_program",
            "open_workspace": "open_program",
            "rerun_workspace": "rerun_workspace",
            "list_workspaces": "list_workspaces",
            "inspect_workspace": "inspect_workspace",
        }.get(normalized, normalized)
        target_text = str(target or safe_args.get("target") or safe_args.get("app") or "").strip()
        if delegate_to_codex:
            safe_args = dict(safe_args)
            safe_args.setdefault("executor", "codex")
            safe_args.setdefault("prompt", str(safe_args.get("prompt") or target_text or target or "").strip())
            if not target_text:
                target_text = str(safe_args.get("prompt") or "Codex fallback")[:80]
        audit_args = {"action": "delegate_to_codex" if delegate_to_codex else normalized, "target": target_text, "args": safe_args}
        if normalized in {"develop_app", "list_workspaces", "inspect_workspace", "rerun_workspace", "update_workspace_summary"}:
            coding_action_name = {
                "develop_app": "submit_task",
                "list_workspaces": "list_workspaces",
                "inspect_workspace": "inspect_workspace",
                "rerun_workspace": "rerun_workspace",
                "update_workspace_summary": "update_workspace_summary",
            }[normalized]
            coding_args = dict(safe_args)
            coding_args["_audit_tool"] = "computer_action"
            coding_args["_computer_action"] = "delegate_to_codex" if delegate_to_codex else normalized
            result = parse_jsonish_value(coding_action(coding_action_name, target_text, coding_args))
            if not isinstance(result, dict):
                result = {"ok": False, "domain": "computer", "action": normalized, "error": str(result)}
            result.setdefault("domain", "computer")
            result["action"] = "delegate_to_codex" if delegate_to_codex else normalized
            result["subdomain"] = "coding"
            label = _coding_executor_label(str(result.get("executor") or "codex"))
            if result.get("ok"):
                result.setdefault("_log_word", f"{label} 开工了" if normalized == "develop_app" else "找到了")
                result.setdefault("_summary", f"{label} 开工了" if normalized == "develop_app" else "找到了")
            else:
                result.setdefault("_log_word", f"{label} 没启动" if normalized == "develop_app" else "没找到")
                result.setdefault("_summary", f"{result['_log_word']}：{str(result.get('error') or '失败')[:40]}")
            return json.dumps(result, ensure_ascii=False)
        if normalized == "open_program":
            query = target_text or str(safe_args.get("query") or "").strip()
            coding_root = _coding_workdir(build_args)
            workspace_root = coding_root / "workspaces" if getattr(build_args, "coding_workdir", None) else coding_root
            workspace_index = CodingWorkspaceIndex(store, workspace_root)
            found = resolve_workspace_program(query, workspace_index.list(limit=100)) if query else None
            if found:
                program = found.get("program") if isinstance(found.get("program"), dict) else {}
                open_method = program.get("open_method") if isinstance(program.get("open_method"), dict) else {}
                entrypoint = str(open_method.get("entrypoint") or "").strip()
                entry = Path(str(found["path"])) / entrypoint if entrypoint else None
                if entry and entry.exists():
                    venv = Path(str((open_method.get("env") or {}).get("VIRTUAL_ENV") or resolve_host_venv()))
                    launch = CodingAppRunner(venv=venv).launch(entry)
                    ok = bool(launch.get("ok"))
                    if ok:
                        manifest = read_manifest(Path(str(found["path"]))) or found
                        manifest = register_workspace_program(manifest, entry=entry, active_venv=venv, launch_result=launch, status="ready")
                        workspace_index.upsert(manifest)
                    result = {
                        "ok": ok,
                        "domain": "computer",
                        "action": normalized,
                        "subdomain": "program",
                        "target": query,
                        "program": _public_program_payload(program),
                        "workspace": _public_program_payload(found),
                        "launch": _public_program_payload(launch),
                        "_log_word": "打开了" if ok else "没打开",
                        "_summary": "打开了" if ok else f"没打开：{str(launch.get('error') or '启动失败')[:40]}",
                    }
                    store.record_tool_event("computer_action", audit_args, result, ok=ok)
                    return json.dumps(result, ensure_ascii=False)
            result = {
                "ok": False,
                "domain": "computer",
                "action": normalized,
                "target": query,
                "error": "no registered program open_method matched",
                "candidates": workspace_index.list(limit=10),
                "_log_word": "没打开",
                "_summary": "没打开：没找到注册的打开方式",
            }
            store.record_tool_event("computer_action", audit_args, result, ok=False)
            return json.dumps(result, ensure_ascii=False)
        if normalized in {"open_app", "focus_app", "close_app"} and target_text:
            coding_root = _coding_workdir(build_args)
            workspace_root = coding_root / "workspaces" if getattr(build_args, "coding_workdir", None) else coding_root
            workspace_index = CodingWorkspaceIndex(store, workspace_root)
            found_program = resolve_workspace_program(target_text, workspace_index.list(limit=100))
            if found_program:
                program = found_program.get("program") if isinstance(found_program.get("program"), dict) else {}
                if normalized == "close_app":
                    workspace_path = Path(str(found_program.get("path") or ""))
                    ok, pid, error = _terminate_program_workspace(workspace_path, program)
                    result = {
                        "ok": ok,
                        "domain": "computer",
                        "action": normalized,
                        "subdomain": "program",
                        "target": target_text,
                        "program": program,
                        "workspace": found_program,
                        "pid": pid,
                        "error": error,
                        "_log_word": "关掉了" if ok else "没关掉",
                        "_summary": "关掉了" if ok else f"没关掉：{error or '没有可关闭进程'}",
                    }
                    store.record_tool_event("computer_action", audit_args, result, ok=ok)
                    return json.dumps(result, ensure_ascii=False)
                open_method = program.get("open_method") if isinstance(program.get("open_method"), dict) else {}
                entrypoint = str(open_method.get("entrypoint") or "").strip()
                entry = Path(str(found_program["path"])) / entrypoint if entrypoint else None
                if entry and entry.exists():
                    venv = Path(str((open_method.get("env") or {}).get("VIRTUAL_ENV") or resolve_host_venv()))
                    launch = CodingAppRunner(venv=venv).launch(entry)
                    ok = bool(launch.get("ok"))
                    if ok:
                        manifest = read_manifest(Path(str(found_program["path"]))) or found_program
                        manifest = register_workspace_program(manifest, entry=entry, active_venv=venv, launch_result=launch, status="ready")
                        workspace_index.upsert(manifest)
                    result = {
                        "ok": ok,
                        "domain": "computer",
                        "action": normalized,
                        "subdomain": "program",
                        "target": target_text,
                        "program": _public_program_payload(program),
                        "workspace": _public_program_payload(found_program),
                        "launch": _public_program_payload(launch),
                        "_log_word": "打开了" if ok else "没打开",
                        "_summary": "打开了" if ok else f"没打开：{str(launch.get('error') or '启动失败')[:40]}",
                    }
                    store.record_tool_event("computer_action", audit_args, result, ok=ok)
                    return json.dumps(result, ensure_ascii=False)
        if normalized in {"open_app", "focus_app", "close_app"}:
            if not target_text:
                result = {"ok": False, "domain": "computer", "action": normalized, "error": "target app is required"}
                store.record_tool_event("computer_action", audit_args, result, ok=False)
                return json.dumps(result, ensure_ascii=False)
            command = "quit" if normalized == "close_app" else "activate"
            script = f'''
            tell application {applescript_quote(target_text)}
              {command}
            end tell
            return {applescript_quote(command + " " + target_text)}
            '''
            osascript_result = run_osascript_tool("computer_action", script, audit_args, store)
            result = {
                "ok": int(osascript_result.get("returncode", 1)) == 0,
                "domain": "computer",
                "action": normalized,
                "target": target_text,
                "result": osascript_result,
            }
            return json.dumps(result, ensure_ascii=False)
        if normalized == "arrange_workspace":
            query = str(safe_args.get("query") or target_text or store.latest_user_transcript() or "").strip()
            app_names = safe_args.get("app_names")
            if app_names is None and target_text:
                app_names = [target_text]
            return arrange_workspace(
                query=query,
                app_names=app_names,
                mode=str(safe_args.get("mode") or "auto"),
                open_if_missing=bool(safe_args.get("open_if_missing", True)),
                max_windows=int(safe_args.get("max_windows") or 4),
            )
        if normalized == "run_osascript":
            script = str(safe_args.get("script") or target_text or "").strip()
            if not script:
                result = {"ok": False, "domain": "computer", "action": normalized, "error": "script is required"}
                store.record_tool_event("computer_action", audit_args, result, ok=False)
                return json.dumps(result, ensure_ascii=False)
            script = normalize_browser_osascript(script)
            result = run_osascript_tool("computer_action", script, audit_args, store)
            return json.dumps({"ok": int(result.get("returncode", 1)) == 0, "domain": "computer", "action": normalized, "result": result}, ensure_ascii=False)
        if normalized == "run_shell":
            command = str(safe_args.get("command") or target_text or "").strip()
            timeout_seconds = max(1, min(int(safe_args.get("timeout_seconds") or 20), 60))
            workdir = Path(str(safe_args.get("cwd") or Path.cwd())).expanduser()
            if not command:
                result = {"ok": False, "domain": "computer", "action": normalized, "error": "command is required"}
                store.record_tool_event("computer_action", audit_args, result, ok=False)
                return json.dumps(result, ensure_ascii=False)
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(workdir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    executable=os.environ.get("SHELL") or "/bin/zsh",
                )
                result = {
                    "ok": proc.returncode == 0,
                    "domain": "computer",
                    "action": normalized,
                    "returncode": proc.returncode,
                    "stdout": (proc.stdout or "")[:12000],
                    "stderr": (proc.stderr or "")[:6000],
                    "cwd": str(workdir),
                }
                ok = proc.returncode == 0
            except Exception as exc:
                result = {"ok": False, "domain": "computer", "action": normalized, "cwd": str(workdir), "error": str(exc)}
                ok = False
            store.record_tool_event("computer_action", audit_args, result, ok=ok)
            return json.dumps(result, ensure_ascii=False)
        if normalized == "screenshot":
            screenshot_dir = args.tool_workdir.resolve() / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            filename = str(safe_args.get("filename") or f"screenshot-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
            if not safe_name.lower().endswith(".png"):
                safe_name += ".png"
            path = screenshot_dir / safe_name
            try:
                proc = subprocess.run(["screencapture", "-x", str(path)], capture_output=True, text=True, timeout=10)
                result = {"ok": proc.returncode == 0, "domain": "computer", "action": normalized, "path": str(path), "stderr": (proc.stderr or "")[:2000]}
                ok = proc.returncode == 0
            except Exception as exc:
                result = {"ok": False, "domain": "computer", "action": normalized, "path": str(path), "error": str(exc)}
                ok = False
            store.record_tool_event("computer_action", audit_args, result, ok=ok)
            return json.dumps(result, ensure_ascii=False)
        if normalized == "computer_use":
            operation = str(safe_args.get("operation") or safe_args.get("op") or "").strip().lower()
            app = str(safe_args.get("app") or target_text or "").strip()
            script = ""
            if operation in {"activate", "focus_app"} and app:
                script = f'tell application {applescript_quote(app)} to activate\nreturn "activated"'
            elif operation in {"keystroke", "type_text"}:
                text_value = str(safe_args.get("text") or "")
                modifiers = _osascript_modifier_list(safe_args.get("modifiers"))
                suffix = f" using {modifiers}" if modifiers else ""
                script = f'''
                tell application "System Events"
                  keystroke {applescript_quote(text_value)}{suffix}
                end tell
                return "keystroke"
                '''
            elif operation in {"key_code", "press_key"}:
                key_code = int(safe_args.get("key_code") or safe_args.get("code") or 0)
                modifiers = _osascript_modifier_list(safe_args.get("modifiers"))
                suffix = f" using {modifiers}" if modifiers else ""
                script = f'''
                tell application "System Events"
                  key code {key_code}{suffix}
                end tell
                return "key_code"
                '''
            elif operation == "menu_click":
                if not app:
                    result = {"ok": False, "domain": "computer", "action": normalized, "error": "app is required for menu_click"}
                    store.record_tool_event("computer_action", audit_args, result, ok=False)
                    return json.dumps(result, ensure_ascii=False)
                menu = str(safe_args.get("menu") or "").strip()
                item = str(safe_args.get("item") or "").strip()
                if not menu or not item:
                    result = {"ok": False, "domain": "computer", "action": normalized, "error": "menu and item are required for menu_click"}
                    store.record_tool_event("computer_action", audit_args, result, ok=False)
                    return json.dumps(result, ensure_ascii=False)
                script = f'''
                tell application {applescript_quote(app)} to activate
                tell application "System Events"
                  tell process {applescript_quote(app)}
                    click menu item {applescript_quote(item)} of menu {applescript_quote(menu)} of menu bar 1
                  end tell
                end tell
                return "menu_click"
                '''
            if not script:
                result = {
                    "ok": False,
                    "domain": "computer",
                    "action": normalized,
                    "operation": operation,
                    "error": "unsupported computer_use operation",
                    "supported_operations": ["activate", "keystroke", "key_code", "menu_click"],
                }
                store.record_tool_event("computer_action", audit_args, result, ok=False)
                return json.dumps(result, ensure_ascii=False)
            result = run_osascript_tool("computer_action", script, audit_args, store)
            return json.dumps({"ok": int(result.get("returncode", 1)) == 0, "domain": "computer", "action": normalized, "operation": operation, "result": result}, ensure_ascii=False)
        result = {
            "ok": False,
            "domain": "computer",
            "action": normalized,
            "target": target_text,
            "error": "unsupported computer action",
            "supported_actions": ["open_app", "close_app", "focus_app", "arrange_workspace", "run_shell", "run_osascript", "computer_use", "screenshot", "develop_app", "delegate_to_codex"],
        }
        store.record_tool_event("computer_action", audit_args, result, ok=False)
        return json.dumps(result, ensure_ascii=False)

    def shell_command(command: str, timeout_seconds: int = 20, cwd: str = "") -> str:
        """Run a local shell command. Use for user-requested local diagnostics; output is capped."""
        timeout_seconds = max(1, min(int(timeout_seconds or 20), 60))
        workdir = Path(cwd).expanduser() if cwd else Path.cwd()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                executable=os.environ.get("SHELL") or "/bin/zsh",
            )
            result = {
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[:12000],
                "stderr": (proc.stderr or "")[:6000],
                "cwd": str(workdir),
            }
            ok = proc.returncode == 0
        except Exception as exc:
            result = {"error": str(exc), "cwd": str(workdir)}
            ok = False
        store.record_tool_event("shell_command", {"command": command, "timeout_seconds": timeout_seconds, "cwd": str(workdir)}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def list_path(path: str = ".", max_entries: int = 100) -> str:
        """List a local directory path."""
        target = Path(path).expanduser()
        max_entries = max(1, min(int(max_entries or 100), 500))
        try:
            entries = []
            for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:max_entries]:
                stat = child.stat()
                entries.append({"name": child.name, "path": str(child), "is_dir": child.is_dir(), "size": stat.st_size})
            result = {"path": str(target), "entries": entries}
            ok = True
        except Exception as exc:
            result = {"path": str(target), "error": str(exc)}
            ok = False
        store.record_tool_event("list_path", {"path": path, "max_entries": max_entries}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def read_path(path: str, max_chars: int = 20000) -> str:
        """Read a local text file, capped by max_chars."""
        target = Path(path).expanduser()
        max_chars = max(1, min(int(max_chars or 20000), 100000))
        try:
            text = target.read_text(encoding="utf-8", errors="replace")[:max_chars]
            result = {"path": str(target), "content": text, "truncated": target.stat().st_size > len(text.encode("utf-8", errors="ignore"))}
            ok = True
        except Exception as exc:
            result = {"path": str(target), "error": str(exc)}
            ok = False
        store.record_tool_event("read_path", {"path": path, "max_chars": max_chars}, {"ok": ok, "path": str(target), "chars": len(result.get("content", ""))}, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def resolve_tool_path(path: str) -> Path:
        raw = Path(path)
        if raw.is_absolute() or str(path).startswith("~"):
            return raw.expanduser()
        return (args.tool_workdir / raw).resolve()

    def write_text_file(path: str, content: str, append: bool = False) -> str:
        """Write or append text to a local file. Relative paths are written under the tool base_dir."""
        target = resolve_tool_path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if append:
                with target.open("a", encoding="utf-8") as f:
                    f.write(content)
            else:
                target.write_text(content, encoding="utf-8")
            result = {"path": str(target), "bytes": target.stat().st_size, "append": bool(append)}
            latest = store.latest_user_transcript()
            if front_note_requested(latest) and not front_note_context_requested(latest):
                note_state = store.update_front_note(
                    action="append" if append else "update",
                    tab="live",
                    content=content,
                    source="write_text_file_mirror",
                    allow_empty=True,
                    visible=True,
                )
                result["front_note_mirror"] = {"tab": "live", "version": (note_state.get("live") or {}).get("version")}
            ok = True
        except Exception as exc:
            result = {"path": str(target), "error": str(exc)}
            ok = False
        store.record_tool_event("write_text_file", {"path": path, "append": append, "content_chars": len(content)}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def launch_python_script(path: str, args: list[str] | None = None) -> str:
        """Launch a Python script from the tool base_dir without blocking. Use for GUI demos, floating windows, or animations."""
        base_dir = args.tool_workdir.resolve()
        target = resolve_tool_path(path).resolve()
        try:
            target.relative_to(base_dir)
            if target.suffix != ".py":
                raise ValueError("only .py scripts can be launched")
            if not target.exists():
                raise FileNotFoundError(str(target))
            log_dir = base_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            stdout_path = log_dir / f"{target.stem}-{stamp}.stdout.log"
            stderr_path = log_dir / f"{target.stem}-{stamp}.stderr.log"
            argv = [sys.executable, str(target), *(str(arg) for arg in (args or []))]
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(base_dir),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    start_new_session=True,
                )
            time.sleep(0.8)
            returncode = proc.poll()
            if returncode is None:
                _reap_background_process(proc)
                result = {
                    "launched": True,
                    "path": str(target),
                    "pid": proc.pid,
                    "base_dir": str(base_dir),
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                }
                ok = True
            else:
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000] if stderr_path.exists() else ""
                stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")[:2000] if stdout_path.exists() else ""
                result = {
                    "launched": False,
                    "path": str(target),
                    "returncode": returncode,
                    "base_dir": str(base_dir),
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                }
                ok = False
        except Exception as exc:
            result = {"launched": False, "path": str(target), "base_dir": str(base_dir), "error": str(exc)}
            ok = False
        store.record_tool_event("launch_python_script", {"path": path, "args": args or []}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def coding_action(action: str = "submit_task", target: str = "", args: dict[str, Any] | str | None = None) -> str:
        """Submit a local Codex coding task asynchronously."""
        normalized = str(action or "").strip().lower().replace("-", "_")
        if normalized in {"", "submit", "submit_task", "codex_develop", "develop", "develop_app", "coding_submit_task"}:
            normalized = "submit_task"
        target_text = str(target or "").strip()
        raw_args = parse_jsonish_value(args) if isinstance(args, str) else args
        safe_args = raw_args if isinstance(raw_args, dict) else {}
        audit_tool = str(safe_args.pop("_audit_tool", "coding_action") or "coding_action")
        computer_action_name = str(safe_args.pop("_computer_action", "") or "")
        audit_args: dict[str, Any] = {"action": computer_action_name or normalized, "target": target_text, "subdomain": "coding", "coding_action": normalized}

        def finish(result: dict[str, Any], ok: bool) -> str:
            if audit_tool == "computer_action":
                result.setdefault("domain", "computer")
                result.setdefault("subdomain", "coding")
                result.setdefault("coding_action", normalized)
                if normalized == "submit_task":
                    label = _coding_executor_label(str(result.get("executor") or "codex"))
                    result.setdefault("_log_word", f"{label} 开工了" if ok else f"{label} 没启动")
                    result.setdefault("_summary", f"{label} 开工了" if ok else f"{label} 没启动：{str(result.get('error') or '失败')[:40]}")
                else:
                    result.setdefault("_log_word", "找到了" if ok else "没找到")
                    result.setdefault("_summary", "找到了" if ok else f"没找到：{str(result.get('error') or '失败')[:40]}")
            store.record_tool_event(audit_tool, audit_args, result, ok=ok)
            return json.dumps(result, ensure_ascii=False)

        coding_root = _coding_workdir(build_args)
        run_root = coding_root / "runs" if getattr(build_args, "coding_workdir", None) else coding_root.parent / "coding_runs"
        workspace_root = coding_root / "workspaces" if getattr(build_args, "coding_workdir", None) else coding_root
        cache_root = _coding_cache_dir(build_args)
        run_root.mkdir(parents=True, exist_ok=True)
        workspace_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "uv").mkdir(parents=True, exist_ok=True)
        (cache_root / "npm").mkdir(parents=True, exist_ok=True)
        workspace_index = CodingWorkspaceIndex(store, workspace_root)
        service_registry = CodingServiceRegistry(workspace_index)

        if normalized in {"list", "list_workspaces"}:
            query = str(safe_args.get("query") or target_text or "").strip()
            workspaces = workspace_index.list(limit=int(safe_args.get("limit") or 20))
            if query:
                ranked = rank_workspaces(query, workspaces)
                items = [{**workspace, "score": round(score, 3)} for score, workspace in ranked[:20]]
            else:
                items = workspaces
            return finish({"ok": True, "action": "list_workspaces", "query": query, "workspaces": items}, True)

        if normalized in {"inspect", "inspect_workspace"}:
            workspace_id = str(safe_args.get("workspace_id") or "").strip()
            path = str(safe_args.get("path") or safe_args.get("cwd") or "").strip()
            query = target_text or str(safe_args.get("query") or "").strip()
            found = workspace_index.get(workspace_id=workspace_id or None, path=path or None)
            if not found and query:
                ranked = rank_workspaces(query, workspace_index.list(limit=100))
                found = ranked[0][1] if ranked else None
                if found:
                    found = {**found, "score": round(ranked[0][0], 3)}
            if not found:
                return finish({"ok": False, "action": "inspect_workspace", "error": "workspace not found", "query": query}, False)
            services = service_registry.probe_services(Path(str(found["path"])))
            if services:
                found["services"] = services
            elif isinstance(found.get("services"), list):
                found["services"] = [
                    {**service, "status": service.get("status") or "down"}
                    for service in found["services"]
                    if isinstance(service, dict)
                ]
            return finish({"ok": True, "action": "inspect_workspace", "workspace": found, "manifest_path": str(Path(str(found["path"])) / ".gjallarhorn" / "workspace.json")}, True)

        if normalized in {"rerun", "rerun_workspace"}:
            workspace_id = str(safe_args.get("workspace_id") or "").strip()
            path = str(safe_args.get("path") or safe_args.get("cwd") or "").strip()
            query = target_text or str(safe_args.get("query") or "").strip()
            found = workspace_index.get(workspace_id=workspace_id or None, path=path or None)
            if not found and query:
                ranked = rank_workspaces(query, workspace_index.list(limit=100))
                found = ranked[0][1] if ranked else None
            if not found:
                return finish({"ok": False, "action": "rerun_workspace", "error": "workspace not found", "query": query}, False)
            runner = CodingAppRunner()
            entry = runner.find_entry({"workspace": found["path"], "cwd": found["path"]})
            if not entry:
                return finish({"ok": False, "action": "rerun_workspace", "workspace": found, "error": "no runnable app.py/main.py entrypoint found"}, False)
            launch = runner.launch(entry)
            if launch.get("ok"):
                services = service_registry.probe_services(Path(str(found["path"])))
                return finish({"ok": True, "action": "rerun_workspace", "workspace": found, "launch": launch, "services": services}, True)
            return finish({"ok": False, "action": "rerun_workspace", "workspace": found, "launch": launch, "error": launch.get("error")}, False)

        if normalized in {"update_workspace_summary", "update_summary"}:
            workspace_id = str(safe_args.get("workspace_id") or "").strip()
            path = str(safe_args.get("path") or safe_args.get("cwd") or "").strip()
            found = workspace_index.get(workspace_id=workspace_id or None, path=path or None)
            if not found:
                return finish({"ok": False, "action": "update_workspace_summary", "error": "workspace not found"}, False)
            manifest = workspace_index.update_from_files(
                Path(str(found["path"])),
                title=str(safe_args.get("title") or found.get("title") or target_text),
                summary=str(safe_args.get("summary") or found.get("summary") or ""),
                services=safe_args.get("services") if isinstance(safe_args.get("services"), list) else None,
            )
            return finish({"ok": True, "action": "update_workspace_summary", "workspace": manifest}, True)

        if normalized != "submit_task":
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                "error": "unsupported coding_action",
                "supported_actions": ["submit_task", "list_workspaces", "inspect_workspace", "rerun_workspace", "update_workspace_summary"],
            }
            return finish(result, False)

        user_prompt = str(safe_args.get("prompt") or target_text).strip()
        if not user_prompt:
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                "error": "args.prompt or target is required",
            }
            return finish(result, False)
        executor_info = _select_coding_executor(safe_args, store)
        executor = str(executor_info.get("executor") or "codex")
        executor_label = _coding_executor_label(executor)
        executor_model = str(executor_info.get("executor_model") or _coding_executor_model(executor))
        executor_bin = str(executor_info.get("executor_bin") or "")
        executor_fields = {
            "executor": executor,
            "executor_model": executor_model,
            "executor_bin": executor_bin,
            "executor_selection_reason": str(executor_info.get("executor_selection_reason") or ""),
            "executor_fallback": bool(executor_info.get("executor_fallback")),
            **({"executor_fallback_from": executor_info.get("fallback_from")} if executor_info.get("fallback_from") else {}),
            **({"last_executor": executor_info.get("last_executor")} if executor_info.get("last_executor") else {}),
        }
        if not executor_info.get("ok"):
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                **executor_fields,
                "error": str(executor_info.get("error") or "coding executor unavailable"),
            }
            audit_args.update({"prompt_chars": len(user_prompt), **executor_fields})
            return finish(result, False)
        try:
            timeout_seconds = float(safe_args.get("timeout_seconds") or 2.0)
        except Exception:
            timeout_seconds = 2.0
        timeout_seconds = max(0.1, min(timeout_seconds, 10.0))

        slug_source = target_text or user_prompt[:80] or "codex"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug_source).strip("-")[:48] or "codex"
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{slug or 'codex'}"
        task_id = str(safe_args.get("task_id") or uuid.uuid4())
        try:
            selector = CodingWorkspaceSelector(workspace_index, workspace_root)
            workspace_decision = selector.select(target=target_text, prompt=user_prompt, args=safe_args, run_id=stamp)
            cwd = workspace_decision.path
        except Exception as exc:
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                **executor_fields,
                "error": f"workspace selection failed: {exc}",
            }
            audit_args.update({"prompt_chars": len(user_prompt), "timeout_seconds": timeout_seconds, **executor_fields})
            return finish(result, False)
        try:
            cwd.mkdir(parents=True, exist_ok=True)
            pyproject_path = cwd / "pyproject.toml"
            if not pyproject_path.exists():
                pyproject_path.write_text(CODING_ACTION_DEFAULT_PYPROJECT, encoding="utf-8")
            staged_kenney_assets = _stage_kenney_platformer_assets(cwd, build_args)
            cwd_exists = cwd.exists() and cwd.is_dir()
        except Exception:
            staged_kenney_assets = None
            cwd_exists = False
        active_venv = resolve_host_venv()
        codex_home = _coding_codex_home(build_args)
        try:
            _ensure_coding_codex_home(codex_home)
        except Exception as exc:
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                "cwd": str(cwd),
                "workspace": str(cwd),
                "workspace_id": workspace_decision.workspace_id,
                "active_venv": str(active_venv),
                "codex_home": str(codex_home),
                **executor_fields,
                "error": f"isolated codex home setup failed: {exc}",
            }
            return finish(result, False)
        service_registry.probe_services(cwd)
        workspace_manifest = read_manifest(cwd) or workspace_decision.manifest
        workspace_context = workspace_context_for_prompt(workspace_manifest)
        prompt = (
            f"{CODING_ACTION_PROMPT_PREFIX}\n"
            f"Task workspace:\n{cwd}\n"
            f"Coding cache root:\n{cache_root}\n"
            f"UV cache:\n{cache_root / 'uv'}\n"
            f"NPM cache:\n{cache_root / 'npm'}\n"
            f"Staged Kenney platformer assets:\n{staged_kenney_assets or 'not available'}\n"
            f"Kenney semantic manifest:\n{(staged_kenney_assets / 'gjallarhorn_asset_manifest.json') if staged_kenney_assets else 'not available'}\n"
            f"Workspace reused: {json.dumps(workspace_decision.reused, ensure_ascii=False)}\n"
            f"Workspace match score: {workspace_decision.score:.3f}\n"
            f"Workspace decision reason: {workspace_decision.reason}\n"
            f"Workspace context and callable local services:\n{workspace_context}\n"
            f"Active host venv:\n{active_venv}\n"
            "Use this directory as the working directory. If this workspace was reused, continue developing it instead of creating a duplicate project.\n"
            "Keep context small: inspect only files needed for this task, do not scan parent repos, and do not include long logs in your final message.\n"
            "A minimal pyproject.toml has already been created here as workspace metadata only. Do not use it to sync dependencies.\n"
            f"When the implementation is ready, run it with `VIRTUAL_ENV={active_venv} uv run --active --no-sync ...` from this directory before finishing. Do not use bare python for dependency-backed code and do not run uv sync.\n\n"
            "Before finishing, ensure `run.sh` exists at the workspace root and launches the runnable app/service from this workspace using the Active host venv. The voice service will use `bash run.sh` for future open_program calls.\n\n"
            f"User task:\n{user_prompt}"
        )
        audit_args.update({
            "cwd": str(cwd),
            "workspace": str(cwd),
            "workspace_id": workspace_decision.workspace_id,
            "workspace_reused": workspace_decision.reused,
            "workspace_score": round(workspace_decision.score, 3),
            "workspace_reason": workspace_decision.reason,
            "active_venv": str(active_venv),
            "codex_home": str(codex_home),
            "coding_root": str(coding_root),
            "coding_cache_dir": str(cache_root),
            "explicit_cwd": bool(safe_args.get("cwd")),
            "prompt_chars": len(user_prompt),
            "enhanced_prompt_chars": len(prompt),
            "timeout_seconds": timeout_seconds,
            **executor_fields,
        })
        if not cwd_exists:
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                "cwd": str(cwd),
                "workspace": str(cwd),
                "workspace_id": workspace_decision.workspace_id,
                **executor_fields,
                "error": "cwd does not exist or is not a directory",
            }
            return finish(result, False)

        if not host_venv_has_module(active_venv, "webview"):
            result = {
                "ok": False,
                "launched": False,
                "action": normalized,
                "target": target_text,
                "cwd": str(cwd),
                "workspace": str(cwd),
                "workspace_id": workspace_decision.workspace_id,
                "active_venv": str(active_venv),
                **executor_fields,
                "error": "host venv is missing pywebview module",
            }
            return finish(result, False)

        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.jsonl"
        stderr_path = run_dir / "stderr.log"
        executor_log_path = run_dir / f"{executor}.log"
        last_message_path = run_dir / "last_message.txt"
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        model = executor_model
        argv = _codex_executor_argv(executor_bin, model, cwd, last_message_path, prompt)
        executor_env = coding_runtime_env(os.environ.copy(), venv=active_venv)
        executor_env["JEN_CODING_CACHE_DIR"] = str(cache_root)
        executor_env["GJALLARHORN_CODING_CACHE_DIR"] = str(cache_root)
        executor_env["UV_CACHE_DIR"] = str(cache_root / "uv")
        executor_env["npm_config_cache"] = str(cache_root / "npm")
        executor_env["NPM_CONFIG_CACHE"] = str(cache_root / "npm")
        executor_env.setdefault("npm_config_prefer_offline", "true")
        executor_env.setdefault("NPM_CONFIG_PREFER_OFFLINE", "true")
        executor_env["CODEX_HOME"] = str(codex_home)
        executor_env["JEN_CODEX_HOME"] = str(codex_home)
        executor_env["GJALLARHORN_CODEX_HOME"] = str(codex_home)
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(cwd),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=executor_env,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            time.sleep(timeout_seconds)
            returncode = proc.poll()
            if returncode is None:
                _reap_background_process(proc)
                task_payload = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "pid": proc.pid,
                    "status": "running",
                    "workspace_id": workspace_decision.workspace_id,
                    "target": target_text,
                    "cwd": str(cwd),
                    "workspace": str(cwd),
                    "executor": executor,
                    "active_venv": str(active_venv),
                    "codex_home": str(codex_home),
                    "model": model,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "executor_log": str(executor_log_path),
                    "last_message_path": str(last_message_path),
                    "prompt_path": str(prompt_path),
                    "next_speech_at": time.time() + 60.0,
                }
                store.create_coding_task(task_payload)
                store.record_coding_task_status(
                    task_id,
                    "running",
                    f"workspace · {'复用' if workspace_decision.reused else '新建'} {workspace_decision.title} · score={workspace_decision.score:.2f}",
                    {
                        "run_id": run_id,
                        "workspace_id": workspace_decision.workspace_id,
                        "workspace": str(cwd),
                        "workspace_reused": workspace_decision.reused,
                        "workspace_score": round(workspace_decision.score, 3),
                        "workspace_reason": workspace_decision.reason,
                        **executor_fields,
                    },
                )
                store.record_coding_task_status(task_id, "running", f"{executor_label} 开工 · {target_text or '开发任务'}", {"run_id": run_id, **executor_fields})
                result = {
                    "ok": True,
                    "launched": True,
                    "task_id": task_id,
                    "pid": proc.pid,
                    "run_id": run_id,
                    "status": "running",
                    "target": target_text,
                    "cwd": str(cwd),
                    "workspace": str(cwd),
                    "workspace_id": workspace_decision.workspace_id,
                    "workspace_reused": workspace_decision.reused,
                    "workspace_score": round(workspace_decision.score, 3),
                    "workspace_reason": workspace_decision.reason,
                    "workspace_manifest_path": str(cwd / ".gjallarhorn" / "workspace.json"),
                    "active_venv": str(active_venv),
                    "codex_home": str(codex_home),
                    **executor_fields,
                    "model": model,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "executor_log": str(executor_log_path),
                    "last_message_path": str(last_message_path),
                    "prompt_path": str(prompt_path),
                }
                ok = True
            else:
                stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")[:2000] if stdout_path.exists() else ""
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000] if stderr_path.exists() else ""
                result = {
                    "ok": False,
                    "launched": False,
                    "task_id": task_id,
                    "run_id": run_id,
                    "target": target_text,
                    "cwd": str(cwd),
                    "workspace": str(cwd),
                    "workspace_id": workspace_decision.workspace_id,
                    **executor_fields,
                    "model": model,
                    "returncode": returncode,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "executor_log": str(executor_log_path),
                    "last_message_path": str(last_message_path),
                    "prompt_path": str(prompt_path),
                }
                ok = False
        except Exception as exc:
            result = {
                "ok": False,
                "launched": False,
                "task_id": task_id,
                "target": target_text,
                "cwd": str(cwd),
                "workspace": str(cwd),
                "workspace_id": workspace_decision.workspace_id,
                    "active_venv": str(active_venv),
                    "codex_home": str(codex_home),
                    **executor_fields,
                "model": model,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "executor_log": str(executor_log_path),
                "last_message_path": str(last_message_path),
                "prompt_path": str(prompt_path),
                "error": str(exc),
            }
            ok = False
        return finish(result, ok)

    def run_ddgs_search(kind: str, query: str, max_results: int) -> tuple[dict[str, Any], bool]:
        max_results = max(1, min(int(max_results or 5), 10))
        try:
            from ddgs import DDGS

            kwargs: dict[str, Any] = {
                "max_results": max_results,
                "backend": args.web_search_backend,
            }
            if args.web_search_region:
                kwargs["region"] = args.web_search_region
            with DDGS(timeout=args.web_search_timeout) as ddgs:
                raw_results = ddgs.news(query, **kwargs) if kind == "news" else ddgs.text(query, **kwargs)
            results = []
            for row in raw_results[:max_results]:
                link = str(row.get("href") or row.get("url") or "")
                results.append({
                    "title": str(row.get("title") or "").strip(),
                    "url": link,
                    "snippet": str(row.get("body") or row.get("snippet") or "").strip(),
                    **({"date": row.get("date")} if row.get("date") else {}),
                    **({"source": row.get("source")} if row.get("source") else {}),
                })
            video_retry_query = video_search_retry_query(query, results) if kind == "web" else ""
            if video_retry_query:
                retry_kwargs = dict(kwargs)
                retry_kwargs["max_results"] = max_results
                with DDGS(timeout=args.web_search_timeout) as retry_ddgs:
                    retry_raw_results = retry_ddgs.text(video_retry_query, **retry_kwargs)
                seen_urls = {str(row.get("url") or "").strip() for row in results}
                for row in retry_raw_results[:max_results]:
                    link = str(row.get("href") or row.get("url") or "").strip()
                    if not link or link in seen_urls:
                        continue
                    seen_urls.add(link)
                    results.append({
                        "title": str(row.get("title") or "").strip(),
                        "url": link,
                        "snippet": str(row.get("body") or row.get("snippet") or "").strip(),
                        "source_query": video_retry_query,
                        **({"date": row.get("date")} if row.get("date") else {}),
                        **({"source": row.get("source")} if row.get("source") else {}),
                    })
                    if len(results) >= max_results:
                        break
            ok = bool(results)
            result = {
                "query": query,
                "backend": args.web_search_backend,
                "region": args.web_search_region,
                "kind": kind,
                "results": results,
            }
            if video_retry_query:
                result["video_retry_query"] = video_retry_query
            if not results:
                result["error"] = "search returned no results"
        except Exception as exc:
            result = {"query": query, "backend": args.web_search_backend, "kind": kind, "error": str(exc)}
            ok = False
        return result, ok

    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web using ddgs meta-search. Use for current or internet-sourced information."""
        result, ok = run_ddgs_search("web", query, max_results)
        store.record_tool_event("web_search", {"query": query, "max_results": max_results}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def search_news(query: str, max_results: int = 5) -> str:
        """Search recent news using ddgs meta-search."""
        result, ok = run_ddgs_search("news", query, max_results)
        retry_query = news_search_retry_query(query)
        if not ok and retry_query and retry_query != query:
            retry_result, retry_ok = run_ddgs_search("news", retry_query, max_results)
            if retry_ok:
                retry_result["original_query"] = query
                retry_result["retry_query"] = retry_query
                result, ok = retry_result, retry_ok
            else:
                result["retry_query"] = retry_query
                result["retry_error"] = retry_result.get("error")
        store.record_tool_event("search_news", {"query": query, "max_results": max_results, **({"retry_query": retry_query} if retry_query else {})}, result, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def fetch_url(url: str, max_chars: int = 12000) -> str:
        """Fetch a URL and return text content with HTML tags stripped when applicable."""
        max_chars = max(1, min(int(max_chars or 12000), 60000))
        req = request.Request(url, headers={"User-Agent": "Mozilla/5.0 JenVoice/0.1"})
        try:
            raw_bytes = urlopen_bytes(req, timeout=20.0, verify_tls=True, label="fetch url")
            text = raw_bytes.decode("utf-8", errors="replace")
            if "<html" in text[:1000].lower() or "<body" in text[:2000].lower():
                text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
                text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", text))
                text = re.sub(r"\s{2,}", " ", text).strip()
            result = {"url": url, "content": text[:max_chars], "truncated": len(text) > max_chars}
            ok = True
        except Exception as exc:
            result = {"url": url, "error": str(exc)}
            ok = False
        store.record_tool_event("fetch_url", {"url": url, "max_chars": max_chars}, {"ok": ok, "url": url, "chars": len(result.get("content", ""))}, ok=ok)
        return json.dumps(result, ensure_ascii=False)

    def open_url_in_browser(url: str, fullscreen: bool = False, video_fullscreen: bool = False) -> str:
        """Open a URL in Google Chrome. Fullscreen is opt-in because it can move windows to another macOS Space."""
        try:
            subprocess.run(["open", "-a", DEFAULT_BROWSER_APP, url], check=True, timeout=8)
            time.sleep(2.0 if looks_like_video_url(url) else 0.6)
            video_fullscreen_result = (
                press_video_fullscreen_shortcut(click_first=True) if video_fullscreen and looks_like_video_url(url) else {"attempted": False}
            )
            fullscreen_result = set_browser_fullscreen() if fullscreen and not video_fullscreen_result.get("ok") else {"attempted": False}
            result = {
                "opened": True,
                "url": url,
                "browser": DEFAULT_BROWSER_APP,
                "fullscreen": fullscreen_result,
                "video_fullscreen": video_fullscreen_result,
            }
            ok = True
        except Exception as exc:
            result = {
                "opened": False,
                "url": url,
                "browser": DEFAULT_BROWSER_APP,
                "fullscreen": {"attempted": bool(fullscreen), "ok": False},
                "video_fullscreen": {"attempted": bool(video_fullscreen), "ok": False},
                "error": str(exc),
            }
            ok = False
        store.record_tool_event(
            "open_url_in_browser",
            {"url": url, "browser": DEFAULT_BROWSER_APP, "fullscreen": fullscreen, "video_fullscreen": video_fullscreen},
            result,
            ok=ok,
        )
        return json.dumps(result, ensure_ascii=False)

    def run_osascript(script: str, timeout_seconds: int = 20, purpose: str = "") -> str:
        """Run AppleScript via osascript to operate this Mac's GUI: apps, windows, menus, keys, tabs, delays, and fullscreen."""
        script = normalize_browser_osascript(script)
        timeout_seconds = max(1, min(int(timeout_seconds or 20), 120))
        purpose = str(purpose or "")[:300]
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            result = {
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "").strip()[:12000],
                "stderr": (proc.stderr or "").strip()[:6000],
                "purpose": purpose,
            }
            ok = proc.returncode == 0
        except Exception as exc:
            result = {"error": str(exc), "purpose": purpose}
            ok = False
        store.record_tool_event(
            "run_osascript",
            {"script_chars": len(script), "timeout_seconds": timeout_seconds, "purpose": purpose},
            result,
            ok=ok,
        )
        return json.dumps(result, ensure_ascii=False)

    def capture_camera_snapshot(
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        warmup_seconds: float = 0.25,
        timeout_seconds: int = 8,
        filename: str = "",
    ) -> str:
        """Capture one still image from the Mac camera without opening Camera.app. Saves the JPEG under tool base_dir/camera."""
        started = time.perf_counter()
        device_index = max(0, min(int(device_index or 0), 10))
        width = max(320, min(int(width or 1280), 3840))
        height = max(240, min(int(height or 720), 2160))
        warmup_seconds = max(0.0, min(float(warmup_seconds or 0.0), 3.0))
        timeout_seconds = max(2, min(int(timeout_seconds or 8), 30))
        camera_dir = args.tool_workdir.resolve() / "camera"
        camera_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
            if not safe_name.lower().endswith((".jpg", ".jpeg")):
                safe_name += ".jpg"
        else:
            safe_name = f"snapshot-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
        output_path = camera_dir / safe_name
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            result = {"ok": False, "error": "ffmpeg not found", "path": str(output_path)}
            store.record_tool_event("capture_camera_snapshot", {"device_index": device_index, "width": width, "height": height}, result, ok=False)
            return json.dumps(result, ensure_ascii=False)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "avfoundation",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            f"{device_index}:none",
            "-t",
            str(max(warmup_seconds, 0.1)),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            exists = output_path.exists() and output_path.stat().st_size > 0
            result = {
                "ok": proc.returncode == 0 and exists,
                "path": str(output_path),
                "device_index": device_index,
                "width": width,
                "height": height,
                "bytes": output_path.stat().st_size if exists else 0,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "").strip()[:3000],
            }
            ok = bool(result["ok"])
        except Exception as exc:
            result = {
                "ok": False,
                "path": str(output_path),
                "device_index": device_index,
                "width": width,
                "height": height,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
            }
            ok = False
        store.record_tool_event(
            "capture_camera_snapshot",
            {
                "device_index": device_index,
                "width": width,
                "height": height,
                "warmup_seconds": warmup_seconds,
                "timeout_seconds": timeout_seconds,
                "filename": filename,
            },
            result,
            ok=ok,
        )
        return json.dumps(result, ensure_ascii=False)

    def arrange_workspace(
        query: str = "",
        app_names = None,
        mode: str = "auto",
        open_if_missing: bool = True,
        max_windows: int = 4,
    ) -> str:
        """Find visible app windows and arrange them. Auto mode: 1 full, 2 side-by-side, 3 spiral, 4 rotating main pane. Parallel mode: equal columns."""
        started = time.perf_counter()
        try:
            mode = normalize_workspace_mode(mode)
            max_windows = max(1, min(int(max_windows or 4), 4))
            app_names = normalize_workspace_app_names(app_names)
            terms = window_query_terms(query, app_names)
            fullscreen_exit = exit_browser_fullscreen_if_needed(terms)
            windows, enum_error = enumerate_visible_windows(terms)
            opened_apps: list[str] = []
            if open_if_missing and terms:
                opened_apps = open_missing_workspace_apps(terms, windows)
                if opened_apps:
                    windows, enum_error = enumerate_visible_windows(terms)
            selected, candidates, terms = match_workspace_windows(windows, query, app_names, max_windows)
            ignored = selected[max_windows:]
            selected = selected[:max_windows]
            expected_window_count = expected_workspace_window_count(terms, max_windows)
            if expected_window_count > 1 and len(selected) < expected_window_count:
                result = {
                    "ok": False,
                    "mode": mode,
                    "query": query,
                    "terms": terms,
                    "opened_apps": opened_apps,
                    "fullscreen_exit": fullscreen_exit,
                    "windows": selected,
                    "layout": [],
                    "actions": [],
                    "candidates": candidates,
                    "error": f"only matched {len(selected)} of {expected_window_count} requested windows",
                }
                store.record_tool_event(
                    "arrange_workspace",
                    {"query": query, "app_names": app_names, "mode": mode, "open_if_missing": open_if_missing, "max_windows": max_windows},
                    result,
                    ok=False,
                )
                return json.dumps(result, ensure_ascii=False)
            if not selected:
                result = {
                    "ok": False,
                    "mode": mode,
                    "query": query,
                    "terms": terms,
                    "opened_apps": opened_apps,
                    "fullscreen_exit": fullscreen_exit,
                    "windows": [],
                    "layout": [],
                    "actions": [],
                    "candidates": candidates,
                    "error": enum_error.get("error") or enum_error.get("stderr") or "no matching windows",
                }
                store.record_tool_event(
                    "arrange_workspace",
                    {"query": query, "app_names": app_names, "mode": mode, "open_if_missing": open_if_missing, "max_windows": max_windows},
                    result,
                    ok=False,
                )
                return json.dumps(result, ensure_ascii=False)
            bounds = desktop_usable_bounds()
            try:
                rotation_index = int(store.session_value("arrange_workspace.rotation_index", "0") or "0")
            except ValueError:
                rotation_index = 0
            ordered, next_rotation_index = rotate_windows_for_workspace(selected, rotation_index, mode)
            rects = workspace_layout_rects(len(ordered), bounds, mode=mode, rotation_index=rotation_index)
            actions: list[dict[str, Any]] = []
            for window, rect in zip(ordered, rects):
                actions.append({
                    "app_name": window["app_name"],
                    "window_index": window["window_index"],
                    "title": window.get("title", ""),
                    "old_position": window.get("position", []),
                    "old_size": window.get("size", []),
                    "new_bounds": rect,
                })
            apply_result = apply_workspace_layout(actions)
            ok = not apply_result.get("error") and int(apply_result.get("returncode", 1)) == 0
            if ok and len(selected) == 4 and mode == "auto":
                store.set_session_value("arrange_workspace.rotation_index", str(next_rotation_index))
            result = {
                "ok": ok,
                "mode": mode,
                "query": query,
                "terms": terms,
                "opened_apps": opened_apps,
                "fullscreen_exit": fullscreen_exit,
                "bounds": bounds,
                "windows": ordered,
                "layout": rects,
                "actions": actions,
                "candidates": candidates,
                "ignored": ignored,
                "rotation_index": rotation_index,
                "next_rotation_index": next_rotation_index if ok and len(selected) == 4 and mode == "auto" else rotation_index,
                "apply_result": apply_result,
            }
            if not ok:
                result["error"] = apply_result.get("error") or apply_result.get("stderr") or "window layout failed"
            store.record_tool_event(
                "arrange_workspace",
                {"query": query, "app_names": app_names, "mode": mode, "open_if_missing": open_if_missing, "max_windows": max_windows},
                result,
                ok=ok,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            result = {
                "ok": False,
                "mode": normalize_workspace_mode(mode),
                "query": query,
                "app_names": normalize_workspace_app_names(app_names),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
            }
            store.record_tool_event("arrange_workspace", result, result, ok=False)
            return json.dumps(result, ensure_ascii=False)

    def calendar_events(days: int = 7) -> str:
        """List upcoming macOS Calendar events via AppleScript."""
        days = max(1, min(int(days or 7), 30))
        script = f'''
        set output to ""
        set nowDate to current date
        set endDate to nowDate + ({days} * days)
        tell application "Calendar"
          repeat with cal in calendars
            try
              set evs to (every event of cal whose start date ≥ nowDate and start date ≤ endDate)
              repeat with ev in evs
                set output to output & (summary of ev as text) & " | " & (start date of ev as text) & " | " & (name of cal as text) & linefeed
              end repeat
            end try
          end repeat
        end tell
        return output
        '''
        result = run_osascript_tool("calendar_events", script, {"days": days}, store)
        return json.dumps(result, ensure_ascii=False)

    def reminders_list() -> str:
        """List incomplete macOS Reminders via AppleScript."""
        script = '''
        set output to ""
        tell application "Reminders"
          repeat with l in lists
            repeat with r in (reminders of l whose completed is false)
              set output to output & (name of r as text) & " | " & (name of l as text) & linefeed
            end repeat
          end repeat
        end tell
        return output
        '''
        result = run_osascript_tool("reminders_list", script, {}, store)
        return json.dumps(result, ensure_ascii=False)

    def mail_message(to: str, subject: str, body: str, send: bool = False) -> str:
        """Create a Mail.app draft, or send it when send=true."""
        action = "send newMessage" if send else "set visible of newMessage to true"
        script = f'''
        tell application "Mail"
          set newMessage to make new outgoing message with properties {{subject:{applescript_quote(subject)}, content:{applescript_quote(body)} & return & return}}
          tell newMessage
            make new to recipient at end of to recipients with properties {{address:{applescript_quote(to)}}}
          end tell
          {action}
        end tell
        return "ok"
        '''
        result = run_osascript_tool("mail_message", script, {"to": to, "subject": subject, "body_chars": len(body), "send": send}, store)
        return json.dumps(result, ensure_ascii=False)

    tool_workdir = args.tool_workdir
    tool_workdir.mkdir(parents=True, exist_ok=True)
    return [
        daily_action,
        computer_action,
        web_search,
        search_news,
        trigger_fast_followup,
    ]
