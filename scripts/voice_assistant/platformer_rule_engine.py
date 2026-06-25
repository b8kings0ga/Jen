from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal


DIRECTIONS = ("top", "right", "bottom", "left")
CELL_TYPES = {
    "air",
    "solid",
    "platform",
    "hazard",
    "water",
    "collectible",
    "spawn",
    "goal",
    "decoration",
}
COLLISION_TYPES = {
    "none",
    "solid",
    "platform",
    "one_way",
    "hazard",
    "collectible",
    "trigger",
    "water",
    "ladder",
}
EVENT_ON = {"player_overlap", "player_collide", "enemy_overlap", "timer"}
EVENT_ACTIONS = {"collect", "damage", "kill", "bounce", "open", "checkpoint", "finish"}

DEFAULT_PHYSICS_PROFILE: dict[str, Any] = {
    "engine": "phaser4_arcade",
    "tile_size": 18,
    "gravity_y": 900,
    "player_max_velocity_x": 220,
    "player_acceleration_x": 1400,
    "player_drag_x": 1800,
    "jump_velocity_y": -420,
    "coyote_time_ms": 90,
    "jump_buffer_ms": 110,
    "max_jump_height_tiles": 4,
    "max_jump_distance_tiles": 5,
    "walk_speed_tiles_per_second": 6,
}


def validation_result(ok: bool, errors: list[str] | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {"ok": ok, "errors": errors or [], "warnings": warnings or []}


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sheet_frame_count(sheet: dict[str, Any]) -> int:
    if "frame_count" in sheet:
        return int(sheet["frame_count"])
    return int(sheet.get("columns") or 0) * int(sheet.get("rows") or 0)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sheets = manifest.get("sheets") if isinstance(manifest.get("sheets"), dict) else {}
    tiles = manifest.get("tiles") if isinstance(manifest.get("tiles"), list) else []
    animations = manifest.get("animations") if isinstance(manifest.get("animations"), dict) else {}
    semantic_groups = manifest.get("semantic_groups") if isinstance(manifest.get("semantic_groups"), dict) else {}

    if not sheets:
        errors.append("manifest.sheets is required")
    if not tiles:
        errors.append("manifest.tiles is required for v2")

    tile_ids: set[str] = set()
    for index, tile in enumerate(tiles):
        prefix = f"tiles[{index}]"
        tile_id = str(tile.get("id") or "")
        if not tile_id:
            errors.append(f"{prefix}.id is required")
        elif tile_id in tile_ids:
            errors.append(f"{prefix}.id duplicate: {tile_id}")
        tile_ids.add(tile_id)

        sheet_name = str(tile.get("sheet") or "")
        sheet = sheets.get(sheet_name)
        if not sheet:
            errors.append(f"{prefix}.sheet unknown: {sheet_name}")
        else:
            frame = tile.get("frame")
            if not isinstance(frame, int) or frame < 0 or frame >= _sheet_frame_count(sheet):
                errors.append(f"{prefix}.frame invalid for sheet {sheet_name}: {frame}")

        cell_type = str(tile.get("cell_type") or "")
        if cell_type not in CELL_TYPES:
            errors.append(f"{prefix}.cell_type invalid: {cell_type}")
        if not tile.get("role"):
            errors.append(f"{prefix}.role is required")

        sockets = tile.get("sockets")
        if not isinstance(sockets, dict):
            errors.append(f"{prefix}.sockets is required")
        else:
            for direction in DIRECTIONS:
                values = sockets.get(direction)
                if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
                    errors.append(f"{prefix}.sockets.{direction} must be a non-empty string list")
        socket_blacklist = tile.get("socket_blacklist", {})
        if socket_blacklist is not None and not isinstance(socket_blacklist, dict):
            errors.append(f"{prefix}.socket_blacklist must be a direction map")
        elif isinstance(socket_blacklist, dict):
            for direction, values in socket_blacklist.items():
                if direction not in DIRECTIONS:
                    errors.append(f"{prefix}.socket_blacklist.{direction} invalid direction")
                elif not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
                    errors.append(f"{prefix}.socket_blacklist.{direction} must be a string list")

        collision = tile.get("collision")
        if not isinstance(collision, dict):
            errors.append(f"{prefix}.collision is required")
        else:
            collision_type = str(collision.get("type") or "")
            if collision_type not in COLLISION_TYPES:
                errors.append(f"{prefix}.collision.type invalid: {collision_type}")

        for event_index, event in enumerate(tile.get("events") or []):
            if event.get("on") not in EVENT_ON:
                errors.append(f"{prefix}.events[{event_index}].on invalid")
            if event.get("action") not in EVENT_ACTIONS:
                errors.append(f"{prefix}.events[{event_index}].action invalid")

        animation = tile.get("animation")
        if animation is not None and animation not in animations:
            errors.append(f"{prefix}.animation unknown: {animation}")

    for name, group in semantic_groups.items():
        sheet_name = str(group.get("sheet") or "")
        sheet = sheets.get(sheet_name)
        if not sheet:
            errors.append(f"semantic_groups.{name}.sheet unknown: {sheet_name}")
            continue
        for frame in group.get("frames") or []:
            if not isinstance(frame, int) or frame < 0 or frame >= _sheet_frame_count(sheet):
                errors.append(f"semantic_groups.{name}.frames invalid frame: {frame}")

    for name, animation in animations.items():
        sheet_name = str(animation.get("sheet") or "")
        sheet = sheets.get(sheet_name)
        if not sheet:
            errors.append(f"animations.{name}.sheet unknown: {sheet_name}")
            continue
        frames = animation.get("frames")
        if not isinstance(frames, list) or not frames:
            errors.append(f"animations.{name}.frames must be non-empty")
        else:
            for frame in frames:
                if not isinstance(frame, int) or frame < 0 or frame >= _sheet_frame_count(sheet):
                    errors.append(f"animations.{name}.frames invalid frame: {frame}")

    for rule_index, rule in enumerate(manifest.get("autotile_rules") or []):
        role_map = rule.get("role_map")
        if not isinstance(role_map, dict) or not role_map:
            errors.append(f"autotile_rules[{rule_index}].role_map is required")
            continue
        for role, tile_id in role_map.items():
            if tile_id not in tile_ids:
                errors.append(f"autotile_rules[{rule_index}].role_map.{role} unknown tile id: {tile_id}")

    return validation_result(not errors, errors, warnings)


def generate_logic_map(options: dict[str, Any]) -> dict[str, Any]:
    width = max(12, int(options.get("width") or 80))
    height = max(8, int(options.get("height") or 18))
    difficulty = str(options.get("difficulty") or "easy")
    theme = str(options.get("theme") or "grass")
    seed = int(options.get("seed") if options.get("seed") is not None else 123)
    rng = random.Random(seed)

    grid = [["air" for _ in range(width)] for _ in range(height)]
    floor_y = height - 3
    for y in range(floor_y, height):
        for x in range(width):
            grid[y][x] = "solid"

    spawn = {"x": 2, "y": floor_y - 1}
    goal = {"x": width - 3, "y": floor_y - 1}
    grid[spawn["y"]][spawn["x"]] = "spawn"
    grid[goal["y"]][goal["x"]] = "goal"

    platform_count = 2 if difficulty == "easy" else 4
    for index in range(platform_count):
        length = 4 if difficulty == "easy" else rng.randint(3, 6)
        x0 = min(width - length - 4, 8 + index * max(8, width // (platform_count + 1)))
        y = max(3, floor_y - 4 - (index % 2))
        for x in range(x0, x0 + length):
            grid[y][x] = "platform"
        coin_x = x0 + length // 2
        if y > 1:
            grid[y - 1][coin_x] = "collectible"

    if difficulty != "easy" and width > 28:
        hazard_x = min(width - 8, width // 2)
        if grid[floor_y - 1][hazard_x] == "air":
            grid[floor_y - 1][hazard_x] = "hazard"

    return {
        "version": 1,
        "width": width,
        "height": height,
        "theme": theme,
        "difficulty": difficulty,
        "seed": seed,
        "grid": grid,
        "spawn": spawn,
        "goal": goal,
    }


def _cell(logic_map: dict[str, Any], x: int, y: int) -> str:
    grid = logic_map["grid"]
    if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
        return "air"
    return str(grid[y][x])


def _support_cell(cell: str) -> bool:
    return cell in {"solid", "platform"}


def _walkable_cell(cell: str) -> bool:
    return cell in {"air", "spawn", "goal", "collectible"}


def _surface_nodes(logic_map: dict[str, Any]) -> list[tuple[int, int]]:
    nodes: list[tuple[int, int]] = []
    for y, row in enumerate(logic_map["grid"]):
        for x, cell in enumerate(row):
            if _walkable_cell(str(cell)) and _support_cell(_cell(logic_map, x, y + 1)):
                nodes.append((x, y))
    return nodes


def validate_playability(logic_map: dict[str, Any], physics: dict[str, Any] | None = None) -> dict[str, Any]:
    physics = {**DEFAULT_PHYSICS_PROFILE, **(physics or {})}
    errors: list[str] = []
    spawn = logic_map.get("spawn")
    goal = logic_map.get("goal")
    if not isinstance(spawn, dict):
        errors.append("spawn is required")
    if not isinstance(goal, dict):
        errors.append("goal is required")
    if errors:
        return validation_result(False, errors)

    spawn_xy = (int(spawn["x"]), int(spawn["y"]))
    goal_xy = (int(goal["x"]), int(goal["y"]))
    if not _support_cell(_cell(logic_map, spawn_xy[0], spawn_xy[1] + 1)):
        errors.append("spawn must stand on solid/platform")
    if not _support_cell(_cell(logic_map, goal_xy[0], goal_xy[1] + 1)):
        errors.append("goal must stand on solid/platform")
    for x in range(max(0, spawn_xy[0] - 1), min(int(logic_map["width"]), spawn_xy[0] + 3)):
        if _cell(logic_map, x, spawn_xy[1]) == "hazard":
            errors.append("spawn area contains hazard")

    nodes = set(_surface_nodes(logic_map))
    if spawn_xy not in nodes:
        errors.append("spawn platform node not found")
    if goal_xy not in nodes:
        errors.append("goal platform node not found")
    if errors:
        return validation_result(False, errors)

    max_dx = int(physics.get("max_jump_distance_tiles") or 5)
    max_up = int(physics.get("max_jump_height_tiles") or 4)
    queue = [spawn_xy]
    seen = {spawn_xy}
    while queue:
        current = queue.pop(0)
        if current == goal_xy:
            return validation_result(True)
        for candidate in nodes:
            if candidate in seen:
                continue
            dx = abs(candidate[0] - current[0])
            dy = current[1] - candidate[1]
            if dx <= max_dx and dy <= max_up and candidate[1] - current[1] <= max_up + 2:
                if not _segment_blocked_by_hazard(logic_map, current, candidate):
                    seen.add(candidate)
                    queue.append(candidate)

    return validation_result(False, ["goal is not reachable from spawn"])


def _segment_blocked_by_hazard(logic_map: dict[str, Any], a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a[1] != b[1]:
        return False
    y = a[1]
    start, end = sorted((a[0], b[0]))
    return any(_cell(logic_map, x, y) == "hazard" for x in range(start + 1, end))


def _role_for_cell(logic_map: dict[str, Any], x: int, y: int, cell_type: str) -> str:
    if cell_type in {"solid", "platform"}:
        above_air = _cell(logic_map, x, y - 1) in {"air", "spawn", "goal", "collectible", "hazard"}
        left_air = _cell(logic_map, x - 1, y) == "air"
        right_air = _cell(logic_map, x + 1, y) == "air"
        if above_air and left_air:
            return "top_left"
        if above_air and right_air:
            return "top_right"
        if above_air:
            return "top"
        return "fill"
    return cell_type


def _tiles_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(tile["id"]): tile for tile in manifest.get("tiles") or []}


def _autotile_rule(manifest: dict[str, Any], theme: str, cell_type: str) -> dict[str, Any] | None:
    for rule in manifest.get("autotile_rules") or []:
        if rule.get("theme") == theme and rule.get("cell_type") == cell_type:
            return rule
    return None


def plan_tiles(logic_map: dict[str, Any], manifest: dict[str, Any], theme: str | None = None) -> dict[str, Any]:
    theme = theme or str(logic_map.get("theme") or "grass")
    tiles = _tiles_by_id(manifest)
    visual: list[list[str | None]] = [[None for _ in range(logic_map["width"])] for _ in range(logic_map["height"])]
    for y, row in enumerate(logic_map["grid"]):
        for x, cell in enumerate(row):
            cell_type = str(cell)
            if cell_type in {"air", "spawn", "goal"}:
                continue
            rule = _autotile_rule(manifest, theme, "solid" if cell_type == "solid" else cell_type)
            if not rule:
                continue
            role = _role_for_cell(logic_map, x, y, cell_type)
            role_map = rule.get("role_map") or {}
            tile_id = role_map.get(role) or role_map.get("fill") or role_map.get("top") or role_map.get("single")
            if tile_id in tiles:
                visual[y][x] = str(tile_id)
    return {"width": logic_map["width"], "height": logic_map["height"], "theme": theme, "tiles": visual}


def _compatible(a: list[str], b: list[str], blacklist_a: list[str] | None = None, blacklist_b: list[str] | None = None) -> bool:
    left = set(a).difference(blacklist_b or [])
    right = set(b).difference(blacklist_a or [])
    return bool(left.intersection(right))


def validate_sockets(visual_map: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    tiles = _tiles_by_id(manifest)
    grid = visual_map["tiles"]
    height = len(grid)
    width = len(grid[0]) if height else 0
    for y in range(height):
        for x in range(width):
            tile_id = grid[y][x]
            if not tile_id:
                continue
            tile = tiles[str(tile_id)]
            right_id = grid[y][x + 1] if x + 1 < width else None
            bottom_id = grid[y + 1][x] if y + 1 < height else None
            right_socket = tiles[str(right_id)]["sockets"]["left"] if right_id else ["air"]
            bottom_socket = tiles[str(bottom_id)]["sockets"]["top"] if bottom_id else ["air"]
            right_blacklist = tiles[str(right_id)].get("socket_blacklist", {}).get("left", []) if right_id else []
            bottom_blacklist = tiles[str(bottom_id)].get("socket_blacklist", {}).get("top", []) if bottom_id else []
            if not _compatible(tile["sockets"]["right"], right_socket, tile.get("socket_blacklist", {}).get("right", []), right_blacklist):
                errors.append(f"socket mismatch at ({x},{y}) right: {tile_id} -> {right_id or 'air'}")
            if not _compatible(tile["sockets"]["bottom"], bottom_socket, tile.get("socket_blacklist", {}).get("bottom", []), bottom_blacklist):
                errors.append(f"socket mismatch at ({x},{y}) bottom: {tile_id} -> {bottom_id or 'air'}")
    return validation_result(not errors, errors)


def generate_collision_layer(visual_map: dict[str, Any], manifest: dict[str, Any], tile_size: int | None = None) -> dict[str, Any]:
    tile_size = int(tile_size or manifest.get("tile_size") or DEFAULT_PHYSICS_PROFILE["tile_size"])
    tiles = _tiles_by_id(manifest)
    objects: list[dict[str, Any]] = []
    grid = visual_map["tiles"]
    for y, row in enumerate(grid):
        run_start: int | None = None
        run_type: str | None = None
        for x in range(len(row) + 1):
            tile_id = row[x] if x < len(row) else None
            tile = tiles.get(str(tile_id)) if tile_id else None
            collision_type = str((tile or {}).get("collision", {}).get("type") or "none")
            mergeable = collision_type in {"solid", "platform", "one_way"}
            if mergeable and run_start is None:
                run_start = x
                run_type = collision_type
            elif (not mergeable or collision_type != run_type) and run_start is not None:
                objects.append({
                    "type": run_type,
                    "x": run_start * tile_size,
                    "y": y * tile_size,
                    "width": (x - run_start) * tile_size,
                    "height": tile_size,
                })
                run_start = x if mergeable else None
                run_type = collision_type if mergeable else None
            if tile and collision_type in {"hazard", "collectible", "trigger", "water", "ladder"}:
                objects.append({
                    "type": collision_type,
                    "x": x * tile_size,
                    "y": y * tile_size,
                    "width": tile_size,
                    "height": tile_size,
                    "tile_id": tile_id,
                })
    return {"objects": objects}


def _events_from_logic(logic_map: dict[str, Any], tile_size: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for y, row in enumerate(logic_map["grid"]):
        for x, cell in enumerate(row):
            if cell == "collectible":
                events.append({"type": "collectible", "x": x * tile_size, "y": y * tile_size, "action": "collect", "value": 1})
            elif cell == "hazard":
                events.append({"type": "hazard", "x": x * tile_size, "y": y * tile_size, "action": "damage", "amount": 1})
    goal = logic_map.get("goal") or {}
    events.append({"type": "goal", "x": int(goal.get("x", 0)) * tile_size, "y": int(goal.get("y", 0)) * tile_size, "action": "finish"})
    return events


def emit_runtime_level(
    logic_map: dict[str, Any],
    visual_map: dict[str, Any],
    collision_layer: dict[str, Any],
    manifest: dict[str, Any],
    physics_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    physics = {**DEFAULT_PHYSICS_PROFILE, **(physics_profile or {})}
    tile_size = int(physics.get("tile_size") or manifest.get("tile_size") or 18)
    return {
        "version": 1,
        "tile_size": tile_size,
        "theme": logic_map.get("theme", "grass"),
        "width": logic_map["width"],
        "height": logic_map["height"],
        "layers": {
            "logic": logic_map["grid"],
            "visual": visual_map["tiles"],
            "collision": collision_layer["objects"],
            "events": _events_from_logic(logic_map, tile_size),
        },
        "spawn": logic_map["spawn"],
        "goal": logic_map["goal"],
        "physics_profile": physics,
        "metadata": {
            "seed": logic_map.get("seed"),
            "difficulty": logic_map.get("difficulty"),
            "manifest_version": manifest.get("version", 2),
        },
    }
