from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image
except Exception:  # pragma: no cover - caller handles missing optional dep.
    Image = None  # type: ignore[assignment]


DIRECTIONS = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}
ROTATE_90 = {"top": "right", "right": "bottom", "bottom": "left", "left": "top"}


@dataclass(frozen=True)
class SheetSpec:
    name: str
    path: str
    tile_width: int
    tile_height: int
    columns: int
    rows: int
    frame_count: int


DEFAULT_SHEETS = {
    "terrain": SheetSpec("terrain", "Tilemap/tilemap_packed.png", 18, 18, 20, 9, 180),
    "backgrounds": SheetSpec("backgrounds", "Tilemap/tilemap-backgrounds_packed.png", 24, 24, 8, 3, 24),
    "characters": SheetSpec("characters", "Tilemap/tilemap-characters_packed.png", 24, 24, 9, 3, 27),
}


def _quantize(value: int, step: int = 24) -> int:
    return max(0, min(255, int(round(value / step) * step)))


def _color_bucket(pixel: tuple[int, int, int, int]) -> str:
    r, g, b, a = pixel
    if a < 32:
        return "transparent"
    if _is_dark(pixel):
        return "outline"
    return f"rgb:{_quantize(r)}:{_quantize(g)}:{_quantize(b)}"


def _is_dark(pixel: tuple[int, int, int, int]) -> bool:
    r, g, b, a = pixel
    return a >= 32 and r <= 76 and g <= 88 and b <= 96


def _dominant(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "transparent"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _stable_digest(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _crop_frame(image: Any, spec: SheetSpec, frame: int) -> Any:
    col = frame % spec.columns
    row = frame // spec.columns
    x = col * spec.tile_width
    y = row * spec.tile_height
    return image.crop((x, y, x + spec.tile_width, y + spec.tile_height)).convert("RGBA")


def _edge_pixels(tile: Any, direction: str) -> list[tuple[int, int, int, int]]:
    width, height = tile.size
    if direction == "top":
        return [tile.getpixel((x, 0)) for x in range(width)]
    if direction == "right":
        return [tile.getpixel((width - 1, y)) for y in range(height)]
    if direction == "bottom":
        return [tile.getpixel((x, height - 1)) for x in range(width)]
    if direction == "left":
        return [tile.getpixel((0, y)) for y in range(height)]
    raise ValueError(f"unknown direction: {direction}")


def _edge_condition(sheet: str, frame: int, direction: str, pixels: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    opaque = [pixel for pixel in pixels if pixel[3] >= 32]
    dark = [pixel for pixel in opaque if _is_dark(pixel)]
    buckets = [_color_bucket(pixel) for pixel in opaque if not _is_dark(pixel)]
    dominant = _dominant(buckets)
    has_outline = bool(opaque) and len(dark) / max(1, len(opaque)) >= 0.38
    open_color_group = "air" if not opaque else dominant
    edge_signature = "|".join(_color_bucket(pixel) for pixel in pixels)
    short_signature = _stable_digest(edge_signature)
    if not opaque:
        connect_policy = "air_only"
        sockets = ["air"]
        confidence = 0.95
    elif has_outline:
        connect_policy = "air_only"
        sockets = ["air"]
        confidence = min(0.99, 0.72 + len(dark) / max(1, len(opaque)) * 0.25)
    else:
        connect_policy = "same_signature_or_color"
        sockets = [f"edge:{short_signature}", f"color:{dominant}"]
        confidence = 0.62 if dominant == "transparent" else 0.82
    return {
        "direction": direction,
        "edge_signature": short_signature,
        "has_outline": has_outline,
        "dominant_color": dominant,
        "open_color_group": open_color_group,
        "connect_policy": connect_policy,
        "sockets": sockets,
        "socket": sockets[0],
        "clearance": 2 if connect_policy == "air_only" else 0,
        "confidence": round(confidence, 3),
        "debug": {"opaque": len(opaque), "dark": len(dark), "length": len(pixels)},
    }


def _rotate_conditions(base: dict[str, Any], turns: int) -> dict[str, Any]:
    conditions = base
    for _ in range(turns):
        conditions = {ROTATE_90[direction]: {**condition, "direction": ROTATE_90[direction]} for direction, condition in conditions.items()}
    return conditions


def _sockets_from_conditions(conditions: dict[str, Any]) -> dict[str, list[str]]:
    return {direction: [str(value) for value in conditions[direction].get("sockets", [conditions[direction]["socket"]])] for direction in DIRECTIONS}


def _blacklist_from_conditions(conditions: dict[str, Any]) -> dict[str, list[str]]:
    blacklist: dict[str, list[str]] = {}
    for direction in DIRECTIONS:
        condition = conditions[direction]
        if condition["connect_policy"] != "air_only":
            blacklist[direction] = ["air"]
    return blacklist


def classify_sheet(asset_dir: Path, spec: SheetSpec) -> dict[int, dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow is required for Kenney tile condition classification")
    image_path = asset_dir / spec.path
    image = Image.open(image_path).convert("RGBA")
    out: dict[int, dict[str, Any]] = {}
    for frame in range(spec.frame_count):
        tile = _crop_frame(image, spec, frame)
        base_condition = {
            direction: _edge_condition(spec.name, frame, direction, _edge_pixels(tile, direction))
            for direction in DIRECTIONS
        }
        rotations: dict[str, Any] = {}
        for turns, name in enumerate(("rot0", "rot90", "rot180", "rot270")):
            rotated = _rotate_conditions(base_condition, turns)
            rotations[name] = {
                "rotation_degrees": turns * 90,
                "condition": rotated,
                "sockets": _sockets_from_conditions(rotated),
                "socket_blacklist": _blacklist_from_conditions(rotated),
            }
        confidence = min(condition["confidence"] for condition in base_condition.values())
        out[frame] = {
            "sheet": spec.name,
            "frame": frame,
            "base_condition": base_condition,
            "rotations": rotations,
            "connect_policy": {direction: base_condition[direction]["connect_policy"] for direction in DIRECTIONS},
            "condition_source": "auto_edge_classifier",
            "condition_confidence": round(confidence, 3),
        }
    return out


def classify_asset_dir(asset_dir: str | Path, sheets: dict[str, SheetSpec] | None = None) -> dict[str, Any]:
    root = Path(asset_dir).expanduser().resolve()
    specs = sheets or DEFAULT_SHEETS
    classified = {name: classify_sheet(root, spec) for name, spec in specs.items()}
    return {
        "version": 1,
        "asset_dir": str(root),
        "classifier": "auto_edge_classifier",
        "sheets": {
            sheet: {str(frame): data for frame, data in frames.items()}
            for sheet, frames in classified.items()
        },
    }


def apply_conditions_to_tiles(tiles: list[dict[str, Any]], classified: dict[str, Any]) -> list[dict[str, Any]]:
    by_sheet = classified.get("sheets", {})
    for tile in tiles:
        sheet = str(tile.get("sheet") or "")
        frame = str(int(tile.get("frame") or 0))
        data = by_sheet.get(sheet, {}).get(frame)
        if not data:
            continue
        tile["condition"] = data["base_condition"]
        tile["rotations"] = data["rotations"]
        tile["connect_policy"] = data["connect_policy"]
        tile["condition_source"] = data["condition_source"]
        tile["condition_confidence"] = data["condition_confidence"]
        tile["sockets"] = data["rotations"]["rot0"]["sockets"]
        tile["socket_blacklist"] = data["rotations"]["rot0"]["socket_blacklist"]
    return tiles


def write_report(classified: dict[str, Any], report_path: Path) -> None:
    payload = json.dumps(classified, ensure_ascii=False)
    report_path.write_text(
        """<!doctype html><meta charset=\"utf-8\"><title>Kenney Tile Conditions</title>
<style>body{font:12px -apple-system,BlinkMacSystemFont,sans-serif;margin:16px;background:#f7f4ec;color:#191a15}pre{white-space:pre-wrap;background:#111;color:#f8f4e8;padding:12px;border-radius:8px}</style>
<h1>Kenney Tile Conditions</h1><p>Generated by auto_edge_classifier.</p><pre id=\"out\"></pre>
<script>document.getElementById('out').textContent=JSON.stringify(""" + payload + """, null, 2)</script>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify Kenney tile edge conditions from packed tilesheets.")
    parser.add_argument("--asset-dir", default="/Users/a1234/Downloads/kenney_pixel-platformer", type=Path)
    parser.add_argument("--out", default=Path("data/kenney_tile_conditions.json"), type=Path)
    parser.add_argument("--report", default=Path("scripts/kenney_tile_conditions_report.html"), type=Path)
    parser.add_argument("--write-manifest-preview", action="store_true", help="Accepted for workflow compatibility; writes classifier JSON/report only.")
    args = parser.parse_args()

    classified = classify_asset_dir(args.asset_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(classified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(classified, args.report)
    total = sum(len(frames) for frames in classified["sheets"].values())
    print(json.dumps({"ok": True, "tiles": total, "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
