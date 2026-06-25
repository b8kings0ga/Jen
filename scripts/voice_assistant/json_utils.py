from __future__ import annotations

import json
import re
from typing import Any


def coerce_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): coerce_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [coerce_json_value(v) for v in value]
        return str(value)


def coerce_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return {str(k): coerce_json_value(v) for k, v in parsed.items()}
    value = coerce_json_value(value)
    return value if isinstance(value, dict) else {"value": value}


def parse_jsonish_value(value: Any) -> Any:
    value = coerce_json_value(value)
    seen = 0
    while isinstance(value, str) and seen < 4:
        text = value.strip()
        if not text or text[0] not in "[{":
            return value
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return value
        seen += 1
    if isinstance(value, dict) and "result" in value and len(value) <= 6:
        inner = value.get("result")
        parsed = parse_jsonish_value(inner)
        if parsed is not inner:
            return parsed
    return value


def parse_json_object(raw: str) -> dict[str, Any] | None:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None
