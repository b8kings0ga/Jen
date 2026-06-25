from __future__ import annotations

import ssl
from urllib import error, request


def urlopen_text(req: request.Request, *, timeout: float, verify_tls: bool, label: str) -> str:
    return urlopen_bytes(req, timeout=timeout, verify_tls=verify_tls, label=label).decode("utf-8")


def urlopen_bytes(req: request.Request, *, timeout: float, verify_tls: bool, label: str) -> bytes:
    context = None if verify_tls else ssl._create_unverified_context()
    try:
        with request.urlopen(req, timeout=timeout, context=context) as response:
            return response.read()
    except TimeoutError as exc:
        raise RuntimeError(f"{label} request timed out after {timeout:.1f}s") from exc
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{label} request failed: {exc.reason}") from exc
