from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request

from voice_assistant.http_client import urlopen_text
from voice_assistant.http_payloads import multipart_form_data

def transcribe_with_gjallarhorn(audio_path: Path, args: argparse.Namespace) -> str:
    fields = {"model": args.asr_model, "response_format": "json"}
    language = str(getattr(args, "language", "") or "").strip()
    if language and language.lower() not in {"auto", "detect", "auto-detect", "autodetect"}:
        fields["language"] = language
    asr_prompt = str(getattr(args, "asr_prompt", "") or "").strip()
    if asr_prompt:
        fields["prompt"] = asr_prompt
    body, content_type = multipart_form_data(
        fields=fields,
        files={"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")},
    )
    req = request.Request(
        f"{args.gjallarhorn_base_url.rstrip('/')}/audio/transcriptions",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": content_type},
    )
    payload = json.loads(urlopen_text(req, timeout=args.asr_timeout, verify_tls=args.verify_tls, label="Gjallarhorn ASR"))
    text = payload.get("text") if isinstance(payload, dict) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError(f"Gjallarhorn ASR response has no text: {payload!r}")
