from __future__ import annotations

import re


DEFAULT_TTS_INSTRUCTIONS = "用自然、清楚、适合实时语音助手的中文口吻朗读；语速稍慢一点，短句之间有自然停顿；不要夸张，不要播音腔。"


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking\b[^>]*>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking\b[^>]*>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def normalize_assistant_text(text: str) -> str:
    text = strip_think_blocks(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_speech_text(text: str) -> str:
    return re.sub(r"[\s，,。.!！?？、；;：:《》<>「」『』\"'“”‘’]+", "", text.strip())


def split_speech_text(text: str, max_chars: int = 45) -> list[str]:
    text = normalize_assistant_text(text)
    if len(text) <= max_chars:
        return [text] if text else []
    text = re.sub(r"(?<![，,。！？!?；;、])(然后|接着|再|最后|同时|并且|但是|所以|不过)", r"，\1", text)
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;，,、])\s*", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts or [text]:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(hard_split_speech_part(part, max_chars=max_chars))
            continue
        candidate = current + part if not current else current + part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def hard_split_speech_part(text: str, max_chars: int = 45) -> list[str]:
    chunks: list[str] = []
    current = ""
    for token in re.split(r"(\s+)", text):
        if not token:
            continue
        if len(current) + len(token) <= max_chars:
            current += token
            continue
        if current:
            chunks.append(current.strip())
            current = ""
        while len(token) > max_chars:
            chunks.append(token[:max_chars].strip())
            token = token[max_chars:]
        current = token
    if current.strip():
        chunks.append(current.strip())
    return chunks


def filler_tts_instructions(tone: str) -> str:
    if tone == "active":
        return "用更明确、稍有行动感的处理中提示语气朗读，清楚但不要播音腔。"
    return "用轻声、自然、短促的接话语气朗读，像真人边思考边回应，不要夸张。"
