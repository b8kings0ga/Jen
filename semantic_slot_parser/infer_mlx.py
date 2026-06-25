#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from labels import fix_bio, labels_to_spans
from model_mlx import CharBiGRUTagger
from resolver import resolve


def load_model(artifact_dir: Path, embedding_dim: int, hidden_dim: int) -> tuple[CharBiGRUTagger, dict[str, int], dict[int, str]]:
    vocab = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
    label_map = json.loads((artifact_dir / "label_map.json").read_text(encoding="utf-8"))
    id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}
    model = CharBiGRUTagger(len(vocab), len(id_to_label), embedding_dim, hidden_dim)
    model.load_weights(str(artifact_dir / "model_mlx.npz"))
    mx.eval(model.parameters())
    return model, vocab, id_to_label


def predict(text: str, model: CharBiGRUTagger, vocab: dict[str, int], id_to_label: dict[int, str]) -> dict[str, object]:
    ids = np.array([[vocab.get(ch, vocab.get("<unk>", 1)) for ch in text]], dtype=np.int32)
    logits = model(mx.array(ids))
    pred = mx.argmax(logits, axis=-1)[0].tolist()
    labels = fix_bio([id_to_label[int(i)] for i in pred])
    spans = labels_to_spans(text, labels)
    return {"text": text, "labels": labels, "spans": spans, "resolved": resolve(spans, text)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    model, vocab, id_to_label = load_model(Path(args.artifact_dir), args.embedding_dim, args.hidden_dim)
    started = time.perf_counter()
    result = predict(args.text, model, vocab, id_to_label)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if args.timing:
        result["latency_ms"] = round(elapsed_ms, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
