#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from infer_mlx import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--valid", default="data/valid.jsonl")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.valid).read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [str(row["text"]) for row in random.choices(rows, k=args.n)]
    model, vocab, id_to_label = load_model(Path(args.artifact_dir), args.embedding_dim, args.hidden_dim)
    for text in samples[:20]:
        predict(text, model, vocab, id_to_label)
    times = []
    for text in samples:
        started = time.perf_counter()
        predict(text, model, vocab, id_to_label)
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    print(
        json.dumps(
            {
                "n": len(times),
                "avg_ms": round(statistics.mean(times), 3),
                "p50_ms": round(times[int(len(times) * 0.50)], 3),
                "p95_ms": round(times[int(len(times) * 0.95)], 3),
                "max_ms": round(max(times), 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
