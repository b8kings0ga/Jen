#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from labels import LABELS, LABEL_TO_ID
from model_mlx import CharBiGRUTagger

PAD = "<pad>"
UNK = "<unk>"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_vocab(rows: list[dict[str, object]]) -> dict[str, int]:
    chars = sorted({ch for row in rows for ch in str(row["text"])})
    vocab = {PAD: 0, UNK: 1}
    vocab.update({ch: i + 2 for i, ch in enumerate(chars)})
    return vocab


def encode_batch(rows: list[dict[str, object]], vocab: dict[str, int]) -> tuple[mx.array, mx.array, mx.array]:
    max_len = max(len(str(row["text"])) for row in rows)
    xs = np.zeros((len(rows), max_len), dtype=np.int32)
    ys = np.zeros((len(rows), max_len), dtype=np.int32)
    mask = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, row in enumerate(rows):
        text = str(row["text"])
        labels = row["labels"]
        for j, ch in enumerate(text):
            xs[i, j] = vocab.get(ch, vocab[UNK])
            ys[i, j] = LABEL_TO_ID[str(labels[j])]
            mask[i, j] = 1.0
    return mx.array(xs), mx.array(ys), mx.array(mask)


def batches(rows: list[dict[str, object]], batch_size: int, shuffle: bool) -> list[list[dict[str, object]]]:
    rows = list(rows)
    if shuffle:
        random.shuffle(rows)
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def accuracy(model: CharBiGRUTagger, rows: list[dict[str, object]], vocab: dict[str, int], batch_size: int) -> float:
    good = 0
    total = 0
    for batch in batches(rows, batch_size, shuffle=False):
        x, y, mask = encode_batch(batch, vocab)
        pred = mx.argmax(model(x), axis=-1)
        ok = (pred == y) * mask.astype(mx.bool_)
        good += int(mx.sum(ok).item())
        total += int(mx.sum(mask).item())
    return good / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--valid", default="data/valid.jsonl")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    train_rows = read_jsonl(Path(args.train))
    valid_rows = read_jsonl(Path(args.valid))
    vocab = build_vocab(train_rows)
    model = CharBiGRUTagger(len(vocab), len(LABELS), args.embedding_dim, args.hidden_dim)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def loss_fn(model: CharBiGRUTagger, x: mx.array, y: mx.array, mask: mx.array) -> mx.array:
        logits = model(x)
        per_token = nn.losses.cross_entropy(logits, y, reduction="none")
        return mx.sum(per_token * mask) / mx.maximum(mx.sum(mask), 1.0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    best_acc = 0.0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        losses = []
        for batch in batches(train_rows, args.batch_size, shuffle=True):
            x, y, mask = encode_batch(batch, vocab)
            loss, grads = loss_and_grad(model, x, y, mask)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss.item()))
        val_acc = accuracy(model, valid_rows, vocab, args.batch_size)
        elapsed = time.perf_counter() - started
        print(f"epoch={epoch} loss={np.mean(losses):.4f} valid_acc={val_acc:.4f} time={elapsed:.2f}s")
        if val_acc >= best_acc:
            best_acc = val_acc
            model.save_weights(str(out / "model_mlx.npz"))
            (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
            label_map = {"label_to_id": LABEL_TO_ID, "id_to_label": {str(i): l for i, l in enumerate(LABELS)}}
            (out / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved best model to {out} valid_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
