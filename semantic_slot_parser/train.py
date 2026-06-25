#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from labels import LABELS, LABEL_TO_ID

PAD = "<pad>"
UNK = "<unk>"


class TorchTagger(nn.Module):
    def __init__(self, vocab_size: int, label_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, label_size)

    def forward(self, x):
        y, _ = self.gru(self.embedding(x))
        return self.proj(y)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vocab(rows: list[dict[str, object]]) -> dict[str, int]:
    chars = sorted({ch for row in rows for ch in str(row["text"])})
    vocab = {PAD: 0, UNK: 1}
    vocab.update({ch: i + 2 for i, ch in enumerate(chars)})
    return vocab


def collate(rows: list[dict[str, object]], vocab: dict[str, int]):
    max_len = max(len(str(row["text"])) for row in rows)
    x = torch.zeros((len(rows), max_len), dtype=torch.long)
    y = torch.zeros((len(rows), max_len), dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        text = str(row["text"])
        labels = row["labels"]
        for j, ch in enumerate(text):
            x[i, j] = vocab.get(ch, vocab[UNK])
            y[i, j] = LABEL_TO_ID[str(labels[j])]
            mask[i, j] = True
    return x, y, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--valid", default="data/valid.jsonl")
    parser.add_argument("--out-dir", default="artifacts_torch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    train_rows = read_jsonl(Path(args.train))
    valid_rows = read_jsonl(Path(args.valid))
    vocab = build_vocab(train_rows)
    model = TorchTagger(len(vocab), len(LABELS), args.embedding_dim, args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        model.train()
        total_loss = 0.0
        loader = DataLoader(train_rows, batch_size=args.batch_size, collate_fn=lambda b: collate(b, vocab))
        for x, y, mask in loader:
            logits = model(x)
            loss = loss_fn(logits[mask], y[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        model.eval()
        good = total = 0
        with torch.no_grad():
            for x, y, mask in DataLoader(valid_rows, batch_size=args.batch_size, collate_fn=lambda b: collate(b, vocab)):
                pred = model(x).argmax(-1)
                good += int(((pred == y) & mask).sum())
                total += int(mask.sum())
        acc = good / max(total, 1)
        print(f"epoch={epoch} loss={total_loss:.4f} valid_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), out / "model.pt")
            (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "label_map.json").write_text(
                json.dumps({"label_to_id": LABEL_TO_ID, "id_to_label": {str(i): l for i, l in enumerate(LABELS)}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(f"saved best model to {out} valid_acc={best:.4f}")


if __name__ == "__main__":
    main()
