from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class CharBiGRUTagger(nn.Module):
    def __init__(self, vocab_size: int, label_size: int, embedding_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.forward_gru = nn.GRU(embedding_dim, hidden_dim)
        self.backward_gru = nn.GRU(embedding_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, label_size)

    def __call__(self, x):
        emb = self.embedding(x)
        fw = self.forward_gru(emb)
        rev = emb[..., ::-1, :]
        bw = self.backward_gru(rev)[..., ::-1, :]
        return self.proj(mx.concatenate([fw, bw], axis=-1))
