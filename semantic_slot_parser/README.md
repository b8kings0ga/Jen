# 中文语音 Slot Parser

一个面向中文 ASR 文本的极速 slot parser。主路径使用 MLX，在 Apple Silicon 上做字符级 BIO 序列标注；纠错、否定、后说覆盖前说由 deterministic resolver 处理。

## 安装

```sh
cd semantic_slot_parser
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 生成数据

```sh
python generate_data.py --train-size 50000 --valid-size 5000
```

输出：

```text
data/train.jsonl
data/valid.jsonl
```

每行格式：

```json
{"text":"今天圣保罗不是不是北京天气怎么样","labels":["B-TIME","I-TIME","B-LOCATION"]}
```

`labels` 长度始终等于 `text` 字符长度。

## MLX 训练

```sh
python train_mlx.py --epochs 15 --embedding-dim 64 --hidden-dim 96
```

输出：

```text
artifacts/model_mlx.npz
artifacts/vocab.json
artifacts/label_map.json
```

## 推理

```sh
python infer_mlx.py "今天圣保罗不是不是北京天气怎么样" --timing
```

输出包含：

- `labels`: 每个字符的 BIO 标签
- `spans`: 聚合后的 span
- `resolved`: resolver 之后的最终 slot JSON

期望结果会同时保留兼容字段和 daily fat tool 字段：

```json
{
  "domain": "daily",
  "action": "weather",
  "target": "北京",
  "content": "",
  "time": "今天",
  "location": "北京",
  "object": "",
  "modifiers": [],
  "cancelled": false,
  "daily_action_call": {
    "action": "weather",
    "target": "北京",
    "args": {
      "time": "今天",
      "content": "",
      "modifiers": []
    }
  }
}
```

## Benchmark

```sh
python bench.py --n 1000
```

输出 avg/p50/p95/max latency。目标是 MacBook Air 本地单句推理低延迟优先，准确率靠数据覆盖和 resolver 保证。

## PyTorch fallback

非 Apple Silicon 或调试对照可以跑：

```sh
python train.py --epochs 10
```

PyTorch fallback 使用同样的轻量 BiGRU + token classification，不作为推荐路径。
