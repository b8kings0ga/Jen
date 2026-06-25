#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from labels import LABELS, bio_labels

TIMES = [
    "今天",
    "明天",
    "后天",
    "现在",
    "今晚",
    "明早",
    "上午",
    "下午",
    "晚上",
    "明天上午",
    "明天下午",
    "明天晚上",
    "这个周末",
    "下周一",
    "三点",
    "下午五点",
    "十分钟后",
]
LOCATIONS = [
    "北京",
    "上海",
    "深圳",
    "杭州",
    "东京",
    "纽约",
    "伦敦",
    "巴黎",
    "瓦纳卡",
    "库斯科",
    "马尔代夫",
    "因斯布鲁克",
    "清迈",
    "胡志明",
    "圣保罗",
    "利马",
    "AvenidaE",
    "RuaGrape",
    "Paulista",
    "Avenida Paulista",
    "Avenida Engenheiro Luís Carlos Berrini",
    "Tokyo",
    "Sao Paulo",
    "我孙子那",
    "我孙子市",
    "我孙子试",
    "我朋友那",
    "公司这边",
    "家这边",
    "楼下",
]
OBJECTS = [
    "特朗普",
    "世界杯",
    "YMCA",
    "清晏世子",
    "Chrome",
    "Camera",
    "PhotoBooth",
    "这个视频",
    "下一集",
    "我的会议",
    "咖啡",
]
ACTIONS = [
    "天气",
    "天气怎么样",
    "温度",
    "温度怎么样",
    "现在几度",
    "会不会下雨",
    "会下雨吗",
    "下不下雨",
    "有雨吗",
    "下雨了吗",
    "下雨了嗎",
    "下雨了么",
    "下雨了吧",
    "下雨嘛",
    "下雨吗",
    "下雨嗎",
    "下雨么",
    "下雨麼",
    "要不要带伞",
    "查一下",
    "打开",
    "关闭",
    "播放",
    "提醒我",
    "记一下",
    "搜索一下",
    "排一下窗口",
]
DAILY_ACTIONS = {
    "weather": ["天气", "天气怎么样", "天氣怎麼樣", "温度", "溫度怎麼樣", "温度怎么样", "现在几度", "會不會下雨", "会不会下雨", "会下雨吗", "會下雨嗎", "下不下雨", "有雨吗", "有雨嗎", "下雨了吗", "下雨了嗎", "下雨了么", "下雨了吧", "下雨嘛", "下雨吗", "下雨嗎", "下雨么", "下雨麼", "要不要带伞"],
    "time": ["几点", "现在几点", "现在时间", "当地时间"],
    "map": ["地图", "地址", "在哪", "怎么走", "路线"],
    "calendar_list": ["看看日历", "查日程", "今天有什么会"],
    "reminder_list": ["看看提醒", "列一下提醒", "我有哪些待办"],
    "reminder_create": ["提醒我", "设个提醒", "加个待办"],
    "note_live": ["写到便签", "贴个便签", "在note写下"],
    "note_context": ["放进上下文", "写到context note", "加到上下文信息"],
    "memory": ["记住", "记得", "以后记得", "帮我记一下"],
}
MODIFIERS = ["当前", "明天", "今天", "路线", "地址", "live", "context", "长期", "列表", "创建"]
CONTENTS = [
    "明天买咖啡",
    "下午给妈妈打电话",
    "这个视频看到第三分钟",
    "Trump新闻要晚点再查",
    "会议材料放桌面",
    "我喜欢圣保罗",
    "下次打开Chrome",
    "明天开会",
]
CORRECTIONS = ["不对", "不是", "不是不是", "说错了", "改成", "等一下", "刚才那个不要"]
NEGATIONS = ["不要", "取消", "别", "算了"]
NOISE = ["呃", "嗯", "那个", "就是", "啊", "然后", "是", "那", "，", ",", "。", ""]


DAILY_FIXTURES: list[list[tuple[str, str | None]]] = [
    [("今天", "TIME"), ("我孙子那", "LOCATION"), ("天气", "ACTION"), (",", None), ("不是不是", "CORRECTION"), (",", None), ("是", None), ("北京", "LOCATION"), (",", None), ("啊", None), ("算了", "NEGATION")],
    [("今天", "TIME"), ("我孙子试", "TARGET"), ("天气怎么样", "ACTION")],
    [("今天", "TIME"), ("我孙子市", "TARGET"), ("天气怎么样", "ACTION")],
    [("今天", "TIME"), ("圣保罗", "TARGET"), ("，", None), ("不对", "CORRECTION"), ("北京", "LOCATION"), ("，", None), ("不对", "CORRECTION"), ("上海", "LOCATION"), ("天气怎么样", "ACTION"), ("，", None), ("然后", None), ("提醒我", "ACTION"), ("明天下午", "TIME"), ("去上海", "CONTENT")],
    [("今天", "TIME"), ("圣保罗", "TARGET"), (",", None), ("啊", None), ("不对不对", "CORRECTION"), ("北京", "LOCATION"), ("啊", None), ("不对", "CORRECTION"), ("上海", "LOCATION"), ("天气怎么样", "ACTION"), (",", None), ("然后", None), ("记得提醒我", "ACTION"), ("明天下午", "TIME"), ("去上海", "CONTENT")],
    [("今天", "TIME"), ("毛罗", "TARGET"), ("，", None), ("不对", "CORRECTION"), ("北京", "LOCATION"), ("天气怎么样", "ACTION"), ("，", None), ("然后", None), ("提醒我", "ACTION"), ("明天下午", "TIME"), ("去北京", "CONTENT")],
    [("今天", "TIME"), ("温度怎么样", "ACTION")],
    [("现在", "TIME"), ("几度", "ACTION")],
    [("这边", "TARGET"), ("下雨吗", "ACTION")],
    [("這邊", "TARGET"), ("下雨嗎", "ACTION")],
    [("明天", "TIME"), ("下雨嗎", "ACTION")],
    [("明天", "TIME"), ("下雨了吗", "ACTION")],
    [("明天", "TIME"), ("下雨了嗎", "ACTION")],
    [("明天", "TIME"), ("会下雨吗", "ACTION")],
    [("明天", "TIME"), ("下不下雨", "ACTION")],
    [("明天", "TIME"), ("有雨吗", "ACTION")],
    [("明天", "TIME"), ("下雨嘛", "ACTION")],
    [("明天", "TIME"), ("下雨了吧", "ACTION")],
    [("圣保罗", "TARGET"), ("下雨了吗", "ACTION")],
    [("聖保羅", "TARGET"), ("下雨麼", "ACTION")],
    [("这里", "TARGET"), ("会不会下雨", "ACTION")],
    [("当地", "TARGET"), ("要不要带伞", "ACTION")],
    [("今天", "TIME"), ("要不要带伞", "ACTION")],
]


def n() -> tuple[str, None]:
    return random.choice(NOISE), None


def maybe_noise(parts: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    out = []
    for part in parts:
        if random.random() < 0.25:
            out.append(n())
        out.append(part)
        if random.random() < 0.12:
            out.append(n())
    return out


def sample_normal() -> tuple[str, list[str]]:
    t = random.choice(TIMES)
    loc = random.choice(LOCATIONS)
    obj = random.choice(OBJECTS)
    action = random.choice(ACTIONS)
    patterns = [
        [(t, "TIME"), (loc, "LOCATION"), (action, "ACTION")],
        [(t, "TIME"), (obj, "OBJECT"), (action, "ACTION")],
        [(loc, "LOCATION"), (t, "TIME"), (action, "ACTION")],
        [(action, "ACTION"), (loc, "LOCATION")],
        [(obj, "OBJECT"), (action, "ACTION")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_daily_weather() -> tuple[str, list[str]]:
    t = random.choice(TIMES)
    loc = random.choice(LOCATIONS)
    old_loc = random.choice([item for item in LOCATIONS if item != loc])
    action = random.choice(DAILY_ACTIONS["weather"])
    modifier = random.choice(["当前", "明天", "今天", ""])
    patterns = [
        [(t, "TIME"), (loc, "TARGET"), (action, "ACTION")],
        [(loc, "TARGET"), (t, "TIME"), (action, "ACTION")],
        [(t, "TIME"), (modifier, "MODIFIER"), (loc, "TARGET"), (action, "ACTION")],
        [(t, "TIME"), (old_loc, "TARGET"), (action, "ACTION"), (random.choice(CORRECTIONS), "CORRECTION"), ("是", None), (loc, "TARGET")],
        [(t, "TIME"), (old_loc, "TARGET"), (random.choice(["不对", "不是", "啊不对不对"]), "CORRECTION"), (loc, "LOCATION"), (action, "ACTION")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_daily_map() -> tuple[str, list[str]]:
    loc = random.choice(LOCATIONS)
    action = random.choice(DAILY_ACTIONS["map"])
    modifier = "路线" if action in {"怎么走", "路线"} else random.choice(["地址", "当前", ""])
    patterns = [
        [(loc, "TARGET"), (action, "ACTION")],
        [(action, "ACTION"), (loc, "TARGET")],
        [(modifier, "MODIFIER"), (loc, "TARGET"), (action, "ACTION")],
        [("去", "ACTION"), (loc, "TARGET"), ("怎么走", "ACTION")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_daily_note() -> tuple[str, list[str]]:
    action_name = random.choice(["note_live", "note_context", "memory"])
    action = random.choice(DAILY_ACTIONS[action_name])
    content = random.choice(CONTENTS)
    modifier = {"note_live": "live", "note_context": "context", "memory": "长期"}[action_name]
    patterns = [
        [(action, "ACTION"), (content, "CONTENT")],
        [(modifier, "MODIFIER"), (action, "ACTION"), (content, "CONTENT")],
        [(action, "ACTION"), ("，", None), (content, "CONTENT")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_daily_reminder() -> tuple[str, list[str]]:
    action_name = random.choice(["reminder_create", "reminder_list", "calendar_list", "time"])
    action = random.choice(DAILY_ACTIONS[action_name])
    content = random.choice(CONTENTS)
    t = random.choice(TIMES)
    if action_name in {"reminder_create"}:
        patterns = [
            [(action, "ACTION"), (t, "TIME"), (content, "CONTENT")],
            [(t, "TIME"), (action, "ACTION"), (content, "CONTENT")],
        ]
    elif action_name in {"calendar_list", "reminder_list"}:
        patterns = [
            [(t, "TIME"), (action, "ACTION")],
            [("列表", "MODIFIER"), (action, "ACTION")],
        ]
    else:
        patterns = [
            [(random.choice(LOCATIONS), "TARGET"), (action, "ACTION")],
            [(t, "TIME"), (action, "ACTION")],
        ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_correction() -> tuple[str, list[str]]:
    t = random.choice(TIMES)
    old_loc, new_loc = random.sample(LOCATIONS, 2)
    old_obj, new_obj = random.sample(OBJECTS, 2)
    corr = random.choice(CORRECTIONS)
    action = random.choice(ACTIONS)
    patterns = [
        [(t, "TIME"), (old_loc, "LOCATION"), (corr, "CORRECTION"), (new_loc, "LOCATION"), (action, "ACTION")],
        [(t, "TIME"), (old_loc, "TARGET"), (corr, "CORRECTION"), (new_loc, "TARGET"), (action, "ACTION")],
        [(t, "TIME"), (old_loc, "LOCATION"), (action, "ACTION"), (",", None), (corr, "CORRECTION"), (",", None), ("是", None), (new_loc, "LOCATION")],
        [(t, "TIME"), (old_loc, "TARGET"), (action, "ACTION"), (",", None), (corr, "CORRECTION"), (",", None), ("是", None), (new_loc, "TARGET")],
        [(old_obj, "OBJECT"), (action, "ACTION"), (corr, "CORRECTION"), (new_obj, "OBJECT"), (action, "ACTION")],
        [(old_obj, "CONTENT"), (action, "ACTION"), (corr, "CORRECTION"), (new_obj, "CONTENT"), (action, "ACTION")],
        [(t, "TIME"), (old_loc, "LOCATION"), (action, "ACTION"), (corr, "CORRECTION"), (new_loc, "LOCATION"), (action, "ACTION")],
        [(old_loc, "LOCATION"), (corr, "CORRECTION"), (new_loc, "LOCATION"), (action, "ACTION")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_repeat() -> tuple[str, list[str]]:
    t = random.choice(TIMES)
    loc = random.choice(LOCATIONS)
    action = random.choice(ACTIONS)
    repeated = random.choice([t, loc, action])
    kind = "TIME" if repeated == t else "LOCATION" if repeated == loc else "ACTION"
    patterns = [
        [(t, "TIME"), (loc, "LOCATION"), (loc, "LOCATION"), (action, "ACTION")],
        [(repeated, kind), (repeated, kind), (action, "ACTION")],
        [(t, "TIME"), (t, "TIME"), (loc, "LOCATION"), (action, "ACTION")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_negation() -> tuple[str, list[str]]:
    neg = random.choice(NEGATIONS)
    action = random.choice(ACTIONS)
    obj = random.choice(OBJECTS)
    loc = random.choice(LOCATIONS)
    patterns = [
        [(neg, "NEGATION"), (action, "ACTION")],
        [(neg, "NEGATION"), (obj, "OBJECT"), (action, "ACTION")],
        [(neg, "NEGATION"), (obj, "CONTENT"), (action, "ACTION")],
        [(loc, "LOCATION"), (neg, "NEGATION"), (action, "ACTION")],
        [(loc, "TARGET"), (neg, "NEGATION"), (action, "ACTION")],
    ]
    return bio_labels("", maybe_noise(random.choice(patterns)))


def sample_one() -> dict[str, object]:
    fn = random.choices(
        [
            sample_daily_weather,
            sample_daily_map,
            sample_daily_note,
            sample_daily_reminder,
            sample_normal,
            sample_correction,
            sample_repeat,
            sample_negation,
        ],
        weights=[0.28, 0.16, 0.16, 0.14, 0.1, 0.1, 0.03, 0.03],
    )[0]
    text, labels = fn()
    if len(text) != len(labels):
        raise AssertionError((text, labels))
    for label in labels:
        if label not in LABELS:
            raise AssertionError(label)
    return {"text": text, "labels": labels}


def fixture_rows() -> list[dict[str, object]]:
    rows = []
    for parts in DAILY_FIXTURES:
        text, labels = bio_labels("", parts)
        rows.append({"text": text, "labels": labels})
    return rows


def write_jsonl(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        fixtures = fixture_rows()
        fixture_budget = min(size, len(fixtures) * 80)
        for idx in range(fixture_budget):
            f.write(json.dumps(fixtures[idx % len(fixtures)], ensure_ascii=False) + "\n")
        for _ in range(max(0, size - fixture_budget)):
            f.write(json.dumps(sample_one(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--valid-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", args.train_size)
    write_jsonl(out / "valid.jsonl", args.valid_size)
    print(f"wrote {out / 'train.jsonl'} and {out / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
