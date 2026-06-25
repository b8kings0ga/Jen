LABELS = [
    "O",
    "B-TIME",
    "I-TIME",
    "B-LOCATION",
    "I-LOCATION",
    "B-OBJECT",
    "I-OBJECT",
    "B-TARGET",
    "I-TARGET",
    "B-CONTENT",
    "I-CONTENT",
    "B-ACTION",
    "I-ACTION",
    "B-MODIFIER",
    "I-MODIFIER",
    "B-CORRECTION",
    "I-CORRECTION",
    "B-NEGATION",
    "I-NEGATION",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
SLOT_TYPES = {"TIME", "LOCATION", "OBJECT", "TARGET", "CONTENT", "ACTION", "MODIFIER", "CORRECTION", "NEGATION"}


def bio_labels(text: str, parts: list[tuple[str, str | None]]) -> tuple[str, list[str]]:
    out = []
    labels = []
    for value, kind in parts:
        if not value:
            continue
        out.append(value)
        if kind is None:
            labels.extend(["O"] * len(value))
        else:
            labels.append(f"B-{kind}")
            labels.extend([f"I-{kind}"] * (len(value) - 1))
    merged = "".join(out)
    if text and merged != text:
        raise ValueError(f"text mismatch: {text!r} != {merged!r}")
    return merged, labels


def fix_bio(labels: list[str]) -> list[str]:
    fixed = []
    prev_type = None
    for label in labels:
        if label == "O" or "-" not in label:
            fixed.append("O")
            prev_type = None
            continue
        prefix, typ = label.split("-", 1)
        if typ not in SLOT_TYPES:
            fixed.append("O")
            prev_type = None
            continue
        if prefix == "I" and prev_type != typ:
            label = f"B-{typ}"
        elif prefix not in {"B", "I"}:
            label = f"B-{typ}"
        fixed.append(label)
        prev_type = typ
    return fixed


def labels_to_spans(text: str, labels: list[str]) -> list[dict[str, str]]:
    labels = fix_bio(labels)
    spans = []
    cur_type = None
    cur_chars = []
    for ch, label in zip(text, labels):
        if label == "O":
            if cur_type:
                spans.append({"text": "".join(cur_chars), "type": cur_type})
                cur_type, cur_chars = None, []
            continue
        prefix, typ = label.split("-", 1)
        if prefix == "B" or typ != cur_type:
            if cur_type:
                spans.append({"text": "".join(cur_chars), "type": cur_type})
            cur_type, cur_chars = typ, [ch]
        else:
            cur_chars.append(ch)
    if cur_type:
        spans.append({"text": "".join(cur_chars), "type": cur_type})
    return spans
