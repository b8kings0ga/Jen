from __future__ import annotations


def filler_stage_for_phrase(phrase: str) -> str:
    if any(marker in phrase for marker in ["眉目", "线索", "明白", "结论", "换个方式", "看到了", "继续"]):
        return "transition"
    if any(marker in phrase for marker in ["还在处理", "再试", "确认", "核对", "不急着下结论", "多确认"]):
        return "blocked"
    if any(marker in phrase for marker in ["查一下", "稍等", "想", "理一下", "整理"]):
        return "working"
    return "opening"
