from __future__ import annotations

import re
from typing import Any

from voice_assistant.json_utils import parse_jsonish_value

TOOL_LOG_WORDS: dict[str, dict[str, str]] = {
    "add_context_note": {"start": "我记一下", "success": "记下来了", "failure": "没记下来"},
    "front_note": {"start": "我贴一下", "success": "贴好了", "failure": "没贴好"},
    "update_task_status": {"start": "我改改", "success": "进度改好了", "failure": "进度没改成"},
    "mark_task_in_progress": {"start": "我挂上", "success": "已经开始跟了", "failure": "没挂上去"},
    "mark_task_blocked": {"start": "我标一下", "success": "卡点记下了", "failure": "卡点没记上"},
    "mark_task_done": {"start": "我收一下", "success": "收好了", "failure": "没收成"},
    "trigger_fast_followup": {"start": "我转述", "success": "排进语音了", "failure": "没转出去"},
    "get_weather": {"start": "我看看", "success": "天气查到了", "failure": "天气没查到"},
    "current_datetime": {"start": "我看下", "success": "时间看到了", "failure": "时间没看成"},
    "daily_action": {"start": "我看一下", "success": "弄好了", "failure": "没弄好"},
    "system_status": {"start": "我看下", "success": "状态看到了", "failure": "状态没看成"},
    "computer_action": {"start": "我操作一下", "success": "电脑处理好了", "failure": "电脑没处理好"},
    "shell_command": {"start": "我跑一下", "success": "跑完了", "failure": "没跑通"},
    "list_path": {"start": "我找找", "success": "找到文件了", "failure": "没找到文件"},
    "read_path": {"start": "我读一下", "success": "读完了", "failure": "没读懂"},
    "write_text_file": {"start": "我写写", "success": "写完了", "failure": "没写明白"},
    "web_search": {"start": "我查查", "success": "搜到了", "failure": "没搜到"},
    "search_news": {"start": "我翻翻", "success": "新闻翻到了", "failure": "新闻没翻到"},
    "fetch_url": {"start": "我看看", "success": "看到了", "failure": "没打开"},
    "open_url_in_browser": {"start": "我打开", "success": "已经打开了", "failure": "没打开"},
    "run_osascript": {"start": "我操作", "success": "电脑动过了", "failure": "电脑没动成"},
    "arrange_workspace": {"start": "我排一下", "success": "排好了", "failure": "没排好"},
    "capture_camera_snapshot": {"start": "我抓一张", "success": "拍到了", "failure": "没拍到"},
    "calendar_events": {"start": "我翻翻", "success": "日程找到了", "failure": "日程没找到"},
    "reminders_list": {"start": "我看看", "success": "提醒看到了", "failure": "提醒没看到"},
    "mail_message": {"start": "我写给", "success": "邮件写好了", "failure": "邮件没写成"},
    "read_file": {"start": "我读读", "success": "小文件读到了", "failure": "小文件没读到"},
    "list_files": {"start": "我翻翻", "success": "文件夹翻到了", "failure": "文件夹没翻开"},
    "back_tool_calling_unavailable": {"start": "我换条路试试", "success": "换路成功了", "failure": "换路没成"},
}
DEFAULT_TOOL_LOG_WORDS = {"start": "动手中", "success": "办妥了", "failure": "卡住了"}
DEFAULT_SILENT_TOOLS = {
    "add_context_note",
    "trigger_fast_followup",
    "update_task_status",
    "mark_task_in_progress",
    "mark_task_blocked",
    "mark_task_done",
}
DEFAULT_TOOL_TASK_LABELS: dict[str, str] = {
    "add_context_note": "记忆",
    "front_note": "便签",
    "update_task_status": "进度",
    "mark_task_in_progress": "进度",
    "mark_task_blocked": "卡点",
    "mark_task_done": "收尾",
    "trigger_fast_followup": "播报",
    "get_weather": "天气",
    "current_datetime": "时间",
    "daily_action": "日常",
    "system_status": "状态",
    "computer_action": "电脑",
    "shell_command": "命令",
    "list_path": "文件",
    "read_path": "阅读",
    "write_text_file": "写入",
    "web_search": "搜索",
    "search_news": "新闻",
    "fetch_url": "网页",
    "open_url_in_browser": "打开",
    "run_osascript": "电脑",
    "arrange_workspace": "窗口",
    "capture_camera_snapshot": "拍照",
    "calendar_events": "日程",
    "reminders_list": "提醒",
    "mail_message": "邮件",
    "read_file": "文件",
    "list_files": "文件夹",
    "back_tool_calling_unavailable": "换路",
}


def tool_log_words(tool_name: str) -> dict[str, str]:
    key = tool_name.split(":", 1)[-1]
    return TOOL_LOG_WORDS.get(key, DEFAULT_TOOL_LOG_WORDS)


def tool_log_label(tool_name: str, status: str) -> str:
    return tool_log_words(tool_name).get(status, DEFAULT_TOOL_LOG_WORDS[status])


def short_tool_error_reason(result: Any) -> str:
    value = parse_jsonish_value(result)
    text = ""
    if isinstance(value, dict):
        spoken = value.get("_spoken_summary") or value.get("spoken_summary")
        if spoken:
            return _clean_spoken_reason(str(spoken))
        for key in ("error", "stderr", "result", "stdout"):
            raw = value.get(key)
            if raw:
                text = str(raw)
                break
    else:
        text = str(value or "")
    text = text.strip()
    if not text:
        return ""
    lower = text.lower()
    if "video url not verified" in lower:
        return "视频链接还没验证"
    if "camera app open requested" in lower:
        return "这是打开相机，不是拍照"
    if "local action not completed" in lower:
        return "前一步还没做完"
    if "input should be a valid list" in lower and "app_names" in lower:
        return "窗口参数格式不对"
    if "no matching windows" in lower:
        return "没找到窗口"
    if "window layout failed" in lower:
        return "窗口没排好"
    if "weather request failed" in lower or "location not found" in lower or "opencage" in lower:
        return "地点没识别对"
    if "errors.pydantic.dev" in lower or "validation error" in lower:
        return "参数格式不对"
    missing = re.search(r"no module named ['\"]?([^'\"\\s]+)", text, flags=re.IGNORECASE)
    if missing:
        return f"缺少 {missing.group(1)}"
    if "outside the allowed base directory" in lower:
        return "路径不在工具目录里"
    if "timed out" in lower or "timeout" in lower:
        return "超时了"
    first_line = re.sub(r"\s+", " ", text.splitlines()[-1]).strip()
    return _clean_spoken_reason(first_line)


def _clean_spoken_reason(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"(/Users|/private|/tmp|/var|/Applications|/System|/Library|data/voice|semantic_slot_parser|scripts/voice_assistant)[^\\s,，。；;:：)\\]}]+", "文件路径", value)
    value = re.sub(r"\\b[A-Za-z_][A-Za-z0-9_]*\\.(py|json|log|txt|sqlite|db|npz|wav|mp3)\\b", "文件", value)
    value = re.sub(r"\\b[A-Za-z_][A-Za-z0-9_]{18,}\\b", "内部标识", value)
    value = re.sub(r"\\{.*?\\}", "参数", value)
    value = re.sub(r"\\[.*?\\]", "列表", value)
    value = re.sub(r"\\s+", " ", value).strip(" ：:，,。;；")
    if len(value) > 36:
        value = value[:36].rstrip(" ：:，,。;；") + "..."
    return value
