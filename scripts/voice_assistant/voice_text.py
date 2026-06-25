from __future__ import annotations

import re

from voice_assistant.speech_text import compact_speech_text, strip_think_blocks

START_SOUND_ECHO_PHRASES = {"你说", "怎么了", "我听着呢"}
ASR_BOUNDARY_TOKENS = [
    "不是",
    "不对",
    "不要",
    "别",
    "但是",
    "不过",
    "然后",
    "接着",
    "再",
    "最后",
    "随便一个",
    "任意一个",
    "哪个都行",
    "都可以",
]

ASR_TEXT_NORMALIZATION = str.maketrans(
    {
        "開": "开",
        "關": "关",
        "嗎": "吗",
        "麼": "么",
        "麽": "么",
        "聖": "圣",
        "羅": "罗",
        "葉": "叶",
        "氣": "气",
        "溫": "温",
        "濕": "湿",
        "風": "风",
        "雲": "云",
        "電": "电",
        "腦": "脑",
        "視": "视",
        "頻": "频",
        "網": "网",
        "頁": "页",
        "會": "会",
        "這": "这",
        "為": "为",
        "沒": "没",
        "還": "还",
        "讓": "让",
        "幫": "帮",
        "寫": "写",
        "說": "说",
        "聽": "听",
        "讀": "读",
        "詢": "询",
        "記": "记",
        "錄": "录",
        "發": "发",
        "現": "现",
        "後": "后",
        "裡": "里",
        "裏": "里",
        "邊": "边",
        "當": "当",
        "個": "个",
        "點": "点",
        "兩": "两",
        "與": "与",
        "對": "对",
        "錯": "错",
        "應": "应",
        "該": "该",
        "實": "实",
        "優": "优",
        "啟": "启",
        "動": "动",
        "運": "运",
        "檔": "档",
        "圖": "图",
        "庫": "库",
        "長": "长",
        "難": "难",
        "語": "语",
        "聲": "声",
        "體": "体",
        "簡": "简",
        "轉": "转",
        "換": "换",
        "處": "处",
        "連": "连",
        "結": "结",
        "鏈": "链",
        "設": "设",
        "備": "备",
        "態": "态",
        "務": "务",
        "時": "时",
        "間": "间",
        "週": "周",
        "曆": "历",
        "標": "标",
        "籤": "签",
        "內": "内",
        "攝": "摄",
        "鏡": "镜",
        "頭": "头",
        "臺": "台",
        "颱": "台",
        "車": "车",
        "馬": "马",
        "國": "国",
        "廣": "广",
        "東": "东",
        "門": "门",
        "學": "学",
        "測": "测",
        "試": "试",
        "數": "数",
        "據": "据",
        "顯": "显",
        "單": "单",
        "雙": "双",
        "軌": "轨",
        "佇": "伫",
        "隊": "队",
        "縣": "县",
        "區": "区",
        "鄉": "乡",
        "鎮": "镇",
        "貝": "贝",
        "爾": "尔",
        "奧": "奥",
        "蘭": "兰",
        "紐": "纽",
        "約": "约",
        "舊": "旧",
        "蘋": "苹",
        "軟": "软",
        "郵": "邮",
        "聯": "联",
        "繫": "系",
        "權": "权",
        "遊": "游",
        "戲": "戏",
        "畫": "画",
        "幀": "帧",
        "絲": "丝",
        "飛": "飞",
        "鳥": "鸟",
        "靈": "灵",
        "鍵": "键",
        "綁": "绑",
        "專": "专",
        "項": "项",
        "預": "预",
        "認": "认",
        "證": "证",
        "驗": "验",
        "剛": "刚",
        "號": "号",
        "碼": "码",
        "熱": "热",
        "刪": "删",
        "創": "创",
        "載": "载",
        "線": "线",
        "佈": "布",
        "協": "协",
        "擴": "扩",
        "彈": "弹",
        "豬": "猪",
        "屍": "尸",
        "壓": "压",
        "縮": "缩",
        "總": "总",
        "狀": "状",
        "場": "场",
        "螢": "屏",
    }
)

ASR_PHRASE_NORMALIZATION = {
    "瀏覽器": "浏览器",
    "網頁": "网页",
    "視訊": "视频",
    "影片": "视频",
    "視窗": "窗口",
    "螢幕": "屏幕",
    "全螢幕": "全屏",
    "便籤": "便签",
    "筆記": "笔记",
    "天氣": "天气",
    "時間": "时间",
    "地圖": "地图",
    "路線": "路线",
    "搜尋": "搜索",
    "查詢": "查询",
    "結果": "结果",
    "失敗": "失败",
    "語音": "语音",
    "輸入": "输入",
    "輸出": "输出",
    "錄音": "录音",
    "選項": "选项",
    "圖示": "图标",
    "標識": "标识",
    "實時": "实时",
    "遠端": "远端",
    "本機": "本机",
    "服務": "服务",
    "關閉": "关闭",
    "並排": "并排",
    "輪轉": "轮转",
    "前臺": "前台",
    "後臺": "后台",
    "語義": "语义",
    "註冊": "注册",
    "對話": "对话",
    "會話": "会话",
    "壓縮": "压缩",
    "啟用": "启用",
    "開啟": "开启",
    "開發": "开发",
    "調試": "调试",
    "優化": "优化",
    "問題": "问题",
    "為什麼": "为什么",
    "怎麼": "怎么",
    "方向鍵": "方向键",
    "應用": "应用",
    "錯誤": "错误",
    "動畫": "动画",
    "遊戲": "游戏",
    "關卡": "关卡",
    "幽靈": "幽灵",
    "啟動": "启动",
    "關掉": "关掉",
    "桌面": "桌面",
    "記住": "记住",
    "記得": "记得",
    "儲存": "储存",
    "緩存": "缓存",
}


def normalize_chinese_asr_variants(text: str) -> str:
    value = str(text or "")
    for traditional, simplified in ASR_PHRASE_NORMALIZATION.items():
        value = value.replace(traditional, simplified)
    return value.translate(ASR_TEXT_NORMALIZATION)

def normalize_for_similarity(text: str) -> str:
    text = strip_think_blocks(text).lower()
    text = re.sub(r"我查到了|有结果了|我先说结论|简单说|总之|好的|好，|嗯|啊|稍等", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    text = text.replace("摄氏度", "c").replace("度", "c").replace("公里每小时", "kmh")
    return text


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_for_similarity(left)
    right_norm = normalize_for_similarity(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        return shorter / longer if longer else 0.0
    left_tokens = similarity_tokens(left_norm)
    right_tokens = similarity_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / union if union else 0.0


def similarity_tokens(text: str) -> set[str]:
    if len(text) <= 2:
        return {text}
    return {text[i:i + 2] for i in range(len(text) - 1)}


def is_tool_route_unavailable(text: str) -> bool:
    compact = text.lower()
    return (
        "no endpoints found that support tool use" in compact
        or "try disabling" in compact and "tool" in compact
        or "route fallback timeout" in compact
        or "context deadline exceeded" in compact
        or "upstream timeout" in compact
        or "gateway_timeout" in compact
        or "api connection error" in compact
        or "connection error" in compact
        or "connection reset by peer" in compact
        or "error code: 504" in compact
        or "error code: 451" in compact
        or "censorship_blocked" in compact
        or "content you provided or machine outputted is blocked" in compact
    )


def is_error_like_followup(text: str) -> bool:
    compact = text.lower()
    markers = [
        "traceback",
        "exception",
        "error",
        "failed",
        "失败",
        "报错",
        "安装失败",
        "pip install",
        "no module named",
        "returned non-zero exit status",
    ]
    return any(marker in compact for marker in markers)


def is_status_only_followup(text: str) -> bool:
    compact = re.sub(r"[\s。.!！,，、：:；;]+", "", text.strip().lower())
    if not compact:
        return True
    exact = {
        "好的已完成",
        "已完成",
        "任务已完成",
        "后台任务已完成",
        "处理完成",
        "完成了",
        "好了",
        "okdone",
        "done",
    }
    if compact in exact:
        return True
    status_markers = [
        "任务已完成",
        "后台任务已完成",
        "后台处理完成",
        "处理完成",
        "已记录",
        "已经记录",
        "状态已更新",
        "进度已更新",
        "已标记",
        "已完成该任务",
    ]
    return any(marker in compact for marker in status_markers)


def classify_filler_stage(user_text: str, context_json: str = "") -> str:
    compact = re.sub(r"\s+", "", user_text.lower())
    if any(marker in compact for marker in ["查", "天气", "搜索", "找", "确认", "核对", "为什么", "怎么回事", "排查", "看看log", "看log"]):
        return "working"
    if any(marker in compact for marker in ["继续", "然后", "接着", "下一步", "开始", "实现", "开发", "改成", "优化"]):
        return "transition"
    if any(marker in compact for marker in ["不行", "失败", "报错", "卡住", "还在", "慢", "问题", "不对"]):
        return "blocked"
    if context_json and any(marker in context_json for marker in ["in_progress", "blocked", "pending"]):
        return "working"
    return "opening"


def suppress_asr_hallucination(text: str) -> str:
    stripped = text.strip()
    compact = compact_speech_text(stripped)
    if compact == "请不吝点赞订阅转发打赏支持明镜与点点栏目":
        return ""
    if sum(marker in compact for marker in ["请不吝点赞", "订阅转发", "打赏支持", "明镜与点点栏目"]) >= 3:
        return ""
    return stripped


def normalize_asr_transcript(text: str) -> str:
    value = normalize_chinese_asr_variants(text).strip()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)
    value = _strip_spoken_fillers(value)
    value = _apply_self_correction_override(value)
    value = _punctuate_spoken_boundaries(value)
    value = re.sub(r"([，,。.!！?？、；;：:]){2,}", r"\1", value)
    value = re.sub(r"\s*([，,。.!！?？、；;：:])\s*", r"\1", value)
    return value.strip(" ，,。.!！?？、；;：:")


def _strip_spoken_fillers(text: str) -> str:
    value = str(text or "")
    filler = r"(?:嗯+|呃+|额+|啊+|唔+|呃嗯|嗯呃|那个|这个|就是|怎么说|怎么讲)"
    value = re.sub(rf"^(?:\s*{filler}\s*[，,。.!！?？、；;：:]*)+", "", value, flags=re.IGNORECASE)
    value = re.sub(rf"([，,。.!！?？、；;：:]\s*){filler}(?=\s*[，,。.!！?？、；;：:])", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(rf"(\s+){filler}(\s+)", r"\1", value, flags=re.IGNORECASE)
    return value.strip()


def _apply_self_correction_override(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    action = r"(?:放一下|播放|打开|放|播|找|搜索|搜)"
    filler = r"(?:嗯+|呃+|额+|啊+|唔+|那个|这个)?"
    pattern = re.compile(
        rf"^(?P<prefix>.*?)(?P<first_action>{action})\s*(?P<first>.+?)"
        rf"[\s，,。.!！?？、；;：:]*{filler}[\s，,。.!！?？、；;：:]*"
        rf"(?:不是|不对)[\s，,。.!！?？、；;：:]*"
        rf"(?P<second_action>{action})\s*(?P<second>.+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(value)
    if not match:
        return value
    second = match.group("second").strip(" ，,。.!！?？、；;：:")
    if not second:
        return value
    compact_second = compact_speech_text(second)
    compact_value = compact_speech_text(value)
    if "官方" in compact_second and any(token in compact_value for token in ["随便一个", "任意一个", "哪个都行", "都可以"]):
        return value
    second_action = match.group("second_action")
    prefix = match.group("prefix").strip(" ，,。.!！?？、；;：:")
    corrected = f"{second_action}{second}"
    return f"{prefix}，{corrected}" if prefix else corrected


def _punctuate_spoken_boundaries(text: str) -> str:
    value = str(text or "")
    for token in sorted(ASR_BOUNDARY_TOKENS, key=len, reverse=True):
        value = re.sub(rf"(?<![，,。.!！?？、；;：:\s])({re.escape(token)})", r"，\1", value, flags=re.IGNORECASE)
        value = re.sub(rf"({re.escape(token)})(?![，,。.!！?？、；;：:\s])", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*[，,]\s*", "", value)
    return value


def suppress_incomplete_fragment(text: str) -> str:
    stripped = text.strip()
    compact = compact_speech_text(stripped)
    return "" if compact in {"不是", "没有", "不对", "算了", "嗯", "啊", "呃", "那个"} else stripped


def is_start_sound_echo(text: str) -> bool:
    return compact_speech_text(text) in START_SOUND_ECHO_PHRASES


def strip_start_sound_echo_prefix(text: str) -> str:
    stripped = text.strip()
    compact = compact_speech_text(stripped)
    for phrase in sorted(START_SOUND_ECHO_PHRASES, key=len, reverse=True):
        if compact.startswith(phrase) and len(compact) > len(phrase) + 1:
            return re.sub(rf"^\s*{re.escape(phrase)}[\s，,。.!！?？、；;：:]*", "", stripped, count=1).strip()
    return stripped


def is_low_information_transcript(text: str) -> bool:
    compact = compact_speech_text(text)
    if not compact:
        return True
    if compact in {"我会", "我就", "我要", "我想", "你看", "这个", "那个", "然后", "就是", "好的", "好吧"}:
        return True
    if len(compact) <= 1:
        return True
    return False
