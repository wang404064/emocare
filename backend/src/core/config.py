"""
EmoCare 配置管理
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 获取backend目录的绝对路径
BACKEND_DIR = Path(__file__).parent.parent.parent
ENV_FILE = BACKEND_DIR / ".env"

# 加载.env文件（如果存在）
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=True)
    print(f"已加载环境变量文件: {ENV_FILE}")
else:
    # 如果.env文件不存在，尝试从当前目录加载
    load_dotenv()
    print(f"警告: .env文件不存在于 {ENV_FILE}，尝试从当前目录加载")


class Settings(BaseSettings):
    """应用配置"""
    
    # LLM配置 — 兼容多家厂商 + 本地模型（代码零改动，改 .env 即可切换）
    # 云端: DashScope(qwen3-max) | DeepSeek(deepseek-chat) | 本地: vLLM(emocare-8b) | Ollama(emocare:latest)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "EMPTY")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen3-max")
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    
    # 情绪分类配置 - 7种情绪（v2.1 合并 shame_guilt→loneliness, hope→joy）
    EMOTION_CATEGORIES: list = [
        "sadness",      # 悲伤
        "anxiety",      # 焦虑
        "anger",        # 愤怒
        "loneliness",   # 孤独/羞耻/内疚
        "hopelessness", # 绝望
        "calm",         # 平静
        "joy"           # 喜悦/希望
    ]
    
    # 危机关键词（强信号 - 直接进入危机流程）
    CRISIS_KEYWORDS: list = [
        "自杀", "想死", "不想活了", "结束生命", "活着没意思",
        "自我伤害", "割腕", "跳楼", "吃药自杀",
        "遗书", "不想醒来", "永远离开", "世界没有我会更好",
        "活不下去"
    ]
    
    # 弱危机信号关键词（需温和关注，不直接进危机流程）
    WEAK_CRISIS_KEYWORDS: list = [
        "没意思", "累了", "不想动", "什么都不想做",
        "好累", "撑不住了", "很绝望", "没人在乎我",
        "没有人理解我", "没人理解", "活得好累"
    ]
    
    # 口语化误报过滤（正则模式，匹配整个字符串）
    # 注意：这些是正则表达式，不是简单子串
    CRISIS_FALSE_POSITIVE_PATTERNS: list = [
        # "想死X了" 句式 — 口语思念/期待，非真实危机
        r"想死.{1,4}了",
        # "X死我了" 句式 — 口语夸张表达
        r".{1,3}死我了",
        # "笑死/吓死/气死/累死/烦死/饿死/困死/吵死" 等口语夸张
        r"(笑|吓|气|累|烦|饿|困|吵|冻|热|疼|撑|咸|辣|酸|苦|闲|忙|穷|丑|胖)死",
        # "死了" 在常见口语搭配中
        r"(开心|高兴|舒服|爽|激动|感动|幸福|便宜|简单|容易|快|慢)死了",
    ]
    
    # 危机热线（真实可用号码）
    CRISIS_HOTLINES: dict = {
        "全国心理援助热线": "400-161-9995",
        "北京心理危机研究与干预中心": "010-82951332",
        "生命热线": "400-821-1215",
        "希望24热线": "400-161-9995",
        "紧急求助": "120 / 110"
    }
    
    # 场景分类
    SCENE_CATEGORIES: list = [
        "work_stress",       # 工作压力
        "relationship",      # 人际关系
        "family",            # 家庭问题
        "health_anxiety",    # 健康焦虑
        "loneliness",        # 孤独感
        "self_doubt",        # 自我怀疑
        "life_meaning",      # 人生意义
        "daily_chat",        # 日常闲聊
        "other"              # 其他
    ]
    
    # 对话历史长度限制
    MAX_HISTORY_LENGTH: int = 20
    
    # 情绪识别器配置
    EMOTION_MODEL_PATH: Optional[str] = os.getenv(
        "EMOTION_MODEL_PATH",
        str(BACKEND_DIR / "models" / "emotion_risk_model_v2")
    )
    EMOTION_MODEL_DEVICE: Optional[str] = os.getenv("EMOTION_MODEL_DEVICE", None)  # None表示自动选择

    # 语音情绪识别配置
    AUDIO_MODEL_NAME: str = os.getenv(
        "AUDIO_MODEL_NAME", "iic/SenseVoiceSmall"
    )
    AUDIO_VAD_MODEL: str = os.getenv("AUDIO_VAD_MODEL", "fsmn-vad")
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_EMOTION_WEIGHT: float = 0.3  # 音频情绪在融合中的默认权重
    AUDIO_MAX_DURATION_SEC: int = 30   # 最长录音时长

    class Config:
        env_file = str(ENV_FILE) if ENV_FILE.exists() else ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# 注入 ModelScope 缓存路径（避免下载到 C 盘）
if os.getenv("MODELSCOPE_CACHE"):
    os.environ.setdefault("MODELSCOPE_CACHE", os.getenv("MODELSCOPE_CACHE"))

# 启动时打印配置信息（用于调试）
def print_config():
    """打印当前配置（隐藏敏感信息）"""
    api_key_display = settings.LLM_API_KEY[:10] + "..." if len(settings.LLM_API_KEY) > 10 else "(未设置)"
    print(f"[配置信息] LLM_API_BASE: {settings.LLM_API_BASE}")
    print(f"[配置信息] LLM_MODEL_NAME: {settings.LLM_MODEL_NAME}")
    print(f"[配置信息] LLM_API_KEY: {api_key_display}")

# 在模块加载时打印配置
print_config()
