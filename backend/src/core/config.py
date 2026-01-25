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
    
    # LLM配置 - Qwen3-8B API
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "EMPTY")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen3-8b")
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    
    # 情绪分类配置 - 9种情绪（与情绪识别器一致，不进行映射）
    EMOTION_CATEGORIES: list = [
        "sadness",      # 悲伤
        "anxiety",      # 焦虑
        "anger",        # 愤怒
        "loneliness",   # 孤独
        "shame_guilt",  # 羞耻/内疚
        "hopelessness", # 绝望
        "hope",         # 希望
        "calm",         # 平静
        "joy"           # 喜悦
    ]
    
    # 危机关键词（硬匹配）
    CRISIS_KEYWORDS: list = [
        "自杀", "想死", "不想活", "结束生命", "活着没意思",
        "自我伤害", "割腕", "跳楼", "吃药自杀", "了结",
        "遗书", "不想醒来", "永远离开", "世界没有我会更好",
        "没有人在乎我", "活不下去", "解脱"
    ]
    
    # 危机热线
    CRISIS_HOTLINES: dict = {
        "全国心理援助热线": "xxxxxxxxxx",
        "北京心理危机研究与干预中心": "010-xxxxxxxxxx",
        "生命热线": "xxxxxxxxxx",
        "希望24热线": "xxxxxxxxxx"
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
        r"E:\project\myProject\NewEmoCare\spec-kit-chinese-main\emotional-support-agent\models\emotion_classifier\emotion_risk_model_v1"
    )
    EMOTION_MODEL_DEVICE: Optional[str] = os.getenv("EMOTION_MODEL_DEVICE", None)  # None表示自动选择
    
    class Config:
        env_file = str(ENV_FILE) if ENV_FILE.exists() else ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# 启动时打印配置信息（用于调试）
def print_config():
    """打印当前配置（隐藏敏感信息）"""
    api_key_display = settings.LLM_API_KEY[:10] + "..." if len(settings.LLM_API_KEY) > 10 else "(未设置)"
    print(f"[配置信息] LLM_API_BASE: {settings.LLM_API_BASE}")
    print(f"[配置信息] LLM_MODEL_NAME: {settings.LLM_MODEL_NAME}")
    print(f"[配置信息] LLM_API_KEY: {api_key_display}")

# 在模块加载时打印配置
print_config()
