"""
EmoCare Core 包
核心配置和状态定义
"""
from .config import settings, Settings
from .state import (
    AgentState, 
    EmotionAnalysis, 
    PerceptionResult, 
    ToolRequest,
    create_initial_state
)

__all__ = [
    "settings",
    "Settings",
    "AgentState",
    "EmotionAnalysis",
    "PerceptionResult",
    "ToolRequest",
    "create_initial_state"
]
