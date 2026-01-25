"""
API 请求/响应模型定义
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息", min_length=1)
    user_id: str = Field(default="anonymous", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")


class EmotionInfo(BaseModel):
    """情绪信息"""
    emotion: str = Field(default="calm", description="情绪类别（9种情绪之一）")
    intensity: float = Field(default=0.5, description="情绪强度")
    confidence: float = Field(default=0.5, description="置信度")


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str = Field(..., description="助手回复")
    emotion: EmotionInfo = Field(default_factory=EmotionInfo, description="情绪分析")
    scene: str = Field(default="other", description="场景识别")
    is_crisis: bool = Field(default=False, description="是否危机状态")
    session_id: str = Field(..., description="会话ID")


class HistoryMessage(BaseModel):
    """历史消息"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")


class SessionHistoryResponse(BaseModel):
    """会话历史响应"""
    session_id: str
    messages: List[HistoryMessage]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    service: str
    version: str
