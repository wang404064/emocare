"""
EmoCare 状态定义
定义 LangGraph 中流转的状态结构
"""
from typing import TypedDict, Optional, List, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class EmotionAnalysis(BaseModel):
    """情绪分析结果"""
    emotion: str = Field(description="识别的情绪类别")
    intensity: float = Field(ge=0, le=1, description="情绪强度 0-1")
    confidence: float = Field(ge=0, le=1, description="识别置信度")


class PerceptionResult(BaseModel):
    """感知Agent输出"""
    emotion: EmotionAnalysis = Field(description="情绪分析")
    crisis_detected: bool = Field(default=False, description="是否检测到危机")
    crisis_keywords_matched: List[str] = Field(default_factory=list, description="匹配到的危机关键词")
    scene_hint: str = Field(default="other", description="场景识别提示")
    scene_confidence: float = Field(ge=0, le=1, default=0.5, description="场景识别置信度")


class ToolRequest(BaseModel):
    """工具调用请求"""
    tool_name: str = Field(description="工具名称")
    parameters: dict = Field(default_factory=dict, description="工具参数")
    reason: str = Field(default="", description="调用原因")


class AgentState(TypedDict):
    """Agent状态 - LangGraph核心状态"""
    # 消息历史 - 使用add_messages reducer自动合并
    messages: Annotated[list, add_messages]
    
    # 当前用户输入
    user_input: str
    
    # 用户ID（用于情绪追踪）
    user_id: str
    
    # 感知结果
    perception: Optional[dict]
    
    # 是否处于危机状态
    is_crisis: bool
    
    # 对话Agent响应
    response: str
    
    # 工具调用请求列表
    tool_requests: List[dict]
    
    # 工具执行结果
    tool_results: List[dict]
    
    # 是否需要工具Agent处理
    needs_tools: bool
    
    # 是否已有工具结果（供对话Agent生成最终回复使用）
    has_tool_results: bool
    
    # 格式化后的工具结果文本（供对话Agent使用）
    tool_results_formatted: str
    
    # 当前对话策略标签（可解释性/可控性）
    current_strategy: str
    
    # 会话元数据
    session_metadata: dict


def create_initial_state(user_input: str, user_id: str = "anonymous") -> AgentState:
    """创建初始状态"""
    return AgentState(
        messages=[],
        user_input=user_input,
        user_id=user_id,
        perception=None,
        is_crisis=False,
        response="",
        tool_requests=[],
        tool_results=[],
        needs_tools=False,
        has_tool_results=False,
        tool_results_formatted="",
        current_strategy="normal_chat",
        session_metadata={}
    )
