"""
LangGraph 路由逻辑
定义条件边的路由决策
"""
from typing import Literal
from loguru import logger

from ..core.state import AgentState


def route_after_perception(state: AgentState) -> Literal["crisis_handler", "conversation"]:
    """
    感知后的路由决策
    根据是否检测到危机信号决定走哪条分支
    """
    is_crisis = state.get("is_crisis", False)
    
    if is_crisis:
        logger.warning("路由决策: 进入危机处理分支")
        return "crisis_handler"
    else:
        logger.info("路由决策: 进入正常对话分支")
        return "conversation"


def route_after_conversation(state: AgentState) -> Literal["tool_agent", "end"]:
    """
    对话后的路由决策
    根据是否需要工具决定是否进入工具Agent
    如果已经有工具结果，直接结束（工具结果已由对话Agent处理）
    """
    # 如果已经有工具结果，说明工具已执行且对话Agent已生成最终回答，直接结束
    has_tool_results = state.get("has_tool_results", False)
    if has_tool_results:
        logger.info("路由决策: 工具已执行，对话Agent已生成最终回答，结束流程")
        return "end"
    
    needs_tools = state.get("needs_tools", False)
    tool_requests = state.get("tool_requests", [])
    
    if needs_tools and tool_requests:
        logger.info(f"路由决策: 进入工具Agent，{len(tool_requests)}个请求")
        return "tool_agent"
    else:
        logger.info("路由决策: 结束流程")
        return "end"
