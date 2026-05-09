"""
LangGraph 路由逻辑
定义条件边的路由决策

危机分级路由说明：
  - crisis_immediate  → crisis_handler  (强危机信号，立即干预)
  - empathy_first_gentle_probe → conversation (弱危机信号，温和探询，由对话Agent的策略控制)
  - 其他策略          → conversation  (正常对话)
"""
from typing import Literal
from loguru import logger

from ..core.state import AgentState


def route_after_perception(state: AgentState) -> Literal["crisis_handler", "conversation"]:
    """
    感知后的路由决策
    基于 current_strategy 做三级路由：
      - crisis_immediate → crisis_handler
      - 其他（含 empathy_first_gentle_probe）→ conversation
    同时兼容旧的 is_crisis bool 标志作为兜底。
    """
    current_strategy = state.get("current_strategy", "normal_chat")
    is_crisis = state.get("is_crisis", False)

    if current_strategy == "crisis_immediate" or is_crisis:
        logger.warning(
            f"路由决策: 进入危机处理分支 (strategy={current_strategy}, is_crisis={is_crisis})"
        )
        return "crisis_handler"
    
    if current_strategy == "empathy_first_gentle_probe":
        logger.warning(
            "路由决策: 检测到弱危机信号，进入对话分支（温和探询策略）"
        )
    else:
        logger.info(f"路由决策: 进入正常对话分支 (strategy={current_strategy})")
    
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
