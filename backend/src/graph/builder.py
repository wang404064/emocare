"""
LangGraph 流程图构建器
构建完整的Agent工作流
"""
from loguru import logger

from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from ..core.state import AgentState
from ..core.config import BACKEND_DIR
from ..agents.perception import run_perception
from ..agents.conversation import run_conversation
from ..agents.tools import run_tool_agent
from .routes import route_after_perception, route_after_conversation
from ..agents.crisis import run_crisis_handler


def extract_user_input(state: AgentState) -> AgentState:
    """
    从 messages 中提取最新的用户输入
    这个节点让图能够直接处理 messages 输入，启用 Chat 功能
    """
    # 优先使用直接传入的 user_input
    user_input = state.get("user_input", "")
    
    # 如果没有传入 user_input，才从 messages 中提取最新的用户消息
    if not user_input:
        messages = state.get("messages", [])
        if messages:
            # 从后往前查找最后一条用户消息（只提取 HumanMessage，忽略 AIMessage）
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_input = msg.content
                    break
    
    return {
        **state,
        "user_input": user_input
    }


def add_messages_to_state(state: AgentState) -> AgentState:
    """
    在流程结束前，将用户输入和响应添加到消息历史
    """
    user_input = state.get("user_input", "")
    response = state.get("response", "")
    messages = list(state.get("messages", []))
    
    # 添加用户消息
    messages.append(HumanMessage(content=user_input))
    
    # 添加助手响应
    if response:
        messages.append(AIMessage(content=response))
    
    return {
        **state,
        "messages": messages
    }


def create_emo_graph():
    """
    创建EmoCare Agent工作流图
    
    流程:
    1. extract_input (从 messages 提取用户输入) 
       ↓
    2. perception (感知Agent) 
       ↓
    3. [条件路由] 
       ├── 危机检测 → crisis_handler → finalize
       └── 正常 → conversation
                    ↓
    4. [条件路由]
       ├── 需要工具 → tool_agent → conversation (根据工具结果生成回答) → finalize
       └── 不需要 → finalize
    """
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("extract_input", extract_user_input)  # 输入适配器 - 支持 messages 输入
    workflow.add_node("perception", run_perception)
    workflow.add_node("crisis_handler", run_crisis_handler)
    workflow.add_node("conversation", run_conversation)
    workflow.add_node("tool_agent", run_tool_agent)
    workflow.add_node("finalize", add_messages_to_state)
    
    # 设置入口点 - 从 extract_input 开始，支持 messages 输入
    workflow.set_entry_point("extract_input")
    
    # extract_input 后进入 perception
    workflow.add_edge("extract_input", "perception")
    
    # 添加边：感知后的条件路由
    workflow.add_conditional_edges(
        "perception",
        route_after_perception,
        {
            "crisis_handler": "crisis_handler",
            "conversation": "conversation"
        }
    )
    
    # 危机处理后直接结束
    workflow.add_edge("crisis_handler", "finalize")
    
    # 对话后的条件路由
    workflow.add_conditional_edges(
        "conversation",
        route_after_conversation,
        {
            "tool_agent": "tool_agent",
            "end": "finalize"
        }
    )
    
    # 工具Agent后回到对话Agent，让对话Agent根据工具结果生成最终回答
    workflow.add_edge("tool_agent", "conversation")
    
    # 最终节点连接到END
    workflow.add_edge("finalize", END)
    
    # 编译图，使用 SQLite 持久化检查点（服务重启不丢失对话历史）
    db_dir = BACKEND_DIR / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "checkpoints.db")

    try:
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        logger.info(f"使用 SQLite 检查点: {db_path}")
    except Exception as e:
        logger.warning(f"SQLite 检查点初始化失败，回退到内存模式: {e}")
        checkpointer = MemorySaver()

    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("EmoCare Graph 已创建")

    return graph
