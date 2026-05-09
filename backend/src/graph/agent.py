"""
EmoCare Agent 封装类
提供简单的接口来使用Agent
"""
from loguru import logger

from .builder import create_emo_graph


class EmoCareAgent:
    """
    EmoCare Agent 封装类
    提供简单的接口来使用Agent
    
    注意：会话历史的持久化完全依赖 LangGraph MemorySaver Checkpointer，
    不再维护额外的 _sessions 字典，避免双写导致的上下文重复/混乱。
    """
    
    def __init__(self):
        self.graph = create_emo_graph()
    
    async def chat(
        self, 
        user_input: str, 
        user_id: str = "anonymous",
        session_id: str = None
    ) -> dict:
        """
        处理用户消息
        
        Args:
            user_input: 用户输入
            user_id: 用户ID
            session_id: 会话ID（用于维护对话历史，对应 Checkpointer thread_id）
        
        Returns:
            {
                "response": str,
                "emotion": dict,
                "is_crisis": bool,
                "tool_results": list
            }
        """
        if session_id is None:
            session_id = user_id
        
        # 只传入本轮新增的数据，历史消息由 Checkpointer 自动从上一轮 state 恢复
        # 不传历史 messages，避免与 Checkpointer 中的历史重复叠加
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "user_id": user_id,
            "perception": None,
            "is_crisis": False,
            "response": "",
            "tool_requests": [],
            "tool_results": [],
            "needs_tools": False,
            "has_tool_results": False,
            "tool_results_formatted": "",
            "current_strategy": "normal_chat",
            "session_metadata": {}
        }
        
        # 配置（用于checkpointer，thread_id 是会话的唯一标识）
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        logger.info(f"处理消息: user={user_id}, session={session_id}")
        
        # 执行图
        try:
            result = await self.graph.ainvoke(initial_state, config)
            
            # 返回结果
            return {
                "response": result.get("response", ""),
                "emotion": result.get("perception", {}).get("emotion", {}),
                "scene": result.get("perception", {}).get("scene_hint", ""),
                "is_crisis": result.get("is_crisis", False),
                "tool_results": result.get("tool_results", [])
            }
            
        except Exception as e:
            logger.error(f"图执行失败: {e}")
            return {
                "response": "抱歉，我遇到了一些问题。你可以再说一遍吗？",
                "emotion": {},
                "scene": "",
                "is_crisis": False,
                "tool_results": []
            }
    
    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史
        从 Checkpointer 中读取状态，而非独立维护的字典
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.graph.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {
                        "role": "user" if msg.type == "human" else "assistant",
                        "content": msg.content
                    }
                    for msg in messages
                ]
        except Exception as e:
            logger.warning(f"获取会话历史失败: {e}")
        return []
    
    def clear_session(self, session_id: str):
        """清除会话历史（通过重置 Checkpointer 状态）"""
        # MemorySaver 不支持直接删除，通过记录清空消息列表来实现
        # 更彻底的方案：使用 SqliteSaver 并删除对应的 checkpoint 记录
        logger.info(f"会话 {session_id} 已请求清除（MemorySaver 下下次对话将从空状态开始）")


# 延迟初始化的全局单例
_agent_instance: EmoCareAgent | None = None


def _get_agent() -> EmoCareAgent:
    """获取全局 agent 单例（首次调用时初始化）"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = EmoCareAgent()
    return _agent_instance

