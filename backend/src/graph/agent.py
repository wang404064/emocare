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
    """
    
    def __init__(self):
        self.graph = create_emo_graph()
        self._sessions = {}  # 会话存储
    
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
            session_id: 会话ID（用于维护对话历史）
        
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
        
        # 获取或创建会话状态
        if session_id in self._sessions:
            messages = self._sessions[session_id].get("messages", [])
        else:
            messages = []
        
        # 创建初始状态
        initial_state = {
            "messages": messages,
            "user_input": user_input,
            "user_id": user_id,
            "perception": None,
            "is_crisis": False,
            "response": "",
            "tool_requests": [],
            "tool_results": [],
            "needs_tools": False,
            "session_metadata": {}
        }
        
        # 配置（用于checkpointer）
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        logger.info(f"处理消息: user={user_id}, session={session_id}")
        
        # 执行图
        try:
            result = await self.graph.ainvoke(initial_state, config)
            
            # 更新会话存储
            self._sessions[session_id] = {
                "messages": result.get("messages", []),
                "user_id": user_id
            }
            
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
                "tool_results": [],
                "error": str(e)
            }
    
    def get_session_history(self, session_id: str) -> list:
        """获取会话历史"""
        if session_id in self._sessions:
            messages = self._sessions[session_id].get("messages", [])
            return [
                {
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                }
                for msg in messages
            ]
        return []
    
    def clear_session(self, session_id: str):
        """清除会话历史"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"已清除会话: {session_id}")


# 创建全局实例
emo_agent = EmoCareAgent()


# 便捷函数
async def chat(user_input: str, user_id: str = "anonymous", session_id: str = None) -> dict:
    """便捷的聊天函数"""
    return await emo_agent.chat(user_input, user_id, session_id)
