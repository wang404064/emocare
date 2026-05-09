"""
EmoCare Graph 包
LangGraph 流程图相关模块

使用延迟加载，避免 import 时触发 LLM client 创建 + BERT 模型加载 + Graph 编译。
"""
from .builder import create_emo_graph
from .agent import EmoCareAgent, _get_agent
from .routes import route_after_conversation

# 延迟获取全局 agent 实例（首个调用者触发初始化）
def _lazy_agent():
    return _get_agent()

# 兼容旧代码的便捷函数
async def chat(user_input: str, user_id: str = "anonymous", session_id: str = None) -> dict:
    return await _get_agent().chat(user_input, user_id, session_id)

__all__ = [
    "create_emo_graph",
    "EmoCareAgent",
    "chat",
    "route_after_conversation",
    "_get_agent"
]
