# EmoCare Agent - 情感陪护助手
# 基于 LangGraph 的多Agent系统
#
# 注意：为避免在模块导入时触发完整的 Agent 初始化（LLM client 创建、BERT 模型加载、
# LangGraph 编译），顶层导入仅暴露配置和状态类型。Agent 实例通过工厂函数延迟创建。

from .core import settings, AgentState

__all__ = ["settings", "AgentState"]
