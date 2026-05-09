"""
危机处理分支 (Crisis Handler)
职责：
- 当检测到危机信号时，提供安全的响应
- 提供专业热线信息
- 使用硬编码的安全模板，不进行任何"建议"
"""
import random
from typing import List
from loguru import logger

from ..core.config import settings
from ..core.state import AgentState


class CrisisHandler:
    """危机处理器 - 提供安全、温和的危机响应"""
    
    # 硬编码的安全响应模板（先共情，后给出资源）
    CRISIS_TEMPLATES = [

        """谢谢你愿意和我说这些。我知道把这些说出来本身就很不容易。

你现在的感受是真实的，也是重要的。不管这一刻有多难熬，请相信——你不是一个人在面对这些。

有一些受过专业训练的人，他们的工作就是在这种时刻倾听和陪伴：

{hotlines}

这些电话不需要你准备什么，你可以在任何感到需要的时候打过去，就只是说说你现在的感受。他们能听懂。

我会一直在这里。你需要的时候，随时回来找我。""",

        """谢谢你信任我，愿意把这些告诉我。我听到你了。

你可能觉得自己被困在了一个看不见出口的地方，但我想让你知道——总有出口，而你不需要一个人找。有人可以陪你一起。

{hotlines}

任何时候，哪怕只是想找人听听自己说话，这些电话都是可以的。

我在这里。现在，也在接下来的任何时候。"""
    ]
    
    def __init__(self):
        self.hotlines = settings.CRISIS_HOTLINES
    
    def format_hotlines(self) -> str:
        """格式化热线信息"""
        lines = []
        for name, number in self.hotlines.items():
            lines.append(f"📞 {name}: {number}")
        return "\n".join(lines)
    
    def generate_response(self, matched_keywords: List[str] = None) -> str:
        """
        生成危机响应
        使用硬编码模板，确保安全性
        """
        template = random.choice(self.CRISIS_TEMPLATES)
        hotlines_text = self.format_hotlines()
        
        response = template.format(hotlines=hotlines_text)
        
        if matched_keywords:
            logger.info(f"危机响应生成，匹配关键词: {matched_keywords}")
        
        return response
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        危机处理主入口
        """
        perception = state.get("perception", {})
        matched_keywords = perception.get("crisis_keywords_matched", [])
        
        logger.warning(f"进入危机处理流程，关键词: {matched_keywords}")
        
        # 生成安全响应
        response = self.generate_response(matched_keywords)
        
        # 更新状态 - 危机情况下不触发工具Agent
        return {
            **state,
            "response": response,
            "needs_tools": False,  # 危机状态下禁用工具
            "tool_requests": []
        }


# 创建全局实例
crisis_handler = CrisisHandler()


async def run_crisis_handler(state: AgentState) -> AgentState:
    """LangGraph节点函数"""
    return await crisis_handler(state)
