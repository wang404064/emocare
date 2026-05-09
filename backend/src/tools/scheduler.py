"""
消息调度工具 (Message Scheduler)
- 主动关怀消息调度
- 提醒功能
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger

# 使用APScheduler进行任务调度
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger


class ProactiveMessageScheduler:
    """主动消息调度器"""

    # 待投递的消息（user_id → [messages]，用于 HTTP 轮询）
    _pending_delivery: dict[str, list[dict]] = {}

    @classmethod
    def get_pending_for_user(cls, user_id: str) -> list[dict]:
        """获取并清空指定用户的待投递消息"""
        msgs = cls._pending_delivery.pop(user_id, [])
        return msgs

    @classmethod
    def enqueue_for_user(cls, user_id: str, message: str):
        """将消息放入待投递队列"""
        cls._pending_delivery.setdefault(user_id, []).append({
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    # 主动关怀消息模板
    PROACTIVE_TEMPLATES = {
        "check_in": [
            "嘿，想起你了。今天感觉怎么样？",
            "路过来看看你。最近还好吗？",
            "Hi~ 今天过得如何？",
        ],
        "encouragement": [
            "新的一天，你已经很棒了 ✨",
            "不管今天怎样，记得你值得被善待",
            "想告诉你：你比自己想象的更坚强",
        ],
        "evening": [
            "忙碌的一天结束了，给自己一个拥抱吧",
            "晚上好，今天辛苦了",
            "夜深了，希望你能好好休息",
        ]
    }
    
    def __init__(self):
        self.name = "proactive_message"
        self.description = "调度主动关怀消息"
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.pending_messages: Dict[str, Dict] = {}  # job_id -> message_info
        self._message_callback = None
    
    def init_scheduler(self, message_callback=None):
        """初始化调度器"""
        if self.scheduler is None:
            self.scheduler = AsyncIOScheduler()
            self.scheduler.start()
            logger.info("消息调度器已启动")
        
        if message_callback:
            self._message_callback = message_callback
    
    def stop_scheduler(self):
        """停止调度器"""
        if self.scheduler:
            self.scheduler.shutdown()
            self.scheduler = None
            logger.info("消息调度器已停止")

    def schedule_check_in(self, user_id: str, delay_minutes: int = 30, message_type: str = "check_in"):
        """为指定用户调度一次主动关怀（应用层自动触发，非 LLM 工具触发）"""
        import random
        if message_type in self.PROACTIVE_TEMPLATES:
            message = random.choice(self.PROACTIVE_TEMPLATES[message_type])
        else:
            message = "Hi，想看看你今天怎么样"
        send_at = datetime.now() + timedelta(minutes=delay_minutes)
        if not self.scheduler:
            self.init_scheduler()
        return self.schedule_message(user_id, message, send_at, message_type)

    async def _send_message(self, job_id: str, user_id: str, message: str):
        """发送消息的回调（WebSocket 推送 + HTTP 轮询双路兜底）"""
        logger.info(f"发送主动消息: user={user_id}, message={message[:30]}...")

        # 入队待投递（供 HTTP 轮询）
        self.enqueue_for_user(user_id, message)

        # WebSocket 实时推送（如果客户端在线）
        if self._message_callback:
            try:
                await self._message_callback(user_id, message)
            except Exception as e:
                logger.warning(f"WebSocket 推送失败（消息已入轮询队列）: {e}")

        if job_id in self.pending_messages:
            del self.pending_messages[job_id]
    
    def schedule_message(
        self,
        user_id: str,
        message: str,
        send_at: datetime,
        message_type: str = "custom"
    ) -> str:
        """调度一条消息"""
        if not self.scheduler:
            self.init_scheduler()
        
        job_id = str(uuid.uuid4())
        
        # 添加任务
        self.scheduler.add_job(
            self._send_message,
            trigger=DateTrigger(run_date=send_at),
            args=[job_id, user_id, message],
            id=job_id
        )
        
        self.pending_messages[job_id] = {
            "user_id": user_id,
            "message": message,
            "send_at": send_at.isoformat(),
            "type": message_type
        }
        
        logger.info(f"已调度消息: job_id={job_id}, send_at={send_at}")
        
        return job_id
    
    def cancel_message(self, job_id: str) -> bool:
        """取消调度的消息"""
        try:
            if self.scheduler and job_id in self.pending_messages:
                self.scheduler.remove_job(job_id)
                del self.pending_messages[job_id]
                logger.info(f"已取消消息: job_id={job_id}")
                return True
        except Exception as e:
            logger.error(f"取消消息失败: {e}")
        return False
    
    def get_pending_messages(self, user_id: str = None) -> List[Dict]:
        """获取待发送的消息"""
        messages = list(self.pending_messages.values())
        if user_id:
            messages = [m for m in messages if m['user_id'] == user_id]
        return messages
    
    async def run(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行工具"""
        params = parameters or {}
        action = params.get("action", "schedule")
        user_id = params.get("user_id", "anonymous")
        
        if action == "schedule":
            # 解析时间
            delay_minutes = params.get("delay_minutes", 60)
            send_at = datetime.now() + timedelta(minutes=delay_minutes)
            
            message = params.get("message", "")
            message_type = params.get("type", "check_in")
            
            # 如果没有提供消息，使用模板
            if not message and message_type in self.PROACTIVE_TEMPLATES:
                import random
                message = random.choice(self.PROACTIVE_TEMPLATES[message_type])
            
            if not message:
                message = "Hi，想看看你今天怎么样"
            
            job_id = self.schedule_message(
                user_id=user_id,
                message=message,
                send_at=send_at,
                message_type=message_type
            )
            
            return {
                "success": True,
                "tool_name": self.name,
                "result": {
                    "action": "scheduled",
                    "job_id": job_id,
                    "send_at": send_at.isoformat()
                }
            }
        
        elif action == "cancel":
            job_id = params.get("job_id")
            success = self.cancel_message(job_id) if job_id else False
            return {
                "success": success,
                "tool_name": self.name,
                "result": {"action": "cancelled", "job_id": job_id}
            }
        
        elif action == "list":
            messages = self.get_pending_messages(user_id)
            return {
                "success": True,
                "tool_name": self.name,
                "result": {"action": "list", "messages": messages}
            }
        
        return {
            "success": False,
            "tool_name": self.name,
            "error": f"Unknown action: {action}"
        }


class ReminderTool:
    """提醒工具"""

    def __init__(self, scheduler: ProactiveMessageScheduler):
        self.name = "reminder"
        self.description = "设置提醒"
        self.scheduler = scheduler
    
    def parse_time_expression(self, expression: str) -> Optional[datetime]:
        """解析时间表达式（简单实现）"""
        now = datetime.now()
        
        # 简单的时间解析
        if "明天" in expression:
            target = now + timedelta(days=1)
            # 默认早上9点
            return target.replace(hour=9, minute=0, second=0, microsecond=0)
        
        if "后天" in expression:
            target = now + timedelta(days=2)
            return target.replace(hour=9, minute=0, second=0, microsecond=0)
        
        if "小时后" in expression or "个小时后" in expression:
            import re
            match = re.search(r'(\d+)\s*(小时|个小时)后', expression)
            if match:
                hours = int(match.group(1))
                return now + timedelta(hours=hours)
        
        if "分钟后" in expression:
            import re
            match = re.search(r'(\d+)\s*分钟后', expression)
            if match:
                minutes = int(match.group(1))
                return now + timedelta(minutes=minutes)
        
        # 默认1小时后
        return now + timedelta(hours=1)
    
    async def run(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行工具"""
        params = parameters or {}
        user_id = params.get("user_id", "anonymous")
        content = params.get("content", "")
        time_expr = params.get("time", "1小时后")
        
        # 解析时间
        remind_at = self.parse_time_expression(time_expr)
        if not remind_at:
            remind_at = datetime.now() + timedelta(hours=1)
        
        # 构建提醒消息
        reminder_message = f"⏰ 提醒：{content}" if content else "⏰ 这是你设置的提醒"
        
        # 使用调度器
        job_id = self.scheduler.schedule_message(
            user_id=user_id,
            message=reminder_message,
            send_at=remind_at,
            message_type="reminder"
        )
        
        logger.info(f"设置提醒: user={user_id}, content={content}, at={remind_at}")
        
        return {
            "success": True,
            "tool_name": self.name,
            "result": {
                "job_id": job_id,
                "remind_at": remind_at.isoformat(),
                "content": content,
                "confirmation": f"好的，我会在 {remind_at.strftime('%m月%d日 %H:%M')} 提醒你"
            }
        }


# 全局实例
proactive_scheduler = ProactiveMessageScheduler()
reminder_tool = ReminderTool(proactive_scheduler)
