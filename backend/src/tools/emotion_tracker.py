"""
情绪追踪工具 (Emotion Tracker Tool)
记录和分析用户的情绪变化
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger


class EmotionTrackerTool:
    """情绪追踪记录工具"""
    
    def __init__(self, storage_path: str = "./data/emotions"):
        self.name = "emotion_tracker"
        self.description = "记录用户情绪用于长期追踪"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_user_file(self, user_id: str) -> Path:
        """获取用户情绪记录文件路径"""
        return self.storage_path / f"{user_id}_emotions.json"
    
    def _load_records(self, user_id: str) -> List[Dict]:
        """加载用户的情绪记录"""
        file_path = self._get_user_file(user_id)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载情绪记录失败: {e}")
                return []
        return []
    
    def _save_records(self, user_id: str, records: List[Dict]):
        """保存情绪记录"""
        file_path = self._get_user_file(user_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存情绪记录失败: {e}")
    
    def record_emotion(
        self, 
        user_id: str, 
        emotion: str, 
        intensity: float,
        scene: str = "other",
        note: str = "",
        recent_conversations: Optional[List[Dict[str, str]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        记录一条情绪
        只记录强度大于0.7的情绪，并保存最近3轮对话的上下文
        情绪记录可以保存多条，不限制数量
        """
        # 只记录强度大于0.7的情绪
        if intensity <= 0.7:
            logger.debug(f"情绪强度 {intensity} 未达到阈值0.7，跳过记录: user={user_id}, emotion={emotion}")
            return None
        
        records = self._load_records(user_id)
        
        # 处理最近3轮对话
        conversation_context = []
        if recent_conversations:
            # 只取最近3轮对话
            conversation_context = recent_conversations[-3:] if len(recent_conversations) > 3 else recent_conversations
        
        new_record = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion,
            "intensity": intensity,
            "scene": scene,
            "note": note,
            "conversation_context": conversation_context  # 保存最近3轮对话的上下文
        }
        
        # 添加新记录（不限制数量）
        records.append(new_record)
        
        self._save_records(user_id, records)
        
        logger.info(f"记录情绪: user={user_id}, emotion={emotion}, intensity={intensity}, 对话轮数={len(conversation_context)}")
        
        return new_record
    
    def get_recent_emotions(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict]:
        """获取最近的情绪记录"""
        records = self._load_records(user_id)
        return records[-limit:] if records else []
    
    def get_emotion_summary(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """获取情绪统计摘要"""
        records = self._load_records(user_id)
        
        if not records:
            return {
                "total_records": 0,
                "emotion_distribution": {},
                "average_intensity": 0,
                "trend": "no_data"
            }
        
        # 筛选指定天数内的记录
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        recent = [
            r for r in records 
            if datetime.fromisoformat(r['timestamp']).timestamp() > cutoff
        ]
        
        if not recent:
            return {
                "total_records": 0,
                "emotion_distribution": {},
                "average_intensity": 0,
                "trend": "no_data"
            }
        
        # 统计情绪分布
        emotion_counts = {}
        total_intensity = 0
        
        for r in recent:
            emotion = r['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_intensity += r['intensity']
        
        avg_intensity = total_intensity / len(recent)
        
        # 简单趋势判断
        if len(recent) >= 3:
            recent_avg = sum(r['intensity'] for r in recent[-3:]) / 3
            earlier_avg = sum(r['intensity'] for r in recent[:3]) / 3
            
            if recent_avg > earlier_avg + 0.1:
                trend = "improving" if recent[-1]['emotion'] in ['joy', 'trust'] else "worsening"
            elif recent_avg < earlier_avg - 0.1:
                trend = "worsening" if recent[-1]['emotion'] in ['joy', 'trust'] else "improving"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "total_records": len(recent),
            "emotion_distribution": emotion_counts,
            "average_intensity": round(avg_intensity, 2),
            "trend": trend
        }
    
    async def run(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行工具"""
        params = parameters or {}
        action = params.get("action", "record")
        user_id = params.get("user_id", "anonymous")
        
        if action == "record":
            record = self.record_emotion(
                user_id=user_id,
                emotion=params.get("emotion", "calm"),
                intensity=params.get("intensity", 0.5),
                scene=params.get("scene", "other"),
                note=params.get("note", ""),
                recent_conversations=params.get("recent_conversations", None)
            )
            if record is None:
                # 情绪强度未达到阈值，不记录
                return {
                    "success": True,
                    "tool_name": self.name,
                    "result": {
                        "action": "skipped",
                        "reason": "情绪强度未达到阈值0.7",
                        "intensity": params.get("intensity", 0.5)
                    }
                }
            return {
                "success": True,
                "tool_name": self.name,
                "result": {"action": "recorded", "record": record}
            }
        
        elif action == "get_recent":
            records = self.get_recent_emotions(
                user_id=user_id,
                limit=params.get("limit", 10)
            )
            return {
                "success": True,
                "tool_name": self.name,
                "result": {"action": "get_recent", "records": records}
            }
        
        elif action == "summary":
            summary = self.get_emotion_summary(
                user_id=user_id,
                days=params.get("days", 7)
            )
            return {
                "success": True,
                "tool_name": self.name,
                "result": {"action": "summary", "summary": summary}
            }
        
        return {
            "success": False,
            "tool_name": self.name,
            "error": f"Unknown action: {action}"
        }


# 全局实例
emotion_tracker = EmotionTrackerTool()
