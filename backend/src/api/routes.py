"""
EmoCare API 路由
提供RESTful接口
"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from ..graph.agent import _get_agent
from .schemas import (
    ChatRequest, 
    ChatResponse, 
    EmotionInfo, 
    HistoryMessage, 
    SessionHistoryResponse, 
    HealthResponse
)
from .websocket import router as ws_router


# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["EmoCare"])


# ==================== API 端点 ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送消息给EmoCare助手
    
    - 自动进行情绪识别
    - 检测危机信号并提供安全响应
    - 支持多轮对话
    """
    try:
        # 确定session_id
        session_id = request.session_id or request.user_id

        logger.info(f"收到消息: user={request.user_id}, session={session_id}")

        # 调用Agent
        agent = _get_agent()
        result = await agent.chat(
            user_input=request.message,
            user_id=request.user_id,
            session_id=session_id
        )

        # 首次会话自动安排回访关怀（30分钟后发送问候）
        history = agent.get_session_history(session_id)
        if len(history) <= 2:  # 仅有本轮 1 user + 1 assistant
            from ..tools import proactive_scheduler
            proactive_scheduler.schedule_check_in(
                user_id=request.user_id,
                delay_minutes=30,
                message_type="check_in"
            )

        # 构建响应
        emotion_data = result.get("emotion", {})

        return ChatResponse(
            response=result.get("response", ""),
            emotion=EmotionInfo(
                emotion=emotion_data.get("emotion", "calm"),
                intensity=emotion_data.get("intensity", 0.5),
                confidence=emotion_data.get("confidence", 0.5)
            ),
            scene=result.get("scene", "other"),
            is_crisis=result.get("is_crisis", False),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """
    获取会话历史
    """
    try:
        history = _get_agent().get_session_history(session_id)
        
        return SessionHistoryResponse(
            session_id=session_id,
            messages=[
                HistoryMessage(role=msg["role"], content=msg["content"])
                for msg in history
            ]
        )
        
    except Exception as e:
        logger.error(f"获取历史错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    清除会话历史
    """
    try:
        _get_agent().clear_session(session_id)
        return {"message": f"会话 {session_id} 已清除"}
        
    except Exception as e:
        logger.error(f"清除会话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查
    """
    return HealthResponse(
        status="healthy",
        service="EmoCare Agent",
        version="1.0.0"
    )


@router.get("/session/{session_id}/proactive")
async def get_proactive_messages(session_id: str):
    """
    获取并清空待投递的主动关怀消息（供前端轮询）
    每 30 秒调用一次即可
    """
    from ..tools.scheduler import ProactiveMessageScheduler
    messages = ProactiveMessageScheduler.get_pending_for_user(session_id)
    return {"messages": messages}


# 包含WebSocket路由
router.include_router(ws_router)
