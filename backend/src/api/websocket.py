"""
WebSocket 支持
"""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from ..graph.agent import emo_agent


router = APIRouter()


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket连接: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket断开: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


ws_manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket实时对话接口
    """
    await ws_manager.connect(websocket, session_id)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_input = message_data.get("message", "")
            user_id = message_data.get("user_id", "anonymous")
            
            if not user_input:
                continue
            
            # 调用Agent
            result = await emo_agent.chat(
                user_input=user_input,
                user_id=user_id,
                session_id=session_id
            )
            
            # 发送响应
            await ws_manager.send_message(session_id, {
                "type": "response",
                "response": result.get("response", ""),
                "emotion": result.get("emotion", {}),
                "scene": result.get("scene", ""),
                "is_crisis": result.get("is_crisis", False)
            })
            
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        ws_manager.disconnect(session_id)
