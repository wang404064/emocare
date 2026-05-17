"""
WebSocket 支持
"""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from ..graph.agent import _get_agent


router = APIRouter()


class ConnectionManager:
    """WebSocket连接管理器（含心跳检测和连接数限制）"""

    MAX_CONNECTIONS = 100
    HEARTBEAT_SECONDS = 30

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        if len(self.active_connections) >= self.MAX_CONNECTIONS:
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "连接数已达上限"})
            await websocket.close()
            return False
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket 连接: {session_id} (当前 {len(self.active_connections)} 个)")
        return True

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket 断开: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception:
                self.disconnect(session_id)


ws_manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket 实时对话接口（含心跳）
    """
    import asyncio as aio

    accepted = await ws_manager.connect(websocket, session_id)
    if not accepted:
        return

    # 心跳任务
    async def heartbeat():
        while True:
            await aio.sleep(ConnectionManager.HEARTBEAT_SECONDS)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    heartbeat_task = aio.create_task(heartbeat())

    try:
        while True:
            # 支持 text JSON 和 binary 两种消息
            message = await websocket.receive()

            if "text" in message:
                try:
                    message_data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws_manager.send_message(session_id, {
                        "type": "error",
                        "message": "消息格式错误，请发送有效的 JSON"
                    })
                    continue

                # 客户端心跳响应
                if message_data.get("type") == "pong":
                    continue

                # 文本路径
                user_input = message_data.get("message", "")
                user_id = message_data.get("user_id", "anonymous")

                if not user_input:
                    continue

                result = await _get_agent().chat(
                    user_input=user_input,
                    user_id=user_id,
                    session_id=session_id
                )

            elif "bytes" in message:
                # 音频路径
                audio_bytes = message["bytes"]
                user_id = "anonymous"  # WebSocket 无 metadata，取默认值

                if not audio_bytes or len(audio_bytes) < 800:
                    await ws_manager.send_message(session_id, {
                        "type": "error",
                        "message": "音频数据过短"
                    })
                    continue

                result = await _get_agent().chat_audio(
                    audio_data=audio_bytes,
                    user_id=user_id,
                    session_id=session_id,
                )

            else:
                continue

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
        logger.error(f"WebSocket 错误: {e}")
        ws_manager.disconnect(session_id)
    finally:
        heartbeat_task.cancel()
