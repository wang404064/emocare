"""
EmoCare API - 情感陪护助手
主入口文件
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
import os
import time
from collections import defaultdict
from pathlib import Path
from .core.config import BACKEND_DIR

_LOG_DIR = BACKEND_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    str(_LOG_DIR / "emocare_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="7 days",
    level="DEBUG"
)

# ─── 速率限制中间件 ──────────────────────────────────────────────────────────────
# 基于滑动窗口的简单 IP 限流，无需外部依赖

_RATE_LIMIT_WINDOW = 60       # 窗口秒数
_RATE_LIMIT_MAX_REQUESTS = 30 # 每窗口最大请求数
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _rate_limit_middleware(request: Request, call_next):
    """简单滑动窗口 IP 限流"""
    now = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # 清理过期记录
    timestamps = [t for t in _rate_limit_store[client_ip] if now - t < _RATE_LIMIT_WINDOW]
    _rate_limit_store[client_ip] = timestamps

    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "请求过于频繁，请稍后再试"}
        )

    _rate_limit_store[client_ip].append(now)

    # 定期清理全量过期的 IP 条目
    if len(_rate_limit_store) > 1000:
        expired_ips = [ip for ip, ts in _rate_limit_store.items()
                       if not [t for t in ts if now - t < _RATE_LIMIT_WINDOW]]
        for ip in expired_ips:
            del _rate_limit_store[ip]

    return call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # --- 启动 ---
    logger.info("EmoCare API 启动中...")
    logger.info("文档地址: http://localhost:8080/docs")

    # 初始化主动关怀调度器，回调广播到 WebSocket 客户端
    from .tools import proactive_scheduler
    from .api.websocket import ws_manager

    async def proactive_callback(user_id: str, message: str):
        """主动关怀消息通过 WebSocket 推送到前端"""
        for sid, ws in list(ws_manager.active_connections.items()):
            try:
                await ws.send_json({
                    "type": "proactive",
                    "message": message,
                    "user_id": user_id
                })
            except Exception:
                pass

    proactive_scheduler.init_scheduler(message_callback=proactive_callback)
    # 存储回调引用供 api 路由使用
    app.state.proactive_callback = proactive_callback
    app.state.proactive_scheduler = proactive_scheduler

    yield
    # --- 关闭 ---
    logger.info("EmoCare API 关闭中...")
    proactive_scheduler.stop_scheduler()


# 从环境变量读取允许的 CORS 来源（生产环境勿用 *）
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",")]

# 创建FastAPI应用
app = FastAPI(
    title="EmoCare API",
    description="""
# EmoCare - 情感陪护助手 API

基于 LangGraph 的多Agent情感陪伴系统

## 功能特性

- 🔍 **情绪感知**: 自动识别用户情绪状态和强度
- 🚨 **危机检测**: 识别危机信号并提供安全响应
- 💬 **共情对话**: 温暖、专业的情感支持对话
- 🛠️ **辅助工具**: 情绪追踪、主动关怀、天气查询、网络搜索

## 架构

```
用户消息 → 感知Agent → [路由] → 对话Agent → [可选]工具Agent → 响应
                         ↓
                    危机处理分支
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 速率限制
app.middleware("http")(_rate_limit_middleware)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from .api.routes import router as api_router
app.include_router(api_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "EmoCare API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )

