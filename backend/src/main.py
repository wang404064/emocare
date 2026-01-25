"""
EmoCare API - 情感陪护助手
主入口文件
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/emocare_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG"
)

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
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("EmoCare API 启动中...")
    logger.info("文档地址: http://localhost:8080/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("EmoCare API 关闭中...")
    # 清理调度器
    from .tools import proactive_scheduler
    proactive_scheduler.stop_scheduler()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
