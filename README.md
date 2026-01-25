# EmoCare - 情感陪护助手

基于 LangGraph 的多Agent中文情感支持对话系统

## 🎯 项目简介

EmoCare 是一个专业的情感支持AI助手，通过多Agent协作架构，提供温暖、专业、非评判的情感陪伴服务。系统集成了情绪识别、危机检测、共情对话和辅助工具等功能。

## ✨ 核心特性

### 🔍 智能感知
- **情绪识别**：9分类情绪识别
- **情绪强度预测**：连续值情绪强度建模（0-1）
- **场景识别**：自动识别用户面临的生活场景（工作压力、人际关系、家庭问题等）

### 🚨 安全机制
- **危机检测**：实时识别自杀、自伤等危机信号
- **安全响应**：提供专业危机干预和转介建议
- **边界保护**：拒绝有害请求，维护对话安全边界

### 💬 共情对话
- **多Agent协作**：感知Agent → 对话Agent → 工具Agent 的智能路由
- **上下文理解**：支持多轮对话，保持对话连贯性
- **策略匹配**：基于情绪和场景动态选择支持策略

### 🛠️ 辅助工具
- **天气查询**：查询指定城市的天气信息
- **互联网搜索**：在互联网上搜索实时信息
- **情绪追踪**：记录和分析用户情绪变化
- **提醒功能**：设置提醒和定时任务
- **主动关怀**：定时关怀提醒和情绪检查

## 🏗️ 系统架构

```
用户消息
    ↓
感知Agent (Perception)
    ├── 情绪识别 (BERT模型)
    ├── 危机检测 (关键词+风险等级)
    └── 场景识别 (LLM分类)
    ↓
[条件路由]
    ├── 危机信号 → 危机处理Agent → 响应
    └── 正常对话 → 对话Agent
                        ↓
                    [条件路由]
                        ├── 需要工具 → 工具Agent → 响应
                        └── 直接响应
```

### Agent 组件

1. **PerceptionAgent (感知Agent)**
   - 情绪识别：使用BERT模型进行8分类+强度预测
   - 危机检测：关键词匹配+风险等级评估
   - 场景识别：LLM辅助场景分类

2. **CrisisAgent (危机处理Agent)**
   - 危机信号识别
   - 安全响应生成
   - 专业资源转介

3. **ConversationAgent (对话Agent)**
   - 基于LLM的共情对话生成
   - 上下文理解和管理
   - 策略选择和应用

4. **ToolAgent (工具Agent)**
   - 天气查询服务
   - 互联网搜索功能
   - 情绪追踪记录
   - 提醒和主动关怀调度

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- CUDA (可选，用于情绪识别模型)

### 安装步骤

1. **克隆项目**
```bash
cd E:\project\agent\emo\backend
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**

创建 `.env` 文件：
```env
# LLM API配置
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=your-api-key
LLM_MODEL_NAME=

# 情绪识别模型路径
EMOTION_MODEL_PATH=your_emotion_recognizer_model_path
EMOTION_MODEL_DEVICE=cuda  # 或 cpu
```

4. **启动服务**
```bash
python -m src.main
```

服务将在 `http://localhost:8080` 启动

### API 文档

启动服务后，访问：
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## 📡 API 接口

### POST `/api/v1/chat`
发送消息给EmoCare助手

**请求体**:
```json
{
  "message": "我最近工作压力很大，感觉很焦虑",
  "user_id": "user123",
  "session_id": "session456"
}
```

**响应**:
```json
{
  "response": "我理解你现在的感受...",
  "emotion": {
    "emotion": "fear",
    "intensity": 0.75,
    "confidence": 0.89
  },
  "scene": "work_stress",
  "is_crisis": false,
  "session_id": "session456"
}
```

### GET `/api/v1/session/{session_id}/history`
获取会话历史

### DELETE `/api/v1/session/{session_id}`
清除会话历史

### GET `/api/v1/health`
健康检查

## 🛠️ 技术栈

- **框架**: LangGraph, LangChain, FastAPI
- **LLM**: Qwen3-8B (通过API调用)
- **情绪识别**: BERT-based模型 (RoBERTa)
- **状态管理**: LangGraph MemorySaver
- **API**: FastAPI + WebSocket
- **日志**: Loguru

## 📁 项目结构

```
emo/
├── backend/
│   ├── src/
│   │   ├── agents/          # Agent实现
│   │   │   ├── perception.py    # 感知Agent
│   │   │   ├── crisis.py        # 危机处理Agent
│   │   │   ├── conversation.py  # 对话Agent
│   │   │   └── tools.py         # 工具Agent
│   │   ├── api/            # API路由
│   │   │   ├── routes.py
│   │   │   ├── schemas.py
│   │   │   └── websocket.py
│   │   ├── core/            # 核心配置
│   │   │   ├── config.py
│   │   │   └── state.py
│   │   ├── graph/           # LangGraph构建
│   │   │   ├── builder.py
│   │   │   ├── agent.py
│   │   │   └── routes.py
│   │   ├── models/          # 模型封装
│   │   │   └── emotion_recognizer.py
│   │   ├── tools/           # 辅助工具
│   │   │   ├── breathing.py
│   │   │   ├── emotion_tracker.py
│   │   │   └── scheduler.py
│   │   └── main.py          # 入口文件
│   ├── data/               # 数据目录
│   ├── dataset/            # 训练数据集
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── langgraph.json
└── README.md
```

## 🔧 开发指南

### 添加新的Agent

1. 在 `src/agents/` 创建新的Agent文件
2. 实现Agent函数，接收 `AgentState` 并返回更新后的状态
3. 在 `src/graph/builder.py` 中注册节点
4. 添加路由逻辑（如需要）

### 添加新的工具

1. 在 `src/tools/` 创建工具文件
2. 实现工具类，继承 `BaseTool` 或实现标准接口
3. 在 `ToolAgent` 中注册工具

## 📊 项目亮点

### 技术亮点
- ✅ 多Agent协作架构，职责清晰
- ✅ 实时情绪识别和强度预测
- ✅ 危机检测和安全响应机制
- ✅ 上下文感知的多轮对话
- ✅ 灵活的Agent路由策略

### 工程亮点
- ✅ 基于LangGraph的状态管理
- ✅ RESTful API + WebSocket支持
- ✅ 完整的日志系统
- ✅ 健康检查和监控

## 🔒 安全与伦理

- **隐私保护**：用户数据本地存储，不泄露
- **危机干预**：自动识别并提供专业资源
- **边界维护**：拒绝有害请求，保护用户安全
- **专业转介**：超出能力范围时建议寻求专业帮助

## 📝 许可证

本项目仅供学习和研究使用。

---

**注意**：本项目是一个情感支持工具，不能替代专业心理咨询。如遇严重心理问题，请及时寻求专业帮助。
