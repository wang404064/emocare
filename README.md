# EmoCare 智能情感陪护助手

> 基于 LLM 的多 Agent 情感支持对话系统。LangGraph 编排 + Qwen3-8B LoRA SFT+DPO 对齐 + Electron 桌面宠物。

---

## 快速启动

### 环境

| 端 | 要求 |
|----|------|
| 后端 | conda 环境 `pytorch-transformer`，Python 3.11，PyTorch 2.0+ |
| 前端 | Node.js 18+，npm |

**后端依赖安装：**

```bash
conda activate pytorch-transformer
cd backend
pip install -r requirements.txt
pip install langgraph-checkpoint-sqlite
```

**前端依赖安装：**

```bash
cd desktop-pet
npm install
# 国内镜像加速：
# set ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ && npm install
```

### 配置

EmoCare 支持两种 LLM 部署模式，取决于 `.env` 配置。代码零改动。

#### 模式 A：云端 API（默认，零部署成本）

使用阿里云百炼 DashScope 的 OpenAI 兼容接口：

```env
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的DashScope_API_Key
LLM_MODEL_NAME=qwen3-max
```

#### 模式 B：本地微调模型

**第一步 — 合并 LoRA 权重：**

训练产出的 LoRA adapter 不能独立运行，需先合并进基座模型：

```python
# merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3-8B-Instruct"
adapter = "./llamafactory/output/lora"    # LoRA 权重目录
output = "./emocare-8b-merged"

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(output)
tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.save_pretrained(output)
```

**第二步 — 启动推理服务：**

```bash
# 方案 1: vLLM（推荐，生产级）
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model ./emocare-8b-merged \
    --served-model-name emocare-8b \
    --port 8000 \
    --max-model-len 4096

# 方案 2: Ollama（最简单）
ollama create emocare -f Modelfile   # FROM ./emocare-8b-merged
ollama serve
```

**第三步 — 切换 `.env`：**

```env
# vLLM
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=EMPTY
LLM_MODEL_NAME=emocare-8b

# Ollama
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=EMPTY
LLM_MODEL_NAME=emocare:latest
```

> 后端使用 LangChain `ChatOpenAI`，天然兼容 OpenAI API 协议。vLLM 和 Ollama 都暴露此协议，切换模型只需改 `.env`，不动任何 Agent 代码。

**BERT 情绪识别模型**权重放在 `backend/models/emotion_risk_model_v1/`（路径已自动配置）。缺失时不阻塞启动，自动降级为默认值。

### 启动

**终端 1 — 后端（端口 8080）：**

```bash
conda activate pytorch-transformer
cd backend
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```

**终端 2 — 前端（Vite 5173 + Electron 桌面窗口）：**

```bash
cd desktop-pet
npx tsc -p tsconfig.main.json          # 仅首次需编译主进程
set NODE_ENV=development&& npx electron .   # 启动桌面宠物
```

或用 `npm run dev` 一键启动（等待 Vite 就绪后自动拉起 Electron）。

访问 http://localhost:8080/docs 查看 API 文档和在线调试。

---

## 功能总览

### 后端 — 多 Agent 对话系统

#### 1. 感知 Agent（Perception）

| 能力 | 实现 |
|------|------|
| 情绪识别 | BERT-base-chinese 双头模型，9 维情绪强度回归 + 3 级风险分类，在线程池异步执行 |
| 危机检测 | 三通道：12 个强信号关键词 + 10 个弱信号 + BERT 风险等级 |
| 误报过滤 | 4 条正则模式覆盖上百种口语组合（"想死你了""开心死了""穷死了"等不触发危机） |
| 场景分析 | LLM 识别 9 类场景（工作压力/人际关系/家庭/健康焦虑/孤独/自我怀疑/人生意义/日常闲聊/其他） |
| 危机二次判断 | **Crisis Judge**：强关键词命中后，独立 LLM 快速判断真实风险（high/low/uncertain），避免口语夸张误触危机流程 |
| 策略选择 | 5 级策略 + Judge 联动：`crisis_immediate` → `empathy_first_gentle_probe` → `empathy_first` → `gentle_explore` → `normal_chat` |

**9 种情绪：** sadness · anxiety · anger · loneliness · shame_guilt · hopelessness · hope · calm · joy

#### 2. 对话 Agent（Conversation）

| 能力 | 实现 |
|------|------|
| 共情生成 | Qwen3-8B（LoRA SFT+DPO 微调），温度 0.7，max_tokens 2048 |
| 策略执行 | 根据感知 Agent 的策略标签，动态切换对话风格（共情优先/温和探询/自然聊天等） |
| 去重后处理 | 自动检测并移除 LLM 输出中的逐句重复 |
| 工具判断 | 独立 LLM 调用判断是否需要触发工具，输出结构化 JSON |

#### 3. 危机处理 Agent（Crisis）

- **Safety-First 模板**：先共情（"谢谢你信任我"）→ 再倾听 → 最后给热线，不机械不冰冷
- **硬编码隔离**：危机状态下不使用 LLM 生成，消除幻觉风险
- **真实热线嵌入**：5 条心理援助热线（400-161-9995 等），2 种安全话术随机选择
- **自动阻断工具链**：危机模式下强制 `needs_tools=False`

#### 4. 危机安全体系

```
用户输入
    ↓
正则误报过滤（"笑死我了"→放行）
    ↓
关键词匹配?
  ├─ 强信号命中 → Crisis Judge (LLM二次判断)
  │     ├─ high      → 硬编码模板+热线（零生成风险）
  │     ├─ uncertain → LLM生成 + safety约束（先倾听，柔性提专业帮助）
  │     └─ low       → 正常对话（Judge认为无实际危险）
  ├─ BERT高风险/弱信号 → LLM生成 + safety约束
  └─ 正常 → 按情绪强度选策略
```

- **Judge API 失败时安全回退为 high**（宁可过度紧张，不可漏报）
- **`uncertain` 级别**：LLM 生成但不机械贴热线——只在对方表达无助时才柔性提专业帮助
- **误报过滤比 Judge 更早拦截**：省一次 LLM 调用

#### 5. 工具 Agent（Tool）

| 工具 | 功能 | 触发条件 |
|------|------|----------|
| 天气查询 | OpenWeatherMap API / 城市差异化 mock | 用户询问天气 |
| 网页搜索 | DuckDuckGo 双路（Instant Answer + HTML 回退） | 用户查找信息 |
| 情绪追踪 | JSON 持久化，强度 >0.7 自动记录，含 3 轮对话上下文 | 每次对话自动触发 |
| 定时提醒 | 中文时间解析（明天/X小时后/X分钟后），APScheduler 执行 | 用户说"提醒我" |
| 主动关怀 | 3 类模板（问候/鼓励/晚安），按用户定时调度，首次对话自动安排 30 分钟后回访 |

#### 6. 基础设施

| 特性 | 实现 |
|------|------|
| 会话持久化 | SqliteSaver（`data/checkpoints.db`），服务重启不丢对话 |
| 日志 | Loguru 双路输出（stdout + 按日轮转文件，保留 7 天） |
| 速率限制 | 滑动窗口 IP 限流中间件，60s/30 次，超 1000 条自动清理 |
| 延迟加载 | Agent 单例首次调用时初始化，import 不触发 heavy lifting |
| BERT 异步 | `run_in_executor` 线程池执行，不阻塞 asyncio 事件循环 |
| BERT 缺失容错 | 无 BERT 模型时自动降级为默认情绪值，服务正常运行 |

---

### 前端 — Electron 桌面宠物

#### 1. 宠物窗口（200×200，右下角）

| 交互 | 行为 |
|------|------|
| 显示 | 透明无边框，始终置顶，不出现在任务栏 |
| 拖拽 | 全窗口可拖拽移动（drag 区域与 click 区域分离），松手 300ms 后自动边缘吸附（40px 阈值） |
| 悬停 | 猫咪放大 1.1 倍 + 显示聊天按钮 |
| 点击 | 切换聊天气泡窗口（宠物左侧弹出，空间不足换右侧） |
| 非悬停 | 鼠标事件穿透（不阻挡桌面操作） |
| 系统托盘 | 右键菜单：显示/隐藏 · 打开聊天 · 退出 |

#### 2. 宠物动画（9 种情绪 × CSS 关键帧）

| 情绪 | 动画 | 效果 |
|------|------|------|
| idle 平静 | float | 3s 缓浮动 |
| happy 开心 | bounce | 0.6s 弹跳旋转 |
| sad 悲伤 | shake-gentle | 2s 轻颤 |
| angry 愤怒 | shake-fast | 0.3s 快速抖动 |
| scared 焦虑 | tremble | 0.15s 高频微颤 |
| thinking 思考 | tilt | 1.5s 歪头 |
| waving 挥手 | wave | 0.8s 挥手 ×3 |
| sleeping 睡眠 | slow-float | 4s 缓浮 + 透明度降低 |
| crisis 危机 | pulse-glow | 1s 脉动缩放 + 红色光晕 |

**空闲检测**：3 分钟无鼠标/键盘操作自动进入睡眠动画。

**粒子效果**：高强度负面情绪时显示 6 色粒子上升。

**情绪色板**：每种情绪对应独立主题色，宠物和阴影实时切换。

#### 3. 聊天窗口（360×500，毛玻璃深色主题）

| 特性 | 实现 |
|------|------|
| 风格 | 深蓝底色 rgba(15,23,42,0.92) + 24px 高斯模糊 + 1px 白色边框 |
| 消息气泡 | 用户（绿色圆角右对齐）· 助手（紫色圆角左对齐） |
| 加载指示 | 3 点脉动动画 |
| 空状态 | 欢迎语 + 旋转樱花 |
| 输入框 | 圆角搜索框样式，聚焦时边框高亮，Enter 发送，最大 500 字 |
| 其他 | 消息淡入动画 · 滚动条美化 · 时间戳 · 清空对话按钮 |

#### 4. 主动关怀推送

| 机制 | 说明 |
|------|------|
| 触发 | 首次对话后自动安排 30 分钟后回访 |
| 投递 | WebSocket 实时推送 + HTTP 轮询（30s）双路兜底 |
| 通知 | 宠物触发 `waving` 动画 3s + 聊天窗口显示关怀消息 |

#### 5. 会话管理

- sessionId 持久化到 localStorage，刷新页面不丢失对话
- 清空对话同步调用后端 `DELETE /session/{id}`
- 多窗口状态共享（zustand）

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Electron 桌面端                        │
│  ┌──────────────┐     IPC      ┌──────────────────────┐ │
│  │  PetWindow   │◄────────────►│     main process     │ │
│  │  (猫咪动画)   │              │  HTTP POST / WS 轮询  │ │
│  └──────┬───────┘              └──────────┬───────────┘ │
│         │ 点击                            │              │
│  ┌──────┴───────┐                         │              │
│  │  ChatWindow  │◄── IPC ────────────────┘              │
│  │  (聊天气泡)   │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ HTTP :8080
┌─────────────────────────────────────────────────────────┐
│                    FastAPI 后端                           │
│                                                         │
│  POST /api/v1/chat                                      │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────┐                                        │
│  │ perception  │ BERT情绪 + 关键词 + 正则误报过滤        │
│  └──────┬──────┘                                        │
│         │ 强关键词命中?                                  │
│         ▼ yes                                           │
│  ┌─────────────┐                                        │
│  │ CrisisJudge │ LLM 二次判断 (high/low/uncertain)      │
│  └──┬───┬───┬──┘                                        │
│     │   │   │                                           │
│ high│   │uncertain/low                                  │
│     ▼   │                                               │
│  ┌──────┐│    ┌──────────────┐    ┌────────────┐       │
│  │crisis││───►│ conversation │───►│  finalize  │       │
│  │硬编码│     │  Qwen3-8B    │    │            │       │
│  └──────┘     └──────┬───────┘    └────────────┘       │
│                      │ 需工具                           │
│                      ▼                                  │
│               ┌──────────────┐                          │
│               │  tool_agent  │                          │
│               │ 5 工具调度   │                          │
│               └──────┬───────┘                          │
│                      ▼                                  │
│               conversation（含工具结果）                │
│                                                         │
│  GET  /api/v1/session/{id}/history    对话历史          │
│  GET  /api/v1/session/{id}/proactive  主动消息轮询       │
│  DEL  /api/v1/session/{id}            清除会话          │
│  WS   /api/v1/ws/{id}                实时对话 + 心跳     │
│  GET  /api/v1/health                 健康检查           │
└─────────────────────────────────────────────────────────┘
```

---

## 数据与训练

| 数据集 | 条数 | 说明 |
|--------|------|------|
| SFT | 3,800 | REBT 理情行为疗法风格对话 |
| DPO | 2,769 | 42 场景 × 18 策略矩阵，4 类困难负样本，5 种错误类型 |

```bash
# 上传服务器 → 训练
tar -czvf emocare.tar.gz llamafactory/
bash llamafactory/setup.sh
bash llamafactory/train.sh
```

从训练到部署的完整流水线：

```
Qwen3-8B
      │
      ├── SFT LoRA (r=64) ──→ adapter_sft/
      │
      ├── DPO LoRA (beta=0.1) ──→ adapter_dpo/ (最终 adapter)
      │
      ├── merge_lora.py ──→ emocare-8b-merged/ (完整模型)
      │
      ├── vLLM / Ollama ──→ localhost:8000 (OpenAI 兼容)
      │
      └── .env 切换 ──→ Agent 无缝调用本地模型
```

详见 `llamafactory/DEPLOY.md`

---

## 技术栈

| 层 | 技术 |
|----|------|
| 基座模型 | Qwen3-8B-Instruct |
| 微调 | LlamaFactory (LoRA r=64, SFT + DPO, beta=0.1) |
| Agent 编排 | LangGraph StateGraph + SqliteSaver |
| 情绪识别 | BERT-base-chinese 双头模型（9 维回归 + 3 级分类） |
| 推理服务 | FastAPI + LangChain ChatOpenAI |
| 前端框架 | Electron + React + TypeScript + Zustand |
| 任务调度 | APScheduler AsyncIOScheduler |
| 评测体系 | Win Rate · Reward Margin · PPL · 安全性 · LLM-as-Judge 5 维 |

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | 发送消息，返回情绪分析 + 场景 + 回复 |
| `GET` | `/api/v1/session/{id}/history` | 获取会话历史（含 timestamp） |
| `GET` | `/api/v1/session/{id}/proactive` | 获取待投递的主动关怀消息（轮询用） |
| `DELETE` | `/api/v1/session/{id}` | 清除会话（本地 + 后端同步） |
| `WS` | `/api/v1/ws/{id}` | WebSocket 实时对话（30s 心跳，100 连接上限） |
| `GET` | `/api/v1/health` | 健康检查 |

---

## 项目结构

```
emocare/
├── backend/
│   ├── src/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── core/                # 配置 / 状态定义
│   │   ├── agents/              # 4 个 Agent（感知/对话/危机/工具）
│   │   ├── graph/               # LangGraph 图构建 / 路由 / Agent 封装
│   │   ├── models/              # BERT 模型定义 + 推理封装
│   │   ├── tools/               # 5 个工具（天气/搜索/情绪追踪/提醒/关怀）
│   │   └── api/                 # REST 路由 / WebSocket / Schemas
│   ├── models/emotion_risk_model_v1/   # BERT 权重
│   ├── data/checkpoints.db             # SQLite 对话持久化
│   ├── dataset/sft/ + dpo/             # 训练数据
│   ├── .env / requirements.txt
│   └── pyproject.toml
│
├── desktop-pet/
│   ├── src/
│   │   ├── main/                # Electron 主进程
│   │   ├── renderer/            # React 渲染进程
│   │   │   ├── components/      # PetWindow / ChatWindow
│   │   │   ├── store/           # Zustand 状态
│   │   │   ├── hooks/           # usePetSprite
│   │   │   └── styles/          # pet.css + chat.css
│   │   └── shared/              # 共享类型
│   └── package.json
│
├── llamafactory/                # 训练配置 + 数据 + 评测脚本
└── README.md
```
