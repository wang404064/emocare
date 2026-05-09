# EmoCare 桌面宠物前端

> 基于 **Electron + React + TypeScript** 的情感陪伴桌面宠物，是 EmoCare 后端的前端界面。

## 功能特性

| 特性 | 说明 |
|------|------|
| 透明无边框 | 宠物悬浮在桌面，不遮挡其他窗口 |
| 始终置顶 | 常驻桌面右下角，随时可交互 |
| 情绪驱动动画 | 后端检测情绪后，宠物实时切换表情和动作 |
| 毛玻璃聊天 | 点击宠物弹出深色毛玻璃聊天气泡 |
| 鼠标穿透 | 非悬停状态下鼠标可穿透宠物区域操作底层窗口 |
| 空闲睡眠 | 3分钟无操作自动切换睡眠动画 |
| 系统托盘 | 托盘图标支持显示/隐藏/退出 |
| 主动关怀 | 后端主动推送关怀消息时宠物挥手提示 |

## 项目结构

```
desktop-pet/
├── src/
│   ├── main/                    # Electron 主进程
│   │   ├── index.ts             # 入口，创建窗口和托盘
│   │   ├── preload.ts           # 安全桥接 IPC API
│   │   ├── ipcHandlers.ts       # 集中注册 IPC 处理器
│   │   ├── apiClient.ts         # 调用后端 HTTP API
│   │   └── windowManager.ts     # 窗口位置管理（待完善）
│   ├── renderer/                # React 渲染进程
│   │   ├── App.tsx              # 根组件（hash 路由）
│   │   ├── main.tsx             # ReactDOM 入口
│   │   ├── components/
│   │   │   ├── PetWindow.tsx    # 宠物主窗口
│   │   │   └── ChatWindow.tsx   # 聊天气泡窗口
│   │   ├── hooks/
│   │   │   └── usePetSprite.ts  # 情绪→动画状态 Hook
│   │   ├── store/
│   │   │   └── chatStore.ts     # Zustand 全局状态
│   │   ├── styles/
│   │   │   ├── pet.css          # 宠物窗口样式（情绪动画）
│   │   │   └── chat.css         # 聊天窗口样式（毛玻璃）
│   │   └── assets/
│   │       └── sprites/         # 放置精灵图/Lottie 文件
│   └── shared/
│       └── types.ts             # 前后端共享类型定义
├── index.html
├── vite.config.ts
├── tsconfig.json                # 渲染进程 TS 配置
├── tsconfig.main.json           # 主进程 TS 配置
└── package.json
```

## 快速开始

### 环境要求

- Node.js >= 18
- npm >= 9

### 安装依赖

```bash
cd desktop-pet
npm install
```

### 开发模式

```bash
# 启动 Vite dev server + tsc watch
npm run dev

# 另开终端启动 Electron（等 Vite 启动后）
npm run electron
```

### 打包

```bash
npm run package
```

输出在 `release/` 目录。

## 情绪 → 宠物表现对应表

| 情绪 | 策略 | 宠物状态 | 动画 | 粒子 |
|------|------|----------|------|------|
| neutral | normal_chat | idle 🐱 | 缓慢浮动 | 无 |
| joy | normal_chat | happy 😸 | 蹦跳 | 无 |
| sadness | empathy_first | sad 😿 | 轻微摇晃 | ✅ |
| anger | gentle_explore | angry 😾 | 快速抖动 | ✅ |
| fear | empathy_first | scared 🙀 | 颤抖 | ✅ |
| 任意 | empathy_first_gentle_probe | 原状态+💙徽标 | 原动画 | - |
| 任意 | crisis_immediate | crisis 🫂 | 脉冲发光 | - |
| 无操作3分钟 | - | sleeping 😴 | 缓慢漂浮 | 无 |

## 后续扩展计划

- [ ] 替换 Emoji 为 Lottie 动画文件（推荐 [LottieFiles](https://lottiefiles.com/)）
- [ ] 添加宠物皮肤切换（设置面板）
- [ ] 情绪历史趋势图（周/月视图）
- [ ] 支持自定义宠物名字
- [ ] macOS 支持（需调整窗口层级策略）
- [ ] 多语言支持

## 架构说明

### 为什么在主进程发 HTTP 请求？

渲染进程（BrowserWindow）受同源策略限制，直接调用 `localhost:8000` 会遇到 CORS 问题。
将 HTTP 请求放在主进程的 `apiClient.ts` 中，通过 IPC 传递数据，彻底规避 CORS。

### 鼠标穿透原理

```
默认状态: setIgnoreMouseEvents(true, { forward: true })
  → 鼠标事件穿透到底层窗口，但仍能追踪到 mousemove

鼠标进入宠物区域: setIgnoreMouseEvents(false)  
  → 恢复正常鼠标事件，可点击交互
```
