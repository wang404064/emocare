/**
 * Preload 脚本
 * 在渲染进程和主进程之间安全地桥接 API
 * contextIsolation=true 时必须使用 contextBridge
 */

import { contextBridge, ipcRenderer } from 'electron'

// 暴露给渲染进程的安全 API
contextBridge.exposeInMainWorld('electronAPI', {
  // ── 窗口控制 ──────────────────────────────────────────
  /** 切换聊天窗口的显示/隐藏 */
  toggleChatWindow: () => ipcRenderer.invoke('window:toggleChat'),
  /** 隐藏聊天窗口 */
  hideChatWindow: () => ipcRenderer.invoke('window:hideChat'),
  /** 宠物窗口拖拽（透明窗口无法通过系统拖拽，需要 IPC） */
  startDrag: (x: number, y: number) =>
    ipcRenderer.invoke('window:startDrag', { x, y }),
  /** 更新宠物形象大小（情绪变化时动态缩放） */
  resizePet: (width: number, height: number) =>
    ipcRenderer.invoke('window:resizePet', { width, height }),

  // ── 聊天 API ──────────────────────────────────────────
  /** 发送消息到后端 */
  sendMessage: (sessionId: string, message: string) =>
    ipcRenderer.invoke('chat:sendMessage', { sessionId, message }),
  /** 开始新会话 */
  newSession: () => ipcRenderer.invoke('chat:newSession'),
  /** 清除会话历史 */
  clearSession: (sessionId: string) =>
    ipcRenderer.invoke('chat:clearSession', { sessionId }),
  /** 开始轮询主动关怀消息 */
  startProactivePolling: (sessionId: string) =>
    ipcRenderer.invoke('proactive:startPolling', sessionId),
  /** 停止主动关怀轮询 */
  stopProactivePolling: () =>
    ipcRenderer.invoke('proactive:stopPolling'),

  // ── 宠物状态同步 ──────────────────────────────────────
  /** 监听来自主进程的情绪状态更新（后端推送） */
  onEmotionUpdate: (callback: (emotion: EmotionState) => void) => {
    ipcRenderer.on('pet:emotionUpdate', (_event, data) => callback(data))
  },
  /** 移除情绪监听 */
  offEmotionUpdate: () => {
    ipcRenderer.removeAllListeners('pet:emotionUpdate')
  },
  /** 监听主动关怀消息 */
  onProactiveMessage: (callback: (message: string) => void) => {
    ipcRenderer.on('pet:proactiveMessage', (_event, msg) => callback(msg))
  },
  /** 移除主动关怀消息监听 */
  offProactiveMessage: () => {
    ipcRenderer.removeAllListeners('pet:proactiveMessage')
  },

  // ── 系统 ──────────────────────────────────────────────
  /** 在鼠标悬停/离开时控制是否穿透 */
  setIgnoreMouseEvents: (ignore: boolean, forward: boolean = true) =>
    ipcRenderer.invoke('window:setIgnoreMouseEvents', { ignore, forward }),
  /** 平台信息 */
  platform: process.platform
})

// 类型声明（供 TypeScript 渲染进程使用）
interface EmotionState {
  emotion: string
  intensity: number
  strategy: string
}

// 为全局 window 添加类型（在 renderer 中使用）
declare global {
  interface Window {
    electronAPI: typeof import('./preload')
  }
}
