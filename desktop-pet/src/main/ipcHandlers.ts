/**
 * IPC 处理器注册
 * 集中管理所有 ipcMain.handle 和 ipcMain.on 注册
 */

import { ipcMain, BrowserWindow, screen } from 'electron'
import { ApiClient } from './apiClient'

const apiClient = new ApiClient()

export function setupIpcHandlers(
  petWindow: BrowserWindow,
  chatWindow: BrowserWindow
): void {

  // ── 主动关怀轮询（每 30 秒检查后端是否有待投递消息） ────────────────────
  let proactivePollTimer: ReturnType<typeof setInterval> | null = null
  let currentSessionId: string | null = null

  ipcMain.handle('proactive:startPolling', (_event, sessionId: string) => {
    currentSessionId = sessionId
    if (proactivePollTimer) clearInterval(proactivePollTimer)
    proactivePollTimer = setInterval(async () => {
      if (!currentSessionId) return
      try {
        const result = await apiClient.getProactiveMessages(currentSessionId)
        for (const msg of result.messages) {
          chatWindow.webContents.send('pet:proactiveMessage', msg.message)
          petWindow.webContents.send('pet:proactiveMessage', msg.message)
        }
      } catch (_) { /* 静默失败，下轮重试 */ }
    }, 30_000)
  })

  ipcMain.handle('proactive:stopPolling', () => {
    currentSessionId = null
    if (proactivePollTimer) {
      clearInterval(proactivePollTimer)
      proactivePollTimer = null
    }
  })

  // ── 窗口控制 ──────────────────────────────────────────────────────────────

  ipcMain.handle('window:toggleChat', () => {
    if (chatWindow.isVisible()) {
      chatWindow.hide()
    } else {
      // 聊天窗口位置：宠物窗口左侧
      const petBounds = petWindow.getBounds()
      chatWindow.setPosition(
        petBounds.x - 370,
        petBounds.y - 300
      )
      chatWindow.show()
      chatWindow.focus()
    }
  })

  ipcMain.handle('window:hideChat', () => {
    chatWindow.hide()
  })

  ipcMain.handle('window:startDrag', (_event, { x, y }: { x: number; y: number }) => {
    // 透明窗口的自定义拖拽
    petWindow.on('will-move', () => {
      // 这里 Electron 会自动处理 -webkit-app-region: drag 的拖拽
      // 这个 handler 可用于额外的边界限制逻辑
    })
  })

  ipcMain.handle('window:resizePet', (_event, { width, height }: { width: number; height: number }) => {
    petWindow.setSize(width, height)
  })

  ipcMain.handle('window:setIgnoreMouseEvents', (
    _event,
    { ignore, forward }: { ignore: boolean; forward: boolean }
  ) => {
    petWindow.setIgnoreMouseEvents(ignore, { forward })
  })

  // ── 聊天 API ──────────────────────────────────────────────────────────────

  ipcMain.handle('chat:sendMessage', async (
    _event,
    { sessionId, message }: { sessionId: string; message: string }
  ) => {
    try {
      const result = await apiClient.chat(sessionId, message)

      // 将情绪状态广播给宠物窗口
      if (result.perception) {
        petWindow.webContents.send('pet:emotionUpdate', {
          emotion: result.perception.emotion?.emotion ?? 'neutral',
          intensity: result.perception.emotion?.intensity ?? 0.5,
          strategy: result.current_strategy ?? 'normal_chat'
        })
      }

      return { success: true, data: result }
    } catch (error: any) {
      return { success: false, error: error.message }
    }
  })

  ipcMain.handle('chat:sendVoiceMessage', async (
    _event,
    { sessionId, audioBase64 }: { sessionId: string; audioBase64: string }
  ) => {
    try {
      const result = await apiClient.sendAudio(sessionId, audioBase64)

      if (result.perception) {
        petWindow.webContents.send('pet:emotionUpdate', {
          emotion: result.perception.emotion?.emotion ?? 'neutral',
          intensity: result.perception.emotion?.intensity ?? 0.5,
          strategy: result.current_strategy ?? 'normal_chat'
        })
      }

      return { success: true, data: result }
    } catch (error: any) {
      return { success: false, error: error.message }
    }
  })

  ipcMain.handle('chat:newSession', () => {
    return { sessionId: `session_${Date.now()}` }
  })

  ipcMain.handle('chat:clearSession', async (
    _event,
    { sessionId }: { sessionId: string }
  ) => {
    try {
      await apiClient.clearSession(sessionId)
      return { success: true }
    } catch (error: any) {
      return { success: false, error: error.message }
    }
  })
}
