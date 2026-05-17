/**
 * 全局状态管理（Zustand）
 * 管理聊天会话和宠物情绪状态
 */

import { create } from 'zustand'
import type {
  Message,
  ChatSession,
  EmotionState,
  EmotionType,
  ConversationStrategy,
  PetSprite
} from '@shared/types'

// ─── 工具函数：情绪 → 精灵图 ──────────────────────────────────────────────────

export function emotionToSprite(
  emotion: EmotionType,
  strategy: ConversationStrategy
): PetSprite {
  if (strategy === 'crisis_immediate') return 'crisis'
  switch (emotion) {
    case 'joy':          return 'happy'
    case 'calm':         return 'idle'
    case 'sadness':      return 'sad'
    case 'anxiety':      return 'scared'
    case 'anger':        return 'angry'
    case 'loneliness':   return 'sad'
    case 'hopelessness': return 'crisis'
    default:             return 'idle'
  }
}

// ─── 聊天 Store ───────────────────────────────────────────────────────────────

interface ChatStore {
  session: ChatSession
  emotionState: EmotionState

  // Actions
  initSession: () => Promise<void>
  sendMessage: (text: string) => Promise<void>
  sendVoiceMessage: (audioBlob: Blob) => Promise<void>
  clearMessages: () => void
  updateEmotion: (state: Omit<EmotionState, 'updatedAt'>) => void
  addProactiveMessage: (text: string) => void
}

const defaultEmotionState: EmotionState = {
  emotion: 'neutral',
  intensity: 0.5,
  strategy: 'normal_chat',
  updatedAt: Date.now()
}

export const useChatStore = create<ChatStore>((set, get) => ({
  session: {
    sessionId: '',
    messages: [],
    isLoading: false
  },
  emotionState: defaultEmotionState,

  // ── 初始化会话 ───────────────────────────────────────────────────────────
  initSession: async () => {
    // 优先恢复上一次的 sessionId，刷新不丢对话
    const storedId = localStorage.getItem('emocare_session_id')
    if (storedId) {
      set((state) => ({
        session: {
          ...state.session,
          sessionId: storedId,
          messages: []
        }
      }))
      window.electronAPI.startProactivePolling(storedId)
      return
    }
    const result = await window.electronAPI.newSession()
    localStorage.setItem('emocare_session_id', result.sessionId)
    set((state) => ({
      session: {
        ...state.session,
        sessionId: result.sessionId,
        messages: []
      }
    }))
    window.electronAPI.startProactivePolling(result.sessionId)
  },

  // ── 发送消息 ─────────────────────────────────────────────────────────────
  sendMessage: async (text: string) => {
    const { session } = get()
    if (!session.sessionId || !text.trim()) return

    // 添加用户消息
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now()
    }

    set((state) => ({
      session: {
        ...state.session,
        messages: [...state.session.messages, userMessage],
        isLoading: true,
        error: undefined
      }
    }))

    try {
      // 通过 Electron IPC 调用（主进程发起 HTTP 请求，避免跨域）
      const result = await window.electronAPI.sendMessage(
        session.sessionId,
        text
      )

      if (result.success && result.data) {
        const assistantMessage: Message = {
          id: `ai_${Date.now()}`,
          role: 'assistant',
          content: result.data.response,
          timestamp: Date.now(),
          emotion: result.data.perception?.emotion?.emotion as EmotionType
        }

        set((state) => ({
          session: {
            ...state.session,
            messages: [...state.session.messages, assistantMessage],
            isLoading: false
          }
        }))
      } else {
        set((state) => ({
          session: {
            ...state.session,
            isLoading: false,
            error: result.error ?? '请求失败，请稍后重试'
          }
        }))
      }
    } catch (err: any) {
      set((state) => ({
        session: {
          ...state.session,
          isLoading: false,
          error: err.message ?? '网络错误'
        }
      }))
    }
  },

  // ── 发送语音消息 ───────────────────────────────────────────────────────────
  sendVoiceMessage: async (audioBlob: Blob) => {
    const { session } = get()
    if (!session.sessionId || session.isLoading) return

    // 将 Blob 转 base64
    const buffer = await audioBlob.arrayBuffer()
    const bytes = new Uint8Array(buffer)
    let binary = ''
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i])
    }
    const audioBase64 = btoa(binary)

    set((state) => ({
      session: {
        ...state.session,
        isLoading: true,
        error: undefined,
      }
    }))

    try {
      const result = await window.electronAPI.sendVoiceMessage(
        session.sessionId,
        audioBase64
      )

      if (result.success && result.data) {
        const assistantMessage: Message = {
          id: `ai_${Date.now()}`,
          role: 'assistant',
          content: result.data.response,
          timestamp: Date.now(),
          emotion: result.data.perception?.emotion?.emotion as EmotionType,
        }

        set((state) => ({
          session: {
            ...state.session,
            messages: [...state.session.messages, assistantMessage],
            isLoading: false,
          },
        }))
      } else {
        set((state) => ({
          session: {
            ...state.session,
            isLoading: false,
            error: result.error ?? '语音处理失败',
          },
        }))
      }
    } catch (err: any) {
      set((state) => ({
        session: {
          ...state.session,
          isLoading: false,
          error: err.message ?? '网络错误',
        },
      }))
    }
  },

  // ── 清空消息 ─────────────────────────────────────────────────────────────
  clearMessages: () => {
    const { session } = get()
    if (session.sessionId) {
      window.electronAPI.clearSession(session.sessionId).catch(() => {})
    }
    set((state) => ({
      session: {
        ...state.session,
        messages: [],
        error: undefined
      }
    }))
  },

  // ── 更新情绪状态（来自 IPC 广播） ────────────────────────────────────────
  updateEmotion: (emotionData) => {
    set({
      emotionState: {
        ...emotionData,
        updatedAt: Date.now()
      }
    })
  },

  // ── 添加主动关怀消息 ──────────────────────────────────────────────────────
  addProactiveMessage: (text: string) => {
    const proactiveMessage: Message = {
      id: `proactive_${Date.now()}`,
      role: 'assistant',
      content: text,
      timestamp: Date.now()
    }
    set((state) => ({
      session: {
        ...state.session,
        messages: [...state.session.messages, proactiveMessage]
      }
    }))
  }
}))
