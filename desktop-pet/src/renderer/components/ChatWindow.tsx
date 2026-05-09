/**
 * ChatWindow 组件
 * 聊天气泡窗口 - 毛玻璃风格
 */

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useChatStore } from '../store/chatStore'
import type { Message } from '@shared/types'
import '../styles/chat.css'

const ChatWindow: React.FC = () => {
  const { session, sendMessage, clearMessages } = useChatStore()
  const [inputText, setInputText] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // 自动滚动到最新消息
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session.messages])

  // 聚焦输入框
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSend = useCallback(async () => {
    const text = inputText.trim()
    if (!text || session.isLoading) return
    setInputText('')
    await sendMessage(text)
  }, [inputText, session.isLoading, sendMessage])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  const handleClose = () => {
    window.electronAPI.hideChatWindow()
  }

  return (
    <div className="chat-container">
      {/* 标题栏（可拖拽） */}
      <div className="chat-header" style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}>
        <div className="chat-header-info">
          <span className="pet-avatar">🐱</span>
          <div>
            <div className="chat-name">小暖</div>
            <div className="chat-status">情感陪伴助手</div>
          </div>
        </div>
        <div className="chat-header-actions" style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
          <button className="icon-btn" onClick={clearMessages} title="清空对话">
            🗑️
          </button>
          <button className="icon-btn close-btn" onClick={handleClose} title="关闭">
            ✕
          </button>
        </div>
      </div>

      {/* 消息列表 */}
      <div className="messages-container">
        {session.messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">🌸</div>
            <div className="empty-text">嗨～我是小暖</div>
            <div className="empty-subtext">今天有什么想聊的吗？</div>
          </div>
        )}

        {session.messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* 加载中气泡 */}
        {session.isLoading && (
          <div className="message ai-message loading-message">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </div>
        )}

        {/* 错误提示 */}
        {session.error && (
          <div className="error-banner">
            ⚠️ {session.error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="input-area">
        <input
          ref={inputRef}
          className="message-input"
          type="text"
          placeholder="说点什么吧..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={session.isLoading}
          maxLength={500}
        />
        <button
          className="send-btn"
          onClick={handleSend}
          disabled={!inputText.trim() || session.isLoading}
          title="发送 (Enter)"
        >
          ↑
        </button>
      </div>
    </div>
  )
}

// ── 消息气泡子组件 ────────────────────────────────────────────────────────────

const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === 'user'
  const time = new Date(message.timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'ai-row'}`}>
      {!isUser && <div className="avatar ai-avatar">🐱</div>}
      <div className={`message ${isUser ? 'user-message' : 'ai-message'}`}>
        <div className="message-content">{message.content}</div>
        <div className="message-time">{time}</div>
      </div>
      {isUser && <div className="avatar user-avatar">😊</div>}
    </div>
  )
}

export default ChatWindow
