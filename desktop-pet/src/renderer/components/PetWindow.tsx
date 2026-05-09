/**
 * PetWindow 组件
 * 宠物主窗口渲染逻辑
 * - 透明背景，只显示宠物形象
 * - 支持拖拽（-webkit-app-region: drag）
 * - 悬停时显示互动按钮，非悬停时鼠标穿透
 * - 点击打开/关闭聊天气泡窗口
 */

import React, { useCallback, useEffect, useState } from 'react'
import { usePetSprite } from '../hooks/usePetSprite'
import { useChatStore } from '../store/chatStore'
import '../styles/pet.css'

/** 宠物精灵图的 CSS 类名映射 */
const SPRITE_CSS_MAP: Record<string, string> = {
  idle:     'sprite-idle',
  happy:    'sprite-happy',
  sad:      'sprite-sad',
  angry:    'sprite-angry',
  scared:   'sprite-scared',
  thinking: 'sprite-thinking',
  waving:   'sprite-waving',
  sleeping: 'sprite-sleeping',
  crisis:   'sprite-crisis'
}

const PetWindow: React.FC = () => {
  const { sprite, scale, showParticles, triggerAnimation } = usePetSprite()
  const { initSession, emotionState } = useChatStore()
  const [isHovered, setIsHovered] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)

  // 初始化聊天会话
  useEffect(() => {
    initSession()
  }, [])

  // 监听来自主进程的情绪更新（后端广播）
  useEffect(() => {
    window.electronAPI.onEmotionUpdate((data) => {
      useChatStore.getState().updateEmotion(data)
    })
    window.electronAPI.onProactiveMessage((msg) => {
      useChatStore.getState().addProactiveMessage(msg)
      triggerAnimation('waving', 3000)
    })
    return () => {
      window.electronAPI.offEmotionUpdate()
      window.electronAPI.offProactiveMessage?.()
    }
  }, [triggerAnimation])

  // 悬停时禁用鼠标穿透，离开时恢复穿透
  const handleMouseEnter = useCallback(() => {
    setIsHovered(true)
    window.electronAPI.setIgnoreMouseEvents(false)
  }, [])

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false)
    window.electronAPI.setIgnoreMouseEvents(true, true)
  }, [])

  // 点击宠物 → 打开/关闭聊天
  const handlePetClick = useCallback(() => {
    triggerAnimation('waving', 1500)
    window.electronAPI.toggleChatWindow()
    setChatOpen((prev) => !prev)
  }, [triggerAnimation])

  const spriteClass = SPRITE_CSS_MAP[sprite] ?? 'sprite-idle'
  const emotionColor = getEmotionAccentColor(emotionState.emotion)

  return (
    <div
      className="pet-container"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
    >
      {/* 宠物形象 — 拖拽由外层容器接管，这里是点击 + 动画区域 */}
      <div
        className={`pet-sprite ${spriteClass} ${isHovered ? 'hovered' : ''}`}
        style={{
          transform: `scale(${scale})`,
          WebkitAppRegion: 'no-drag',
          filter: emotionState.strategy === 'crisis_immediate'
            ? 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.8))'
            : `drop-shadow(0 4px 12px ${emotionColor}55)`
        } as React.CSSProperties}
        onClick={handlePetClick}
        title={`小暖 - ${sprite}`}
      >
        <PetEmoji sprite={sprite} />
      </div>

      {/* 粒子效果 */}
      {showParticles && (
        <div className="particles">
          {Array.from({ length: 6 }).map((_, i) => (
            <span key={i} className={`particle particle-${i}`} />
          ))}
        </div>
      )}

      {/* 悬停菜单 */}
      {isHovered && (
        <div className="pet-menu" style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
          <button
            className="pet-menu-btn"
            onClick={handlePetClick}
            title={chatOpen ? '关闭聊天' : '开始聊天'}
          >
            {chatOpen ? '✕' : '💬'}
          </button>
        </div>
      )}

      {/* 弱危机信号提示徽标 */}
      {emotionState.strategy === 'empathy_first_gentle_probe' && (
        <div className="care-badge" title="小暖正在关心你">💙</div>
      )}
    </div>
  )
}

// ── 占位宠物形象（基于情绪的 Emoji，可替换为真实精灵图）────────────────────

const SPRITE_EMOJI: Record<string, string> = {
  idle:     '🐱',
  happy:    '😸',
  sad:      '😿',
  angry:    '😾',
  scared:   '🙀',
  thinking: '🤔',
  waving:   '👋',
  sleeping: '😴',
  crisis:   '🫂'
}

const PetEmoji: React.FC<{ sprite: string }> = ({ sprite }) => (
  <span className="pet-emoji" role="img" aria-label={sprite}>
    {SPRITE_EMOJI[sprite] ?? '🐱'}
  </span>
)

// ── 工具函数：情绪 → 主题色 ────────────────────────────────────────────────

function getEmotionAccentColor(emotion: string): string {
  const colors: Record<string, string> = {
    joy:           '#f59e0b',
    hope:          '#fbbf24',
    calm:          '#22d3ee',
    sadness:       '#6366f1',
    anxiety:       '#a78bfa',
    anger:         '#ef4444',
    loneliness:    '#8b5cf6',
    shame_guilt:   '#f97316',
    hopelessness:  '#991b1b'
  }
  return colors[emotion] ?? '#94a3b8'
}

export default PetWindow
