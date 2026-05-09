/**
 * usePetSprite Hook
 * 根据情绪状态驱动宠物动画和外观变化
 * 动画通过 CSS 类名切换实现，也可替换为 Lottie/Spine
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useChatStore, emotionToSprite } from '../store/chatStore'
import type { PetSprite, PetAppearance } from '@shared/types'

const IDLE_TIMEOUT_MS = 3 * 60 * 1000  // 3 分钟无操作后切换为睡眠动画

export function usePetSprite(): PetAppearance & {
  triggerAnimation: (sprite: PetSprite, durationMs?: number) => void
} {
  const emotionState = useChatStore((s) => s.emotionState)
  const [currentSprite, setCurrentSprite] = useState<PetSprite>('idle')
  const [showParticles, setShowParticles] = useState(false)
  const [bubbleText, setBubbleText] = useState<string | undefined>()

  // 用 ref 跟踪可变值，避免闭包陈旧和 effect 重复注册
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const currentSpriteRef = useRef(currentSprite)
  currentSpriteRef.current = currentSprite

  // ── 情绪变化 → 切换精灵图 ─────────────────────────────────────────────
  useEffect(() => {
    const targetSprite = emotionToSprite(emotionState.emotion, emotionState.strategy)
    setCurrentSprite(targetSprite)

    setShowParticles(
      emotionState.intensity > 0.7 &&
      ['sad', 'scared', 'angry', 'crisis'].includes(targetSprite)
    )
  }, [emotionState])

  // ── 空闲检测 → 睡眠动画 ───────────────────────────────────────────────
  useEffect(() => {
    const resetIdle = () => {
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current)
        idleTimerRef.current = null
      }
      if (currentSpriteRef.current === 'sleeping') {
        setCurrentSprite('idle')
      }
      idleTimerRef.current = setTimeout(() => {
        setCurrentSprite('sleeping')
      }, IDLE_TIMEOUT_MS)
    }

    // 初始启动 timer
    idleTimerRef.current = setTimeout(() => {
      setCurrentSprite('sleeping')
    }, IDLE_TIMEOUT_MS)

    window.addEventListener('mousemove', resetIdle)
    window.addEventListener('keydown', resetIdle)

    return () => {
      window.removeEventListener('mousemove', resetIdle)
      window.removeEventListener('keydown', resetIdle)
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current)
        idleTimerRef.current = null
      }
    }
  }, [])  // 空依赖，只注册一次，通过 ref 读取最新值

  // ── 临时触发动画（用于点击/消息到来等） ──────────────────────────────
  const triggerAnimation = useCallback(
    (sprite: PetSprite, durationMs: number = 2000) => {
      setCurrentSprite(sprite)
      setTimeout(() => {
        const restore = emotionToSprite(emotionState.emotion, emotionState.strategy)
        setCurrentSprite(restore)
      }, durationMs)
    },
    [emotionState]
  )

  return {
    sprite: currentSprite,
    scale: 1 + emotionState.intensity * 0.15,
    showParticles,
    bubbleText,
    triggerAnimation
  }
}
