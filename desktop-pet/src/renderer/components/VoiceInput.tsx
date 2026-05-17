/**
 * VoiceInput — 语音输入组件
 * 按住录音，松开发送。带音量波形 + 状态指示。
 */
import React, { useState, useRef, useCallback, useEffect } from 'react'

type VoiceState = 'idle' | 'recording' | 'processing'

interface Props {
  disabled: boolean
  onAudioReady: (blob: Blob) => Promise<void>
}

const VoiceInput: React.FC<Props> = ({ disabled, onAudioReady }) => {
  const [state, setState] = useState<VoiceState>('idle')
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number>(0)

  // 波形 canvas
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // ── 清理资源 ──────────────────────────────────
  const cleanup = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = 0
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    mediaRecorderRef.current = null
    analyserRef.current = null
  }, [])

  useEffect(() => {
    return cleanup
  }, [cleanup])

  // ── 开始录音 ──────────────────────────────────
  const startRecording = useCallback(async () => {
    if (disabled || state !== 'idle') return
    setError(null)
    chunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      streamRef.current = stream

      // 音频分析器（波形）
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      analyser.minDecibels = -90
      analyser.maxDecibels = -10
      source.connect(analyser)
      analyserRef.current = analyser

      // 波形绘制
      const drawWave = () => {
        if (!analyserRef.current || !canvasRef.current) return
        const ctx = canvasRef.current.getContext('2d')
        if (!ctx) return

        const data = new Uint8Array(analyserRef.current.frequencyBinCount)
        analyserRef.current.getByteFrequencyData(data)

        const w = canvasRef.current.width
        const h = canvasRef.current.height
        ctx.clearRect(0, 0, w, h)

        const barCount = 20
        const barW = (w / barCount) * 0.7
        const gap = (w / barCount) * 0.3
        for (let i = 0; i < barCount; i++) {
          const idx = Math.floor((i / barCount) * data.length)
          const val = data[idx] / 255
          const barH = Math.max(3, val * h * 0.9)
          const x = i * (barW + gap)
          const y = h - barH

          // 渐变色：底部亮紫，顶部暗
          const gradient = ctx.createLinearGradient(0, y, 0, h)
          gradient.addColorStop(0, 'rgba(139, 92, 246, 0.9)')
          gradient.addColorStop(1, 'rgba(99, 102, 241, 0.3)')
          ctx.fillStyle = gradient

          ctx.beginPath()
          ctx.roundRect(x, y, barW, barH, [2, 2, 0, 0])
          ctx.fill()
        }
        animationRef.current = requestAnimationFrame(drawWave)
      }

      // MediaRecorder
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        drawWave && cancelAnimationFrame(animationRef.current)
        const blob = new Blob(chunksRef.current, { type: mimeType })
        chunksRef.current = []

        if (blob.size < 200) {
          setError('录音过短，请重试')
          setState('idle')
          cleanup()
          return
        }

        setState('processing')
        try {
          await onAudioReady(blob)
        } catch {
          setError('识别失败，请重试')
        } finally {
          setState('idle')
          cleanup()
        }
      }

      recorder.start()
      setState('recording')
      drawWave()

      // 最长录音 30 秒后自动停止
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          mediaRecorderRef.current.stop()
        }
      }, 30_000)

    } catch (err: any) {
      if (err.name === 'NotAllowedError') {
        setError('麦克风权限被拒绝，请在系统设置中允许')
      } else if (err.name === 'NotFoundError') {
        setError('未检测到麦克风设备')
      } else {
        setError(err.message || '录音启动失败')
      }
      setState('idle')
    }
  }, [disabled, state, cleanup, onAudioReady])

  // ── 停止录音 ──────────────────────────────────
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  // ── 渲染 ──────────────────────────────────────
  const isIdle = state === 'idle'
  const isRecording = state === 'recording'
  const isProcessing = state === 'processing'

  return (
    <div className="voice-input-wrapper">
      {/* 错误提示 */}
      {error && (
        <div className="voice-error">
          {error}
          <button className="voice-error-dismiss" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}

      {/* 波形 Canvas */}
      {isRecording && (
        <canvas ref={canvasRef} className="voice-wave" width={200} height={60} />
      )}

      {/* 处理中 */}
      {isProcessing && (
        <div className="voice-processing">
          <span className="voice-spinner" />
          <span>识别中...</span>
        </div>
      )}

      {/* 录音按钮 */}
      <button
        className={`voice-btn ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
        disabled={disabled || isProcessing}
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        onMouseLeave={stopRecording}
        onTouchStart={startRecording}
        onTouchEnd={stopRecording}
        title={isRecording ? '松开发送' : isProcessing ? '识别中...' : '按住说话'}
        aria-label="语音输入"
      >
        {isRecording ? '🎤' : isProcessing ? '⏳' : '🎙️'}
      </button>
    </div>
  )
}

export default VoiceInput
