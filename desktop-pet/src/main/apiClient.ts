/**
 * API 客户端（主进程）
 * 负责与 EmoCare 后端通信，避免跨域问题（在主进程发起请求）
 */

import https from 'https'
import http from 'http'

// 后端 API 地址（可通过环境变量覆盖）
const BACKEND_URL = process.env.EMOCARE_API_URL ?? 'http://localhost:8080'

interface ChatResponse {
  response: string
  session_id: string
  perception?: {
    emotion?: {
      emotion: string
      intensity: number
    }
  }
  current_strategy?: string
}

export class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = BACKEND_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * 发送聊天消息
   */
  async chat(sessionId: string, message: string): Promise<ChatResponse> {
    return this.post<ChatResponse>('/api/v1/chat', {
      session_id: sessionId,
      message: message
    })
  }

  /**
   * 发送语音消息 (multipart/form-data)
   */
  async sendAudio(sessionId: string, audioBase64: string): Promise<ChatResponse> {
    // 将 base64 解码为 binary
    const binaryStr = atob(audioBase64)
    const bytes = new Uint8Array(binaryStr.length)
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i)
    }

    const boundary = '----EmoCare' + Date.now()
    const body = this._buildMultipart(boundary, bytes, sessionId)
    return this._postMultipart<ChatResponse>('/api/v1/chat/audio', body, boundary)
  }

  /**
   * 清除会话历史
   */
  async clearSession(sessionId: string): Promise<void> {
    await this.delete(`/api/v1/session/${sessionId}`)
  }

  /**
   * 获取待投递的主动关怀消息（轮询用）
   */
  async getProactiveMessages(sessionId: string): Promise<{ messages: Array<{ message: string; timestamp: string }> }> {
    return this.get<{ messages: Array<{ message: string; timestamp: string }> }>(`/api/v1/session/${sessionId}/proactive`)
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.get('/api/v1/health')
      return true
    } catch {
      return false
    }
  }

  // ── 内部方法 ─────────────────────────────────────────────────────────────

  private _buildMultipart(boundary: string, fileBytes: Uint8Array, sessionId: string): Uint8Array {
    const encoder = new TextEncoder()
    const parts: Uint8Array[] = []
    const crlf = encoder.encode('\r\n')

    // 音频文件
    parts.push(encoder.encode(`--${boundary}`))
    parts.push(crlf)
    parts.push(encoder.encode(
      'Content-Disposition: form-data; name="audio"; filename="recording.webm"'
    ))
    parts.push(crlf)
    parts.push(encoder.encode('Content-Type: audio/webm'))
    parts.push(crlf)
    parts.push(crlf)
    parts.push(fileBytes)
    parts.push(crlf)

    // session_id
    parts.push(encoder.encode(`--${boundary}`))
    parts.push(crlf)
    parts.push(encoder.encode(
      'Content-Disposition: form-data; name="session_id"'
    ))
    parts.push(crlf)
    parts.push(crlf)
    parts.push(encoder.encode(sessionId))
    parts.push(crlf)

    // user_id
    parts.push(encoder.encode(`--${boundary}`))
    parts.push(crlf)
    parts.push(encoder.encode(
      'Content-Disposition: form-data; name="user_id"'
    ))
    parts.push(crlf)
    parts.push(crlf)
    parts.push(encoder.encode('anonymous'))
    parts.push(crlf)

    // 结束
    parts.push(encoder.encode(`--${boundary}--`))
    parts.push(crlf)

    const totalLen = parts.reduce((s, p) => s + p.length, 0)
    const result = new Uint8Array(totalLen)
    let offset = 0
    for (const p of parts) {
      result.set(p, offset)
      offset += p.length
    }
    return result
  }

  private _postMultipart<T>(path: string, body: Uint8Array, boundary: string): Promise<T> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path)
      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': body.length,
        },
      }

      const lib = url.protocol === 'https:' ? https : http
      const req = lib.request(options, (res) => {
        let respBody = ''
        res.on('data', (chunk) => (respBody += chunk))
        res.on('end', () => {
          try {
            resolve(JSON.parse(respBody) as T)
          } catch (e) {
            reject(new Error(`JSON parse error: ${respBody}`))
          }
        })
      })

      req.on('error', reject)
      req.write(body)
      req.end()
    })
  }

  private post<T>(path: string, body: object): Promise<T> {
    return new Promise((resolve, reject) => {
      const data = JSON.stringify(body)
      const url = new URL(this.baseUrl + path)
      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data)
        }
      }

      const lib = url.protocol === 'https:' ? https : http
      const req = lib.request(options, (res) => {
        let body = ''
        res.on('data', (chunk) => (body += chunk))
        res.on('end', () => {
          try {
            resolve(JSON.parse(body) as T)
          } catch (e) {
            reject(new Error(`JSON parse error: ${body}`))
          }
        })
      })

      req.on('error', reject)
      req.write(data)
      req.end()
    })
  }

  private get<T>(path: string): Promise<T> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path)
      const lib = url.protocol === 'https:' ? https : http
      lib.get(url.toString(), (res) => {
        let body = ''
        res.on('data', (chunk) => (body += chunk))
        res.on('end', () => {
          try {
            resolve(JSON.parse(body) as T)
          } catch (e) {
            reject(new Error(`JSON parse error: ${body}`))
          }
        })
      }).on('error', reject)
    })
  }

  private delete<T>(path: string): Promise<T> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path)
      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname,
        method: 'DELETE'
      }
      const lib = url.protocol === 'https:' ? https : http
      const req = lib.request(options, (res) => {
        let body = ''
        res.on('data', (chunk) => (body += chunk))
        res.on('end', () => {
          try {
            resolve(JSON.parse(body) as T)
          } catch (e) {
            reject(new Error(`JSON parse error: ${body}`))
          }
        })
      })
      req.on('error', reject)
      req.end()
    })
  }
}
