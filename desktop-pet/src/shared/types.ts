/**
 * 前后端共享类型定义
 */

/** 9种情绪类型（与后端BERT模型输出一致） */
export type EmotionType =
  | 'sadness'       // 悲伤
  | 'anxiety'       // 焦虑
  | 'anger'         // 愤怒
  | 'loneliness'    // 孤独
  | 'shame_guilt'   // 羞耻/内疚
  | 'hopelessness'  // 绝望
  | 'hope'          // 希望
  | 'calm'          // 平静
  | 'joy'           // 喜悦

export type ConversationStrategy =
  | 'normal_chat'
  | 'gentle_explore'
  | 'empathy_first'
  | 'empathy_first_gentle_probe'
  | 'crisis_immediate'

/** 宠物外观状态（由情绪驱动） */
export interface PetAppearance {
  /** 当前动画帧关键字 */
  sprite: PetSprite
  /** 缩放比例（情绪强度驱动） */
  scale: number
  /** 是否显示粒子效果 */
  showParticles: boolean
  /** 气泡文本（主动关怀或提示） */
  bubbleText?: string
}

export type PetSprite =
  | 'idle'       // 默认待机
  | 'happy'      // 开心
  | 'sad'        // 难过
  | 'angry'      // 生气
  | 'scared'     // 害怕
  | 'thinking'   // 思考/等待
  | 'waving'     // 挥手（欢迎）
  | 'sleeping'   // 睡眠（长时间无操作）
  | 'crisis'     // 担忧/危机关怀

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  emotion?: EmotionType
}

export interface ChatSession {
  sessionId: string
  messages: Message[]
  isLoading: boolean
  error?: string
}

export interface EmotionState {
  emotion: EmotionType
  intensity: number       // 0~1
  strategy: ConversationStrategy
  updatedAt: number       // timestamp
}
