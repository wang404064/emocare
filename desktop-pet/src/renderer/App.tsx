/**
 * App.tsx - 渲染进程根组件
 * 根据 URL hash 决定渲染宠物窗口还是聊天窗口
 * - #/chat  → ChatWindow（聊天气泡）
 * - 默认    → PetWindow（桌面宠物）
 */

import React from 'react'
import PetWindow from './components/PetWindow'
import ChatWindow from './components/ChatWindow'

const App: React.FC = () => {
  const hash = window.location.hash

  if (hash === '#chat' || hash === '#/chat') {
    return <ChatWindow />
  }

  return <PetWindow />
}

export default App
