/**
 * 窗口管理器
 * 负责宠物窗口和聊天窗口的位置、尺寸管理
 * 确保窗口不超出屏幕边界
 */

import { BrowserWindow, screen } from 'electron'

export class PetWindowManager {
  private petWindow: BrowserWindow
  private chatWindow: BrowserWindow

  constructor(petWindow: BrowserWindow, chatWindow: BrowserWindow) {
    this.petWindow = petWindow
    this.chatWindow = chatWindow
  }

  /**
   * 将聊天窗口定位到宠物旁边（自动选择最优方向）
   */
  positionChatNearPet(): void {
    const { width: screenW, height: screenH } =
      screen.getPrimaryDisplay().workAreaSize
    const petBounds = this.petWindow.getBounds()
    const chatSize = this.chatWindow.getSize()

    let chatX = petBounds.x - chatSize[0] - 10  // 默认在宠物左侧
    let chatY = petBounds.y + petBounds.height - chatSize[1]  // 底部对齐

    // 左侧放不下，改到右侧
    if (chatX < 0) {
      chatX = petBounds.x + petBounds.width + 10
    }

    // 右侧也放不下，居中
    if (chatX + chatSize[0] > screenW) {
      chatX = Math.max(0, (screenW - chatSize[0]) / 2)
    }

    // 上方越界
    if (chatY < 0) {
      chatY = petBounds.y
    }

    // 下方越界
    if (chatY + chatSize[1] > screenH) {
      chatY = screenH - chatSize[1] - 10
    }

    this.chatWindow.setPosition(Math.round(chatX), Math.round(chatY))
  }

  /**
   * 将宠物窗口吸附到屏幕边缘（防止拖出屏幕）
   */
  snapPetToEdge(): void {
    const { width: screenW, height: screenH } =
      screen.getPrimaryDisplay().workAreaSize
    const petBounds = this.petWindow.getBounds()

    const snapThreshold = 40  // 距离边缘多少像素内触发吸附

    let { x, y } = petBounds

    if (x < snapThreshold) x = 0
    if (y < snapThreshold) y = 0
    if (x + petBounds.width > screenW - snapThreshold)
      x = screenW - petBounds.width
    if (y + petBounds.height > screenH - snapThreshold)
      y = screenH - petBounds.height

    this.petWindow.setPosition(x, y)
  }

  /**
   * 重置宠物到默认位置（右下角）
   */
  resetToDefaultPosition(): void {
    const { width: screenW, height: screenH } =
      screen.getPrimaryDisplay().workAreaSize
    const petSize = this.petWindow.getSize()

    this.petWindow.setPosition(
      screenW - petSize[0] - 20,
      screenH - petSize[1] - 20
    )
  }
}
