/**
 * Electron 主进程入口
 * 负责：
 * - 创建透明无边框置顶窗口（桌面宠物）
 * - 创建聊天气泡窗口
 * - 注册 IPC 通信处理器
 * - 系统托盘管理
 */

import {
  app,
  BrowserWindow,
  Tray,
  Menu,
  ipcMain,
  screen,
  nativeImage,
  shell
} from 'electron'
import path from 'path'
import { PetWindowManager } from './windowManager'
import { ApiClient } from './apiClient'
import { setupIpcHandlers } from './ipcHandlers'

let petWindow: BrowserWindow | null = null
let chatWindow: BrowserWindow | null = null
let tray: Tray | null = null
let windowMgr: PetWindowManager | null = null

const isDev = process.env.NODE_ENV === 'development'

// ─── 边缘吸附 ──────────────────────────────────────────────────────────────────
let snapDebounceTimer: ReturnType<typeof setTimeout> | null = null

function enableEdgeSnap(win: BrowserWindow): void {
  win.on('move', () => {
    if (snapDebounceTimer) clearTimeout(snapDebounceTimer)
    snapDebounceTimer = setTimeout(() => {
      if (windowMgr && win && !win.isDestroyed()) {
        windowMgr.snapPetToEdge()
      }
    }, 300)
  })
}

// ─── 宠物主窗口 ────────────────────────────────────────────────────────────────
function createPetWindow(): BrowserWindow {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize

  petWindow = new BrowserWindow({
    width: 200,
    height: 200,
    // 初始位置：右下角
    x: width - 220,
    y: height - 220,
    // 桌面宠物关键配置
    transparent: true,          // 透明背景
    frame: false,               // 无标题栏
    alwaysOnTop: true,          // 始终置顶
    resizable: false,
    skipTaskbar: true,          // 不出现在任务栏
    hasShadow: false,
    // 允许鼠标事件穿透背景透明区域
    // (通过 setIgnoreMouseEvents 动态控制)
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  // 开发模式加载 Vite dev server，生产模式加载构建产物
  if (isDev) {
    petWindow.loadURL('http://localhost:5173')
    // petWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    petWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }

  enableEdgeSnap(petWindow)

  petWindow.on('closed', () => { petWindow = null })

  return petWindow
}

// ─── 聊天气泡窗口 ──────────────────────────────────────────────────────────────
function createChatWindow(): BrowserWindow {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize

  chatWindow = new BrowserWindow({
    width: 360,
    height: 500,
    x: width - 590,
    y: height - 540,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: true,
    show: false,  // 默认隐藏，点击宠物后显示
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  if (isDev) {
    chatWindow.loadURL('http://localhost:5173/#/chat')
  } else {
    chatWindow.loadFile(path.join(__dirname, '../renderer/index.html'), {
      hash: 'chat'
    })
  }

  chatWindow.on('closed', () => { chatWindow = null })

  return chatWindow
}

// ─── 系统托盘 ──────────────────────────────────────────────────────────────────
function createTray(): Tray {
  // 托盘图标（使用简单占位图标，可替换为实际 ico 文件）
  const icon = nativeImage.createFromDataURL(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABXUlEQVRYR+2Xy0rDQBSGk8mkSdOLrRakoIIKLhTduvEB3PoAvoBv4MKFuHDhA/gELhSvuBIEQRBEsQiKFy9QqLZp0iZpkk4mGReKkHRy0kUX/tmdc/h/ZuacM2OMMdYXSikRQiAiokQJAVR5nuf5YRiWoigipZQyxhhj9lMul7MgCEJVVVXVdV01TVP1fd82TXPPsiy3JELIugkhpNI0LUzTNDYMQ62qKvPegzRNl+bfB0II9X0/Z4TjuEHXdXshhCCEkBBCCIQQ4nme5zmO4wghFELIhRAI4bquOwF2vV7feJ7X930fAM/zFhSPEAKO4xBjDMYY2LYN2OOHaTAzxu46cM/33N97/p4FBoNBEATBRqPRgFKKVqtVoF6vBxBCJEmSaJrmJABwHCeEYRiGYRiGEAQhCCGE8zwfRVFUVVX1BgAAAABJRU5ErkJggg=='
  )

  tray = new Tray(icon)
  const contextMenu = Menu.buildFromTemplate([
    {
      label: '显示/隐藏小暖',
      click: () => {
        if (petWindow) {
          if (petWindow.isVisible()) {
            petWindow.hide()
          } else {
            petWindow.show()
          }
        }
      }
    },
    {
      label: '打开聊天',
      click: () => {
        if (chatWindow) {
          chatWindow.show()
          chatWindow.focus()
        }
      }
    },
    { type: 'separator' },
    {
      label: '退出',
      click: () => {
        app.quit()
      }
    }
  ])

  tray.setToolTip('EmoCare 小暖')
  tray.setContextMenu(contextMenu)
  tray.on('double-click', () => {
    if (chatWindow) {
      chatWindow.isVisible() ? chatWindow.hide() : chatWindow.show()
    }
  })

  return tray
}

// ─── 应用生命周期 ──────────────────────────────────────────────────────────────
app.whenReady().then(() => {
  createPetWindow()
  createChatWindow()
  createTray()
  setupIpcHandlers(petWindow!, chatWindow!)
  windowMgr = new PetWindowManager(petWindow!, chatWindow!)
})

app.on('window-all-closed', () => {
  // macOS 保留行为，Windows/Linux 直接退出
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (!petWindow) createPetWindow()
})

// 导出以供测试
export { petWindow, chatWindow }
