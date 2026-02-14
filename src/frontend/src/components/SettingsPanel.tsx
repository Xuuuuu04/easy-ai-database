import React, { useState, useEffect } from 'react'
import { API_BASE } from '../config'

/**
 * 系统设置与说明面板。
 */
export const SettingsPanel: React.FC = () => {
  const [baseUrl, setBaseUrl] = useState(API_BASE)
  const [health, setHealth] = useState<{ status: string; mock_mode: boolean } | null>(null)
  const [docCount, setDocCount] = useState(0)
  const [chatCount, setChatCount] = useState(0)
  const [refreshing, setRefreshing] = useState(false)
  const [lastCheckedAt, setLastCheckedAt] = useState<string>('')

  const refreshOverview = async () => {
    setRefreshing(true)
    const [h, docs, chats] = await Promise.all([
      fetch(`${API_BASE}/health`)
        .then((r) => r.json())
        .catch(() => null),
      fetch(`${API_BASE}/kb/documents`)
        .then((r) => r.json())
        .catch(() => []),
      fetch(`${API_BASE}/chat/history`)
        .then((r) => r.json())
        .catch(() => []),
    ])

    setHealth(h)
    setDocCount(docs ? docs.length : 0)
    setChatCount(chats ? chats.length : 0)
    setLastCheckedAt(new Date().toLocaleTimeString())
    setRefreshing(false)
  }

  useEffect(() => {
    void refreshOverview()
  }, [])

  return (
    <div className="panel settings-panel">
      <header className="panel-hero">
        <div>
          <p className="eyebrow">系统设置</p>
          <h2>运行配置与本地安全</h2>
          <p className="hero-subtitle">所有数据仅保存在本地环境，配置集中在本机。</p>
        </div>
        <div className="hero-badges">
          <span className="hero-badge">本地存储</span>
          <span className="hero-badge">模型可控</span>
        </div>
      </header>

      <div className="settings-grid">
        <section className="setting-card">
          <h3>API Base URL</h3>
          <p className="hint">当前 API 地址通过环境变量配置。</p>
          <div className="input-with-hint">
            <input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              disabled
              className="readonly-input"
            />
            <p className="hint">
              目前 API 地址通过环境变量 <code>VITE_API_BASE</code> 配置。
              <br />
              后端模型配置 (LLM, Embeddings) 请修改后端 <code>.env</code> 文件。
            </p>
          </div>
        </section>

        <section className="setting-card">
          <div className="setting-headline">
            <h3>系统状态</h3>
            <button type="button" className="refresh-btn" onClick={() => void refreshOverview()} disabled={refreshing}>
              {refreshing ? '刷新中...' : '刷新'}
            </button>
          </div>
          {lastCheckedAt && <p className="hint">上次检查：{lastCheckedAt}</p>}
          <div className="about-grid">
            <div>
              <span className="about-label">后端连接</span>
              <div className="status-indicator status-with-gap">
                <div className={`status-dot ${health ? 'online' : 'offline'}`} />
                <span className="about-value">{health ? '在线' : '离线'}</span>
              </div>
            </div>
            <div>
              <span className="about-label">运行模式</span>
              <span className="about-value">{health?.mock_mode ? 'Mock' : 'Real'}</span>
            </div>
            <div>
              <span className="about-label">文档数量</span>
              <span className="about-value">{docCount}</span>
            </div>
            <div>
              <span className="about-label">对话数量</span>
              <span className="about-value">{chatCount}</span>
            </div>
          </div>
        </section>

        <section className="setting-card">
          <h3>关于</h3>
          <p>本机知识库 AI 助手 v0.1.0</p>
          <div className="about-grid">
            <div>
              <span className="about-label">运行方式</span>
              <span className="about-value">本地优先</span>
            </div>
            <div>
              <span className="about-label">数据路径</span>
              <span className="about-value">./data</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
