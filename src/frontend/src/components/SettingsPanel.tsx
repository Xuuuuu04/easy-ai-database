import React, { useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../config'

type HealthData = { status: string; mock_mode: boolean }

type SettingsPayload = {
  env_file: string
  variables: Record<string, string>
  mcp_tools_enabled: boolean
  deployment_url: string
  mcp_endpoint: string
  mcp_commands: {
    claude_code: string
    codex: string
  }
}

export const SettingsPanel: React.FC = () => {
  const [baseUrl, setBaseUrl] = useState(API_BASE)
  const [health, setHealth] = useState<HealthData | null>(null)
  const [docCount, setDocCount] = useState(0)
  const [chatCount, setChatCount] = useState(0)
  const [refreshing, setRefreshing] = useState(false)
  const [lastCheckedAt, setLastCheckedAt] = useState('')

  const [settingsLoading, setSettingsLoading] = useState(true)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsError, setSettingsError] = useState('')
  const [settingsNotice, setSettingsNotice] = useState('')
  const [settingsPayload, setSettingsPayload] = useState<SettingsPayload | null>(null)
  const [variables, setVariables] = useState<Record<string, string>>({})
  const [newKey, setNewKey] = useState('')
  const [newValue, setNewValue] = useState('')

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
    setDocCount(Array.isArray(docs) ? docs.length : 0)
    setChatCount(Array.isArray(chats) ? chats.length : 0)
    setLastCheckedAt(new Date().toLocaleTimeString())
    setRefreshing(false)
  }

  const refreshSettings = async () => {
    setSettingsLoading(true)
    setSettingsError('')
    try {
      const res = await fetch(`${API_BASE}/settings/env`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = (await res.json()) as SettingsPayload
      setSettingsPayload(data)
      setVariables(data.variables || {})
    } catch (error) {
      setSettingsError(`读取设置失败：${String(error)}`)
    } finally {
      setSettingsLoading(false)
    }
  }

  const sortedEntries = useMemo(() => {
    return Object.entries(variables).sort(([a], [b]) => a.localeCompare(b))
  }, [variables])

  useEffect(() => {
    void refreshOverview()
    void refreshSettings()
  }, [])

  const setVar = (key: string, value: string) => {
    setVariables((prev) => ({ ...prev, [key]: value }))
    setSettingsNotice('')
  }

  const removeVar = (key: string) => {
    setVariables((prev) => {
      const next = { ...prev }
      delete next[key]
      return next
    })
    setSettingsNotice('')
  }

  const addVar = () => {
    const key = newKey.trim()
    if (!key) {
      return
    }
    setVariables((prev) => ({ ...prev, [key]: newValue }))
    setNewKey('')
    setNewValue('')
    setSettingsNotice('')
  }

  const saveSettings = async () => {
    setSettingsSaving(true)
    setSettingsError('')
    setSettingsNotice('')

    try {
      const res = await fetch(`${API_BASE}/settings/env`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variables }),
      })
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}))
        throw new Error(payload.detail || `HTTP ${res.status}`)
      }

      const data = (await res.json()) as SettingsPayload
      setSettingsPayload(data)
      setVariables(data.variables || {})
      setSettingsNotice('配置已保存，后端已重新加载设置。')
      await refreshOverview()
    } catch (error) {
      setSettingsError(`保存失败：${String(error)}`)
    } finally {
      setSettingsSaving(false)
    }
  }

  const copyCommand = async (command: string) => {
    if (!command) {
      return
    }
    try {
      await navigator.clipboard.writeText(command)
      setSettingsNotice('MCP 命令已复制到剪贴板。')
    } catch {
      setSettingsNotice('无法自动复制，请手动复制命令。')
    }
  }

  return (
    <div className="panel settings-panel">
      <header className="panel-hero">
        <div>
          <p className="eyebrow">系统设置</p>
          <h2>运行配置与 MCP 集成</h2>
          <p className="hero-subtitle">配置直接读写根目录 .env，保存后立即在后端生效。</p>
        </div>
      </header>

      <div className="settings-grid">
        <section className="setting-card">
          <h3>连接概览</h3>
          <p className="hint">当前 API 地址通过前端环境变量配置。</p>
          <div className="input-with-hint">
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} disabled className="readonly-input" />
          </div>
          <p className="hint">VITE_API_BASE 仅影响前端请求目标，模型与系统参数在下方环境变量中维护。</p>
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

        <section className="setting-card full-width-card">
          <div className="setting-headline">
            <h3>环境变量（.env）</h3>
            <button type="button" className="refresh-btn" onClick={() => void saveSettings()} disabled={settingsLoading || settingsSaving}>
              {settingsSaving ? '保存中...' : '保存配置'}
            </button>
          </div>
          {settingsPayload && <p className="hint">配置文件：<code>{settingsPayload.env_file}</code></p>}
          {settingsError && <p className="settings-error">{settingsError}</p>}
          {settingsNotice && <p className="settings-notice">{settingsNotice}</p>}

          {settingsLoading ? (
            <p className="hint">正在读取环境变量...</p>
          ) : (
            <>
              <div className="setting-row-inline">
                <label htmlFor="mcp-enabled">启用 MCP 工具调用</label>
                <select
                  id="mcp-enabled"
                  value={variables.MCP_TOOLS_ENABLED ?? '1'}
                  onChange={(e) => setVar('MCP_TOOLS_ENABLED', e.target.value)}
                >
                  <option value="1">启用</option>
                  <option value="0">禁用</option>
                </select>
              </div>

              <div className="env-table">
                <div className="env-table-header">
                  <span>KEY</span>
                  <span>VALUE</span>
                  <span>ACTION</span>
                </div>
                {sortedEntries.map(([key, value]) => (
                  <div className="env-table-row" key={key}>
                    <input value={key} disabled className="readonly-input env-key" />
                    <input value={value} onChange={(e) => setVar(key, e.target.value)} className="env-value" />
                    <button type="button" className="danger-btn" onClick={() => removeVar(key)}>
                      删除
                    </button>
                  </div>
                ))}
              </div>

              <div className="env-add-row">
                <input
                  placeholder="新增 KEY，例如 MCP_AUTH_TOKEN"
                  value={newKey}
                  onChange={(e) => setNewKey(e.target.value)}
                />
                <input placeholder="VALUE" value={newValue} onChange={(e) => setNewValue(e.target.value)} />
                <button type="button" className="refresh-btn" onClick={addVar}>
                  添加
                </button>
              </div>
            </>
          )}
        </section>

        <section className="setting-card full-width-card">
          <h3>MCP 安装命令</h3>
          <p className="hint">基于 DEPLOYMENT_URL 自动生成，可直接用于 Claude Code 与 CodeX。</p>
          <div className="command-block">
            <span className="about-label">Claude Code</span>
            <code>{settingsPayload?.mcp_commands.claude_code || '请先设置 DEPLOYMENT_URL 并保存。'}</code>
            <button
              type="button"
              className="refresh-btn"
              onClick={() => void copyCommand(settingsPayload?.mcp_commands.claude_code || '')}
              disabled={!settingsPayload?.mcp_commands.claude_code}
            >
              复制
            </button>
          </div>
          <div className="command-block">
            <span className="about-label">CodeX</span>
            <code>{settingsPayload?.mcp_commands.codex || '请先设置 DEPLOYMENT_URL 并保存。'}</code>
            <button
              type="button"
              className="refresh-btn"
              onClick={() => void copyCommand(settingsPayload?.mcp_commands.codex || '')}
              disabled={!settingsPayload?.mcp_commands.codex}
            >
              复制
            </button>
          </div>
          {settingsPayload?.mcp_endpoint && (
            <p className="hint">
              MCP Endpoint: <code>{settingsPayload.mcp_endpoint}</code>
            </p>
          )}
        </section>
      </div>
    </div>
  )
}
