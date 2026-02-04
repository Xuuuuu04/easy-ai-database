import React from 'react'

/**
 * 应用头部组件的属性。
 */
interface HeaderProps {
  activeTab: 'chat' | 'kb' | 'settings'
  onTabChange: (tab: 'chat' | 'kb' | 'settings') => void
}

/**
 * 应用顶部导航栏。
 */
export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <span className="mark-core" />
          <span className="mark-ring" />
        </div>
        <div className="brand-text">
          <p className="brand-kicker">Local Knowledge Studio</p>
          <h1>本机知识库助手</h1>
        </div>
      </div>

      <div className="header-meta">
        <span className="meta-pill">离线优先</span>
        <span className="meta-pill muted">数据仅本地</span>
      </div>

      <nav className="main-nav" aria-label="Primary">
        <button
          type="button"
          className={`nav-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => onTabChange('chat')}
        >
          对话
        </button>
        <button
          type="button"
          className={`nav-btn ${activeTab === 'kb' ? 'active' : ''}`}
          onClick={() => onTabChange('kb')}
        >
          知识库
        </button>
        <button
          type="button"
          className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => onTabChange('settings')}
        >
          设置
        </button>
      </nav>
    </header>
  )
}
