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
    <aside className="app-sidebar" aria-label="Primary navigation">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <span className="mark-core" />
          <span className="mark-ring" />
        </div>
        <div className="brand-text">
          <p className="brand-kicker">easy-ai-database</p>
          <h1>轻量 AIRAG 知识库</h1>
        </div>
      </div>

      <div className="sidebar-meta">
        <span className="meta-pill">离线优先</span>
        <span className="meta-pill muted">数据仅本地</span>
      </div>

      <nav className="side-nav" aria-label="Primary">
        <button
          type="button"
          className={`side-nav-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => onTabChange('chat')}
        >
          <span aria-hidden="true">💬</span>
          <span>对话</span>
        </button>
        <button
          type="button"
          className={`side-nav-btn ${activeTab === 'kb' ? 'active' : ''}`}
          onClick={() => onTabChange('kb')}
        >
          <span aria-hidden="true">📚</span>
          <span>知识库</span>
        </button>
        <button
          type="button"
          className={`side-nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => onTabChange('settings')}
        >
          <span aria-hidden="true">⚙️</span>
          <span>设置</span>
        </button>
      </nav>

      <p className="sidebar-footnote">Workspace Ready</p>
    </aside>
  )
}
