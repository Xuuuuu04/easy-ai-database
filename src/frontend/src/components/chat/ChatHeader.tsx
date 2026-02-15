import React from 'react'

export const ChatHeader: React.FC = () => {
  return (
    <header className="panel-hero">
      <div>
        <p className="eyebrow">对话中心</p>
        <h2>在本地知识里快速定位证据与答案</h2>
        <p className="hero-subtitle">
          通过 RAG 或 Agent 模式连接你的私有资料库，输出可追溯的解释。
        </p>
      </div>
      <div className="hero-badges">
        <span className="hero-badge">证据驱动</span>
        <span className="hero-badge">可追溯来源</span>
        <span className="hero-badge">本地处理</span>
      </div>
    </header>
  )
}
