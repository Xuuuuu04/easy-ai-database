import React from 'react'
import { KnowledgeBaseItem } from './types'

interface KnowledgeBaseHeaderProps {
  kbId: number
  knowledgeBases: KnowledgeBaseItem[]
  onKbChange: (kbId: number) => void
  busy: boolean
  kbBusy: boolean
  handleReindexKnowledgeBase: () => void
  handleDeleteKnowledgeBase: () => void
  newKbName: string
  setNewKbName: (name: string) => void
  newKbDescription: string
  setNewKbDescription: (desc: string) => void
  creatingKb: boolean
  handleCreateKnowledgeBase: () => void
  docsCount: number
  fileCount: number
  urlCount: number
}

export const KnowledgeBaseHeader: React.FC<KnowledgeBaseHeaderProps> = ({
  kbId,
  knowledgeBases,
  onKbChange,
  busy,
  kbBusy,
  handleReindexKnowledgeBase,
  handleDeleteKnowledgeBase,
  newKbName,
  setNewKbName,
  newKbDescription,
  setNewKbDescription,
  creatingKb,
  handleCreateKnowledgeBase,
  docsCount,
  fileCount,
  urlCount,
}) => {
  return (
    <>
      <header className="panel-hero">
        <div>
          <p className="eyebrow">知识库</p>
          <h2>把资料变成可检索的本地档案</h2>
          <p className="hero-subtitle">
            支持文件与 URL 导入，自动切分与向量化，随时可追溯来源。
          </p>
        </div>
        <div className="hero-badges">
          <span className="hero-badge">文件导入</span>
          <span className="hero-badge">网页抓取</span>
          <span className="hero-badge">本地索引</span>
        </div>
      </header>

      <div className="kb-tenant-toolbar">
        <label className="kb-switcher" htmlFor="kb-switcher">
          <span>当前知识库</span>
          <select
            id="kb-switcher"
            value={kbId}
            onChange={(e) => onKbChange(Number(e.target.value))}
            disabled={busy || kbBusy}
          >
            {knowledgeBases.length > 0 ? (
              <>
                {!knowledgeBases.some((kb) => kb.id === kbId) && (
                  <option value={kbId}>{`知识库 #${kbId}`}</option>
                )}
                {knowledgeBases.map((kb) => (
                  <option key={kb.id} value={kb.id}>
                    {kb.name || `知识库 #${kb.id}`}
                  </option>
                ))}
              </>
            ) : (
              <option value={kbId}>{`知识库 #${kbId}`}</option>
            )}
          </select>
        </label>

        <div className="kb-ops-group">
          <button className="refresh-btn" onClick={handleReindexKnowledgeBase} disabled={busy || kbBusy}>
            重建当前知识库索引
          </button>
          <button className="delete-btn" onClick={handleDeleteKnowledgeBase} disabled={busy || kbBusy || kbId === 1}>
            删除当前知识库
          </button>
        </div>
      </div>

      <div className="kb-create-row">
        <input
          value={newKbName}
          onChange={(e) => setNewKbName(e.target.value)}
          placeholder="新知识库名称"
          disabled={creatingKb || busy || kbBusy}
        />
        <input
          value={newKbDescription}
          onChange={(e) => setNewKbDescription(e.target.value)}
          placeholder="描述（可选）"
          disabled={creatingKb || busy || kbBusy}
        />
        <button className="refresh-btn" onClick={handleCreateKnowledgeBase} disabled={creatingKb || !newKbName.trim() || busy || kbBusy}>
          {creatingKb ? '创建中...' : '创建知识库'}
        </button>
      </div>

      <div className="kb-stats">
        <div className="stat-card">
          <p>文档总数</p>
          <h3>{docsCount}</h3>
        </div>
        <div className="stat-card">
          <p>文件资料</p>
          <h3>{fileCount}</h3>
        </div>
        <div className="stat-card">
          <p>网页来源</p>
          <h3>{urlCount}</h3>
        </div>
      </div>
    </>
  )
}
