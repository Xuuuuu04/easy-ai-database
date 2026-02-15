import React from 'react'
import { DocItem } from './types'

interface DocumentListProps {
  docs: DocItem[]
  loading: boolean
  error: string
  loadDocs: () => Promise<void>
  selectedDocIds: number[]
  setSelectedDocIds: (ids: number[] | ((prev: number[]) => number[])) => void
  handleReindexDocument: (id: number) => void
  handleDelete: (id: number) => void
  handleBatchReindex: () => void
  handleBatchDelete: () => void
  busy: boolean
  kbBusy: boolean
}

export const DocumentList: React.FC<DocumentListProps> = ({
  docs,
  loading,
  error,
  loadDocs,
  selectedDocIds,
  setSelectedDocIds,
  handleReindexDocument,
  handleDelete,
  handleBatchReindex,
  handleBatchDelete,
  busy,
  kbBusy,
}) => {
  return (
    <div className="doc-list-section">
      <div className="doc-toolbar">
        <h3>已索引文档</h3>
        <div className="doc-toolbar-actions">
          <span className="doc-count">{docs.length} 项</span>
          <button
            className="refresh-btn"
            onClick={handleBatchReindex}
            disabled={!selectedDocIds.length || busy || kbBusy}
          >
            批量重建索引
          </button>
          <button
            className="delete-btn"
            onClick={handleBatchDelete}
            disabled={!selectedDocIds.length || busy || kbBusy}
          >
            批量删除
          </button>
        </div>
      </div>
      {error && (
        <div className="error-message">
          <span>{error}</span>
          <button onClick={loadDocs} className="retry-btn">
            重试
          </button>
        </div>
      )}
      {loading ? (
        <div className="loading-indicator">加载中...</div>
      ) : (
        <div className="doc-grid">
          {docs.map((doc) => (
            <div key={doc.id} className="doc-card">
              <input
                type="checkbox"
                checked={selectedDocIds.includes(doc.id)}
                onChange={(e) => {
                  setSelectedDocIds((prev) => {
                    if (e.target.checked) {
                      return Array.from(new Set([...prev, doc.id]))
                    }
                    return prev.filter((value) => value !== doc.id)
                  })
                }}
              />
              <div className="doc-icon">{doc.source_type === 'url' ? '🌐' : '📄'}</div>
              <div className="doc-info">
                <div className="doc-title" title={doc.title}>
                  {doc.title}
                </div>
                <div className="doc-meta">
                  <span className="doc-type">{doc.source_type}</span>
                  <span className="doc-date">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
              <button
                className="refresh-btn"
                onClick={() => void handleReindexDocument(doc.id)}
                title="重建索引"
              >
                重建
              </button>
              <button
                className="delete-btn"
                onClick={() => void handleDelete(doc.id)}
                title="删除"
              >
                删除
              </button>
            </div>
          ))}
          {docs.length === 0 && <div className="empty-docs">暂无文档，请上传或导入。</div>}
        </div>
      )}
    </div>
  )
}
