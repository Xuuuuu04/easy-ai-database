import React, { useEffect, useState } from 'react'
import { API_BASE } from '../config'

/**
 * 后端返回的文档记录。
 */
type DocItem = {
  id: number
  title: string
  source_type: string
  source_ref: string
  created_at: string
}

/**
 * 知识库管理界面：上传文件、导入 URL、查看索引列表。
 */
export const KnowledgeBasePanel: React.FC = () => {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [urlToIngest, setUrlToIngest] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)

  /**
   * 从后端加载已索引文档列表。
   */
  const loadDocs = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/kb/documents`)
      const data = await res.json()
      setDocs(data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocs()
  }, [])

  /**
   * 上传文件并刷新列表。
   */
  const handleUpload = async (file: File) => {
    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)
    try {
      await fetch(`${API_BASE}/ingest/file`, {
        method: 'POST',
        body: formData,
      })
      await loadDocs()
    } finally {
      setUploading(false)
    }
  }

  /**
   * 导入 URL 并刷新列表。
   */
  const handleIngestUrl = async () => {
    if (!urlToIngest.trim()) return
    setUploading(true)
    try {
      await fetch(`${API_BASE}/ingest/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlToIngest }),
      })
      setUrlToIngest('')
      await loadDocs()
    } finally {
      setUploading(false)
    }
  }

  /**
   * 删除文档并刷新列表。
   */
  const handleDelete = async (id: number) => {
    if (!confirm('确定要删除这个文档吗？')) return
    await fetch(`${API_BASE}/kb/documents/${id}`, { method: 'DELETE' })
    await loadDocs()
  }

  /**
   * 统计 URL 与文件来源数量。
   */
  const urlCount = docs.filter((doc) => doc.source_type === 'url').length
  const fileCount = docs.length - urlCount

  return (
    <div className="panel kb-panel">
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

      <div className="kb-stats">
        <div className="stat-card">
          <p>文档总数</p>
          <h3>{docs.length}</h3>
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

      <div className="kb-actions">
        <div className="action-card">
          <div className="action-header">
            <h3>上传文件</h3>
            <span className="action-note">PDF / DOCX / TXT</span>
          </div>
          <p className="hint">自动解析与切分后进入本地索引。</p>
          <label className="file-upload-btn">
            <span>{uploading ? '处理中...' : '选择文件'}</span>
            <input
              type="file"
              disabled={uploading}
              onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
              style={{ display: 'none' }}
            />
          </label>
        </div>

        <div className="action-card">
          <div className="action-header">
            <h3>导入 URL</h3>
            <span className="action-note">网页抓取</span>
          </div>
          <p className="hint">抓取网页正文内容并写入索引。</p>
          <div className="url-input-group">
            <input
              value={urlToIngest}
              onChange={(e) => setUrlToIngest(e.target.value)}
              placeholder="https://example.com"
              disabled={uploading}
            />
            <button onClick={handleIngestUrl} disabled={uploading || !urlToIngest}>
              {uploading ? '...' : '导入'}
            </button>
          </div>
        </div>
      </div>

      <div className="doc-list-section">
        <div className="doc-toolbar">
          <h3>已索引文档</h3>
          <span className="doc-count">{docs.length} 项</span>
        </div>
        {loading ? (
          <div className="loading-indicator">加载中...</div>
        ) : (
          <div className="doc-grid">
            {docs.map((doc) => (
              <div key={doc.id} className="doc-card">
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
                  className="delete-btn"
                  onClick={() => handleDelete(doc.id)}
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
    </div>
  )
}
