import React from 'react'

interface UrlIngestProps {
  busy: boolean
  urlToIngest: string
  setUrlToIngest: (url: string) => void
  urlIngesting: boolean
  handleIngestUrl: () => Promise<void>
}

export const UrlIngest: React.FC<UrlIngestProps> = ({
  busy,
  urlToIngest,
  setUrlToIngest,
  urlIngesting,
  handleIngestUrl,
}) => {
  return (
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
          disabled={busy || urlIngesting}
        />
        <button onClick={() => void handleIngestUrl()} disabled={busy || urlIngesting || !urlToIngest}>
          {urlIngesting ? '导入中...' : '导入'}
        </button>
      </div>
    </div>
  )
}
