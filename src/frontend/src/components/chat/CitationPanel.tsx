import React from 'react'
import { Citation, PreviewPayload } from './types'

interface CitationPanelProps {
  sourceGroups: Array<{ source: string; folderLabel: string; fileLabel: string; items: Array<{ key: string; citation: Citation }> }>
  sourcesPanelOpen: boolean
  setSourcesPanelOpen: (open: boolean | ((prev: boolean) => boolean)) => void
  expandedSources: Record<string, boolean>
  setExpandedSources: (updater: (prev: Record<string, boolean>) => Record<string, boolean>) => void
  selectedCitationKey: string | null
  setSelectedCitationKey: (key: string | null) => void
  previewLoading: boolean
  previewError: string
  previewData: PreviewPayload | null
  setPreviewData: (data: PreviewPayload | null) => void
  setPreviewError: (error: string) => void
  loadCitationPreview: (key: string, citation: Citation) => void
}

export const CitationPanel: React.FC<CitationPanelProps> = ({
  sourceGroups,
  sourcesPanelOpen,
  setSourcesPanelOpen,
  expandedSources,
  setExpandedSources,
  selectedCitationKey,
  setSelectedCitationKey,
  previewLoading,
  previewError,
  previewData,
  setPreviewData,
  setPreviewError,
  loadCitationPreview,
}) => {
  const showCitationPreview = Boolean(selectedCitationKey || previewLoading || previewError || previewData)

  if (sourceGroups.length === 0) return null

  return (
    <div className="result-card citations-card">
      <button
        type="button"
        className="citations-toggle"
        aria-expanded={sourcesPanelOpen}
        onClick={() => {
          setSourcesPanelOpen((prev) => {
            const next = !prev
            if (!next) {
              setSelectedCitationKey(null)
              setPreviewData(null)
              setPreviewError('')
            }
            return next
          })
        }}
      >
        <h3>参考来源</h3>
        <span className="citations-toggle-meta">{sourceGroups.length} 个来源</span>
        <span className="citations-toggle-icon">{sourcesPanelOpen ? '▾' : '▸'}</span>
      </button>

      {sourcesPanelOpen && (
        <div className={`citation-explorer ${showCitationPreview ? 'mode-split' : 'mode-list'}`}>
          <div className="citation-tree" role="tree" aria-label="citation tree">
            {sourceGroups.map((group) => {
              const expanded = expandedSources[group.source] ?? false
              return (
                <div key={group.source} className="citation-source-group">
                  <button
                    type="button"
                    className="citation-folder"
                    aria-expanded={expanded}
                    onClick={() =>
                      setExpandedSources((prev) => ({
                        ...prev,
                        [group.source]: !expanded,
                      }))
                    }
                  >
                    <span className="folder-caret">{expanded ? '▾' : '▸'}</span>
                    <span className="folder-label">{group.folderLabel || 'local'}</span>
                  </button>
                  {expanded && (
                    <div className="citation-children" role="group">
                      {group.items.map((entry) => (
                        <button
                          key={entry.key}
                          type="button"
                          className={`citation-leaf ${selectedCitationKey === entry.key ? 'active' : ''}`}
                          onClick={() => void loadCitationPreview(entry.key, entry.citation)}
                        >
                          <span className="leaf-name">
                            {group.fileLabel}
                            {entry.citation.page && <span className="page-tag">P.{entry.citation.page}</span>}
                          </span>
                          <span className="leaf-snippet">{entry.citation.snippet}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {showCitationPreview && (
            <div className="citation-preview-panel">
              <button
                type="button"
                className="citation-preview-close"
                onClick={() => {
                  setSelectedCitationKey(null)
                  setPreviewData(null)
                  setPreviewError('')
                }}
              >
                关闭预览
              </button>
              {!selectedCitationKey && !previewLoading && !previewError && (
                <div className="citation-preview-empty">点击左侧引用可查看完整预览</div>
              )}
              {previewLoading && <div className="citation-preview-loading">预览加载中...</div>}
              {previewError && <div className="citation-preview-error">{previewError}</div>}
              {previewData && !previewLoading && !previewError && (
                <div className="citation-preview-content">
                  <div className="citation-preview-head">
                    <span>{previewData.kind === 'url' ? '网页预览' : '文件预览'}</span>
                    <code>{previewData.source}</code>
                  </div>
                  <pre className={`citation-preview-text type-${previewData.preview_type}`}>
                    {previewData.content || '文件无可预览文本内容。'}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
