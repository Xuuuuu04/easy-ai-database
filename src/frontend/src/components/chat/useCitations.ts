import { useState, useMemo, useEffect } from 'react'
import { API_BASE } from '../../config'
import { Citation, PreviewPayload } from './types'

export const useCitations = (kbId: number, citations: Citation[], activeQuestion: string) => {
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const [sourcesPanelOpen, setSourcesPanelOpen] = useState(false)
  const [selectedCitationKey, setSelectedCitationKey] = useState<string | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const [previewData, setPreviewData] = useState<PreviewPayload | null>(null)

  useEffect(() => {
    setSourcesPanelOpen(false)
    setExpandedSources({})
    setSelectedCitationKey(null)
    setPreviewData(null)
    setPreviewError('')
  }, [activeQuestion])

  const sourceGroups = useMemo(() => {
    const grouped = new Map<string, { source: string; folderLabel: string; fileLabel: string; items: Array<{ key: string; citation: Citation }> }>()
    for (let idx = 0; idx < citations.length; idx += 1) {
      const citation = citations[idx]
      const source = (citation.source || 'local').trim() || 'local'

      let folderLabel = 'local'
      let fileLabel = source

      if (source.startsWith('http://') || source.startsWith('https://')) {
        try {
          const url = new URL(source)
          const pathParts = url.pathname.split('/').filter(Boolean)
          fileLabel = pathParts[pathParts.length - 1] || url.hostname
          folderLabel = `${url.hostname}/${pathParts.slice(0, -1).join('/') || ''}`.replace(/\/$/, '')
        } catch {
          folderLabel = 'url'
          fileLabel = source
        }
      } else if (source.includes('/')) {
        const parts = source.split('/').filter(Boolean)
        fileLabel = parts[parts.length - 1] || source
        folderLabel = parts.slice(0, -1).join('/') || 'local'
      }

      if (!grouped.has(source)) {
        grouped.set(source, {
          source,
          folderLabel,
          fileLabel,
          items: [],
        })
      }
      const key = `${source}::${idx}`
      grouped.get(source)!.items.push({ key, citation })
    }

    return Array.from(grouped.values())
  }, [citations])

  const loadCitationPreview = async (key: string, citation: Citation) => {
    setSelectedCitationKey(key)
    setPreviewLoading(true)
    setPreviewError('')
    if (!citation.source || citation.source === 'local') {
      setPreviewData({
        source: citation.source || 'local',
        kind: 'local',
        preview_type: 'text',
        content: citation.snippet,
      })
      setPreviewLoading(false)
      return
    }

    try {
      const endpoint = `${API_BASE}/kb/preview?kb_id=${kbId}&source=${encodeURIComponent(citation.source)}`
      const res = await fetch(endpoint)
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || '预览加载失败')
      }
      const data = await res.json()
      setPreviewData(data)
    } catch (previewErr) {
      setPreviewData(null)
      if (previewErr instanceof Error) {
        setPreviewError(previewErr.message)
      } else {
        setPreviewError('预览加载失败')
      }
    } finally {
      setPreviewLoading(false)
    }
  }

  return {
    expandedSources,
    setExpandedSources,
    sourcesPanelOpen,
    setSourcesPanelOpen,
    selectedCitationKey,
    setSelectedCitationKey,
    previewLoading,
    previewError,
    previewData,
    setPreviewData,
    setPreviewError,
    sourceGroups,
    loadCitationPreview,
  }
}
