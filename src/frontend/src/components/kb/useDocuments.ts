import { useState, useEffect } from 'react'
import { API_BASE } from '../../config'
import { DocItem } from './types'

export const useDocuments = (kbId: number) => {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([])

  const loadDocs = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/kb/documents?kb_id=${kbId}`)
      if (!res.ok) throw new Error('Failed to fetch')
      const data = await res.json()
      setDocs(data)
      setSelectedDocIds([])
    } catch (e) {
      setError('加载文档列表失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocs()
  }, [kbId])

  const handleDelete = async (id: number) => {
    if (!confirm('确定要删除这个文档吗？')) return
    await fetch(`${API_BASE}/kb/documents/${id}?kb_id=${kbId}`, { method: 'DELETE' })
    await loadDocs()
  }

  const handleReindexDocument = async (id: number) => {
    setError('')
    const res = await fetch(`${API_BASE}/kb/documents/${id}/reindex?kb_id=${kbId}`, {
      method: 'POST',
    })
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}))
      setError(payload.detail || '文档重建索引失败')
      return
    }
    await loadDocs()
  }

  const handleBatchDelete = async () => {
    if (!selectedDocIds.length) return
    if (!confirm(`确定删除选中的 ${selectedDocIds.length} 个文档吗？`)) return

    setError('')
    const res = await fetch(`${API_BASE}/kb/documents/batch-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kb_id: kbId, document_ids: selectedDocIds }),
    })
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}))
      setError(payload.detail || '批量删除失败')
      return
    }
    await loadDocs()
  }

  const handleBatchReindex = async () => {
    if (!selectedDocIds.length) return
    setError('')
    const res = await fetch(`${API_BASE}/kb/documents/reindex-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kb_id: kbId, document_ids: selectedDocIds }),
    })
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}))
      setError(payload.detail || '批量重建索引失败')
      return
    }
    await loadDocs()
  }

  return {
    docs,
    loading,
    error,
    setError,
    selectedDocIds,
    setSelectedDocIds,
    loadDocs,
    handleDelete,
    handleReindexDocument,
    handleBatchDelete,
    handleBatchReindex,
  }
}
