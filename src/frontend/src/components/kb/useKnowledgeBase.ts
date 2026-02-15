import { useState } from 'react'
import { API_BASE } from '../../config'

export const useKnowledgeBase = (
  kbId: number,
  onKbChange: (kbId: number) => void,
  onKnowledgeBasesUpdated: () => Promise<void>
) => {
  const [creatingKb, setCreatingKb] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [newKbDescription, setNewKbDescription] = useState('')
  const [kbBusy, setKbBusy] = useState(false)
  const [error, setError] = useState('')

  const handleCreateKnowledgeBase = async () => {
    const trimmed = newKbName.trim()
    if (!trimmed) return

    setCreatingKb(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/kb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed, description: newKbDescription.trim() }),
      })
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}))
        throw new Error(payload.detail || '创建知识库失败')
      }
      const created = await res.json()
      await onKnowledgeBasesUpdated()
      onKbChange(created.id)
      setNewKbName('')
      setNewKbDescription('')
    } catch (createErr) {
      setError(createErr instanceof Error ? createErr.message : '创建知识库失败')
    } finally {
      setCreatingKb(false)
    }
  }

  const handleDeleteKnowledgeBase = async () => {
    if (kbId === 1) {
      setError('默认知识库不支持删除')
      return
    }
    if (!confirm('确定删除当前知识库及其全部文档和会话吗？')) return

    setKbBusy(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/kb/${kbId}`, { method: 'DELETE' })
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}))
        throw new Error(payload.detail || '删除知识库失败')
      }
      await onKnowledgeBasesUpdated()
      onKbChange(1)
    } catch (deleteErr) {
      setError(deleteErr instanceof Error ? deleteErr.message : '删除知识库失败')
    } finally {
      setKbBusy(false)
    }
  }

  const handleReindexKnowledgeBase = async (loadDocs: () => Promise<void>) => {
    setKbBusy(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/kb/${kbId}/reindex`, { method: 'POST' })
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}))
        throw new Error(payload.detail || '重建知识库索引失败')
      }
      await loadDocs()
    } catch (reindexErr) {
      setError(reindexErr instanceof Error ? reindexErr.message : '重建知识库索引失败')
    } finally {
      setKbBusy(false)
    }
  }

  return {
    creatingKb,
    newKbName,
    setNewKbName,
    newKbDescription,
    setNewKbDescription,
    kbBusy,
    error,
    setError,
    handleCreateKnowledgeBase,
    handleDeleteKnowledgeBase,
    handleReindexKnowledgeBase,
  }
}
