import React, { useEffect, useRef, useState } from 'react'
import { API_BASE } from '../config'

const MAX_UPLOAD_CONCURRENCY = 6

const SUPPORTED_EXTENSIONS = [
  'pdf', 'docx', 'doc', 'txt',
  'xlsx', 'xls', 'csv',
  'pptx',
  'md', 'markdown',
  'html', 'htm',
  'json', 'xml',
  'rtf',
  'py', 'js', 'ts', 'java', 'go', 'rs', 'c', 'cpp', 'h',
  'sh', 'yaml', 'yml', 'toml', 'ini', 'cfg'
]

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

type UploadPhase = 'queued' | 'uploading' | 'indexing' | 'done' | 'failed' | 'skipped'

type DuplicatePolicy = 'ask' | 'skip' | 'keep'

type UploadTask = {
  id: number
  name: string
  size: number
  phase: UploadPhase
  progress: number
  message: string
}

type PreparedUpload = {
  file: File
  task: UploadTask
  supported: boolean
  duplicateBatch: boolean
  duplicateExisting: boolean
}

type KnowledgeBaseItem = {
  id: number
  name: string
  description?: string
  document_count?: number
}

interface KnowledgeBasePanelProps {
  kbId: number
  knowledgeBases: KnowledgeBaseItem[]
  onKbChange: (kbId: number) => void
  onKnowledgeBasesUpdated: () => Promise<void>
}

/**
 * 知识库管理界面：上传文件、导入 URL、查看索引列表。
 */
export const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({
  kbId,
  knowledgeBases,
  onKbChange,
  onKnowledgeBasesUpdated,
}) => {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [urlToIngest, setUrlToIngest] = useState('')
  const [loading, setLoading] = useState(false)
  const [batchUploading, setBatchUploading] = useState(false)
  const [urlIngesting, setUrlIngesting] = useState(false)
  const [uploadTasks, setUploadTasks] = useState<UploadTask[]>([])
  const [duplicatePolicy, setDuplicatePolicy] = useState<DuplicatePolicy>('ask')
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([])
  const [creatingKb, setCreatingKb] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [newKbDescription, setNewKbDescription] = useState('')
  const [kbBusy, setKbBusy] = useState(false)
  const [error, setError] = useState('')
  const filePickerRef = useRef<HTMLInputElement | null>(null)
  const folderPickerRef = useRef<HTMLInputElement | null>(null)

  const busy = batchUploading || urlIngesting

  const updateUploadTask = (id: number, patch: Partial<UploadTask>) => {
    setUploadTasks((prev) => prev.map((task) => (task.id === id ? { ...task, ...patch } : task)))
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getPhaseLabel = (phase: UploadPhase): string => {
    if (phase === 'queued') return '等待上传'
    if (phase === 'uploading') return '上传中'
    if (phase === 'indexing') return '等待索引'
    if (phase === 'done') return '已完成'
    if (phase === 'skipped') return '已跳过'
    return '失败'
  }

  const getDisplayName = (file: File): string => {
    const fileWithPath = file as File & { webkitRelativePath?: string }
    if (fileWithPath.webkitRelativePath && fileWithPath.webkitRelativePath.trim()) {
      return fileWithPath.webkitRelativePath
    }
    return file.name
  }

  /**
   * 从后端加载已索引文档列表。
   */
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

  useEffect(() => {
    const folderInput = folderPickerRef.current
    if (!folderInput) return
    folderInput.setAttribute('webkitdirectory', '')
    folderInput.setAttribute('directory', '')
    folderInput.setAttribute('mozdirectory', '')
  }, [])

  /**
   * 上传文件并刷新列表。
   */
  const uploadSingleFile = (
    taskId: number,
    file: File,
    allowDuplicate: boolean
  ): Promise<void> =>
    new Promise((resolve) => {
      updateUploadTask(taskId, {
        phase: 'uploading',
        progress: 0,
        message: '正在上传文件...'
      })

      const formData = new FormData()
      formData.append('file', file)

      const xhr = new XMLHttpRequest()
      const uploadUrl = `${API_BASE}/ingest/file?kb_id=${kbId}&allow_duplicate=${allowDuplicate ? '1' : '0'}`
      xhr.open('POST', uploadUrl)

      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return
        const nextProgress = Math.min(99, Math.round((event.loaded / event.total) * 100))
        updateUploadTask(taskId, {
          phase: 'uploading',
          progress: nextProgress,
          message: `正在上传 ${nextProgress}%`
        })
      }

      xhr.upload.onload = () => {
        updateUploadTask(taskId, {
          phase: 'indexing',
          progress: 100,
          message: '上传完成，等待索引...'
        })
      }

      xhr.onload = () => {
        let payload: { detail?: string; chunks?: number } = {}
        try {
          payload = JSON.parse(xhr.responseText || '{}')
        } catch {
          payload = {}
        }

        if (xhr.status >= 200 && xhr.status < 300) {
          const chunks = typeof payload.chunks === 'number' ? payload.chunks : 0
          updateUploadTask(taskId, {
            phase: 'done',
            progress: 100,
            message: chunks > 0 ? `索引完成，共 ${chunks} 个分块` : '索引完成'
          })
        } else {
          updateUploadTask(taskId, {
            phase: 'failed',
            progress: 100,
            message: payload.detail || '上传失败，请重试'
          })
        }
        resolve()
      }

      xhr.onerror = () => {
        updateUploadTask(taskId, {
          phase: 'failed',
          progress: 100,
          message: '网络错误，请检查后端服务'
        })
        resolve()
      }

      xhr.send(formData)
    })

  const runParallelUploads = async (
    items: Array<{ taskId: number; file: File; allowDuplicate: boolean }>
  ) => {
    let cursor = 0
    const workerCount = Math.min(MAX_UPLOAD_CONCURRENCY, items.length)
    const workers = Array.from({ length: workerCount }, async () => {
      while (true) {
        const current = items[cursor]
        cursor += 1
        if (!current) break
        await uploadSingleFile(current.taskId, current.file, current.allowDuplicate)
      }
    })
    await Promise.all(workers)
  }

  const handleBatchUpload = async (fileList: FileList) => {
    const selectedFiles = Array.from(fileList)
    if (!selectedFiles.length) return

    setError('')
    const now = Date.now()
    const existingTitles = new Set(docs.map((doc) => doc.title.trim().toLowerCase()))
    const seenInBatch = new Set<string>()

    const prepared: PreparedUpload[] = selectedFiles.map((file, index) => {
      const ext = file.name.split('.').pop()?.toLowerCase() || ''
      const supported = SUPPORTED_EXTENSIONS.includes(ext)
      const fingerprint = `${file.name.toLowerCase()}|${file.size}|${file.lastModified}`
      const duplicateBatch = supported && seenInBatch.has(fingerprint)
      if (supported && !duplicateBatch) {
        seenInBatch.add(fingerprint)
      }
      const duplicateExisting = supported && existingTitles.has(file.name.trim().toLowerCase())
      const id = now + index

      let phase: UploadPhase = 'queued'
      let message = '等待上传...'
      if (!supported) {
        phase = 'failed'
        message = '文件类型不支持'
      } else if (duplicateBatch) {
        phase = 'skipped'
        message = '同批次重复文件，已自动去重'
      }

      const task: UploadTask = {
        id,
        name: getDisplayName(file),
        size: file.size,
        phase,
        progress: phase === 'queued' ? 0 : 100,
        message,
      }

      return { file, task, supported, duplicateBatch, duplicateExisting }
    })

    setUploadTasks((prev) => [...prepared.map((item) => item.task), ...prev])

    let uploadables = prepared.filter((item) => item.supported && !item.duplicateBatch)
    if (!uploadables.length) {
      setError('没有可上传文件（可能都不支持或已去重）。')
      return
    }

    const duplicateExistingItems = uploadables.filter((item) => item.duplicateExisting)
    let effectivePolicy = duplicatePolicy
    if (duplicatePolicy === 'ask' && duplicateExistingItems.length > 0) {
      const keep = window.confirm(
        `检测到 ${duplicateExistingItems.length} 个文件与现有文档重名。点击“确定”保留并继续上传，点击“取消”自动跳过重名文件。`
      )
      effectivePolicy = keep ? 'keep' : 'skip'
    }

    if (effectivePolicy === 'skip' && duplicateExistingItems.length > 0) {
      const skipIdSet = new Set(duplicateExistingItems.map((item) => item.task.id))
      setUploadTasks((prev) =>
        prev.map((task) => {
          if (!skipIdSet.has(task.id)) return task
          return {
            ...task,
            phase: 'skipped',
            progress: 100,
            message: '与已有文档重名，已按策略跳过',
          }
        })
      )
      uploadables = uploadables.filter((item) => !item.duplicateExisting)
    }

    if (!uploadables.length) {
      return
    }

    setBatchUploading(true)
    try {
      await runParallelUploads(
        uploadables.map((item) => ({
          taskId: item.task.id,
          file: item.file,
          allowDuplicate: effectivePolicy === 'keep' && item.duplicateExisting,
        }))
      )
      await loadDocs()
    } finally {
      setBatchUploading(false)
    }
  }

  const clearFinishedTasks = () => {
    setUploadTasks((prev) =>
      prev.filter(
        (task) => task.phase !== 'done' && task.phase !== 'failed' && task.phase !== 'skipped'
      )
    )
  }

  /**
   * 导入 URL 并刷新列表。
   */
  const handleIngestUrl = async () => {
    if (!urlToIngest.trim()) return
    setUrlIngesting(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/ingest/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlToIngest, kb_id: kbId }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setError(data.detail || 'URL 导入失败，请重试')
        return
      }
      setUrlToIngest('')
      await loadDocs()
    } catch {
      setError('网络错误，请检查后端服务')
    } finally {
      setUrlIngesting(false)
    }
  }

  /**
   * 删除文档并刷新列表。
   */
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

  const handleReindexKnowledgeBase = async () => {
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

      <div className="kb-tenant-toolbar">
        <label className="kb-switcher" htmlFor="kb-switcher">
          <span>当前知识库</span>
          <select
            id="kb-switcher"
            value={kbId}
            onChange={(e) => onKbChange(Number(e.target.value))}
            disabled={busy || kbBusy}
          >
            {knowledgeBases.map((kb) => (
              <option key={kb.id} value={kb.id}>
                {kb.name || `知识库 #${kb.id}`}
              </option>
            ))}
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
            <span className="action-note">支持文件夹递归、并行上传与状态跟踪</span>
          </div>
          <p className="hint">状态分为等待上传、上传中、等待索引、已完成、已跳过、失败。重名文件可询问是否保留。</p>

          <div className="duplicate-policy-row">
            <label htmlFor="duplicate-policy">重名处理策略</label>
            <select
              id="duplicate-policy"
              value={duplicatePolicy}
              onChange={(e) => setDuplicatePolicy(e.target.value as DuplicatePolicy)}
              disabled={busy}
            >
              <option value="ask">发现重名时询问</option>
              <option value="skip">自动跳过重名</option>
              <option value="keep">保留并继续上传</option>
            </select>
          </div>

          <div className="upload-picker-actions">
            <button
              type="button"
              className="picker-btn"
              onClick={() => filePickerRef.current?.click()}
              disabled={busy}
            >
              {batchUploading ? '处理中...' : '选择文件（可多选）'}
            </button>
            <button
              type="button"
              className="picker-btn folder"
              onClick={() => folderPickerRef.current?.click()}
              disabled={busy}
            >
              选择文件夹（自动递归）
            </button>
          </div>

          <input
            ref={filePickerRef}
            type="file"
            multiple
            disabled={busy}
            onChange={(e) => {
              if (e.target.files) {
                void handleBatchUpload(e.target.files)
              }
              e.currentTarget.value = ''
            }}
            style={{ display: 'none' }}
          />
          <input
            ref={folderPickerRef}
            type="file"
            multiple
            disabled={busy}
            onChange={(e) => {
              if (e.target.files) {
                void handleBatchUpload(e.target.files)
              }
              e.currentTarget.value = ''
            }}
            style={{ display: 'none' }}
          />

          {uploadTasks.length > 0 && (
            <div className="upload-queue">
              <div className="upload-queue-head">
                <span>上传任务</span>
                <button type="button" onClick={clearFinishedTasks} className="retry-btn" disabled={busy}>
                  清空已结束
                </button>
              </div>
              <div className="upload-task-list">
                {uploadTasks.map((task) => (
                  <div key={task.id} className={`upload-task-item phase-${task.phase}`}>
                    <div className="upload-task-main">
                      <div className="upload-task-title" title={task.name}>{task.name}</div>
                      <div className="upload-task-meta">
                        <span>{formatFileSize(task.size)}</span>
                        <span className={`upload-phase-tag phase-${task.phase}`}>{getPhaseLabel(task.phase)}</span>
                      </div>
                    </div>
                    <div className="upload-task-progress">
                      <div className="upload-task-progress-bar" style={{ width: `${task.progress}%` }} />
                    </div>
                    <div className="upload-task-message">{task.message}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
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
              disabled={busy}
            />
            <button onClick={handleIngestUrl} disabled={busy || !urlToIngest}>
              {urlIngesting ? '导入中...' : '导入'}
            </button>
          </div>
        </div>
      </div>

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
    </div>
  )
}
