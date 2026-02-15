import { useState } from 'react'
import { API_BASE } from '../../config'
import { DocItem, DuplicatePolicy, PreparedUpload, UploadPhase, UploadTask } from './types'

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

export const useUpload = (kbId: number, docs: DocItem[], loadDocs: () => Promise<void>) => {
  const [batchUploading, setBatchUploading] = useState(false)
  const [uploadTasks, setUploadTasks] = useState<UploadTask[]>([])
  const [duplicatePolicy, setDuplicatePolicy] = useState<DuplicatePolicy>('ask')
  const [error, setError] = useState('')

  const updateUploadTask = (id: number, patch: Partial<UploadTask>) => {
    setUploadTasks((prev) => prev.map((task) => (task.id === id ? { ...task, ...patch } : task)))
  }

  const getDisplayName = (file: File): string => {
    const fileWithPath = file as File & { webkitRelativePath?: string }
    if (fileWithPath.webkitRelativePath && fileWithPath.webkitRelativePath.trim()) {
      return fileWithPath.webkitRelativePath
    }
    return file.name
  }

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

  return {
    batchUploading,
    uploadTasks,
    duplicatePolicy,
    setDuplicatePolicy,
    error,
    setError,
    handleBatchUpload,
    clearFinishedTasks,
  }
}
