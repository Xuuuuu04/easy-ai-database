import React, { useRef } from 'react'
import { DuplicatePolicy, UploadPhase, UploadTask } from './types'

interface UploadZoneProps {
  busy: boolean
  batchUploading: boolean
  duplicatePolicy: DuplicatePolicy
  setDuplicatePolicy: (policy: DuplicatePolicy) => void
  handleBatchUpload: (files: FileList) => void
  uploadTasks: UploadTask[]
  clearFinishedTasks: () => void
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  busy,
  batchUploading,
  duplicatePolicy,
  setDuplicatePolicy,
  handleBatchUpload,
  uploadTasks,
  clearFinishedTasks,
}) => {
  const filePickerRef = useRef<HTMLInputElement | null>(null)
  const folderPickerRef = useRef<HTMLInputElement | null>(null)

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

  return (
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
        // @ts-expect-error: webkitdirectory is not standard but supported
        webkitdirectory=""
        directory=""
        mozdirectory=""
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
  )
}
