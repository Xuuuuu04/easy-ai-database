export type DocItem = {
  id: number
  title: string
  source_type: string
  source_ref: string
  created_at: string
}

export type UploadPhase = 'queued' | 'uploading' | 'indexing' | 'done' | 'failed' | 'skipped'

export type DuplicatePolicy = 'ask' | 'skip' | 'keep'

export type UploadTask = {
  id: number
  name: string
  size: number
  phase: UploadPhase
  progress: number
  message: string
}

export type PreparedUpload = {
  file: File
  task: UploadTask
  supported: boolean
  duplicateBatch: boolean
  duplicateExisting: boolean
}

export type KnowledgeBaseItem = {
  id: number
  name: string
  description?: string
  document_count?: number
}
