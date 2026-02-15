export type Citation = { source: string | null; page: number | null; snippet: string }

export type PreviewPayload = {
  source: string
  kind: 'file' | 'url' | 'local'
  preview_type: 'text' | 'markdown' | 'code'
  content: string
}

export type KnowledgeBaseItem = {
  id: number
  name: string
  document_count?: number
}

export type ChatStep = {
  round?: number
  tool: string
  input: string
  output: string
}

export type StreamPhase = 'idle' | 'retrieving' | 'ranking' | 'grounding' | 'streaming'

export type ChatMode = 'rag' | 'agent'
