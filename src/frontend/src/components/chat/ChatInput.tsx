import React from 'react'
import { ChatMode, KnowledgeBaseItem } from './types'

interface ChatInputProps {
  question: string
  setQuestion: (q: string) => void
  submitQuestion: (q?: string) => void
  loading: boolean
  kbId: number
  knowledgeBases: KnowledgeBaseItem[]
  onKbChange: (kbId: number) => void
  mode: ChatMode
  setMode: (mode: ChatMode) => void
  historyOpen: boolean
  setHistoryOpen: (open: boolean) => void
  setChatId: (id: number | null) => void
}

export const ChatInput: React.FC<ChatInputProps> = ({
  question,
  setQuestion,
  submitQuestion,
  loading,
  kbId,
  knowledgeBases,
  onKbChange,
  mode,
  setMode,
  historyOpen,
  setHistoryOpen,
  setChatId,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submitQuestion()
    }
  }

  return (
    <section className="chat-input-area composer-area">
      <div className="composer-toolbar">
        <button className="history-toggle" onClick={() => setHistoryOpen(!historyOpen)}>
          📜 历史
        </button>

        <label className="kb-switcher" htmlFor="chat-kb-switcher">
          <span>知识库</span>
          <select
            id="chat-kb-switcher"
            value={kbId}
            onChange={(e) => {
              const nextKbId = Number(e.target.value)
              setChatId(null)
              onKbChange(nextKbId)
            }}
            disabled={loading}
          >
            {knowledgeBases.length > 0 ? (
              knowledgeBases.map((kb) => (
                <option key={kb.id} value={kb.id}>
                  {kb.name || `知识库 #${kb.id}`}
                </option>
              ))
            ) : (
              <option value={kbId}>{`知识库 #${kbId}`}</option>
            )}
          </select>
        </label>

        <div className="mode-toggle" role="tablist" aria-label="模式选择">
          <button
            type="button"
            className={mode === 'rag' ? 'active' : ''}
            aria-pressed={mode === 'rag'}
            onClick={() => setMode('rag')}
          >
            RAG 问答
          </button>
          <button
            type="button"
            className={mode === 'agent' ? 'active' : ''}
            aria-pressed={mode === 'agent'}
            onClick={() => setMode('agent')}
          >
            Agent 智能体
          </button>
        </div>
      </div>

      <div className="input-wrapper">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="给本地知识库发消息... (Shift+Enter 换行)"
          rows={3}
        />
        <button onClick={() => void submitQuestion()} disabled={loading || !question.trim()} className="send-btn" aria-label={loading ? '正在处理' : '发送消息'}>
          {loading ? <span className="spinner" aria-hidden="true" /> : <span className="send-icon">↗</span>}
        </button>
      </div>

      <div className="chat-hints">
        <span>Shift+Enter 换行</span>
        <span>{loading ? '正在流式输出中' : '回答将流式呈现'}</span>
      </div>
    </section>
  )
}
