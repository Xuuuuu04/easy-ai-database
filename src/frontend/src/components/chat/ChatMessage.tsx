import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ExecutionTimeline, ChatStep } from '../ExecutionTimeline'
import { ChatMode, Citation, StreamPhase } from './types'

const Skeleton = ({ height = 20 }: { height?: number }) => (
  <div className="skeleton" style={{ height }} />
)

interface ChatMessageProps {
  activeQuestion: string
  answer: string
  loading: boolean
  citations: Citation[]
  steps: ChatStep[]
  mode: ChatMode
  streamPhase: StreamPhase
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  activeQuestion,
  answer,
  loading,
  citations,
  steps,
  mode,
  streamPhase,
}) => {
  if (!activeQuestion && !answer && !loading && steps.length === 0) {
    return (
      <div className="empty-state">
        <div className="icon">✦</div>
        <p>准备好回答你的问题了</p>
      </div>
    )
  }

  return (
    <>
      {activeQuestion && (
        <article className="chat-message user-message">
          <div className="chat-message-role">你</div>
          <div className="chat-bubble">{activeQuestion}</div>
        </article>
      )}

      {(activeQuestion && (loading || Boolean(answer) || citations.length > 0 || steps.length > 0)) && (
        <article className="chat-message assistant-message">
          <div className="chat-message-role">助手</div>
          <div className="chat-bubble assistant-bubble">
            <div className={`mode-context-badge ${mode === 'agent' ? 'agent' : 'rag'}`}>
              {mode === 'agent' ? 'Agent 研究模式 · 多轮工具探索' : 'RAG 问答模式 · 快速答案'}
            </div>

            <ExecutionTimeline 
              steps={steps} 
              phase={streamPhase} 
              loading={loading}
              mode={mode}
            />

            {loading && !answer && streamPhase !== 'streaming' && (
              <div className="skeleton-group">
                <Skeleton height={16} />
                <Skeleton height={16} />
                <Skeleton height={16} />
              </div>
            )}

            <div className={`markdown-body ${loading ? 'is-streaming' : ''}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {answer || (loading ? (mode === 'agent' ? '正在研究并组织结论...' : '正在组织答案...') : '未返回文本回答，请查看参考来源。')}
              </ReactMarkdown>
              {loading && <span className="stream-cursor" aria-hidden="true">|</span>}
            </div>
          </div>
        </article>
      )}
    </>
  )
}
