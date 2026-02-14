import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { API_BASE } from '../config'
import { ChatHistorySidebar } from './ChatHistorySidebar'

/**
 * 骨架屏组件
 */
const Skeleton = ({ height = 20 }: { height?: number }) => (
  <div className="skeleton" style={{ height }} />
)

/**
 * RAG/Agent 响应的引用信息。
 */
type Citation = { source: string | null; page: number | null; snippet: string }

type PreviewPayload = {
  source: string
  kind: 'file' | 'url' | 'local'
  preview_type: 'text' | 'markdown' | 'code'
  content: string
}

type KnowledgeBaseItem = {
  id: number
  name: string
  document_count?: number
}

interface ChatPanelProps {
  kbId: number
  knowledgeBases: KnowledgeBaseItem[]
  onKbChange: (kbId: number) => void
}

/**
 * 后端返回的 Agent 步骤详情。
 */
type ChatStep = {
  tool: string
  input: string
  output: string
  citations?: Citation[]
}

/**
 * RAG 与 Agent 对话界面。
 */
export const ChatPanel: React.FC<ChatPanelProps> = ({ kbId, knowledgeBases, onKbChange }) => {
  const [question, setQuestion] = useState('')
  const [activeQuestion, setActiveQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [steps, setSteps] = useState<ChatStep[]>([])
  const [mode, setMode] = useState<'rag' | 'agent'>('rag')
  const [loading, setLoading] = useState(false)
  const [streamPhase, setStreamPhase] = useState<'idle' | 'retrieving' | 'ranking' | 'grounding' | 'streaming'>('idle')
  const [error, setError] = useState('')
  const [chatId, setChatId] = useState<number | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const [sourcesPanelOpen, setSourcesPanelOpen] = useState(false)
  const [selectedCitationKey, setSelectedCitationKey] = useState<string | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const [previewData, setPreviewData] = useState<PreviewPayload | null>(null)
  const feedEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!loading) return
    if (streamPhase === 'retrieving') {
      const timer = window.setTimeout(() => setStreamPhase((prev) => (prev === 'retrieving' ? 'ranking' : prev)), 650)
      return () => window.clearTimeout(timer)
    }
    if (streamPhase === 'ranking') {
      const timer = window.setTimeout(() => setStreamPhase((prev) => (prev === 'ranking' ? 'grounding' : prev)), 650)
      return () => window.clearTimeout(timer)
    }
    return
  }, [loading, streamPhase])

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

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [answer, loading, activeQuestion, steps.length, citations.length])

  const appendAnswerAnimated = async (text: string) => {
    const chunkSize = 18
    for (let cursor = 0; cursor < text.length; cursor += chunkSize) {
      const piece = text.slice(cursor, cursor + chunkSize)
      if (!piece) continue
      setAnswer((prev) => prev + piece)
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 16)
      })
    }
  }

  /**
   * 将用户问题提交到选定的后端接口。
   */
  const submitQuestion = async (questionOverride?: string) => {
    const sourceQuestion = typeof questionOverride === 'string' ? questionOverride : question
    const trimmedQuestion = sourceQuestion.trim()
    if (!trimmedQuestion) return

    setActiveQuestion(trimmedQuestion)
    if (typeof questionOverride !== 'string') {
      setQuestion('')
    }
    setAnswer('')
    setCitations([])
    setSteps([])
    setError('')
    setLoading(true)
    setStreamPhase('retrieving')

    try {
      const endpoint = mode === 'rag' ? '/chat/rag' : '/chat/agent'
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion, chat_id: chatId, kb_id: kbId, stream: true }),
      })

      if (!res.ok) {
        throw new Error('请求失败')
      }

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('无法读取响应')
      }

      let buffer = ''
      let receivedStreamChunk = false

      const handleStreamPayload = async (payload: Record<string, unknown>) => {
        const chunk = typeof payload.chunk === 'string' ? payload.chunk : ''
        if (chunk) {
          if (!receivedStreamChunk) {
            setStreamPhase('streaming')
            receivedStreamChunk = true
          }
          await appendAnswerAnimated(chunk)
        }

        if (!payload.done) {
          return
        }

        if (!receivedStreamChunk) {
          const fullAnswer = typeof payload.answer === 'string' ? payload.answer : ''
          if (fullAnswer) {
            setStreamPhase('streaming')
            await appendAnswerAnimated(fullAnswer)
          }
        }

        setCitations(Array.isArray(payload.citations) ? (payload.citations as Citation[]) : [])
        setSteps(Array.isArray(payload.steps) ? (payload.steps as ChatStep[]) : [])
        if (typeof payload.chat_id === 'number') {
          setChatId(payload.chat_id)
        }
        setStreamPhase('idle')
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        while (true) {
          const eventBoundary = buffer.indexOf('\n\n')
          if (eventBoundary < 0) {
            break
          }

          const rawEvent = buffer.slice(0, eventBoundary)
          buffer = buffer.slice(eventBoundary + 2)

          const payloadText = rawEvent
            .split('\n')
            .map((line) => line.trimEnd())
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trimStart())
            .join('\n')

          if (!payloadText) {
            continue
          }

          try {
            const payload = JSON.parse(payloadText) as Record<string, unknown>
            await handleStreamPayload(payload)
          } catch (parseError) {
            if (!(parseError instanceof SyntaxError)) {
              throw parseError
            }
          }
        }
      }

      if (buffer.trim()) {
        const payloadText = buffer
          .split('\n')
          .map((line) => line.trimEnd())
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n')

        if (payloadText) {
          try {
            const payload = JSON.parse(payloadText) as Record<string, unknown>
            await handleStreamPayload(payload)
          } catch (parseError) {
            if (!(parseError instanceof SyntaxError)) {
              throw parseError
            }
          }
        }
      }
    } catch (e) {
      setError('请求失败，请检查后端服务或模型配置。')
      setStreamPhase('idle')
    } finally {
      setLoading(false)
      setStreamPhase('idle')
    }
  }

  /**
   * Enter 提交，Shift+Enter 换行。
   */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submitQuestion()
    }
  }

  const showCitationPreview = Boolean(selectedCitationKey || previewLoading || previewError || previewData)

  return (
    <div className="panel chat-panel">
      <header className="panel-hero">
        <div>
          <p className="eyebrow">对话中心</p>
          <h2>在本地知识里快速定位证据与答案</h2>
          <p className="hero-subtitle">
            通过 RAG 或 Agent 模式连接你的私有资料库，输出可追溯的解释。
          </p>
        </div>
        <div className="hero-badges">
          <span className="hero-badge">证据驱动</span>
          <span className="hero-badge">可追溯来源</span>
          <span className="hero-badge">本地处理</span>
        </div>
      </header>

      <div className="chat-shell">
        <section className="chat-content chat-thread" aria-live="polite">
          {!activeQuestion && !answer && !loading && steps.length === 0 && !error && (
            <div className="empty-state">
              <div className="icon">✦</div>
              <p>准备好回答你的问题了</p>
            </div>
          )}

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
                {loading && streamPhase === 'retrieving' && (
                  <div className="retrieval-status" role="status">
                    <span className="retrieval-label">检索处理中</span>
                    <div className="retrieval-stage-track" aria-hidden="true">
                      <span className={`retrieval-stage ${streamPhase === 'retrieving' ? 'active' : ''}`}>召回</span>
                      <span className={`retrieval-stage ${streamPhase === 'ranking' ? 'active' : ''}`}>重排</span>
                      <span className={`retrieval-stage ${streamPhase === 'grounding' ? 'active' : ''}`}>组织</span>
                    </div>
                  </div>
                )}

                {loading && (streamPhase === 'ranking' || streamPhase === 'grounding') && (
                  <div className="retrieval-status" role="status">
                    <span className="retrieval-label">{streamPhase === 'ranking' ? '正在重排证据' : '正在组织答案结构'}</span>
                    <span className="retrieval-dots" aria-hidden="true">
                      <i />
                      <i />
                      <i />
                    </span>
                  </div>
                )}

                {loading && !answer && streamPhase !== 'streaming' && (
                  <div className="skeleton-group">
                    <Skeleton height={16} />
                    <Skeleton height={16} />
                    <Skeleton height={16} />
                  </div>
                )}

                <div className={`markdown-body ${loading ? 'is-streaming' : ''}`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {answer || (loading ? '正在组织答案...' : '未返回文本回答，请查看参考来源。')}
                  </ReactMarkdown>
                  {loading && <span className="stream-cursor" aria-hidden="true">|</span>}
                </div>
              </div>
            </article>
          )}

          {steps.length > 0 && (
            <div className="result-card steps-card">
              <h3>推理步骤</h3>
              {steps.map((step, idx) => (
                <div key={idx} className="step-item">
                  <div className="step-header">
                    <span className="step-tool">{step.tool}</span>
                  </div>
                  <div className="step-body">
                    <div className="step-row">
                      <span className="label">Input:</span> {step.input}
                    </div>
                    <div className="step-row">
                      <span className="label">Output:</span> {step.output}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {sourceGroups.length > 0 && (
            <div className="result-card citations-card">
              <button
                type="button"
                className="citations-toggle"
                aria-expanded={sourcesPanelOpen}
                onClick={() => {
                  setSourcesPanelOpen((prev) => {
                    const next = !prev
                    if (!next) {
                      setSelectedCitationKey(null)
                      setPreviewData(null)
                      setPreviewError('')
                    }
                    return next
                  })
                }}
              >
                <h3>参考来源</h3>
                <span className="citations-toggle-meta">{sourceGroups.length} 个来源</span>
                <span className="citations-toggle-icon">{sourcesPanelOpen ? '▾' : '▸'}</span>
              </button>

              {sourcesPanelOpen && (
              <div className={`citation-explorer ${showCitationPreview ? 'mode-split' : 'mode-list'}`}>
                <div className="citation-tree" role="tree" aria-label="citation tree">
                  {sourceGroups.map((group) => {
                    const expanded = expandedSources[group.source] ?? false
                    return (
                      <div key={group.source} className="citation-source-group">
                        <button
                          type="button"
                          className="citation-folder"
                          aria-expanded={expanded}
                          onClick={() =>
                            setExpandedSources((prev) => ({
                              ...prev,
                              [group.source]: !expanded,
                            }))
                          }
                        >
                          <span className="folder-caret">{expanded ? '▾' : '▸'}</span>
                          <span className="folder-label">{group.folderLabel || 'local'}</span>
                        </button>
                        {expanded && (
                          <div className="citation-children" role="group">
                            {group.items.map((entry) => (
                              <button
                                key={entry.key}
                                type="button"
                                className={`citation-leaf ${selectedCitationKey === entry.key ? 'active' : ''}`}
                                onClick={() => void loadCitationPreview(entry.key, entry.citation)}
                              >
                                <span className="leaf-name">
                                  {group.fileLabel}
                                  {entry.citation.page && <span className="page-tag">P.{entry.citation.page}</span>}
                                </span>
                                <span className="leaf-snippet">{entry.citation.snippet}</span>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>

                {showCitationPreview && <div className="citation-preview-panel">
                  <button
                    type="button"
                    className="citation-preview-close"
                    onClick={() => {
                      setSelectedCitationKey(null)
                      setPreviewData(null)
                      setPreviewError('')
                    }}
                  >
                    关闭预览
                  </button>
                  {!selectedCitationKey && !previewLoading && !previewError && (
                    <div className="citation-preview-empty">点击左侧引用可查看完整预览</div>
                  )}
                  {previewLoading && <div className="citation-preview-loading">预览加载中...</div>}
                  {previewError && <div className="citation-preview-error">{previewError}</div>}
                  {previewData && !previewLoading && !previewError && (
                    <div className="citation-preview-content">
                      <div className="citation-preview-head">
                        <span>{previewData.kind === 'url' ? '网页预览' : '文件预览'}</span>
                        <code>{previewData.source}</code>
                      </div>
                      <pre className={`citation-preview-text type-${previewData.preview_type}`}>
                        {previewData.content || '文件无可预览文本内容。'}
                      </pre>
                    </div>
                  )}
                </div>}
              </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-message">
              <span>{error}</span>
              <button onClick={() => submitQuestion(activeQuestion)} className="retry-btn" disabled={loading || !activeQuestion}>
                重试
              </button>
            </div>
          )}

          <div ref={feedEndRef} />
        </section>

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
                {knowledgeBases.map((kb) => (
                  <option key={kb.id} value={kb.id}>
                    {kb.name || `知识库 #${kb.id}`}
                  </option>
                ))}
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
      </div>

      <ChatHistorySidebar
        isOpen={historyOpen}
        kbId={kbId}
        onClose={() => setHistoryOpen(false)}
        onSelect={(id) => {
          setChatId(id)
          setHistoryOpen(false)
        }}
        currentChatId={chatId}
      />
    </div>
  )
}
