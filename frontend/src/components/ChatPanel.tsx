import React, { useState } from 'react'
import { API_BASE } from '../config'

/**
 * RAG/Agent 响应的引用信息。
 */
type Citation = { source: string | null; page: number | null; snippet: string }

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
export const ChatPanel: React.FC = () => {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [steps, setSteps] = useState<ChatStep[]>([])
  const [mode, setMode] = useState<'rag' | 'agent'>('rag')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  /**
   * 将用户问题提交到选定的后端接口。
   */
  const submitQuestion = async () => {
    if (!question.trim()) return
    setAnswer('')
    setCitations([])
    setSteps([])
    setError('')
    setLoading(true)

    try {
      const endpoint = mode === 'rag' ? '/chat/rag' : '/chat/agent'
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      setAnswer(data.answer || '')
      setCitations(data.citations || [])
      setSteps(data.steps || [])
    } catch (e) {
      setError('请求失败，请检查后端服务或模型配置。')
    } finally {
      setLoading(false)
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

      <div className="chat-grid">
        <section className="chat-input-area">
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

          <div className="input-wrapper">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题... (Shift+Enter 换行)"
              rows={3}
            />
            <button onClick={submitQuestion} disabled={loading} className="send-btn">
              {loading ? <span className="spinner" aria-hidden="true" /> : <span className="send-icon">↗</span>}
              <span>{loading ? '思考中' : '发送'}</span>
            </button>
          </div>

          <div className="chat-hints">
            <span>Shift+Enter 换行</span>
            <span>支持长文本输入</span>
          </div>

          {error && <div className="error-message">{error}</div>}
        </section>

        <section className="chat-content">
          {answer && (
            <div className="result-card answer-card">
              <h3>回答</h3>
              <div className="markdown-body">{answer}</div>
            </div>
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

          {citations.length > 0 && (
            <div className="result-card citations-card">
              <h3>参考来源</h3>
              <div className="citation-list">
                {citations.map((c, idx) => (
                  <div key={idx} className="citation-item">
                    <div className="citation-source">
                      <span className="file-icon">📄</span>
                      {c.source}
                      {c.page && <span className="page-tag">P.{c.page}</span>}
                    </div>
                    <div className="citation-snippet">{c.snippet}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!answer && !loading && steps.length === 0 && (
            <div className="empty-state">
              <div className="icon">✦</div>
              <p>准备好回答你的问题了</p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
