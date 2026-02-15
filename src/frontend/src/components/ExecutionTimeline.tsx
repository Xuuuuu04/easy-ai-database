import React, { useEffect, useMemo, useState } from 'react'

type CitationLike = { source?: string | null; page?: number | null; snippet?: string }
type HitPreview = { source?: string | null; page?: number | null; snippet?: string }

export type ChatStep = {
  tool: string
  input: string
  output: string
  citations?: CitationLike[]
  round?: number
  status?: 'ok' | 'error' | string
  error?: string
  rewrites?: string[]
  keywords?: string[]
  top_hits?: HitPreview[]
  preview_type?: string
  source_kind?: string
}

interface ExecutionTimelineProps {
  steps: ChatStep[]
  phase: 'idle' | 'retrieving' | 'ranking' | 'grounding' | 'streaming'
  loading: boolean
  mode: 'rag' | 'agent'
}

const TOOL_META: Record<string, { label: string; icon: string; tone: string }> = {
  search_kb: { label: '检索知识库', icon: '🔎', tone: 'search' },
  read_source: { label: '查看源文件', icon: '📄', tone: 'source' },
  fetch_url: { label: '抓取网页', icon: '🌐', tone: 'fetch' },
  summarize: { label: '综合总结', icon: '🧠', tone: 'summary' },
}

const phaseLabel = (phase: ExecutionTimelineProps['phase'], mode: ExecutionTimelineProps['mode']) => {
  if (mode === 'rag') {
    if (phase === 'retrieving') return 'RAG 问答：检索证据中'
    if (phase === 'ranking') return 'RAG 问答：筛选证据中'
    if (phase === 'grounding') return 'RAG 问答：组织答案中'
    if (phase === 'streaming') return 'RAG 问答：输出答案中'
    return 'RAG 问答'
  }

  if (phase === 'retrieving') return 'Agent 研究：多轮探索中'
  if (phase === 'ranking') return 'Agent 研究：评估线索中'
  if (phase === 'grounding') return 'Agent 研究：综合结论中'
  if (phase === 'streaming') return 'Agent 研究：撰写回答中'
  return 'Agent 研究流程'
}

export const ExecutionTimeline: React.FC<ExecutionTimelineProps> = ({ steps, phase, loading, mode }) => {
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (mode === 'agent' && (loading || steps.length > 0)) {
      setExpanded(true)
    }
  }, [loading, mode, steps.length])

  const hasSteps = steps.length > 0
  const isProcessing = loading && phase !== 'idle'

  const rounds = useMemo(() => {
    const groups = new Map<number, ChatStep[]>()
    for (const step of steps) {
      const round = typeof step.round === 'number' && step.round > 0 ? step.round : 1
      if (!groups.has(round)) groups.set(round, [])
      groups.get(round)!.push(step)
    }
    return Array.from(groups.entries()).sort((a, b) => a[0] - b[0])
  }, [steps])

  if (mode === 'rag') {
    if (!isProcessing) return null
    return (
      <div className="execution-timeline mode-rag compact">
        <div className="timeline-header passive">
          <div className="status-row">
            <span className="timeline-spinner" />
            <span className="timeline-label">{phaseLabel(phase, mode)}</span>
          </div>
        </div>
      </div>
    )
  }

  if (!hasSteps && !isProcessing) return null

  return (
    <div className={`execution-timeline mode-agent ${expanded ? 'expanded' : 'collapsed'}`}>
      <button
        type="button"
        className="timeline-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <div className="timeline-status">
          <div className="status-row">
            {isProcessing ? <span className="timeline-spinner" /> : <span className="timeline-icon-check">✓</span>}
            <span className="timeline-label">
              {isProcessing ? phaseLabel(phase, mode) : `Agent 研究完成：${steps.length} 步`}
            </span>
          </div>
        </div>
        <span className="timeline-toggle-icon">{expanded ? '收起' : '展开'}</span>
      </button>

      {expanded && (
        <div className="timeline-body">
          {rounds.map(([round, roundSteps]) => (
            <section key={round} className="agent-round-block">
              <div className="agent-round-title">第 {round} 轮探索</div>
              <div className="timeline-track">
                {roundSteps.map((step, idx) => {
                  const meta = TOOL_META[step.tool] || {
                    label: step.tool || '工具执行',
                    icon: '⚙️',
                    tone: 'default',
                  }
                  const hasError = step.status === 'error' || Boolean(step.error)
                  return (
                    <div key={`${round}-${idx}-${step.tool}`} className={`timeline-item completed tone-${meta.tone} ${hasError ? 'is-error' : ''}`}>
                      <div className="timeline-marker">
                        <div className="marker-dot" />
                        {idx < roundSteps.length - 1 && <div className="marker-line" />}
                      </div>
                      <div className="timeline-content">
                        <div className="step-tool-name">
                          <span className="tool-icon">{meta.icon}</span>
                          <span>{meta.label}</span>
                          {hasError ? <span className="step-status error">失败</span> : <span className="step-status ok">成功</span>}
                        </div>

                        <div className="step-meta-row">
                          <span className="meta-chip">查询：{step.input || '（空）'}</span>
                          {Array.isArray(step.citations) && step.citations.length > 0 && (
                            <span className="meta-chip accent">引用 {step.citations.length}</span>
                          )}
                          {step.preview_type && <span className="meta-chip">预览 {step.preview_type}</span>}
                        </div>

                        {Array.isArray(step.keywords) && step.keywords.length > 0 && (
                          <div className="step-list-inline">
                            <span className="inline-label">关键词</span>
                            <div className="chip-list">
                              {step.keywords.slice(0, 8).map((kw) => (
                                <span key={kw} className="tiny-chip">{kw}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {Array.isArray(step.rewrites) && step.rewrites.length > 0 && (
                          <div className="step-list-inline">
                            <span className="inline-label">检索改写</span>
                            <div className="chip-list">
                              {step.rewrites.slice(0, 3).map((rw, rwIdx) => (
                                <span key={`${rw}-${rwIdx}`} className="tiny-chip muted">{rw}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {Array.isArray(step.top_hits) && step.top_hits.length > 0 && (
                          <div className="step-hit-list">
                            <span className="inline-label">命中片段</span>
                            {step.top_hits.slice(0, 2).map((hit, hitIdx) => (
                              <div key={`${hit.source || 'local'}-${hitIdx}`} className="hit-row">
                                <span className="hit-source">{hit.source || 'local'}</span>
                                <span className="hit-snippet">{hit.snippet || ''}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {hasError && <div className="step-error">{step.error || '工具执行失败'}</div>}

                        {step.output && (
                          <details className="step-output-block">
                            <summary>查看输出</summary>
                            <pre>{step.output}</pre>
                          </details>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>
          ))}

          {isProcessing && (
            <div className="timeline-item processing">
              <div className="timeline-marker">
                <div className="marker-dot pulse" />
              </div>
              <div className="timeline-content">
                <div className="step-tool-name processing-text">{phaseLabel(phase, mode)}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
