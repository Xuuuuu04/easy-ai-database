import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ExecutionTimeline } from './ExecutionTimeline'

describe('ExecutionTimeline', () => {
  it('shows rag processing status while loading', () => {
    render(
      <ExecutionTimeline
        steps={[]}
        phase="retrieving"
        loading
        mode="rag"
      />
    )

    expect(screen.getByText('RAG 问答：检索证据中')).toBeInTheDocument()
  })

  it('shows agent steps when completed', () => {
    render(
      <ExecutionTimeline
        steps={[
          {
            tool: 'search_kb',
            input: '什么是泛型接口',
            output: '命中证据',
            round: 1,
            status: 'ok',
            citations: [{ source: 'doc.md', page: 1, snippet: 'snippet' }],
          },
        ]}
        phase="idle"
        loading={false}
        mode="agent"
      />
    )

    expect(screen.getByText('Agent 研究完成：1 步')).toBeInTheDocument()
    expect(screen.getByText('第 1 轮探索')).toBeInTheDocument()
  })
})
