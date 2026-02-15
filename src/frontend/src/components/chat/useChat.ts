import { useState, useEffect, useRef } from 'react'
import { API_BASE } from '../../config'
import { ChatStep, Citation, StreamPhase, ChatMode } from './types'

export const useChat = (kbId: number) => {
  const [question, setQuestion] = useState('')
  const [activeQuestion, setActiveQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [steps, setSteps] = useState<ChatStep[]>([])
  const [mode, setMode] = useState<ChatMode>('rag')
  const [loading, setLoading] = useState(false)
  const [streamPhase, setStreamPhase] = useState<StreamPhase>('idle')
  const [error, setError] = useState('')
  const [chatId, setChatId] = useState<number | null>(null)

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

  const upsertStep = (incoming: ChatStep) => {
    setSteps((prev) => {
      const signature = `${incoming.round ?? 0}|${incoming.tool}|${incoming.input}|${incoming.output}`
      const foundIdx = prev.findIndex((item) => `${item.round ?? 0}|${item.tool}|${item.input}|${item.output}` === signature)
      if (foundIdx >= 0) {
        const next = [...prev]
        next[foundIdx] = incoming
        return next
      }
      return [...prev, incoming]
    })
  }

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
        const event = typeof payload.event === 'string' ? payload.event : ''

        if (event === 'agent_research_start') {
          setStreamPhase('retrieving')
        }

        if (event === 'agent_summary_start') {
          setStreamPhase('grounding')
        }

        if (event === 'agent_step') {
          if (Array.isArray(payload.steps)) {
            setSteps(payload.steps as ChatStep[])
          } else if (payload.step && typeof payload.step === 'object') {
            upsertStep(payload.step as ChatStep)
          }
        }

        const chunk = typeof payload.chunk === 'string' ? payload.chunk : ''
        if (chunk) {
          if (!receivedStreamChunk) {
            setStreamPhase('streaming')
            receivedStreamChunk = true
          }
          await appendAnswerAnimated(chunk)
        }

        if (Array.isArray(payload.citations)) {
          setCitations(payload.citations as Citation[])
        }
        if (Array.isArray(payload.steps) && payload.steps.length > 0) {
          setSteps(payload.steps as ChatStep[])
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

  return {
    question,
    setQuestion,
    activeQuestion,
    setActiveQuestion,
    answer,
    citations,
    steps,
    mode,
    setMode,
    loading,
    streamPhase,
    error,
    chatId,
    setChatId,
    submitQuestion,
  }
}
