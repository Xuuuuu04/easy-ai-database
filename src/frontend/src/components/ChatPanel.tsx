import React, { useEffect, useRef, useState } from 'react'
import { ChatHistorySidebar } from './ChatHistorySidebar'
import { KnowledgeBaseItem } from './chat/types'
import { useChat } from './chat/useChat'
import { useCitations } from './chat/useCitations'
import { ChatHeader } from './chat/ChatHeader'
import { ChatMessage } from './chat/ChatMessage'
import { CitationPanel } from './chat/CitationPanel'
import { ChatInput } from './chat/ChatInput'

interface ChatPanelProps {
  kbId: number
  knowledgeBases: KnowledgeBaseItem[]
  onKbChange: (kbId: number) => void
}

/**
 * RAG 与 Agent 对话界面。
 */
export const ChatPanel: React.FC<ChatPanelProps> = ({ kbId, knowledgeBases, onKbChange }) => {
  const {
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
  } = useChat(kbId)

  const {
    expandedSources,
    setExpandedSources,
    sourcesPanelOpen,
    setSourcesPanelOpen,
    selectedCitationKey,
    setSelectedCitationKey,
    previewLoading,
    previewError,
    previewData,
    setPreviewData,
    setPreviewError,
    sourceGroups,
    loadCitationPreview,
  } = useCitations(kbId, citations, activeQuestion)

  const [historyOpen, setHistoryOpen] = useState(false)
  const feedEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [answer, loading, activeQuestion, steps.length, citations.length])

  return (
    <div className="panel chat-panel">
      <ChatHeader />

      <div className="chat-shell">
        <section className="chat-content chat-thread" aria-live="polite">
          <ChatMessage
            activeQuestion={activeQuestion}
            answer={answer}
            loading={loading}
            citations={citations}
            steps={steps}
            mode={mode}
            streamPhase={streamPhase}
          />

          <CitationPanel
            sourceGroups={sourceGroups}
            sourcesPanelOpen={sourcesPanelOpen}
            setSourcesPanelOpen={setSourcesPanelOpen}
            expandedSources={expandedSources}
            setExpandedSources={setExpandedSources}
            selectedCitationKey={selectedCitationKey}
            setSelectedCitationKey={setSelectedCitationKey}
            previewLoading={previewLoading}
            previewError={previewError}
            previewData={previewData}
            setPreviewData={setPreviewData}
            setPreviewError={setPreviewError}
            loadCitationPreview={loadCitationPreview}
          />

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

        <ChatInput
          question={question}
          setQuestion={setQuestion}
          submitQuestion={submitQuestion}
          loading={loading}
          kbId={kbId}
          knowledgeBases={knowledgeBases}
          onKbChange={onKbChange}
          mode={mode}
          setMode={setMode}
          historyOpen={historyOpen}
          setHistoryOpen={setHistoryOpen}
          setChatId={setChatId}
        />
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
