import { useEffect, useState } from 'react'
import { API_BASE } from './config'
import { Header } from './components/Header'
import { ChatPanel } from './components/ChatPanel'
import { KnowledgeBasePanel } from './components/KnowledgeBasePanel'
import { SettingsPanel } from './components/SettingsPanel'
import './styles.css'

type KnowledgeBaseItem = {
  id: number
  name: string
  description?: string
  slug?: string
  created_at?: string
  document_count?: number
}

/**
 * 应用外壳与标签页导航。
 */
export default function App() {
  const [tab, setTab] = useState<'chat' | 'kb' | 'settings'>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseItem[]>([])
  const [activeKbId, setActiveKbId] = useState(1)

  const loadKnowledgeBases = async () => {
    try {
      const res = await fetch(`${API_BASE}/kb`)
      if (!res.ok) {
        return
      }
      const data = (await res.json()) as KnowledgeBaseItem[]
      const normalized = data.map((item) => {
        const name = typeof item.name === 'string' ? item.name.trim() : ''
        return {
          ...item,
          name: name || `知识库 #${item.id}`,
        }
      })
      setKnowledgeBases(normalized)
      if (normalized.length > 0 && !normalized.some((item) => item.id === activeKbId)) {
        setActiveKbId(normalized[0].id)
      }
    } catch {
    }
  }

  useEffect(() => {
    void loadKnowledgeBases()
  }, [])

  return (
    <div className="app-shell">
      <div className={`sidebar-wrap ${sidebarOpen ? 'open' : 'collapsed'}`}>
        <Header activeTab={tab} onTabChange={setTab} />
      </div>

      <div className="workspace">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={() => setSidebarOpen((prev) => !prev)}
          aria-label={sidebarOpen ? '收起侧边栏' : '展开侧边栏'}
        >
          {sidebarOpen ? '←' : '→'}
        </button>

        <main className="app-content">
          <div className={`tab-surface tab-${tab}`}>
            {tab === 'chat' && (
              <ChatPanel
                kbId={activeKbId}
                knowledgeBases={knowledgeBases}
                onKbChange={setActiveKbId}
              />
            )}
            {tab === 'kb' && (
              <KnowledgeBasePanel
                kbId={activeKbId}
                knowledgeBases={knowledgeBases}
                onKbChange={setActiveKbId}
                onKnowledgeBasesUpdated={loadKnowledgeBases}
              />
            )}
            {tab === 'settings' && <SettingsPanel />}
          </div>
        </main>
      </div>
    </div>
  )
}
