import { useState } from 'react'
import { Header } from './components/Header'
import { ChatPanel } from './components/ChatPanel'
import { KnowledgeBasePanel } from './components/KnowledgeBasePanel'
import { SettingsPanel } from './components/SettingsPanel'
import './styles.css'

/**
 * 应用外壳与标签页导航。
 */
export default function App() {
  const [tab, setTab] = useState<'chat' | 'kb' | 'settings'>('chat')

  return (
    <div className="app-container">
      <Header activeTab={tab} onTabChange={setTab} />
      
      <main className="app-content">
        {tab === 'chat' && <ChatPanel />}
        {tab === 'kb' && <KnowledgeBasePanel />}
        {tab === 'settings' && <SettingsPanel />}
      </main>
    </div>
  )
}
