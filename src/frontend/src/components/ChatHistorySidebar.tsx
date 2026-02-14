import React, { useState, useEffect } from 'react';
import { API_BASE } from '../config';

interface Chat {
  id: number;
  title: string;
  created_at: string;
}

interface ChatHistorySidebarProps {
  isOpen: boolean;
  kbId: number;
  onClose: () => void;
  onSelect: (chatId: number) => void;
  currentChatId?: number | null;
}

export const ChatHistorySidebar: React.FC<ChatHistorySidebarProps> = ({
  isOpen,
  kbId,
  onClose,
  onSelect,
  currentChatId,
}) => {
  const [history, setHistory] = useState<Chat[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchHistory();
    }
  }, [isOpen, kbId]);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/chat/history?kb_id=${kbId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch chat history');
      }
      const data = await response.json();
      if (Array.isArray(data)) {
        setHistory(data);
      } else {
        setHistory([]);
        console.error('Unexpected API response format:', data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return dateString;
    }
  };

  return (
    <>
      <div 
        className={`history-overlay ${isOpen ? 'visible' : ''}`} 
        onClick={onClose}
        aria-hidden={!isOpen}
      />
      
      <aside 
        className={`history-sidebar ${isOpen ? 'open' : ''}`}
        aria-label="Chat History"
      >
        <div className="history-sidebar-header">
          <h3>历史会话</h3>
          <button
            onClick={onClose}
            className="history-close-btn"
            aria-label="Close sidebar"
          >
            ✕
          </button>
        </div>

        <div className="history-list">
          {loading ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--bg-ink-muted)' }}>
              加载中...
            </div>
          ) : error ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--danger)' }}>
              {error}
              <button
                onClick={fetchHistory}
                style={{
                  display: 'block',
                  margin: '10px auto',
                  padding: '4px 8px',
                  fontSize: '0.875rem',
                  color: 'var(--accent)',
                  background: 'none',
                  border: '1px solid var(--accent)',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                重试
              </button>
            </div>
          ) : history.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--bg-ink-muted)' }}>
              暂无历史会话
            </div>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {history.map((chat) => (
                <li 
                  key={chat.id} 
                  className={`history-item ${currentChatId === chat.id ? 'active' : ''}`}
                  onClick={() => onSelect(chat.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      onSelect(chat.id);
                    }
                  }}
                >
                  <div className="history-item-title">
                    {chat.title || '无标题会话'}
                  </div>
                  <div className="history-item-time">
                    {formatDate(chat.created_at)}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
};
