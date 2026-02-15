import { useState } from 'react'
import { API_BASE } from '../../config'

export const useUrlIngest = (kbId: number, loadDocs: () => Promise<void>) => {
  const [urlToIngest, setUrlToIngest] = useState('')
  const [urlIngesting, setUrlIngesting] = useState(false)
  const [urlError, setUrlError] = useState('')

  const handleIngestUrl = async () => {
    if (!urlToIngest.trim()) return
    setUrlIngesting(true)
    setUrlError('')
    try {
      const res = await fetch(`${API_BASE}/ingest/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlToIngest, kb_id: kbId }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setUrlError(data.detail || 'URL 导入失败，请重试')
        return
      }
      setUrlToIngest('')
      await loadDocs()
    } catch {
      setUrlError('网络错误，请检查后端服务')
    } finally {
      setUrlIngesting(false)
    }
  }

  return {
    urlToIngest,
    setUrlToIngest,
    urlIngesting,
    urlError,
    setUrlError,
    handleIngestUrl,
  }
}
