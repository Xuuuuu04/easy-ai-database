/**
 * 后端 API 基础地址。
 */
const rawApiBase = (import.meta.env.VITE_API_BASE || '').trim()
const normalizedApiBase = rawApiBase.replace(/\/+$/, '')
const isLocalBrowser =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')

export const API_BASE = normalizedApiBase || (isLocalBrowser ? 'http://localhost:8000' : '')
