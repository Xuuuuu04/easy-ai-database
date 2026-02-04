import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

/**
 * 启动 React 应用。
 */
const root = createRoot(document.getElementById('root') as HTMLElement)
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
