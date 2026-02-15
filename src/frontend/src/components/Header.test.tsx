import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Header } from './Header'

describe('Header', () => {
  it('renders navigation tabs and active state', () => {
    const onTabChange = vi.fn()
    render(<Header activeTab="chat" onTabChange={onTabChange} />)

    expect(screen.getByText('本机知识库助手')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '对话' })).toHaveClass('active')
    expect(screen.getByRole('button', { name: '知识库' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '设置' })).toBeInTheDocument()
  })
})
