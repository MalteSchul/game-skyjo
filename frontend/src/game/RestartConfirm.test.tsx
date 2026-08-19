import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import RestartConfirm from './RestartConfirm'

describe('RestartConfirm', () => {
  it('renders the confirm dialog when open (happy path)', () => {
    render(<RestartConfirm open onConfirm={vi.fn()} onCancel={vi.fn()} />)

    expect(screen.getByRole('dialog', { name: 'Restart the game?' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Restart' })).toBeInTheDocument()
  })

  it('renders nothing when closed (sad path)', () => {
    render(<RestartConfirm open={false} onConfirm={vi.fn()} onCancel={vi.fn()} />)

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('cancels via the backdrop and the close button, but not via clicks inside the modal, and confirms via the Restart button (bad path)', () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()
    const { rerender } = render(<RestartConfirm open onConfirm={onConfirm} onCancel={onCancel} />)

    fireEvent.click(screen.getByText(/abandons the current match/))
    expect(onCancel).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onCancel).toHaveBeenCalledTimes(1)

    rerender(<RestartConfirm open onConfirm={onConfirm} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: 'Restart' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })
})
