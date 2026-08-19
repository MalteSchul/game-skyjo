import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import KeyboardHelp from './KeyboardHelp'

describe('KeyboardHelp', () => {
  it('lists the shortcut groups and entries when open (happy path)', () => {
    render(<KeyboardHelp open onClose={vi.fn()} />)

    expect(screen.getByRole('dialog', { name: 'Keyboard shortcuts' })).toBeInTheDocument()
    expect(screen.getByText('Draw from the stock (or place the drawn card)')).toBeInTheDocument()
    expect(screen.getByText('Show or hide this shortcuts list')).toBeInTheDocument()
  })

  it('renders nothing when closed (sad path)', () => {
    render(<KeyboardHelp open={false} onClose={vi.fn()} />)

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('closes via the backdrop and the close button, but not via clicks inside the modal (bad path)', () => {
    const onClose = vi.fn()
    render(<KeyboardHelp open onClose={onClose} />)

    fireEvent.click(screen.getByText('Draw from the stock (or place the drawn card)'))
    expect(onClose).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
