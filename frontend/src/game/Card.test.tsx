import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Card from './Card'

describe('Card', () => {
  it('renders a revealed value and invokes onClick when clicked (happy path)', () => {
    const onClick = vi.fn()
    render(<Card card={{ value: 7, face_up: true }} onClick={onClick} />)

    const button = screen.getByRole('button', { name: 'card 7' })
    expect(button).toHaveTextContent('7')

    fireEvent.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('renders a face-down card without leaking its value (sad path: no value to show)', () => {
    render(<Card card={{ value: null, face_up: false }} />)

    const button = screen.getByRole('button', { name: 'face-down card' })
    expect(button).toHaveTextContent('')
  })

  it('renders a cleared column slot with no button and no click handler (bad path: no interaction possible)', () => {
    render(<Card card={null} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('disables the button when no onClick handler is provided', () => {
    render(<Card card={{ value: 3, face_up: true }} />)

    expect(screen.getByRole('button')).toBeDisabled()
  })
})
