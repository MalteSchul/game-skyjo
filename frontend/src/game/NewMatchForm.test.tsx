import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import NewMatchForm from './NewMatchForm'

describe('NewMatchForm', () => {
  it('submits the chosen player count, seed, and trimmed player names (happy path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Players'), { target: { value: '4' } })
    fireEvent.change(screen.getByLabelText('Seed (optional)'), { target: { value: '42' } })
    fireEvent.change(screen.getByLabelText('Player 1 name'), { target: { value: 'Ada' } })
    fireEvent.change(screen.getByLabelText('Player 3 name'), { target: { value: '  Grace  ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(4, 42, ['Ada', '', 'Grace', ''])
  })

  it('submits undefined for an empty seed and blank names instead of NaN or throwing (sad path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''])
  })

  it('preserves already-typed names when the player count changes (bad path: no data loss on resize)', () => {
    render(<NewMatchForm onCreate={vi.fn()} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 1 name'), { target: { value: 'Ada' } })
    fireEvent.change(screen.getByLabelText('Players'), { target: { value: '3' } })

    expect(screen.getByLabelText('Player 1 name')).toHaveValue('Ada')
    expect(screen.getByLabelText('Player 3 name')).toHaveValue('')
  })

  it('lets the player count field be cleared and retyped without snapping back mid-edit (bad path)', () => {
    render(<NewMatchForm onCreate={vi.fn()} submitting={false} />)
    const input = screen.getByLabelText('Players') as HTMLInputElement

    // Simulate select-all + backspace: the field goes genuinely empty rather
    // than being silently clamped back to a default while still focused.
    fireEvent.change(input, { target: { value: '' } })
    expect(input.value).toBe('')

    // Typing '4' onto the now-empty field should replace, not concatenate
    // onto a value the field had already snapped back to.
    fireEvent.change(input, { target: { value: `${input.value}4` } })
    expect(input.value).toBe('4')
  })

  it('clamps an out-of-range player count on blur rather than on every keystroke (sad path)', () => {
    render(<NewMatchForm onCreate={vi.fn()} submitting={false} />)
    const input = screen.getByLabelText('Players') as HTMLInputElement

    fireEvent.change(input, { target: { value: '15' } })
    expect(input.value).toBe('15')
    fireEvent.blur(input)

    expect(input.value).toBe('8')
    expect(screen.getByLabelText('Player 8 name')).toBeInTheDocument()
  })

  it('disables the submit button while a request is in flight (bad path: prevents double-submit)', () => {
    render(<NewMatchForm onCreate={vi.fn()} submitting />)

    expect(screen.getByRole('button', { name: 'Starting…' })).toBeDisabled()
  })
})
