import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { getMctsModels } from '../api/matchClient'
import NewMatchForm from './NewMatchForm'

vi.mock('../api/matchClient', () => ({
  getMctsModels: vi.fn().mockResolvedValue([]),
}))

describe('NewMatchForm', () => {
  it('submits the chosen player count, seed, and trimmed player names (happy path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Players'), { target: { value: '4' } })
    fireEvent.change(screen.getByLabelText('Seed (optional)'), { target: { value: '42' } })
    fireEvent.change(screen.getByLabelText('Player 1 name'), { target: { value: 'Ada' } })
    fireEvent.change(screen.getByLabelText('Player 3 name'), { target: { value: '  Grace  ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(
      4,
      42,
      ['Ada', '', 'Grace', ''],
      ['human', 'human', 'human', 'human'],
      [null, null, null, null],
      [null, null, null, null],
    )
  })

  it('submits undefined for an empty seed and blank names instead of NaN or throwing (sad path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''], ['human', 'human'], [null, null], [null, null])
  })

  it('lets a seat be set to a random bot and submits the chosen player types (happy path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 2 type'), { target: { value: 'random_bot' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''], ['human', 'random_bot'], [null, null], [null, null])
  })

  it('lets a seat be set to a thinking bot and submits the chosen player types (happy path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 2 type'), { target: { value: 'thinking_bot' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''], ['human', 'thinking_bot'], [null, null], [null, null])
  })

  it('lets a seat be set to an MCTS bot and submits the chosen player types (happy path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 2 type'), { target: { value: 'mcts_bot' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''], ['human', 'mcts_bot'], [null, null], [null, null])
  })

  it('lets an MCTS bot seat pick a specific model once one is available (happy path)', async () => {
    vi.mocked(getMctsModels).mockResolvedValueOnce(['strong'])
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 2 type'), { target: { value: 'mcts_bot' } })
    fireEvent.change(await screen.findByLabelText('Player 2 model'), { target: { value: 'strong' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(
      2,
      undefined,
      ['', ''],
      ['human', 'mcts_bot'],
      [null, 'strong'],
      [null, null],
    )
  })

  it('lets an MCTS bot seat set a custom simulation count (happy path)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 2 type'), { target: { value: 'mcts_bot' } })
    fireEvent.change(screen.getByLabelText('Player 2 simulations'), { target: { value: '50' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''], ['human', 'mcts_bot'], [null, null], [null, 50])
  })

  it('does not show a model or simulations picker for a non-MCTS seat (bad path)', () => {
    render(<NewMatchForm onCreate={vi.fn()} submitting={false} />)

    expect(screen.queryByLabelText('Player 1 model')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Player 1 simulations')).not.toBeInTheDocument()
  })

  it('preserves already-chosen player types when the player count changes (bad path: no data loss on resize)', () => {
    const onCreate = vi.fn()
    render(<NewMatchForm onCreate={onCreate} submitting={false} />)

    fireEvent.change(screen.getByLabelText('Player 2 type'), { target: { value: 'random_bot' } })
    fireEvent.change(screen.getByLabelText('Players'), { target: { value: '3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(
      3,
      undefined,
      ['', '', ''],
      ['human', 'random_bot', 'human'],
      [null, null, null],
      [null, null, null],
    )
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
