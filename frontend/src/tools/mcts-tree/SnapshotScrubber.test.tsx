import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SnapshotScrubber from './SnapshotScrubber'

const ORDER = ['0', '10', '50']
const VISITS = [0, 10, 50]

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
})

describe('SnapshotScrubber', () => {
  it('shows the active snapshot label and calls onChange when the slider moves (happy path)', () => {
    const onChange = vi.fn()
    render(<SnapshotScrubber order={ORDER} visitCounts={VISITS} activeIndex={1} onChange={onChange} />)

    expect(screen.getByText('N=10 · 10 visits')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Simulation snapshot'), { target: { value: '2' } })
    expect(onChange).toHaveBeenCalledWith(2)
  })

  it('advances one step per interval while playing, then stops at the last snapshot (happy path)', () => {
    const onChange = vi.fn()
    const { rerender } = render(<SnapshotScrubber order={ORDER} visitCounts={VISITS} activeIndex={0} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'Play through snapshots' }))
    vi.advanceTimersByTime(900)
    expect(onChange).toHaveBeenCalledWith(1)

    // Simulate the parent applying that change, then reaching the end.
    rerender(<SnapshotScrubber order={ORDER} visitCounts={VISITS} activeIndex={2} onChange={onChange} />)
    onChange.mockClear()
    vi.advanceTimersByTime(5000)
    expect(onChange).not.toHaveBeenCalled()
  })

  it('disables the play button when there is only one snapshot to show (sad path: nothing to scrub through)', () => {
    render(<SnapshotScrubber order={['30']} visitCounts={[30]} activeIndex={0} onChange={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Play through snapshots' })).toBeDisabled()
  })

  it('does not crash and renders no tick marks for an empty snapshot list (bad path)', () => {
    render(<SnapshotScrubber order={[]} visitCounts={[]} activeIndex={0} onChange={vi.fn()} />)
    expect(screen.queryAllByRole('button')).toHaveLength(1)
  })
})
