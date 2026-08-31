import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import DecisionScrubber from './DecisionScrubber'

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
})

describe('DecisionScrubber', () => {
  it('shows the active step label and calls onChange when the slider moves (happy path)', () => {
    const onChange = vi.fn()
    render(<DecisionScrubber count={10} activeIndex={3} overrideIndices={[]} roundBoundaryIndices={[]} onChange={onChange} />)

    expect(screen.getByText('step 3 / 9')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Decision step'), { target: { value: '5' } })
    expect(onChange).toHaveBeenCalledWith(5)
  })

  it('advances one step per interval while playing, then stops at the last step (happy path)', () => {
    const onChange = vi.fn()
    const { rerender } = render(
      <DecisionScrubber count={3} activeIndex={0} overrideIndices={[]} roundBoundaryIndices={[]} onChange={onChange} />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Play through the game' }))
    vi.advanceTimersByTime(250)
    expect(onChange).toHaveBeenCalledWith(1)

    rerender(<DecisionScrubber count={3} activeIndex={2} overrideIndices={[]} roundBoundaryIndices={[]} onChange={onChange} />)
    onChange.mockClear()
    vi.advanceTimersByTime(5000)
    expect(onChange).not.toHaveBeenCalled()
  })

  it('disables the play button for a single-decision game (sad path: nothing to scrub through)', () => {
    render(<DecisionScrubber count={1} activeIndex={0} overrideIndices={[]} roundBoundaryIndices={[]} onChange={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Play through the game' })).toBeDisabled()
  })

  it('does not crash for an empty game (bad path)', () => {
    render(<DecisionScrubber count={0} activeIndex={0} overrideIndices={[]} roundBoundaryIndices={[]} onChange={vi.fn()} />)
    expect(screen.getByText('step 0 / 0')).toBeInTheDocument()
  })

  it('jumps to the marked step when an override or round-boundary marker is clicked (happy path)', () => {
    const onChange = vi.fn()
    render(<DecisionScrubber count={10} activeIndex={0} overrideIndices={[4]} roundBoundaryIndices={[7]} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: /Jump to step 4.*search overrode/ }))
    expect(onChange).toHaveBeenCalledWith(4)

    fireEvent.click(screen.getByRole('button', { name: /Jump to step 7.*round boundary/ }))
    expect(onChange).toHaveBeenCalledWith(7)
  })

  it('renders no markers when nothing overrode the prior and no round closed (bad path)', () => {
    render(<DecisionScrubber count={10} activeIndex={0} overrideIndices={[]} roundBoundaryIndices={[]} onChange={vi.fn()} />)
    expect(screen.queryAllByRole('button', { name: /Jump to step/ })).toHaveLength(0)
  })

  it('hides the heuristic-diff lane by default even when indices are supplied (happy path)', () => {
    render(
      <DecisionScrubber count={10} activeIndex={0} overrideIndices={[]} roundBoundaryIndices={[]} heuristicDiffIndices={[3]} onChange={vi.fn()} />,
    )
    expect(screen.queryByRole('button', { name: /Jump to step 3/ })).not.toBeInTheDocument()
  })

  it('shows the heuristic-diff lane once enabled, alongside an override at the same step (happy path)', () => {
    const onChange = vi.fn()
    render(
      <DecisionScrubber
        count={10}
        activeIndex={0}
        overrideIndices={[3]}
        roundBoundaryIndices={[]}
        heuristicDiffIndices={[3]}
        showHeuristicDiffs
        onChange={onChange}
      />,
    )

    // Step 3 is both an override and a heuristic-diff — expect two distinct
    // marker buttons, not one clobbering the other.
    expect(screen.getByRole('button', { name: /Jump to step 3.*search overrode/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Jump to step 3.*differs from the heuristic/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Jump to step 3.*differs from the heuristic/ }))
    expect(onChange).toHaveBeenCalledWith(3)
  })

  it('hides the chosen-differs lane by default even when indices are supplied (happy path)', () => {
    render(
      <DecisionScrubber count={10} activeIndex={0} overrideIndices={[]} roundBoundaryIndices={[]} chosenDiffersIndices={[5]} onChange={vi.fn()} />,
    )
    expect(screen.queryByRole('button', { name: /Jump to step 5/ })).not.toBeInTheDocument()
  })

  it('shows the chosen-differs lane once enabled, as a third mark independent of override/heuristic lanes (happy path)', () => {
    const onChange = vi.fn()
    render(
      <DecisionScrubber
        count={10}
        activeIndex={0}
        overrideIndices={[3]}
        roundBoundaryIndices={[]}
        heuristicDiffIndices={[3]}
        showHeuristicDiffs
        chosenDiffersIndices={[3]}
        showChosenDiffers
        onChange={onChange}
      />,
    )

    expect(screen.getByRole('button', { name: /Jump to step 3.*search overrode/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Jump to step 3.*differs from the heuristic/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Jump to step 3.*search itself most favored/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Jump to step 3.*search itself most favored/ }))
    expect(onChange).toHaveBeenCalledWith(3)
  })
})
