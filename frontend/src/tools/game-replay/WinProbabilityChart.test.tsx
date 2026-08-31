import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import WinProbabilityChart from './WinProbabilityChart'
import type { WinProbSeries } from './replayUtils'

const SERIES: WinProbSeries[] = [
  { seat: 0, seatName: 'bootstrap', points: [{ step: 0, p: 0.5 }, { step: 2, p: 0.7 }] },
  { seat: 1, seatName: 'iter5', points: [{ step: 1, p: 0.4 }] },
]

describe('WinProbabilityChart', () => {
  it('renders a legend entry for every seat (happy path)', () => {
    render(<WinProbabilityChart series={SERIES} totalSteps={3} activeStep={0} />)

    expect(screen.getByText('bootstrap')).toBeInTheDocument()
    expect(screen.getByText('iter5')).toBeInTheDocument()
  })

  it('shows a fallback message instead of an empty plot for a game with no decisions (sad path)', () => {
    render(<WinProbabilityChart series={[]} totalSteps={0} activeStep={0} />)
    expect(screen.getByText('No decisions to chart.')).toBeInTheDocument()
  })

  it('shows the nearest point for each seat in the tooltip when hovering (happy path)', () => {
    render(<WinProbabilityChart series={SERIES} totalSteps={3} activeStep={0} />)
    const svg = screen.getByRole('img', { name: /Win probability per seat/ })
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 640,
      height: 240,
      right: 640,
      bottom: 240,
      x: 0,
      y: 0,
      toJSON: () => {},
    })

    fireEvent.mouseMove(svg, { clientX: 640 })

    expect(screen.getByText('step 2')).toBeInTheDocument()
    expect(screen.getByText('70%')).toBeInTheDocument()
  })

  it('calls onScrub with the hovered step on click when provided (happy path)', () => {
    const onScrub = vi.fn()
    render(<WinProbabilityChart series={SERIES} totalSteps={3} activeStep={0} onScrub={onScrub} />)
    const svg = screen.getByRole('img', { name: /Win probability per seat/ })
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 640,
      height: 240,
      right: 640,
      bottom: 240,
      x: 0,
      y: 0,
      toJSON: () => {},
    })

    fireEvent.mouseMove(svg, { clientX: 0 })
    fireEvent.click(svg)

    expect(onScrub).toHaveBeenCalledWith(0)
  })
})
