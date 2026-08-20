import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import MovePreferenceChart from './MovePreferenceChart'
import type { TrendSeries } from './treeUtils'

function series(key: string, label: string, points: { snapshotKey: string; share: number }[]): TrendSeries {
  return { key, label, points: points.map((p) => ({ n: Number(p.snapshotKey), visitCount: 0, ...p })) }
}

const ORDER = ['0', '10']
const TWO_SERIES = [
  series('a', 'Draw stock', [{ snapshotKey: '0', share: 0.5 }, { snapshotKey: '10', share: 0.8 }]),
  series('b', 'Draw discard', [{ snapshotKey: '0', share: 0.5 }, { snapshotKey: '10', share: 0.2 }]),
]

describe('MovePreferenceChart', () => {
  it('renders a legend entry for every series (happy path)', () => {
    render(<MovePreferenceChart title="Root" subtitle="root actions" series={TWO_SERIES} order={ORDER} />)

    expect(screen.getByText('Draw stock')).toBeInTheDocument()
    expect(screen.getByText('Draw discard')).toBeInTheDocument()
  })

  it('shows a fallback message instead of an empty plot when there is nothing to chart (sad path)', () => {
    render(<MovePreferenceChart title="Root" subtitle="root actions" series={[]} order={ORDER} />)
    expect(screen.getByText('No actions to chart yet at this node.')).toBeInTheDocument()
  })

  it('folds more than six actions into a single "Other" legend entry (bad path: wide branching)', () => {
    const many = Array.from({ length: 8 }, (_, i) =>
      series(`action-${i}`, `Action ${i}`, [{ snapshotKey: '10', share: (8 - i) / 36 }]),
    )
    render(<MovePreferenceChart title="Root" subtitle="root actions" series={many} order={ORDER} />)

    expect(screen.getByText('Action 0')).toBeInTheDocument()
    expect(screen.getByText('Other (2)')).toBeInTheDocument()
    expect(screen.queryByText('Action 7')).not.toBeInTheDocument()
  })

  it('shows the values for every series at the hovered snapshot in the tooltip', () => {
    const { container } = render(<MovePreferenceChart title="Root" subtitle="root actions" series={TWO_SERIES} order={ORDER} />)
    const svg = screen.getByRole('img', { name: /Root: visit share/ })
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

    fireEvent.mouseMove(svg, { clientX: 640, clientY: 100 })

    const tooltip = container.querySelector('.mtx-chart-tooltip')
    expect(tooltip).not.toBeNull()
    expect(within(tooltip as HTMLElement).getByText('N=10')).toBeInTheDocument()
    expect(within(tooltip as HTMLElement).getByText('80%')).toBeInTheDocument()
    expect(within(tooltip as HTMLElement).getByText('20%')).toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', () => {
    const onClose = vi.fn()
    render(<MovePreferenceChart title="Root" subtitle="root actions" series={TWO_SERIES} order={ORDER} onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: 'close' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
