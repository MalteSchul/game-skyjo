import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReplayBoard from './ReplayBoard'
import type { BoardOut } from '../../api/types'

function board(cards: BoardOut['cards']): BoardOut {
  return { cards }
}

describe('ReplayBoard', () => {
  it('marks a face-down card revealed only via the toggle with the secret-card overlay (happy path)', () => {
    render(
      <ReplayBoard
        board={board([{ value: 7, face_up: false }])}
        name="bootstrap"
        isActing={false}
        revealHidden
      />,
    )

    expect(screen.getByLabelText('card 7')).toBeInTheDocument()
    expect(screen.getByTitle(/Still face-down in the real game/)).toBeInTheDocument()
  })

  it('does not mark a genuinely face-up card, even with the toggle on (bad path)', () => {
    const { container } = render(
      <ReplayBoard board={board([{ value: 7, face_up: true }])} name="bootstrap" isActing={false} revealHidden />,
    )

    expect(container.querySelector('.gr-secret-card')).not.toBeInTheDocument()
  })

  it('does not reveal or mark a face-down card when the toggle is off (bad path)', () => {
    const { container } = render(
      <ReplayBoard
        board={board([{ value: 7, face_up: false }])}
        name="bootstrap"
        isActing={false}
        revealHidden={false}
      />,
    )

    expect(screen.getByLabelText('face-down card')).toBeInTheDocument()
    expect(container.querySelector('.gr-secret-card')).not.toBeInTheDocument()
  })

  it('does not crash on a cleared (null) slot (bad path)', () => {
    const { container } = render(<ReplayBoard board={board([null])} name="bootstrap" isActing={false} revealHidden />)

    expect(screen.getByLabelText('cleared slot')).toBeInTheDocument()
    expect(container.querySelector('.gr-secret-card')).not.toBeInTheDocument()
  })

  it('shows the "Acting" chip only for the acting seat (happy path)', () => {
    render(<ReplayBoard board={board([])} name="bootstrap" isActing revealHidden />)
    expect(screen.getByText('Acting')).toBeInTheDocument()
  })
})
