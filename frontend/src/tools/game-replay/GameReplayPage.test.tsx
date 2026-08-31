import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import GameReplayPage from './GameReplayPage'
import type { GameReplay } from './types'

function boardState(overrides: Partial<GameReplay['decisions'][number]['board_state']> = {}) {
  return {
    boards: [{ cards: [] }, { cards: [] }],
    stock: [1, 2, 3],
    discard: [5],
    current_player: 0,
    drawn_card: null,
    finisher: null,
    players_awaiting_final_turn: [],
    round_scores: null,
    total_scores: [0, 0],
    phase: 'awaiting_draw',
    reshuffle_seed: 1,
    target_score: 100,
    ...overrides,
  }
}

const REPLAY: GameReplay = {
  schema_version: 1,
  seat_names: ['bootstrap', 'iter5'],
  checkpoint_paths: ['a.pt', 'b.pt'],
  seed: 0,
  num_simulations: 100,
  c_puct: 1.5,
  final_total_scores: [80, 95],
  final_ranks: [0, 1],
  winner_name: 'bootstrap',
  rounds_played: 1,
  decisions: [
    {
      step: 0,
      actor_seat: 0,
      actor_name: 'bootstrap',
      phase: 'awaiting_draw',
      total_scores: [0, 0],
      board_state: boardState(),
      raw_policy_priors: [{ action: { type: 'DRAW_STOCK', position: null }, prior: 1 }],
      raw_prior_favorite: { type: 'DRAW_STOCK', position: null },
      raw_rank_probs: [[0.5, 0.5], [0.5, 0.5]],
      raw_points_pred: [0, 0],
      mcts_num_simulations_requested: 100,
      mcts_visit_counts: [{ action: { type: 'DRAW_STOCK', position: null }, visit_count: 100 }],
      mcts_root_value: [0, 0],
      reused_tree_visits: 0,
      chosen_action: { type: 'DRAW_STOCK', position: null },
      search_overrode_prior: false,
      heuristic_action: { type: 'DRAW_STOCK', position: null },
      heuristic_action_representative: { type: 'DRAW_STOCK', position: null },
    },
    {
      step: 1,
      actor_seat: 1,
      actor_name: 'iter5',
      phase: 'awaiting_placement',
      total_scores: [0, 0],
      board_state: boardState({ current_player: 1 }),
      raw_policy_priors: [{ action: { type: 'PLACE', position: 0 }, prior: 0.4 }],
      raw_prior_favorite: { type: 'PLACE', position: 0 },
      raw_rank_probs: [[0.4, 0.6], [0.6, 0.4]],
      raw_points_pred: [0, 0],
      mcts_num_simulations_requested: 100,
      mcts_visit_counts: [{ action: { type: 'PLACE', position: 1 }, visit_count: 100 }],
      mcts_root_value: [0, 0],
      reused_tree_visits: 0,
      chosen_action: { type: 'PLACE', position: 1 },
      search_overrode_prior: true,
      heuristic_action: { type: 'PLACE', position: 0 },
      heuristic_action_representative: { type: 'PLACE', position: 0 },
    },
  ],
}

function uploadFile(input: HTMLElement, contents: string, name = 'game.json') {
  const file = new File([contents], name, { type: 'application/json' })
  fireEvent.change(input, { target: { files: [file] } })
}

afterEach(() => {
  document.body.classList.remove('mcts-tools-page')
})

describe('GameReplayPage', () => {
  it('loads a replay file and shows the first decision (happy path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(REPLAY))

    expect(await screen.findByText('bootstrap vs iter5')).toBeInTheDocument()
    expect(screen.getByText('step 0 / 1')).toBeInTheDocument()
    expect(screen.getByText('1 where search overrode the raw prior')).toBeInTheDocument()
  })

  it('steps to the next decision when the scrubber changes (happy path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(REPLAY))
    await screen.findByText('step 0 / 1')

    fireEvent.change(screen.getByLabelText('Decision step'), { target: { value: '1' } })

    expect(screen.getByText('step 1 / 1')).toBeInTheDocument()
    expect(screen.getByText('search overrode prior')).toBeInTheDocument()
  })

  it('restricts override markers to one seat when that seat is selected in the toggle (happy path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(REPLAY))
    await screen.findByText('step 0 / 1')

    // Only decision 1 (iter5) overrode the prior in this fixture — selecting
    // the other seat (bootstrap) should hide that marker entirely.
    fireEvent.click(screen.getByRole('button', { name: 'bootstrap' }))
    expect(screen.queryByRole('button', { name: /Jump to step/ })).not.toBeInTheDocument()
    expect(screen.getByText('0 where search overrode the raw prior (bootstrap only)')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'iter5' }))
    expect(screen.getByRole('button', { name: /Jump to step 1/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Both' }))
    expect(screen.getByText('1 where search overrode the raw prior')).toBeInTheDocument()
  })

  it('shows heuristic-diff markers only once the checkbox is checked (happy path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(REPLAY))
    await screen.findByText('step 0 / 1')

    const checkbox = screen.getByRole('checkbox', { name: /Show diffs from heuristic/ })
    expect(checkbox).not.toBeChecked()
    expect(screen.queryByRole('button', { name: /differs from the heuristic/ })).not.toBeInTheDocument()

    fireEvent.click(checkbox)

    expect(screen.getByRole('button', { name: /Jump to step 1.*differs from the heuristic/ })).toBeInTheDocument()
    expect(screen.getByText('1 differ from the heuristic', { exact: false })).toBeInTheDocument()
  })

  it('shows tau-sampling-deviation markers only once that checkbox is checked (happy path)', async () => {
    // Needs its own fixture: REPLAY's decisions each have a single-action
    // mcts_visit_counts, so there's no "different top pick" to sample away
    // from - and hasTrainingSelfPlayData needs pi_target present somewhere.
    const replayWithSampling: GameReplay = {
      ...REPLAY,
      decisions: [
        REPLAY.decisions[0],
        {
          ...REPLAY.decisions[1],
          mcts_visit_counts: [
            { action: { type: 'PLACE', position: 1 }, visit_count: 90 },
            { action: { type: 'PLACE', position: 0 }, visit_count: 10 },
          ],
          chosen_action: { type: 'PLACE', position: 0 },
          pi_target: [
            { action: { type: 'PLACE', position: 1 }, prior: 0.9 },
            { action: { type: 'PLACE', position: 0 }, prior: 0.1 },
          ],
        },
      ],
    }
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(replayWithSampling))
    await screen.findByText('step 0 / 1')

    const checkbox = screen.getByRole('checkbox', { name: /Show tau-sampling deviations/ })
    expect(checkbox).not.toBeChecked()
    expect(screen.queryByRole('button', { name: /search itself most favored/ })).not.toBeInTheDocument()

    fireEvent.click(checkbox)

    expect(screen.getByRole('button', { name: /Jump to step 1.*search itself most favored/ })).toBeInTheDocument()
    expect(screen.getByText('1 tau-sampling deviations', { exact: false })).toBeInTheDocument()
  })

  it('hides the tau-sampling-deviation checkbox for an eval-style replay with no training self-play data (bad path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(REPLAY))
    await screen.findByText('step 0 / 1')

    expect(screen.queryByRole('checkbox', { name: /Show tau-sampling deviations/ })).not.toBeInTheDocument()
  })

  it('hides the heuristic checkbox entirely for a replay with no heuristic data (bad path)', async () => {
    const replayWithoutHeuristic: GameReplay = {
      ...REPLAY,
      decisions: REPLAY.decisions.map((d) => ({ ...d, heuristic_action_representative: undefined })),
    }
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify(replayWithoutHeuristic))
    await screen.findByText('step 0 / 1')

    expect(screen.queryByRole('checkbox', { name: /Show diffs from heuristic/ })).not.toBeInTheDocument()
  })

  it('shows the JSON parse error inline instead of crashing on malformed input (sad path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), '{not valid json')

    expect(await screen.findByRole('alert')).toHaveTextContent("isn't valid JSON")
  })

  it('shows a schema error for valid JSON that has no decisions (bad path)', async () => {
    render(<GameReplayPage />)
    uploadFile(screen.getByLabelText('Load game replay file'), JSON.stringify({ decisions: [] }))

    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't find a game replay")
  })
})
