import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import DecisionDetail from './DecisionDetail'
import type { DecisionReplay } from './types'

function decision(overrides: Partial<DecisionReplay> = {}): DecisionReplay {
  return {
    step: 4,
    actor_seat: 1,
    actor_name: 'iter5_seat1',
    phase: 'awaiting_draw',
    total_scores: [10, 20],
    board_state: {
      boards: [{ cards: [] }, { cards: [] }],
      stock: [],
      discard: [],
      current_player: 1,
      drawn_card: null,
      finisher: null,
      players_awaiting_final_turn: [],
      round_scores: null,
      total_scores: [10, 20],
      phase: 'awaiting_draw',
      reshuffle_seed: 1,
      target_score: 100,
    },
    raw_policy_priors: [
      { action: { type: 'DRAW_STOCK', position: null }, prior: 0.7 },
      { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.3 },
    ],
    raw_prior_favorite: { type: 'DRAW_STOCK', position: null },
    raw_rank_probs: [[0.6, 0.4], [0.4, 0.6]],
    raw_points_pred: [0.1, -0.1],
    mcts_num_simulations_requested: 10,
    mcts_visit_counts: [
      { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 7 },
      { action: { type: 'DRAW_STOCK', position: null }, visit_count: 3 },
    ],
    mcts_root_value: [0.2, -0.2],
    reused_tree_visits: 0,
    chosen_action: { type: 'DRAW_DISCARD', position: null },
    search_overrode_prior: true,
    ...overrides,
  }
}

describe('DecisionDetail', () => {
  it('shows both the chosen action and the raw prior favorite, and flags the override (happy path)', () => {
    render(<DecisionDetail decision={decision()} seatNames={['bootstrap', 'iter5_seat1']} />)

    expect(screen.getByText('search overrode prior')).toBeInTheDocument()
    expect(screen.getByText('Draw discard', { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText('Draw stock', { selector: 'strong' })).toBeInTheDocument()
  })

  it('renders a policy row for every action either level considered (happy path)', () => {
    render(<DecisionDetail decision={decision()} seatNames={['bootstrap', 'iter5_seat1']} />)

    expect(screen.getAllByText(/Draw stock|Draw discard/).length).toBeGreaterThanOrEqual(2)
  })

  it('does not flag an override when search\'s own visit distribution agreed with the raw prior (bad path)', () => {
    // The override badge is driven by search's own top-visited action vs
    // the raw prior's favorite (searchOverrodeRawPrior), not by chosen_action
    // (a tau-sample that can differ from the favorite by pure chance) - so
    // getting a genuine "no override" case means making mcts_visit_counts
    // itself agree with raw_prior_favorite, not just changing chosen_action.
    render(
      <DecisionDetail
        decision={decision({
          chosen_action: { type: 'DRAW_STOCK', position: null },
          mcts_visit_counts: [
            { action: { type: 'DRAW_STOCK', position: null }, visit_count: 7 },
            { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 3 },
          ],
          search_overrode_prior: false,
        })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.queryByText('search overrode prior')).not.toBeInTheDocument()
  })

  it('shows what the heuristic would have played and notes it differs from both levels (happy path)', () => {
    render(
      <DecisionDetail
        decision={decision({
          heuristic_action: { type: 'PLACE', position: 2 },
          heuristic_action_representative: { type: 'PLACE', position: 2 },
        })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.getByText('Place → r1c3', { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText('(differs from both)', { selector: 'em' })).toBeInTheDocument()
  })

  it('notes when the heuristic matches what was actually played (happy path)', () => {
    render(
      <DecisionDetail
        decision={decision({
          heuristic_action: { type: 'DRAW_DISCARD', position: null },
          heuristic_action_representative: { type: 'DRAW_DISCARD', position: null },
        })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.getByText('(matches what was played)', { selector: 'em' })).toBeInTheDocument()
  })

  it('compares the representative, not the raw action - an equivalent-but-literally-different pick still counts as a match (happy path)', () => {
    // e.g. an initial-flip decision: the heuristic's raw pick (a real,
    // specific slot) differs from chosen_action, but its representative -
    // what the net's own choice also collapses to - matches.
    render(
      <DecisionDetail
        decision={decision({
          chosen_action: { type: 'FLIP_INITIAL', position: 0 },
          heuristic_action: { type: 'FLIP_INITIAL', position: 5 },
          heuristic_action_representative: { type: 'FLIP_INITIAL', position: 0 },
        })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.getByText('Flip r2c2', { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText('(matches what was played)', { selector: 'em' })).toBeInTheDocument()
  })

  it('says nothing about the heuristic for a replay exported before that field existed (bad path)', () => {
    render(
      <DecisionDetail
        decision={decision({ heuristic_action: undefined, heuristic_action_representative: undefined })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.queryByText(/Heuristic would play/)).not.toBeInTheDocument()
  })

  it('shows the noised-prior and pi columns plus tau/tied-group badges for a training self-play recording (happy path)', () => {
    render(
      <DecisionDetail
        decision={decision({
          dirichlet_noised_priors: [
            { action: { type: 'DRAW_STOCK', position: null }, prior: 0.6 },
            { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.4 },
          ],
          pi_target: [
            { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.95 },
            { action: { type: 'DRAW_STOCK', position: null }, prior: 0.05 },
          ],
          tau: 0.1,
          tied_group_size: 3,
        })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.getByText('+Dirichlet noise')).toBeInTheDocument()
    expect(screen.getByText('pi (tau=0.1)')).toBeInTheDocument()
    expect(screen.getByText('tau=0.1')).toBeInTheDocument()
    expect(screen.getByText('3 tied positions')).toBeInTheDocument()
  })

  it('falls back to the plain 3-column table for an eval-style recording with no self-play detail (bad path)', () => {
    render(<DecisionDetail decision={decision()} seatNames={['bootstrap', 'iter5_seat1']} />)

    expect(screen.queryByText('+Dirichlet noise')).not.toBeInTheDocument()
    expect(screen.queryByText(/^tau=/)).not.toBeInTheDocument()
    expect(screen.queryByText(/tied positions/)).not.toBeInTheDocument()
  })

  it('shows the initial-vs-final value comparison when both fields are present (happy path)', () => {
    render(
      <DecisionDetail
        decision={decision({
          initial_action_values: [
            { action: { type: 'DRAW_STOCK', position: null }, value: -0.2 },
            { action: { type: 'DRAW_DISCARD', position: null }, value: 0.1 },
          ],
          final_action_values: [
            { action: { type: 'DRAW_STOCK', position: null }, value: 0.5 },
            { action: { type: 'DRAW_DISCARD', position: null }, value: 0.1 },
          ],
        })}
        seatNames={['bootstrap', 'iter5_seat1']}
      />,
    )

    expect(screen.getByText(/How MCTS values each action/)).toBeInTheDocument()
    expect(screen.getByText('-0.20')).toBeInTheDocument()
    expect(screen.getByText('+0.50')).toBeInTheDocument()
  })

  it('hides the value comparison section for a replay exported before these fields existed (bad path)', () => {
    render(<DecisionDetail decision={decision()} seatNames={['bootstrap', 'iter5_seat1']} />)

    expect(screen.queryByText(/How MCTS values each action/)).not.toBeInTheDocument()
  })
})
