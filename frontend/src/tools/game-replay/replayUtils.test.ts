import { describe, expect, it } from 'vitest'
import {
  actionLabel,
  buildPolicyComparison,
  buildValueComparison,
  buildWinProbSeries,
  chosenDiffersFromTopVisit,
  chosenDiffersFromTopVisitStepIndices,
  hasHeuristicData,
  hasTrainingSelfPlayData,
  heuristicDiffStepIndices,
  overrideStepIndices,
  sameAction,
  searchOverrodeRawPrior,
} from './replayUtils'
import type { DecisionReplay, GameReplay } from './types'

function decision(overrides: Partial<DecisionReplay> = {}): DecisionReplay {
  return {
    step: 0,
    actor_seat: 0,
    actor_name: 'p0',
    phase: 'awaiting_draw',
    total_scores: [0, 0],
    board_state: {
      boards: [{ cards: [] }, { cards: [] }],
      stock: [],
      discard: [],
      current_player: 0,
      drawn_card: null,
      finisher: null,
      players_awaiting_final_turn: [],
      round_scores: null,
      total_scores: [0, 0],
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
    raw_points_pred: [0, 0],
    mcts_num_simulations_requested: 10,
    mcts_visit_counts: [
      { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 7 },
      { action: { type: 'DRAW_STOCK', position: null }, visit_count: 3 },
    ],
    mcts_root_value: [0, 0],
    reused_tree_visits: 0,
    chosen_action: { type: 'DRAW_DISCARD', position: null },
    search_overrode_prior: true,
    ...overrides,
  }
}

describe('actionLabel', () => {
  it('renders known action types with position where relevant (happy path)', () => {
    expect(actionLabel({ type: 'DRAW_STOCK', position: null })).toBe('Draw stock')
    expect(actionLabel({ type: 'PLACE', position: 3 })).toBe('Place → slot 3')
  })

  it('falls back to a generic label for an unknown action type (bad path)', () => {
    expect(actionLabel({ type: 'SOMETHING_NEW', position: 2 })).toBe('SOMETHING_NEW @ 2')
    expect(actionLabel({ type: 'SOMETHING_NEW', position: null })).toBe('SOMETHING_NEW')
  })
})

describe('sameAction', () => {
  it('compares by type and position, not object identity (happy path)', () => {
    expect(sameAction({ type: 'PLACE', position: 1 }, { type: 'PLACE', position: 1 })).toBe(true)
    expect(sameAction({ type: 'PLACE', position: 1 }, { type: 'PLACE', position: 2 })).toBe(false)
  })
})

describe('buildWinProbSeries', () => {
  it('builds one series per seat containing only that seat\'s own decisions (happy path)', () => {
    const replay: GameReplay = {
      schema_version: 1,
      seat_names: ['a', 'b'],
      checkpoint_paths: ['a.pt', 'b.pt'],
      seed: 0,
      num_simulations: 10,
      c_puct: 1.5,
      final_total_scores: null,
      final_ranks: null,
      winner_name: null,
      rounds_played: 0,
      decisions: [
        decision({ step: 0, actor_seat: 0, raw_rank_probs: [[0.6, 0.4], [0.4, 0.6]] }),
        decision({ step: 1, actor_seat: 1, raw_rank_probs: [[0.3, 0.7], [0.7, 0.3]] }),
      ],
    }

    const series = buildWinProbSeries(replay)

    expect(series).toHaveLength(2)
    // raw_rank_probs[i][0] = P(player i finishes at rank 0, i.e. wins) - not
    // the acting player's own perspective, so this reads row `seat`, not row
    // `actor_seat` (they're equal here only because each decision's own
    // actor is the seat whose series it belongs to).
    expect(series[0].points).toEqual([{ step: 0, p: 0.6 }])
    expect(series[1].points).toEqual([{ step: 1, p: 0.7 }])
  })
})

describe('buildPolicyComparison', () => {
  it('pairs prior and visit share by action, sorted by visit share descending (happy path)', () => {
    const rows = buildPolicyComparison(decision())

    expect(rows.map((r) => r.label)).toEqual(['Draw discard', 'Draw stock'])
    expect(rows[0].visitShare).toBeCloseTo(0.7)
    expect(rows[0].priorShare).toBeCloseTo(0.3)
    expect(rows[1].visitShare).toBeCloseTo(0.3)
    expect(rows[1].priorShare).toBeCloseTo(0.7)
  })

  it('includes an action with a prior but zero search visits (bad path: search never explored it)', () => {
    const d = decision({
      raw_policy_priors: [
        { action: { type: 'DRAW_STOCK', position: null }, prior: 0.9 },
        { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.1 },
      ],
      mcts_visit_counts: [{ action: { type: 'DRAW_STOCK', position: null }, visit_count: 10 }],
    })

    const rows = buildPolicyComparison(d)

    const discardRow = rows.find((r) => r.label === 'Draw discard')
    expect(discardRow?.visitShare).toBe(0)
    expect(discardRow?.priorShare).toBeCloseTo(0.1)
  })

  it('leaves noisedPriorShare/piShare undefined for an eval-style decision with no self-play detail (bad path)', () => {
    const rows = buildPolicyComparison(decision())

    expect(rows[0].noisedPriorShare).toBeUndefined()
    expect(rows[0].piShare).toBeUndefined()
  })

  it('adds noisedPriorShare/piShare, defaulting to 0 for an action missing from one distribution, when self-play detail is present (happy path)', () => {
    const d = decision({
      raw_policy_priors: [
        { action: { type: 'DRAW_STOCK', position: null }, prior: 0.9 },
        { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.1 },
      ],
      dirichlet_noised_priors: [{ action: { type: 'DRAW_STOCK', position: null }, prior: 0.6 }],
      pi_target: [{ action: { type: 'DRAW_DISCARD', position: null }, prior: 1.0 }],
    })

    const rows = buildPolicyComparison(d)

    const stockRow = rows.find((r) => r.label === 'Draw stock')
    const discardRow = rows.find((r) => r.label === 'Draw discard')
    expect(stockRow?.noisedPriorShare).toBeCloseTo(0.6)
    expect(stockRow?.piShare).toBe(0)
    expect(discardRow?.noisedPriorShare).toBe(0)
    expect(discardRow?.piShare).toBeCloseTo(1.0)
  })
})

describe('buildValueComparison', () => {
  it('pairs initial and final Q-values by action, sorted by final value descending (happy path)', () => {
    const d = decision({
      initial_action_values: [
        { action: { type: 'DRAW_STOCK', position: null }, value: -0.2 },
        { action: { type: 'DRAW_DISCARD', position: null }, value: 0.1 },
      ],
      final_action_values: [
        { action: { type: 'DRAW_STOCK', position: null }, value: 0.5 },
        { action: { type: 'DRAW_DISCARD', position: null }, value: 0.1 },
      ],
    })

    const rows = buildValueComparison(d)

    expect(rows.map((r) => r.label)).toEqual(['Draw stock', 'Draw discard'])
    expect(rows[0]).toMatchObject({ initialValue: -0.2, finalValue: 0.5 })
    expect(rows[1]).toMatchObject({ initialValue: 0.1, finalValue: 0.1 })
  })

  it('keeps negative values as-is rather than treating them as missing (happy path)', () => {
    const d = decision({
      initial_action_values: [{ action: { type: 'DRAW_STOCK', position: null }, value: -0.7 }],
      final_action_values: [{ action: { type: 'DRAW_STOCK', position: null }, value: -0.9 }],
    })

    const rows = buildValueComparison(d)

    expect(rows).toEqual([{ action: { type: 'DRAW_STOCK', position: null }, label: 'Draw stock', initialValue: -0.7, finalValue: -0.9 }])
  })

  it('defaults a missing side to 0 when an action appears in only one of the two lists (bad path)', () => {
    const d = decision({
      initial_action_values: [{ action: { type: 'DRAW_STOCK', position: null }, value: 0.3 }],
      final_action_values: [],
    })

    const rows = buildValueComparison(d)

    expect(rows).toEqual([{ action: { type: 'DRAW_STOCK', position: null }, label: 'Draw stock', initialValue: 0.3, finalValue: 0 }])
  })

  it('returns an empty list, not zeros, for a replay exported before these fields existed (bad path)', () => {
    expect(buildValueComparison(decision())).toEqual([])
  })
})

describe('searchOverrodeRawPrior', () => {
  it('is true when search\'s most-visited action differs from, and out-shares, the raw favorite (happy path)', () => {
    // defaults: DRAW_DISCARD gets 7/10 visits (70%) vs its own 30% raw
    // prior, while the raw favorite DRAW_STOCK only got 3/10 visits.
    expect(searchOverrodeRawPrior(decision())).toBe(true)
  })

  it('is false when search\'s most-visited action already is the raw favorite (bad path)', () => {
    const d = decision({
      mcts_visit_counts: [
        { action: { type: 'DRAW_STOCK', position: null }, visit_count: 7 },
        { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 3 },
      ],
    })
    expect(searchOverrodeRawPrior(d)).toBe(false)
  })

  it('ignores chosen_action entirely - a tau-sample landing on a low-probability action is not an override (bad path)', () => {
    const d = decision({
      mcts_visit_counts: [
        { action: { type: 'DRAW_STOCK', position: null }, visit_count: 9 },
        { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 1 },
      ],
      chosen_action: { type: 'DRAW_DISCARD', position: null },
    })
    expect(searchOverrodeRawPrior(d)).toBe(false)
  })

  it('is false when the top-visited action differs from the favorite but did not actually out-share its own raw prior (bad path)', () => {
    // Needs a third action to construct: with only two actions, a
    // non-favorite winning the visit-count race always implies it out-shared
    // its own (necessarily-minority) raw prior - the two conditions can only
    // diverge with >=3 actions splitting the raw prior mass. Here DRAW_STOCK
    // is the favorite (50%), but DRAW_DISCARD (45% raw prior) ends up the
    // most-visited action at only 40% share - down from its own prior,
    // because a third option (PLACE) plus DRAW_STOCK's own explored share
    // ate into it. Search topped the visit count with a different action
    // than the favorite, but didn't push its share above its own prior.
    const d = decision({
      raw_policy_priors: [
        { action: { type: 'DRAW_STOCK', position: null }, prior: 0.5 },
        { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.45 },
        { action: { type: 'PLACE', position: 0 }, prior: 0.05 },
      ],
      raw_prior_favorite: { type: 'DRAW_STOCK', position: null },
      mcts_visit_counts: [
        { action: { type: 'DRAW_STOCK', position: null }, visit_count: 30 },
        { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 40 },
        { action: { type: 'PLACE', position: 0 }, visit_count: 30 },
      ],
    })
    expect(searchOverrodeRawPrior(d)).toBe(false)
  })
})

describe('overrideStepIndices', () => {
  // decision()'s own defaults already put search's top-visited action
  // (DRAW_DISCARD, 7 visits, 70% share) at odds with the raw favorite
  // (DRAW_STOCK, 30% raw prior share) - a genuine override under the new
  // definition. To get a "no override" case, search's top pick has to
  // actually agree with the favorite - it's not enough to just change
  // chosen_action, since chosen_action (a tau-sample) no longer drives this
  // determination at all - that's the whole point of the fix.
  const NO_OVERRIDE_VISIT_COUNTS = [
    { action: { type: 'DRAW_STOCK', position: null }, visit_count: 7 },
    { action: { type: 'DRAW_DISCARD', position: null }, visit_count: 3 },
  ]

  it('lists indices where search itself chose differently from the raw prior (happy path)', () => {
    const replay: GameReplay = {
      schema_version: 1,
      seat_names: ['a', 'b'],
      checkpoint_paths: ['a.pt', 'b.pt'],
      seed: 0,
      num_simulations: 10,
      c_puct: 1.5,
      final_total_scores: null,
      final_ranks: null,
      winner_name: null,
      rounds_played: 0,
      decisions: [
        decision({ step: 0, mcts_visit_counts: NO_OVERRIDE_VISIT_COUNTS }),
        decision({ step: 1 }),
        decision({ step: 2, mcts_visit_counts: NO_OVERRIDE_VISIT_COUNTS }),
      ],
    }

    expect(overrideStepIndices(replay)).toEqual([1])
  })

  it('does not count a decision where only the tau-sampled chosen_action differs, not search\'s own top pick (bad path)', () => {
    // search's visit distribution agrees with the raw prior (DRAW_STOCK on
    // top both times), but a tau-sample still landed on the low-probability
    // DRAW_DISCARD - sampling variance, not an override.
    const replay: GameReplay = {
      schema_version: 1,
      seat_names: ['a', 'b'],
      checkpoint_paths: ['a.pt', 'b.pt'],
      seed: 0,
      num_simulations: 10,
      c_puct: 1.5,
      final_total_scores: null,
      final_ranks: null,
      winner_name: null,
      rounds_played: 0,
      decisions: [
        decision({
          mcts_visit_counts: NO_OVERRIDE_VISIT_COUNTS,
          chosen_action: { type: 'DRAW_DISCARD', position: null },
        }),
      ],
    }

    expect(overrideStepIndices(replay)).toEqual([])
  })

  it('restricts to one seat\'s own overrides when given a seat filter (happy path)', () => {
    const replay: GameReplay = {
      schema_version: 1,
      seat_names: ['iter5', 'bootstrap'],
      checkpoint_paths: ['a.pt', 'b.pt'],
      seed: 0,
      num_simulations: 10,
      c_puct: 1.5,
      final_total_scores: null,
      final_ranks: null,
      winner_name: null,
      rounds_played: 0,
      decisions: [
        decision({ step: 0, actor_seat: 0 }),
        decision({ step: 1, actor_seat: 1 }),
        decision({ step: 2, actor_seat: 0, mcts_visit_counts: NO_OVERRIDE_VISIT_COUNTS }),
      ],
    }

    expect(overrideStepIndices(replay, 0)).toEqual([0])
    expect(overrideStepIndices(replay, 1)).toEqual([1])
    expect(overrideStepIndices(replay, null)).toEqual([0, 1])
  })
})

describe('chosenDiffersFromTopVisit', () => {
  it('is true when the played action is not the most-visited one (happy path)', () => {
    // defaults: DRAW_DISCARD is the most-visited action (7 of 10), while
    // chosen_action is also DRAW_DISCARD by default - override to a
    // different, low-visit action to exercise a tau-sample landing elsewhere.
    const d = decision({ chosen_action: { type: 'DRAW_STOCK', position: null } })
    expect(chosenDiffersFromTopVisit(d)).toBe(true)
  })

  it('is false when the played action is the most-visited one, regardless of the raw prior (bad path)', () => {
    // This is deliberately independent of searchOverrodeRawPrior: here the
    // most-visited action (DRAW_DISCARD) differs from the raw favorite
    // (DRAW_STOCK, an override by that measure) but chosen_action still
    // matches what search itself settled on.
    const d = decision({ chosen_action: { type: 'DRAW_DISCARD', position: null } })
    expect(chosenDiffersFromTopVisit(d)).toBe(false)
  })
})

describe('chosenDiffersFromTopVisitStepIndices', () => {
  it('lists indices where the played action differs from search\'s own top pick, respecting the seat filter (happy path)', () => {
    const replay = replayOf([
      decision({ step: 0, actor_seat: 0, chosen_action: { type: 'DRAW_STOCK', position: null } }),
      decision({ step: 1, actor_seat: 1, chosen_action: { type: 'DRAW_DISCARD', position: null } }),
    ])

    expect(chosenDiffersFromTopVisitStepIndices(replay)).toEqual([0])
    expect(chosenDiffersFromTopVisitStepIndices(replay, 0)).toEqual([0])
    expect(chosenDiffersFromTopVisitStepIndices(replay, 1)).toEqual([])
  })
})

describe('hasTrainingSelfPlayData', () => {
  it('is true once at least one decision carries pi_target (happy path)', () => {
    const replay = replayOf([decision({ pi_target: [{ action: { type: 'DRAW_STOCK', position: null }, prior: 1 }] })])
    expect(hasTrainingSelfPlayData(replay)).toBe(true)
  })

  it('is false for an eval-style recording with no pi_target anywhere (bad path)', () => {
    const replay = replayOf([decision({ pi_target: undefined })])
    expect(hasTrainingSelfPlayData(replay)).toBe(false)
  })
})

function replayOf(decisions: DecisionReplay[]): GameReplay {
  return {
    schema_version: 1,
    seat_names: ['iter5', 'bootstrap'],
    checkpoint_paths: ['a.pt', 'b.pt'],
    seed: 0,
    num_simulations: 10,
    c_puct: 1.5,
    final_total_scores: null,
    final_ranks: null,
    winner_name: null,
    rounds_played: 0,
    decisions,
  }
}

describe('hasHeuristicData', () => {
  it('is true once at least one decision carries a heuristic_action_representative (happy path)', () => {
    expect(
      hasHeuristicData(replayOf([decision({ heuristic_action_representative: { type: 'DRAW_STOCK', position: null } })])),
    ).toBe(true)
  })

  it('is false for a replay exported before heuristic_action_representative existed (bad path)', () => {
    expect(
      hasHeuristicData(
        replayOf([
          decision({ heuristic_action_representative: undefined }),
          decision({ heuristic_action_representative: null }),
        ]),
      ),
    ).toBe(false)
  })
})

describe('heuristicDiffStepIndices', () => {
  it('lists indices where what was played differs from the heuristic\'s representative (happy path)', () => {
    const replay = replayOf([
      decision({
        step: 0,
        chosen_action: { type: 'DRAW_STOCK', position: null },
        heuristic_action_representative: { type: 'DRAW_STOCK', position: null },
      }),
      decision({
        step: 1,
        chosen_action: { type: 'DRAW_STOCK', position: null },
        heuristic_action_representative: { type: 'DRAW_DISCARD', position: null },
      }),
    ])

    expect(heuristicDiffStepIndices(replay)).toEqual([1])
  })

  it('compares against the representative, not the raw action, so equivalent picks do not count as a diff (happy path)', () => {
    // e.g. an initial-flip decision: the heuristic's raw pick (any real
    // slot) differs from chosen_action, but its representative - what the
    // net's own choice also collapses to - matches, so this is NOT a diff.
    const replay = replayOf([
      decision({
        step: 0,
        chosen_action: { type: 'FLIP_INITIAL', position: 0 },
        heuristic_action: { type: 'FLIP_INITIAL', position: 5 },
        heuristic_action_representative: { type: 'FLIP_INITIAL', position: 0 },
      }),
    ])

    expect(heuristicDiffStepIndices(replay)).toEqual([])
  })

  it('respects the seat filter the same way overrideStepIndices does (happy path)', () => {
    const replay = replayOf([
      decision({
        step: 0,
        actor_seat: 0,
        chosen_action: { type: 'DRAW_STOCK', position: null },
        heuristic_action_representative: { type: 'DRAW_DISCARD', position: null },
      }),
      decision({
        step: 1,
        actor_seat: 1,
        chosen_action: { type: 'DRAW_STOCK', position: null },
        heuristic_action_representative: { type: 'DRAW_DISCARD', position: null },
      }),
    ])

    expect(heuristicDiffStepIndices(replay, 0)).toEqual([0])
    expect(heuristicDiffStepIndices(replay, 1)).toEqual([1])
  })

  it('returns no indices for a replay with no heuristic data instead of throwing (bad path)', () => {
    const replay = replayOf([decision({ heuristic_action_representative: undefined })])
    expect(heuristicDiffStepIndices(replay)).toEqual([])
  })
})
