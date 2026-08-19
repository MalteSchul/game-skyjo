"""AlphaZero-style MCTS + self-play training for Skyjo.

Everything here operates on the full `GameState` from `skyjo.domain.engine`
rather than the redacted `Turn` a `Bot` sees: MCTS needs to branch/simulate
forward, which requires knowing hidden card values. That makes this a
perfect-information ("cheating") search over the true state, a standard
bootstrap simplification for AlphaZero applied to hidden-information games -
it does not yet model opponents' uncertainty about our hand.

Player count is fixed for the lifetime of a match (Skyjo never adds/removes
players mid-game), so `N_act` is constant across an entire MCTS tree/episode.
"""
