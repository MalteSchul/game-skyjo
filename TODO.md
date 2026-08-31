# TODO

## ~~Self-play decision loops (repeated discard/swap cycles)~~ - resolved

Bots got stuck repeating the same decision for hundreds of game-states: take
the discard, swap it for a same-or-near-value (±1) card on the board, discard
that; the opposing bot mirrors it. Root-caused to `rl_run_selfplay_v6`'s
`self_play/avg_moves_per_game` roughly doubling starting at iteration 40
(~700-800 -> ~1500-1950).

Fixed at the rules level, not the policy level: `engine.legal_actions()` now
refuses to place a card drawn from the discard pile onto a face-up card of
equal or lower value, unless doing so completes that column's clear (see
`GameState.drawn_card_source` and the `_would_clear_column` check in
`backend/src/skyjo/domain/engine.py`). A discard swap can now only ever
strictly improve a face-up slot, which bounds how many times any given slot
can be "worked on" instead of just vetoing exact repeats - and it applies
uniformly to every caller (humans, `HeuristicBot`, self-play, `MctsBot`) since
they all go through `legal_actions()`.
