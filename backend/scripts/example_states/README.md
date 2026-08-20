# Example GameStates

Checked-in `GameState` JSON files (see `domain.state_json`) for feeding into
`scripts/dump_mcts_tree.py --state-file` without hand-crafting or
regenerating a position each time.

## midgame_awaiting_draw.json

2-player match, both players finished their two initial flips, player 0 to
act (`awaiting_draw`). Generated with:

```python
from skyjo.domain.engine import new_match, apply_action, legal_actions, ActionType

state = new_match(player_count=2, seed=5)
while state.phase == "initial_flip":
    action = next(a for a in legal_actions(state) if a.type is ActionType.FLIP_INITIAL)
    state = apply_action(state, action)
```

Covered by `tests/test_state_json.py::test_example_fixture_midgame_awaiting_draw_loads_correctly`,
which fails if a `GameState`/`state_json` schema change makes this file stale.
