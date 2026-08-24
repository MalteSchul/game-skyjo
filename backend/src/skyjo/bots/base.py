"""Bot interface: anything that can play a turn from a Turn view.

A Bot only ever sees a Turn, never GameState - the same redacted information
a human player has. That keeps a bot's decision from accidentally depending
on information it shouldn't have, and lets any Bot slot into the match store
or a future Gym-style training loop unmodified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from skyjo.domain.engine import Action
from skyjo.domain.observation import Turn

# Called with a fraction in [0, 1] while a slow bot (e.g. MCTS) is still
# deciding, so the API can surface live progress to a polling client.
ProgressReporter = Callable[[float], None]


class Bot(Protocol):
    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action: ...


@runtime_checkable
class ObservesActions(Protocol):
    """Optional `Bot` extension for one that keeps state across turns (e.g.
    `MctsBot`'s cached search tree) and needs to track the match's actual
    path to keep that state valid - not just its own turns, since another
    seat's action (human or bot) advances the real game just the same.

    `MatchStore` calls `observe_transition` on every seat's bot after every
    action actually applied to a match - including the acting bot's own -
    via an `isinstance` check against this Protocol (`@runtime_checkable`
    makes that work without every `Bot` needing a no-op override). A
    stateless bot (`HeuristicBot`, `RandomBot`) simply never implements it.
    """

    def observe_transition(self, turn_before: Turn, action: Action, turn_after: Turn) -> None: ...


@runtime_checkable
class ExposesSearchTree(Protocol):
    """Optional `Bot` extension for one that can export the search tree(s)
    behind its most recent decision, e.g. `MctsBot` (see
    `rl.tree_export.tree_to_dict`) - for a "why did it pick this" UI, not
    anything the bot itself needs. A bot with no real search (`HeuristicBot`,
    `RandomBot`) simply never implements it.
    """

    def last_tree_snapshots(self) -> dict[str, dict] | None: ...
