"""A bot with simple, fixed heuristics - no search, no lookahead.

Rules of thumb, in order of what each decision point cares about:

- Draw: take the discard if it's a low card (<= _LOW_CARD) or matches the
  value of one of our own face-up cards (progress towards a column clear);
  otherwise draw from stock, since an unknown card is better odds than a
  mediocre known one.
- Placement: swap the drawn card into our worst known face-up slot if that's
  an improvement; otherwise place it over a face-down card, since a known
  low/medium value beats an unknown one; discard-and-reveal only when every
  board slot is already face-up and would get worse.

This is meant to comfortably beat RandomBot, not to play well.
"""

from __future__ import annotations

import random

from skyjo.bots.base import ProgressReporter
from skyjo.domain.engine import Action, ActionType
from skyjo.domain.observation import Turn

_LOW_CARD = 3


class HeuristicBot:
    def __init__(self, seed: int | None = None) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("HeuristicBot: seed must be an int if provided")
        self._random = random.Random(seed)

    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action:
        if turn.phase == "initial_flip":
            return self._random.choice(turn.legal_actions)
        if turn.phase == "awaiting_draw":
            return self._choose_draw(turn)
        return self._choose_placement(turn)

    def _choose_draw(self, turn: Turn) -> Action:
        draw_discard = Action(ActionType.DRAW_DISCARD)
        if draw_discard not in turn.legal_actions:
            return Action(ActionType.DRAW_STOCK)

        own_face_up_values = {
            card.value
            for card in turn.boards[turn.acting_player].cards
            if card is not None and card.face_up
        }
        assert turn.discard_top is not None
        if turn.discard_top <= _LOW_CARD or turn.discard_top in own_face_up_values:
            return draw_discard
        return Action(ActionType.DRAW_STOCK)

    def _choose_placement(self, turn: Turn) -> Action:
        assert turn.drawn_card is not None
        board = turn.boards[turn.acting_player].cards

        worst_position = None
        worst_value = None
        face_down_position = None
        for i, card in enumerate(board):
            if card is None:
                continue
            if card.face_up:
                if worst_value is None or card.value > worst_value:
                    worst_position = i
                    worst_value = card.value
            elif face_down_position is None:
                face_down_position = i

        if worst_value is not None and turn.drawn_card < worst_value:
            return Action(ActionType.PLACE, position=worst_position)
        if face_down_position is not None:
            return Action(ActionType.PLACE, position=face_down_position)
        # Board is entirely face-up and the drawn card wouldn't improve it -
        # nothing left to do but place it somewhere.
        return Action(ActionType.PLACE, position=worst_position)
