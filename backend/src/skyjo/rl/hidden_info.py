"""Redacts `GameState` for search, and resolves Skyjo's chance events (stock
draws, card reveals, the end-of-round reveal-everything) by sampling from the
pool of cards that are still genuinely unknown - never by reading their true
value.

Skyjo's hidden information is symmetric: nobody, not even a card's own owner,
knows a face-down value (see `domain.observation`). So every unresolved card
- whichever player's board it's on, or wherever it sits in the stock - is
modeled as an exchangeable draw from one shared pool: the deck's full
composition minus whatever is genuinely public right now (every face-up
board card, plus the *entire* discard pile - not just its top, since every
discard is a public event). That symmetry is what makes chance-node sampling
exact for a snapshot with no move history, without needing per-player belief
tracking the way genuinely asymmetric hidden information (e.g. poker's hole
cards) would require.

One deliberate simplification: this pool treats "in the stock" and "hidden
on some board" as one undifferentiated pool, even right after a reshuffle
(where the new stock's composition is briefly *more* narrowly known than the
general pool, to an observer who tracked full discard history). `Turn` is a
snapshot, not a log - it doesn't carry that history either - so this models
an agent whose belief state is exactly "what the current Turn shows," which
is the same modeling boundary `Observation`/`Turn` already draw elsewhere.

`GameState` objects handled by this module are always in one of two states:
freshly converted from a `Turn` (see `gamestate_from_turn`) or the result of
`rescrub`, applied after every transition. Both leave every face-down card
holding `HIDDEN_SENTINEL` rather than a real value - so even a bug that reads
a face-down `.value` before it's resolved produces an obviously-invalid card
(-999) instead of silently leaking the truth.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from skyjo.domain.deck import CARD_COUNTS
from skyjo.domain.engine import (
    Action,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
)
from skyjo.domain.observation import BoardView, Turn

# Outside the real card range [-2, 12], so a leaked sentinel is obviously
# wrong rather than a plausible-looking card.
HIDDEN_SENTINEL = -999

_REVEAL_TYPES = (ActionType.FLIP_INITIAL, ActionType.DISCARD_AND_REVEAL, ActionType.DRAW_STOCK)
_CLOSING_TYPES = (ActionType.PLACE, ActionType.DISCARD_AND_REVEAL)


def unknown_card_counts(turn: Turn) -> Counter[int]:
    """Deck composition minus every card whose value is currently public.

    Public = every face-up board card (across all players) + the entire
    discard pile + the drawn card in hand, if any. The drawn card matters:
    it's already been resolved to a real value (via its own earlier reveal),
    so it is genuinely known - but it sits in `turn.drawn_card`, not in
    `boards` or `discard`, until it's placed or discarded. Omitting it here
    let the same value be sampled again for some other still-hidden card,
    minting a phantom duplicate that only trips the conservation assert once
    enough of them accumulate for one value to exceed the deck's true count.
    Everything else - remaining stock, still-face-down board cards - is, by
    conservation, exactly what's left; see module docstring for why they're
    pooled together rather than tracked separately.
    """
    return _unknown_from_boards_and_discard(turn.boards, turn.discard, turn.drawn_card)


def _unknown_from_boards_and_discard(
    boards: tuple[BoardView, ...], discard: tuple[int, ...], drawn_card: int | None = None
) -> Counter[int]:
    counts: Counter[int] = Counter(dict(CARD_COUNTS))
    for board in boards:
        for card in board.cards:
            if card is not None and card.value is not None:
                counts[card.value] -= 1
    for value in discard:
        counts[value] -= 1
    if drawn_card is not None:
        counts[drawn_card] -= 1

    assert all(count >= 0 for count in counts.values()), (
        "_unknown_from_boards_and_discard: a card is counted as both public and "
        "still-unknown - board/discard bookkeeping is inconsistent"
    )
    return counts


def sample_reveal(counts: Counter[int], rng: np.random.Generator) -> int:
    values = [value for value, count in counts.items() if count > 0]
    if not values:
        raise ValueError("sample_reveal: no unknown cards remain to sample from")
    weights = np.array([counts[v] for v in values], dtype=np.float64)
    probs = weights / weights.sum()
    index = rng.choice(len(values), p=probs)
    return values[index]


def is_reveal(state: GameState, action: Action) -> bool:
    """Whether resolving `action` requires sampling a previously-unknown value.

    True for FLIP_INITIAL/DISCARD_AND_REVEAL/DRAW_STOCK unconditionally, and
    for PLACE only when its target position is still face-down - the old
    card sitting there becomes the new discard top either way, so a hidden
    one must be resolved before it's exposed. DRAW_DISCARD and PLACE onto an
    already-face-up position never reveal anything new.
    """
    if action.type in _REVEAL_TYPES:
        return True
    if action.type is ActionType.PLACE:
        card = state.boards[state.current_player].cards[action.position]
        return card is not None and not card.face_up
    return False


def will_close_round(state: GameState, action: Action) -> bool:
    """Whether `action` is the last-awaiting player's final turn.

    Mirrors `engine._finish_turn`'s own closing condition, computable purely
    from public fields (finisher, players_awaiting_final_turn, current
    player) - no hidden info needed to predict it in advance.
    """
    if action.type not in _CLOSING_TYPES:
        return False
    return state.finisher is not None and state.players_awaiting_final_turn == frozenset({state.current_player})


def resolve_reveal(state: GameState, action: Action, value: int) -> GameState:
    """Primes the board position `action` is about to reveal with `value`,
    so the real engine transition resolves it to `value` instead of whatever
    placeholder sits there. DRAW_STOCK has no single slot to prime this way
    (an empty stock reshuffles inside `apply_action` itself) - see
    `resolve_drawn_stock_card` for how that one is handled instead.
    """
    if action.type not in (ActionType.FLIP_INITIAL, ActionType.DISCARD_AND_REVEAL, ActionType.PLACE):
        raise AssertionError(f"resolve_reveal: {action.type} is not resolved this way")
    return _with_board_card_value(state, state.current_player, action.position, value)


def resolve_drawn_stock_card(state_after_draw: GameState, value: int) -> GameState:
    """Post-hoc fixup for DRAW_STOCK: `apply_action` already popped a
    placeholder value (and, if the stock was empty, already reshuffled the -
    still fully public - discard pile into a fresh stock via OS randomness,
    since redacted states always carry `reshuffle_seed=None`). We only need
    to swap the resulting `drawn_card` in for the properly sampled one.
    """
    return replace(state_after_draw, drawn_card=value)


def resolve_round_close(
    state: GameState, rng: np.random.Generator, *, already_resolved: tuple[int, int] | None = None
) -> GameState:
    """Bulk-resolves every still-hidden board card, needed before an action
    that will trigger `engine._score_and_close_round` - it reveals everyone's
    remaining cards at once to compute round scores, so they all need a
    plausible value first. `already_resolved`, if given, is a (player,
    position) slot some earlier `resolve_reveal` call already primed on this
    same action; it's excluded from both the sample pool and the refill so
    it isn't redrawn out from under that value.
    """
    boards_as_views = tuple(BoardView.from_board(b) for b in state.boards)
    counts = _unknown_from_boards_and_discard(boards_as_views, state.discard, state.drawn_card)
    if already_resolved is not None:
        player, position = already_resolved
        resolved_value = state.boards[player].cards[position].value
        counts = counts.copy()
        counts[resolved_value] -= 1

    pool = [value for value, count in counts.items() for _ in range(max(count, 0))]
    rng.shuffle(pool)
    draw = iter(pool)

    new_boards = []
    for player, board in enumerate(state.boards):
        new_cards = []
        for position, card in enumerate(board.cards):
            if card is not None and not card.face_up and (player, position) != already_resolved:
                card = replace(card, value=next(draw))
            new_cards.append(card)
        new_boards.append(replace(board, cards=tuple(new_cards)))
    return replace(state, boards=tuple(new_boards))


def rescrub(state: GameState) -> GameState:
    """Re-blanks every still-face-down card and the stock back to the hidden
    sentinel, and clears `reshuffle_seed`. Idempotent, and safe to call after
    *any* transition unconditionally: it only ever touches fields that are
    still secret (face_up=False cards, stock contents), never a value that
    was legitimately just resolved (those are face_up=True by the time this
    runs) or genuinely public fields (discard, scores, phase, ...).
    """
    boards = tuple(_rescrub_board(board) for board in state.boards)
    stock = (HIDDEN_SENTINEL,) * len(state.stock)
    return replace(state, boards=boards, stock=stock, reshuffle_seed=None)


def _rescrub_board(board: PlayerBoard) -> PlayerBoard:
    cards = tuple(
        card if card is None or card.face_up else Card(value=HIDDEN_SENTINEL, face_up=False) for card in board.cards
    )
    return replace(board, cards=cards)


def _with_board_card_value(state: GameState, player: int, position: int, value: int) -> GameState:
    board = state.boards[player]
    new_card = replace(board.cards[position], value=value)
    new_cards = board.cards[:position] + (new_card,) + board.cards[position + 1 :]
    new_board = replace(board, cards=new_cards)
    return replace(state, boards=state.boards[:player] + (new_board,) + state.boards[player + 1 :])


def gamestate_from_turn(turn: Turn) -> GameState:
    """Builds the redacted `GameState` a search tree is rooted at: everything
    `Turn` reports stays as-is (all genuinely public), and everything it
    redacts to `None` becomes `HIDDEN_SENTINEL` instead, so `apply_action`'s
    real transition logic can still run - it never actually reads a
    face-down `.value` except at the moment this module resolves it.
    """
    boards = tuple(_board_from_view(view) for view in turn.boards)
    return GameState(
        boards=boards,
        stock=(HIDDEN_SENTINEL,) * turn.stock_count,
        discard=turn.discard,
        current_player=turn.acting_player,
        drawn_card=turn.drawn_card,
        finisher=turn.finisher,
        players_awaiting_final_turn=turn.players_awaiting_final_turn,
        round_scores=None,
        total_scores=turn.total_scores,
        phase=turn.phase,
        reshuffle_seed=None,
        target_score=turn.target_score,
    )


def _board_from_view(view: BoardView) -> PlayerBoard:
    cards = tuple(
        None
        if card_view is None
        else Card(value=card_view.value, face_up=True)
        if card_view.face_up
        else Card(value=HIDDEN_SENTINEL, face_up=False)
        for card_view in view.cards
    )
    return PlayerBoard(cards=cards)
