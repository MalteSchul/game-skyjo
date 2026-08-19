import random

# TODO: duplicated in frontend/src/game/deck.ts — no shared source of truth yet.
# Official Skyjo deck: 150 cards.
CARD_COUNTS: tuple[tuple[int, int], ...] = (
    (-2, 5),
    (-1, 10),
    (0, 15),
    *((value, 10) for value in range(1, 13)),
)

DECK_SIZE = sum(count for _, count in CARD_COUNTS)


def build_deck(seed: int | None = None) -> list[int]:
    """Builds and shuffles the standard 150-card Skyjo deck.

    Pass a seed for a reproducible shuffle (e.g. in tests); omit it for a random one.
    """
    if seed is not None and not isinstance(seed, int):
        raise TypeError("build_deck: seed must be an int if provided")

    deck = [value for value, count in CARD_COUNTS for _ in range(count)]

    rng = random.Random(seed)
    rng.shuffle(deck)
    return deck
