from collections import Counter

import pytest

from skyjo_rl.deck import DECK_SIZE, build_deck


def test_happy_path_builds_official_150_card_deck():
    deck = build_deck(seed=42)

    assert len(deck) == DECK_SIZE == 150

    counts = Counter(deck)
    assert counts[-2] == 5
    assert counts[-1] == 10
    assert counts[0] == 15
    for value in range(1, 13):
        assert counts[value] == 10


def test_happy_path_same_seed_produces_same_shuffle_order():
    assert build_deck(seed=7) == build_deck(seed=7)


def test_sad_path_rejects_non_int_seed():
    with pytest.raises(TypeError):
        build_deck(seed="not-a-seed")  # type: ignore[arg-type]


def test_bad_path_seed_zero_is_a_valid_falsy_seed():
    deck = build_deck(seed=0)

    assert len(deck) == 150
    assert Counter(deck)[0] == 15
    assert deck == build_deck(seed=0)
