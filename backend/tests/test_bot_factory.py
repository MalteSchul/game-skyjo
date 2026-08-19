import pytest

from skyjo.bots.factory import create_bot
from skyjo.bots.random_bot import RandomBot


def test_human_seats_have_no_bot():
    assert create_bot("human") is None


def test_random_bot_type_returns_a_random_bot_instance():
    bot = create_bot("random_bot", seed=5)

    assert isinstance(bot, RandomBot)


def test_rejects_an_unknown_player_type():
    with pytest.raises(ValueError):
        create_bot("grandmaster")
