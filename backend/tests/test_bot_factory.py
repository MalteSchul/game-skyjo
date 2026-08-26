import pytest

from skyjo.bots.factory import create_bot
from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.bots.mcts_bot import MctsBot
from skyjo.bots.random_bot import RandomBot
from skyjo.bots.thinking_bot import ThinkingBot


def test_human_seats_have_no_bot():
    assert create_bot("human") is None


def test_random_bot_type_returns_a_random_bot_instance():
    bot = create_bot("random_bot", seed=5)

    assert isinstance(bot, RandomBot)


def test_thinking_bot_type_returns_a_thinking_bot_instance():
    bot = create_bot("thinking_bot", seed=5)

    assert isinstance(bot, ThinkingBot)


def test_heuristic_bot_type_returns_a_heuristic_bot_instance():
    bot = create_bot("heuristic_bot", seed=5)

    assert isinstance(bot, HeuristicBot)


def test_mcts_bot_type_returns_an_mcts_bot_instance():
    bot = create_bot("mcts_bot", seed=5)

    assert isinstance(bot, MctsBot)


def test_mcts_bot_type_with_an_unknown_model_raises():
    with pytest.raises(ValueError):
        create_bot("mcts_bot", seed=5, mcts_model="does-not-exist")


def test_mcts_bot_type_uses_the_given_num_simulations():
    bot = create_bot("mcts_bot", seed=5, num_simulations=7)

    assert bot._num_simulations == 7


def test_mcts_bot_type_defaults_num_simulations_when_not_given():
    from skyjo.bots.mcts_bot import default_num_simulations

    bot = create_bot("mcts_bot", seed=5)

    assert bot._num_simulations == default_num_simulations()


def test_rejects_an_unknown_player_type():
    with pytest.raises(ValueError):
        create_bot("grandmaster")
