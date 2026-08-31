"""Plays ONE fixed trajectory - every actual move is the raw network policy's
own greedy pick (no search at all) - so every decision point is identical
across conditions. At each of those same decision points, runs MCTS under
several (num_simulations, add_root_noise) configurations *without* acting on
any of them, and records whether that search's own greedy pick would have
disagreed with the move actually played.

This is deliberately different from play_and_record_game.py: that lets each
search decide the game's own moves, so different configs' games diverge
after the first disagreement and are no longer comparable position-for-
position. Fixing the trajectory to the raw policy's own choice is what makes
a fair, apples-to-apples comparison across configs possible.

Usage:
    uv run python scripts/compare_search_configs.py --checkpoint <path> --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyjo.domain.engine import Action, apply_action, force_close_round, new_match, start_next_round  # noqa: E402
from skyjo.domain.observation import Turn  # noqa: E402
from skyjo.rl.evaluator import make_network_evaluator  # noqa: E402
from skyjo.rl.game_recorder import load_net  # noqa: E402
from skyjo.rl.mcts import greedy_action, run_mcts  # noqa: E402

DEFAULT_CONFIGS = [(50, False), (50, True), (100, False), (100, True), (200, False), (200, True)]


def action_key(a: Action) -> tuple[str, int | None]:
    return (a.type, a.position)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--round-max-steps", type=int, default=500)
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    net = load_net(args.checkpoint)
    evaluate = make_network_evaluator(net)
    rng = np.random.default_rng(args.seed)

    configs = DEFAULT_CONFIGS
    disagree_counts = dict.fromkeys(configs, 0)
    disagreement_log: dict[tuple[int, bool], list[tuple[int, str]]] = {c: [] for c in configs}

    state = new_match(player_count=2, seed=args.seed)
    round_step = 0
    round_count = 0
    total_decisions = 0

    for step in range(args.max_steps):
        if state.phase == "round_over":
            round_count += 1
            if round_count >= args.max_rounds:
                break
            state = start_next_round(state)
            round_step = 0
            continue
        if state.phase == "game_over":
            break
        if round_step >= args.round_max_steps:
            state = force_close_round(state)
            round_step = 0
            continue

        turn = Turn.from_state(state)
        priors, _value = evaluate(state)
        raw_favorite = max(priors, key=priors.get)

        for sims, noise in configs:
            root = run_mcts(
                turn,
                evaluate,
                num_simulations=sims,
                c_puct=args.c_puct,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_epsilon=args.dirichlet_epsilon,
                add_root_noise=noise,
                rng=rng,
            )
            search_pick = greedy_action(root, rng)
            if action_key(search_pick) != action_key(raw_favorite):
                disagree_counts[(sims, noise)] += 1
                disagreement_log[(sims, noise)].append(
                    (step, f"raw={action_key(raw_favorite)} search={action_key(search_pick)} phase={state.phase}")
                )

        total_decisions += 1
        if total_decisions % 50 == 0:
            print(f"...{total_decisions} decisions so far (step {step}, round {round_count})", flush=True)
        state = apply_action(state, raw_favorite)
        round_step += 1

    print()
    print(f"trajectory: {total_decisions} decisions, {round_count} rounds, final total_scores={state.total_scores}")
    print()
    print(f"{'sims':>5} {'noise':>6} | {'disagreements':>13} | {'rate':>7}")
    for sims, noise in configs:
        n = disagree_counts[(sims, noise)]
        print(f"{sims:>5} {str(noise):>6} | {n:>13} | {100*n/total_decisions:>6.1f}%")

    print()
    print("first 5 disagreements per config:")
    for sims, noise in configs:
        print(f"-- sims={sims} noise={noise} --")
        for step, desc in disagreement_log[(sims, noise)][:5]:
            print(f"  step={step}: {desc}")


if __name__ == "__main__":
    main()
