"""Same fixed-trajectory idea as compare_search_configs.py, but aimed at a
different question: when a 50-sim search and a 100-sim search (both
noise-free) actually disagree with EACH OTHER, which one is right?

Judged by a stronger, independent 400-sim search's own final Q-value
(mean_value at the root, from that judge's own edges) for each of the two
candidate actions - a much deeper search is a reasonable proxy for "closer to
the true value" without needing ground truth. Only decisions where 50 and
100 disagree are worth judging; where they agree there's nothing to compare.

Usage:
    uv run python scripts/compare_sim_quality.py --checkpoint <path> --seed 0
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

JUDGE_SIMS = 400


def action_key(a: Action) -> tuple[str, int | None]:
    return (a.type, a.position)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--round-max-steps", type=int, default=500)
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    net = load_net(args.checkpoint)
    evaluate = make_network_evaluator(net)
    rng = np.random.default_rng(args.seed)

    state = new_match(player_count=2, seed=args.seed)
    round_step = 0
    round_count = 0
    total_decisions = 0
    judged: list[dict] = []

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
        actor = state.current_player

        root50 = run_mcts(turn, evaluate, num_simulations=50, c_puct=args.c_puct, add_root_noise=False, rng=rng)
        pick50 = greedy_action(root50, rng)
        root100 = run_mcts(turn, evaluate, num_simulations=100, c_puct=args.c_puct, add_root_noise=False, rng=rng)
        pick100 = greedy_action(root100, rng)

        if action_key(pick50) != action_key(pick100):
            judge = run_mcts(
                turn, evaluate, num_simulations=JUDGE_SIMS, c_puct=args.c_puct, add_root_noise=False, rng=rng
            )
            judge_values = {a: e.mean_value()[actor] for a, e in judge.edges.items()}
            v50 = judge_values.get(pick50)
            v100 = judge_values.get(pick100)
            judged.append(
                {
                    "step": step,
                    "pick50": action_key(pick50),
                    "pick100": action_key(pick100),
                    "v50_per_judge": v50,
                    "v100_per_judge": v100,
                    "winner": "100" if (v100 or -99) > (v50 or -99) else ("50" if v50 is not None else "?"),
                }
            )

        total_decisions += 1
        if total_decisions % 50 == 0:
            print(f"...{total_decisions} decisions (step {step}, round {round_count}, {len(judged)} disagreements so far)", flush=True)
        state = apply_action(state, raw_favorite)
        round_step += 1

    print()
    print(f"trajectory: {total_decisions} decisions, {round_count} rounds, final total_scores={state.total_scores}")
    print(f"50-vs-100 disagreements: {len(judged)} / {total_decisions}")
    print()
    wins_100 = sum(1 for j in judged if j["winner"] == "100")
    wins_50 = sum(1 for j in judged if j["winner"] == "50")
    print(f"judge (400 sims) preferred 100-sim's pick: {wins_100}")
    print(f"judge (400 sims) preferred 50-sim's pick:  {wins_50}")
    print()
    for j in judged:
        print(
            f"step={j['step']:>4}  50->{j['pick50']}  (judge={j['v50_per_judge']})   "
            f"100->{j['pick100']}  (judge={j['v100_per_judge']})   winner={j['winner']}"
        )


if __name__ == "__main__":
    main()
