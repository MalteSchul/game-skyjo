"""Round-robin tournament between bots - checkpoint-backed `MctsBot`s, and/or
`RandomBot`/`HeuristicBot` as reference points. Every pairing plays
`--games-per-pairing` 2-player games with seats alternated (so neither
entrant always moves first), and results are tallied into a win-rate table.

Usage:
  uv run python scripts/tournament.py \
      --entrant bare_selfplay=scripts/output/checkpoints/comparison/bare_selfplay_2iter.pt \
      --entrant heuristic_only=scripts/output/checkpoints/comparison/heuristic_only_3000games.pt \
      --entrant random=random --entrant heuristic=heuristic \
      --games-per-pairing 20 --num-simulations 20 --workers 6

Each `--entrant name=spec` is either `random`, `heuristic`, or a checkpoint
path (loaded into a fresh `AlphaZeroNet` and wrapped in `MctsBot`). Network
architecture (`--trunk-dim`/`--residual-blocks`) must match whatever produced
the checkpoints being compared - this script doesn't store that per-checkpoint,
so mixing checkpoints trained with different architectures needs separate runs.
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from skyjo.bots.base import Bot
from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.bots.mcts_bot import MctsBot
from skyjo.bots.random_bot import RandomBot
from skyjo.domain.engine import apply_action, new_match, start_next_round
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS, final_ranks


def _build_bot(spec: str, seed: int, *, num_simulations: int, trunk_dim: int, residual_blocks: int) -> Bot:
    if spec == "random":
        return RandomBot(seed=seed)
    if spec == "heuristic":
        return HeuristicBot(seed=seed)
    net = AlphaZeroNet(trunk_dim=trunk_dim, num_residual_blocks=residual_blocks)
    load_checkpoint(spec, net)
    evaluate = make_network_evaluator(net)
    return MctsBot(evaluate=evaluate, num_simulations=num_simulations, seed=seed)


def _play_one_game(bots: list[Bot], seed: int, max_steps: int) -> tuple[int, ...]:
    state = new_match(player_count=len(bots), seed=seed)
    for _ in range(max_steps):
        if state.phase == "round_over":
            state = start_next_round(state)
            continue
        if state.phase == "game_over":
            break
        turn = Turn.from_state(state)
        action = bots[turn.acting_player].choose_action(turn)
        state = apply_action(state, action)
    else:
        raise RuntimeError(f"tournament game did not finish within {max_steps} steps")
    return tuple(final_ranks(state.total_scores))


@dataclass(frozen=True)
class _MatchJob:
    name_a: str
    name_b: str
    spec_a: str
    spec_b: str
    seed: int
    num_simulations: int
    trunk_dim: int
    residual_blocks: int
    max_steps: int


def _run_match(job: _MatchJob) -> tuple[str, str, tuple[int, int]] | None:
    kwargs = {
        "num_simulations": job.num_simulations,
        "trunk_dim": job.trunk_dim,
        "residual_blocks": job.residual_blocks,
    }
    bot_a = _build_bot(job.spec_a, job.seed * 2, **kwargs)
    bot_b = _build_bot(job.spec_b, job.seed * 2 + 1, **kwargs)
    try:
        ranks = _play_one_game([bot_a, bot_b], job.seed, job.max_steps)
    except RuntimeError as exc:
        print(f"_run_match: {job.name_a} vs {job.name_b} seed={job.seed} failed, skipping: {exc}")
        return None
    return job.name_a, job.name_b, ranks


def _build_jobs(
    entrants: dict[str, str], games_per_pairing: int, num_simulations: int, trunk_dim: int, residual_blocks: int, max_steps: int
) -> list[_MatchJob]:
    jobs = []
    seed = 0
    for name_a, name_b in itertools.combinations(entrants, 2):
        for g in range(games_per_pairing):
            seed += 1
            if g % 2 == 0:
                jobs.append(
                    _MatchJob(name_a, name_b, entrants[name_a], entrants[name_b], seed, num_simulations, trunk_dim, residual_blocks, max_steps)
                )
            else:
                jobs.append(
                    _MatchJob(name_b, name_a, entrants[name_b], entrants[name_a], seed, num_simulations, trunk_dim, residual_blocks, max_steps)
                )
    return jobs


def _parse_entrant(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--entrant must be name=spec, got {raw!r}")
    name, spec = raw.split("=", 1)
    return name, spec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrant", action="append", type=_parse_entrant, required=True, dest="entrants")
    parser.add_argument("--games-per-pairing", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--max-steps-per-game", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entrants = dict(args.entrants)
    if len(entrants) < 2:
        raise SystemExit("need at least two --entrant to run a tournament")

    jobs = _build_jobs(
        entrants, args.games_per_pairing, args.num_simulations, args.trunk_dim, args.residual_blocks, args.max_steps_per_game
    )
    print(f"{len(jobs)} games across {len(entrants)} entrants ({len(entrants) * (len(entrants) - 1) // 2} pairings)")

    if args.workers <= 1:
        results = [_run_match(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_run_match, jobs))

    wins: dict[str, int] = defaultdict(int)
    games_played: dict[str, int] = defaultdict(int)
    rank_sum: dict[str, float] = defaultdict(float)
    pairwise_wins: dict[tuple[str, str], int] = defaultdict(int)
    failed = 0

    for result in results:
        if result is None:
            failed += 1
            continue
        name_a, name_b, (rank_a, rank_b) = result
        games_played[name_a] += 1
        games_played[name_b] += 1
        rank_sum[name_a] += rank_a
        rank_sum[name_b] += rank_b
        winner, loser = (name_a, name_b) if rank_a < rank_b else (name_b, name_a)
        wins[winner] += 1
        pairwise_wins[(winner, loser)] += 1

    print(f"\n{failed} game(s) failed and were skipped\n")
    print(f"{'entrant':<20}{'games':>8}{'wins':>8}{'win%':>8}{'avg_rank':>10}")
    for name in sorted(entrants, key=lambda n: -wins[n] / max(games_played[n], 1)):
        played = games_played[name]
        win_pct = 100 * wins[name] / played if played else float("nan")
        avg_rank = rank_sum[name] / played if played else float("nan")
        print(f"{name:<20}{played:>8}{wins[name]:>8}{win_pct:>7.1f}%{avg_rank:>10.3f}")

    print("\npairwise win counts (row beat column):")
    names = sorted(entrants)
    header = " " * 20 + "".join(f"{n[:12]:>14}" for n in names)
    print(header)
    for row in names:
        cells = "".join(f"{pairwise_wins[(row, col)]:>14}" if row != col else f"{'-':>14}" for col in names)
        print(f"{row[:19]:<20}{cells}")


if __name__ == "__main__":
    main()
