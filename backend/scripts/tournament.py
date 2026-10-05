"""Round-robin tournament between bots - checkpoint-backed nets (searched via
MCTS), and/or `RandomBot`/`HeuristicBot` as reference points. Every pairing
plays `--games-per-pairing` 2-player games with seats alternated (so neither
entrant always moves first), and results are tallied into a win-rate table.

Games within a pairing are played in batches of `--batch-size` (default 20):
`--batch-size 1` reproduces the original one-game-at-a-time behavior; a
larger batch drives that many games concurrently and batches every round's
pending MCTS leaf evaluations into one network call - the same win
self-play/eval already get from batching (see rl.mcts.run_mcts_batch), and
for the same reason: batch-of-1 network calls are dominated by per-call
dispatch overhead, not real compute, so batching cuts per-game wall time
substantially once a checkpoint entrant is involved. `random`/`heuristic`
entrants don't call a network at all, so batching them gains nothing, but
they still play correctly within a batch alongside a checkpoint entrant.

The decider abstraction (checkpoint-vs-checkpoint/heuristic/random, batched)
and the batched two-sided game loop both live in `rl.match`, shared with
`rl.selfplay`'s pool-opponent self-play and `rl.match.evaluate_vs_decider` -
this script is a thin CLI front-end over that shared machinery, not a
separate implementation of it.

Usage:
  uv run python scripts/tournament.py \
      --entrant bare_selfplay=scripts/output/checkpoints/comparison/bare_selfplay_2iter.pt \
      --entrant heuristic_only=scripts/output/checkpoints/comparison/heuristic_only_3000games.pt \
      --entrant random=random --entrant heuristic=heuristic \
      --games-per-pairing 20 --num-simulations 20 --batch-size 20 --workers 6

Each `--entrant name=spec` is either `random`, `heuristic`, or a checkpoint
path (loaded into a fresh `AlphaZeroNet`, searched via MCTS). Network
architecture (`--trunk-dim`/`--residual-blocks`) must match whatever produced
the checkpoints being compared - this script doesn't store that per-checkpoint,
so mixing checkpoints trained with different architectures needs separate runs.
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from skyjo.rl.match import build_decider, play_batch
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS

# Same safety valves as `rl.selfplay.generate_episode`, re-enabled here with
# finite defaults (matching the values the training pipeline itself passes):
# Skyjo's finisher-doubling penalty gives a well-searched bot real incentive
# to never be the one who ends a round, so a round can legitimately stall
# forever without one. `rl.match.play_batch`, unlike `generate_episode`, has
# no per-round budget at all otherwise - only the whole-game `max_steps` - so
# a single stalled round could silently burn the entire budget before the
# game errors out.
DEFAULT_ROUND_MAX_STEPS = 500
DEFAULT_MAX_ROUNDS = 10
DEFAULT_BATCH_SIZE = 20


@dataclass(frozen=True)
class _BatchJob:
    name_a: str
    name_b: str
    spec_a: str
    spec_b: str
    seeds: tuple[int, ...]
    forward: tuple[bool, ...]
    num_simulations: int
    c_puct: float
    network_kwargs: dict[str, Any]
    max_steps: int
    round_max_steps: int
    max_rounds: int
    cap_root_lead_a: bool
    cap_root_lead_b: bool


def _run_batch_job(job: _BatchJob) -> list[tuple[str, str, tuple[int, int], tuple[int, int]] | None]:
    kwargs = {
        "num_simulations": job.num_simulations,
        "c_puct": job.c_puct,
        "network_kwargs": job.network_kwargs,
    }
    seeds = list(job.seeds)
    decider_a = build_decider(job.spec_a, [s * 2 for s in seeds], cap_root_lead=job.cap_root_lead_a, **kwargs)
    decider_b = build_decider(job.spec_b, [s * 2 + 1 for s in seeds], cap_root_lead=job.cap_root_lead_b, **kwargs)
    outcomes = play_batch(
        decider_a, decider_b, seeds, list(job.forward), job.max_steps, job.round_max_steps, job.max_rounds
    )

    results: list[tuple[str, str, tuple[int, int], tuple[int, int]] | None] = []
    for seed, forward, outcome in zip(seeds, job.forward, outcomes, strict=True):
        if outcome is None:
            print(f"_run_batch_job: {job.name_a} vs {job.name_b} seed={seed} did not finish, skipping")
            results.append(None)
            continue
        ranks, points = outcome
        if forward:
            results.append((job.name_a, job.name_b, ranks, points))
        else:
            # This game's seat 0 is entrant B (forward=False) - ranks/points
            # are already seat-indexed, so swap them back to (name_a, name_b)
            # order so every result tuple downstream means the same thing
            # regardless of which seat either entrant actually played.
            results.append((job.name_a, job.name_b, (ranks[1], ranks[0]), (points[1], points[0])))
    return results


def _chunks(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _build_jobs(
    entrants: dict[str, str],
    games_per_pairing: int,
    batch_size: int,
    num_simulations: int,
    c_puct: float,
    network_kwargs: dict[str, Any],
    max_steps: int,
    round_max_steps: int,
    max_rounds: int,
    cap_root_lead: set[str],
) -> list[_BatchJob]:
    jobs = []
    seed = 0
    for name_a, name_b in itertools.combinations(entrants, 2):
        pairing_seeds = []
        pairing_forward = []
        for g in range(games_per_pairing):
            seed += 1
            pairing_seeds.append(seed)
            pairing_forward.append(g % 2 == 0)
        for seed_chunk, forward_chunk in zip(_chunks(pairing_seeds, batch_size), _chunks(pairing_forward, batch_size), strict=True):
            jobs.append(
                _BatchJob(
                    name_a,
                    name_b,
                    entrants[name_a],
                    entrants[name_b],
                    tuple(seed_chunk),
                    tuple(forward_chunk),
                    num_simulations,
                    c_puct,
                    network_kwargs,
                    max_steps,
                    round_max_steps,
                    max_rounds,
                    name_a in cap_root_lead,
                    name_b in cap_root_lead,
                )
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
    parser.add_argument(
        "--cap-root-lead",
        type=str,
        default="",
        help="Comma-separated entrant names whose search should use cap_root_lead=True (see "
        "rl.mcts.run_mcts_batch). Ignored for random/heuristic entrants. Lets the same checkpoint be entered "
        "twice under different names to compare with/without it, e.g. --entrant capped=ckpt.pt "
        "--entrant uncapped=ckpt.pt --cap-root-lead capped",
    )
    parser.add_argument("--games-per-pairing", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How many of a pairing's games to play concurrently, batching every decision round's MCTS leaf "
        "evaluations into one network call (see rl.mcts.run_mcts_batch). 1 plays games fully sequentially, "
        "one at a time - the original behavior, and the only way to get per-game progress lines as they "
        "land rather than in bursts of --batch-size.",
    )
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--max-steps-per-game", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--round-max-steps",
        type=int,
        default=DEFAULT_ROUND_MAX_STEPS,
        help="Force-close a round that runs this long without closing naturally (see "
        "rl.selfplay.generate_episode's identical valve - Skyjo's finisher-doubling penalty means a "
        "well-searched bot can have real incentive to stall a round indefinitely).",
    )
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def _print_progress(
    result: tuple[str, str, tuple[int, int], tuple[int, int]] | None, completed: int, total: int
) -> None:
    """One line per finished game, flushed immediately. With --batch-size 1
    these land one at a time as each game completes; with a larger batch
    size, a whole job's games land together in a burst once that job's batch
    finishes (see module docstring) - the per-game line format is unchanged
    either way, only the timing of when lines appear.
    """
    if result is None:
        print(f"[{completed}/{total}] failed, skipped (see error above)", flush=True)
        return
    name_a, name_b, (rank_a, rank_b), (points_a, points_b) = result
    winner = name_a if rank_a < rank_b else name_b
    print(
        f"[{completed}/{total}] {name_a} vs {name_b}: ranks={rank_a},{rank_b} "
        f"points={points_a},{points_b} winner={winner}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    entrants = dict(args.entrants)
    if len(entrants) < 2:
        raise SystemExit("need at least two --entrant to run a tournament")

    cap_root_lead = {name.strip() for name in args.cap_root_lead.split(",") if name.strip()}
    unknown = cap_root_lead - set(entrants)
    if unknown:
        raise SystemExit(f"--cap-root-lead names not among --entrant names: {sorted(unknown)}")

    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}
    jobs = _build_jobs(
        entrants,
        args.games_per_pairing,
        max(1, args.batch_size),
        args.num_simulations,
        args.c_puct,
        network_kwargs,
        args.max_steps_per_game,
        args.round_max_steps,
        args.max_rounds,
        cap_root_lead,
    )
    total_games = sum(len(job.seeds) for job in jobs)
    print(
        f"{total_games} games across {len(entrants)} entrants "
        f"({len(entrants) * (len(entrants) - 1) // 2} pairings) in {len(jobs)} batch job(s)"
    )

    results: list[tuple[str, str, tuple[int, int], tuple[int, int]] | None] = []
    if args.workers <= 1:
        for job in jobs:
            for result in _run_batch_job(job):
                results.append(result)
                _print_progress(result, len(results), total_games)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_batch_job, job) for job in jobs]
            for future in as_completed(futures):
                for result in future.result():
                    results.append(result)
                    _print_progress(result, len(results), total_games)

    wins: dict[str, int] = defaultdict(int)
    games_played: dict[str, int] = defaultdict(int)
    rank_sum: dict[str, float] = defaultdict(float)
    points_sum: dict[str, float] = defaultdict(float)
    pairwise_wins: dict[tuple[str, str], int] = defaultdict(int)
    failed = 0

    for result in results:
        if result is None:
            failed += 1
            continue
        name_a, name_b, (rank_a, rank_b), (points_a, points_b) = result
        games_played[name_a] += 1
        games_played[name_b] += 1
        rank_sum[name_a] += rank_a
        rank_sum[name_b] += rank_b
        points_sum[name_a] += points_a
        points_sum[name_b] += points_b
        winner, loser = (name_a, name_b) if rank_a < rank_b else (name_b, name_a)
        wins[winner] += 1
        pairwise_wins[(winner, loser)] += 1

    print(f"\n{failed} game(s) failed and were skipped\n")
    print(f"{'entrant':<20}{'games':>8}{'wins':>8}{'win%':>8}{'avg_rank':>10}{'avg_points':>12}")
    for name in sorted(entrants, key=lambda n: -wins[n] / max(games_played[n], 1)):
        played = games_played[name]
        win_pct = 100 * wins[name] / played if played else float("nan")
        avg_rank = rank_sum[name] / played if played else float("nan")
        avg_points = points_sum[name] / played if played else float("nan")
        print(f"{name:<20}{played:>8}{wins[name]:>8}{win_pct:>7.1f}%{avg_rank:>10.3f}{avg_points:>12.2f}")

    print("\npairwise win counts (row beat column):")
    names = sorted(entrants)
    header = " " * 20 + "".join(f"{n[:12]:>14}" for n in names)
    print(header)
    for row in names:
        cells = "".join(f"{pairwise_wins[(row, col)]:>14}" if row != col else f"{'-':>14}" for col in names)
        print(f"{row[:19]:<20}{cells}")


if __name__ == "__main__":
    main()
