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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from skyjo.bots.base import Bot
from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.bots.mcts_bot import MctsBot
from skyjo.bots.random_bot import RandomBot
from skyjo.domain.engine import apply_action, force_close_round, new_match, start_next_round
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS, final_ranks

# Same safety valves as `rl.selfplay.generate_episode`, re-enabled here with
# finite defaults (matching the values the training pipeline itself passes):
# Skyjo's finisher-doubling penalty gives a well-searched bot real incentive
# to never be the one who ends a round, so a round can legitimately stall
# forever without one. `_play_one_game`, unlike `generate_episode`, has no
# per-round budget at all otherwise - only the whole-game `max_steps` - so a
# single stalled round could silently burn the entire budget before the game
# errors out.
DEFAULT_ROUND_MAX_STEPS = 500
DEFAULT_MAX_ROUNDS = 10


def _build_bot(
    spec: str, seed: int, *, num_simulations: int, trunk_dim: int, residual_blocks: int, cap_root_lead: bool = False
) -> Bot:
    if spec == "random":
        return RandomBot(seed=seed)
    if spec == "heuristic":
        return HeuristicBot(seed=seed)
    net = AlphaZeroNet(trunk_dim=trunk_dim, num_residual_blocks=residual_blocks)
    load_checkpoint(spec, net)
    evaluate = make_network_evaluator(net)
    return MctsBot(evaluate=evaluate, num_simulations=num_simulations, seed=seed, cap_root_lead=cap_root_lead)


def _play_one_game(
    bots: list[Bot],
    seed: int,
    max_steps: int,
    round_max_steps: int = DEFAULT_ROUND_MAX_STEPS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Returns (ranks, points) - `points` is each seat's raw `total_scores`
    at game end, since rank alone (0 = best) can't tell a narrow win from a
    blowout.
    """
    state = new_match(player_count=len(bots), seed=seed)
    round_step = 0
    round_count = 0
    for _ in range(max_steps):
        if state.phase == "round_over":
            round_count += 1
            if round_count >= max_rounds:
                break
            state = start_next_round(state)
            round_step = 0
            continue
        if state.phase == "game_over":
            break
        if round_step >= round_max_steps:
            state = force_close_round(state)
            round_step = 0
            continue
        turn = Turn.from_state(state)
        action = bots[turn.acting_player].choose_action(turn)
        state = apply_action(state, action)
        round_step += 1
    else:
        raise RuntimeError(f"tournament game did not finish within {max_steps} steps")
    return tuple(final_ranks(state.total_scores)), tuple(state.total_scores)


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
    round_max_steps: int
    max_rounds: int
    cap_root_lead_a: bool
    cap_root_lead_b: bool


def _run_match(job: _MatchJob) -> tuple[str, str, tuple[int, int], tuple[int, int]] | None:
    kwargs = {
        "num_simulations": job.num_simulations,
        "trunk_dim": job.trunk_dim,
        "residual_blocks": job.residual_blocks,
    }
    bot_a = _build_bot(job.spec_a, job.seed * 2, cap_root_lead=job.cap_root_lead_a, **kwargs)
    bot_b = _build_bot(job.spec_b, job.seed * 2 + 1, cap_root_lead=job.cap_root_lead_b, **kwargs)
    try:
        ranks, points = _play_one_game(
            [bot_a, bot_b], job.seed, job.max_steps, job.round_max_steps, job.max_rounds
        )
    except RuntimeError as exc:
        print(f"_run_match: {job.name_a} vs {job.name_b} seed={job.seed} failed, skipping: {exc}")
        return None
    return job.name_a, job.name_b, ranks, points


def _build_jobs(
    entrants: dict[str, str],
    games_per_pairing: int,
    num_simulations: int,
    trunk_dim: int,
    residual_blocks: int,
    max_steps: int,
    round_max_steps: int,
    max_rounds: int,
    cap_root_lead: set[str],
) -> list[_MatchJob]:
    jobs = []
    seed = 0
    for name_a, name_b in itertools.combinations(entrants, 2):
        for g in range(games_per_pairing):
            seed += 1
            args = (seed, num_simulations, trunk_dim, residual_blocks, max_steps, round_max_steps, max_rounds)
            if g % 2 == 0:
                jobs.append(
                    _MatchJob(
                        name_a,
                        name_b,
                        entrants[name_a],
                        entrants[name_b],
                        *args,
                        name_a in cap_root_lead,
                        name_b in cap_root_lead,
                    )
                )
            else:
                jobs.append(
                    _MatchJob(
                        name_b,
                        name_a,
                        entrants[name_b],
                        entrants[name_a],
                        *args,
                        name_b in cap_root_lead,
                        name_a in cap_root_lead,
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
        help="Comma-separated entrant names whose MctsBot should search with cap_root_lead=True (see "
        "bots.mcts_bot.MctsBot). Ignored for random/heuristic entrants. Lets the same checkpoint be entered "
        "twice under different names to compare with/without it, e.g. --entrant capped=ckpt.pt "
        "--entrant uncapped=ckpt.pt --cap-root-lead capped",
    )
    parser.add_argument("--games-per-pairing", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=20)
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
    """One line per finished game, flushed immediately - `main`'s dispatch
    loop drives games via `as_completed`/one-at-a-time specifically so this
    can report as each game lands rather than only once the whole batch is
    done, which otherwise leaves a many-game run with no visible progress at
    all until it either finishes or errors.
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

    jobs = _build_jobs(
        entrants,
        args.games_per_pairing,
        args.num_simulations,
        args.trunk_dim,
        args.residual_blocks,
        args.max_steps_per_game,
        args.round_max_steps,
        args.max_rounds,
        cap_root_lead,
    )
    print(f"{len(jobs)} games across {len(entrants)} entrants ({len(entrants) * (len(entrants) - 1) // 2} pairings)")

    results: list[tuple[str, str, tuple[int, int], tuple[int, int]] | None] = []
    if args.workers <= 1:
        for job in jobs:
            result = _run_match(job)
            results.append(result)
            _print_progress(result, len(results), len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_match, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _print_progress(result, len(results), len(jobs))

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
