"""Plays one 2-player game between two checkpoints (or one checkpoint
against itself, for a self-play mirror match) with real MCTS search on both
seats, recording every decision - ground-truth state, raw policy prior,
full search visit distribution, rank/points estimates - via
`rl.game_recorder.play_and_record`. Writes the result as JSON for the
frontend's game-replay tool (`/tools/game-replay`) and, optionally, as a
pickle for further ad-hoc analysis in Python.

Usage:
  uv run python scripts/play_and_record_game.py --checkpoint-a models/bootstrap.pt --num-simulations 200

  # two different checkpoints, named for the replay UI:
  uv run python scripts/play_and_record_game.py \
      --checkpoint-a models/bootstrap.pt --name-a bootstrap \
      --checkpoint-b scripts/output/pipeline/exp1/rl_checkpoints_vsheuristic_nogate_v3/checkpoint_000006.pt --name-b iter5 \
      --num-simulations 1000 --out scripts/output/bootstrap_vs_iter5.json

`--checkpoint-b`/`--name-b` default to `--checkpoint-a`/"seat1" - a mirror
match against itself, since that's the most common case (comparing a
checkpoint's own play against itself with the "which net is better"
confound removed).
"""

from __future__ import annotations

import argparse
import json
import pickle

from skyjo.rl.game_record_export import game_record_to_dict
from skyjo.rl.game_recorder import play_and_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", required=True, help="checkpoint (.pt) for seat 0")
    parser.add_argument("--checkpoint-b", help="checkpoint (.pt) for seat 1 (default: same as --checkpoint-a)")
    parser.add_argument("--name-a", default="seat0", help="display name for seat 0 in the replay UI")
    parser.add_argument("--name-b", default="seat1", help="display name for seat 1 in the replay UI")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--single-threaded",
        dest="single_threaded",
        action="store_true",
        default=True,
        help="pin torch to one thread for the run's duration, for bit-reproducible reruns of the same seed (default: on)",
    )
    parser.add_argument("--no-single-threaded", dest="single_threaded", action="store_false")
    parser.add_argument(
        "--out",
        default="scripts/output/game_record.json",
        help="gitignored by default (scripts/output/) - JSON for the frontend's game-replay tool",
    )
    parser.add_argument("--out-pkl", help="also pickle the raw GameRecord here, for further ad-hoc analysis in Python")
    args = parser.parse_args()

    checkpoint_b = args.checkpoint_b or args.checkpoint_a

    record = play_and_record(
        args.name_a,
        args.checkpoint_a,
        args.name_b,
        checkpoint_b,
        seed=args.seed,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        max_rounds=args.max_rounds,
        deterministic_torch=args.single_threaded,
    )

    with open(args.out, "w") as f:
        json.dump(game_record_to_dict(record), f)
    print(
        f"final={record.final_total_scores} winner={record.winner_name} "
        f"decisions={len(record.decisions)} rounds={record.rounds_played}"
    )
    print(f"wrote {args.out}")

    if args.out_pkl:
        with open(args.out_pkl, "wb") as f:
            pickle.dump(record, f)
        print(f"wrote {args.out_pkl}")


if __name__ == "__main__":
    main()
