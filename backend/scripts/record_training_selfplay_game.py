"""Plays one self-play game with the EXACT mechanics real training uses -
Dirichlet root noise, tau-tempered sampling, tied-action widening - via
`rl.game_recorder.record_training_selfplay_game`. Unlike
`play_and_record_game.py` (evaluation-style: greedy, no noise, two possibly-
different checkpoints), this reads num_simulations/tau/c_puct/dirichlet
settings/round_max_steps/max_rounds straight from the checkpoint's own saved
`extra.config` (see `rl.loop.run_training_loop`'s checkpoint writes) - there
is nothing to configure, since the whole point is to reproduce what that
checkpoint's own self-play actually looked like.

Writes the same JSON schema `play_and_record_game.py` does, with a few
extra fields populated: `dirichlet_noised_priors`, `pi_target`, `tau`,
`tied_group_size` per decision - see `rl.game_recorder.DecisionRecord`'s
docstrings. `heuristic_action`/`heuristic_action_representative` are null
throughout (no heuristic reference in this mode). The frontend's game-replay
tool (`/tools/game-replay`) shows these as an extra comparison column and a
tau/tied-group note whenever they're present, so this loads there like any
other recording, just with more detail visible.

Usage:
  uv run python scripts/record_training_selfplay_game.py \
      --checkpoint scripts/output/pipeline/exp1/rl_checkpoints_selfplay_v6/latest.pt
"""

from __future__ import annotations

import argparse
import json
import pickle

from skyjo.rl.game_record_export import game_record_to_dict
from skyjo.rl.game_recorder import record_training_selfplay_game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint (.pt) to self-play - its own saved config is used verbatim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="override the checkpoint's own saved tau (default: unset, use the checkpoint's value) - "
        "e.g. --tau 1.0 for pi exactly proportional to raw visit counts, no sharpening",
    )
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
        default="scripts/output/training_selfplay_game.json",
        help="gitignored by default (scripts/output/) - JSON for the frontend's game-replay tool",
    )
    parser.add_argument("--out-pkl", help="also pickle the raw GameRecord here, for further ad-hoc analysis in Python")
    args = parser.parse_args()

    record = record_training_selfplay_game(
        args.checkpoint,
        seed=args.seed,
        tau=args.tau,
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
