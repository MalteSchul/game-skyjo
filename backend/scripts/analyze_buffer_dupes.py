"""Ad-hoc analysis: how many duplicate GameStates sit in a saved replay
buffer, bucketed by how far into the game each state is (revealed-card count
across all boards). Early states (e.g. the very first flip) are structurally
near-identical across many games and are of little training interest, so raw
dedup counts are misleading without this breakdown.

Usage: python -m scripts.analyze_buffer_dupes <path-to-buffer_latest.pkl>
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path


def revealed_count(state) -> int:
    return sum(
        1
        for board in state.boards
        for card in board.cards
        if card is not None and card.face_up
    )


def bucket(n_revealed: int) -> str:
    if n_revealed <= 2:
        return "00-02 (initial flip)"
    if n_revealed <= 5:
        return "03-05"
    if n_revealed <= 8:
        return "06-08"
    if n_revealed <= 11:
        return "09-11"
    return "12+ (board full / col cleared)"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "backend/models/buffer_latest.pkl")
    with open(path, "rb") as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples from {path}")

    state_counts: Counter = Counter(s.state for s in samples)
    by_bucket_total: Counter = Counter()
    by_bucket_unique_states: dict[str, set] = {}

    for s in samples:
        b = bucket(revealed_count(s.state))
        by_bucket_total[b] += 1
        by_bucket_unique_states.setdefault(b, set()).add(s.state)

    total_unique = len(state_counts)
    total_dupes = len(samples) - total_unique

    print(f"\nOverall: {total_unique} unique states, {total_dupes} duplicate rows "
          f"({total_dupes / len(samples):.1%} of buffer)")

    print(f"\n{'bucket':<32}{'rows':>8}{'unique':>8}{'dupe rows':>10}{'dupe %':>8}")
    order = ["00-02 (initial flip)", "03-05", "06-08", "09-11", "12+ (board full / col cleared)"]
    for b in order:
        rows = by_bucket_total.get(b, 0)
        unique = len(by_bucket_unique_states.get(b, ()))
        dupe_rows = rows - unique
        pct = dupe_rows / rows if rows else 0.0
        print(f"{b:<32}{rows:>8}{unique:>8}{dupe_rows:>10}{pct:>8.1%}")

    top = state_counts.most_common(10)
    print("\nTop 10 most-repeated states (count, revealed cards):")
    for state, count in top:
        print(f"  {count:>6}x  revealed={revealed_count(state)}  phase={state.phase}")


if __name__ == "__main__":
    main()
