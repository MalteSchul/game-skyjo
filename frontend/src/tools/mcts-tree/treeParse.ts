import type { SnapshotMap, TreeNode } from './types'

export class TreeParseError extends Error {}

function looksLikeTreeNode(value: unknown): value is TreeNode {
  if (typeof value !== 'object' || value === null) return false
  const kind = (value as { kind?: unknown }).kind
  return kind === 'decision' || kind === 'chance'
}

/** Accepts either shape `dump_mcts_tree.py` can produce: a dict of
 * simulation-count → tree (the normal `--sim-points` output), or a single
 * bare tree with a top-level `kind` field (what you get from serializing one
 * `tree_to_dict()` call yourself). Anything else throws `TreeParseError` with
 * a message meant to be shown directly to the person loading the file. */
export function normalizeSnapshots(data: unknown): SnapshotMap {
  if (looksLikeTreeNode(data)) return { '0': data }

  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new TreeParseError(
      'Couldn\'t find any MCTS tree snapshots in this file — expected the JSON from dump_mcts_tree.py: a dict of simulation-count → tree, or a single tree with a top-level "kind" field.',
    )
  }

  const out: SnapshotMap = {}
  for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
    if (looksLikeTreeNode(value)) out[key] = value
  }

  if (Object.keys(out).length === 0) {
    throw new TreeParseError(
      'Couldn\'t find any MCTS tree snapshots in this file — expected the JSON from dump_mcts_tree.py: a dict of simulation-count → tree, or a single tree with a top-level "kind" field.',
    )
  }
  return out
}

/** Numeric-aware sort so `"2"` sorts before `"10"`; falls back to lexical
 * ordering for any non-numeric snapshot key (shouldn't normally occur, but
 * a hand-edited file could have one). */
export function sortedSnapshotKeys(snapshots: SnapshotMap): string[] {
  return Object.keys(snapshots).sort((a, b) => {
    const na = Number(a)
    const nb = Number(b)
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb
    return a.localeCompare(b)
  })
}
