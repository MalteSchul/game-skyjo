const STORAGE_KEY = 'skyjo:activeMatchId'

/** Reads the locally remembered match id, if any. Swallows storage access
 * errors (e.g. disabled storage in a locked-down browser) and treats them as "none saved". */
export function loadStoredMatchId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function storeMatchId(matchId: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, matchId)
  } catch {
    // Ignore — persistence is a nice-to-have, not required for the match to work.
  }
}

export function clearStoredMatchId(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Ignore, see storeMatchId.
  }
}
