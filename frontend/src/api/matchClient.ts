import type { ActionRequest, MatchHistoryOut, MatchStateOut, NewMatchRequest } from './types'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/** Thrown for any non-2xx response; `detail` is FastAPI's error body message, when present. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
  }
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  // A FormData body (see uploadMctsModel) must not get an explicit
  // Content-Type - the browser needs to set its own multipart boundary.
  const isFormData = init?.body instanceof FormData
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: isFormData ? init?.headers : { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError(0, 'Could not reach the server. Check your connection and try again.')
  }

  if (!response.ok) {
    const detail = await response
      .json()
      .then((body) => (typeof body?.detail === 'string' ? body.detail : response.statusText))
      .catch(() => response.statusText)
    throw new ApiError(response.status, detail)
  }

  return response.json() as Promise<T>
}

export function createMatch(newMatch: NewMatchRequest): Promise<MatchStateOut> {
  return apiRequest<MatchStateOut>('/matches', { method: 'POST', body: JSON.stringify(newMatch) })
}

export function getMctsModels(): Promise<string[]> {
  return apiRequest<string[]>('/matches/mcts-models')
}

/** Uploads a checkpoint file, returning the model name it's now selectable
 * under (see GET /matches/mcts-models). */
export function uploadMctsModel(file: File): Promise<string> {
  const formData = new FormData()
  formData.append('file', file)
  return apiRequest<string>('/matches/mcts-models', { method: 'POST', body: formData })
}

export function getMatch(matchId: string): Promise<MatchStateOut> {
  return apiRequest<MatchStateOut>(`/matches/${matchId}`)
}

export function applyAction(matchId: string, action: ActionRequest): Promise<MatchStateOut> {
  return apiRequest<MatchStateOut>(`/matches/${matchId}/actions`, {
    method: 'POST',
    body: JSON.stringify(action),
  })
}

export function startNextRound(matchId: string): Promise<MatchStateOut> {
  return apiRequest<MatchStateOut>(`/matches/${matchId}/next-round`, { method: 'POST' })
}

export function getMatchHistory(matchId: string): Promise<MatchHistoryOut> {
  return apiRequest<MatchHistoryOut>(`/matches/${matchId}/history`)
}

export function gotoMatchHistoryNode(matchId: string, nodeId: string): Promise<MatchStateOut> {
  return apiRequest<MatchStateOut>(`/matches/${matchId}/history/${nodeId}/goto`, { method: 'POST' })
}

// Untyped: this is the same schema-less `tree_export.tree_to_dict` shape
// `tools/mcts-tree` already parses defensively (see treeParse.ts) rather than
// trusts — no reason to duplicate that structural typing here too.
export function getMctsTree(matchId: string, nodeId: string): Promise<unknown> {
  return apiRequest<unknown>(`/matches/${matchId}/history/${nodeId}/mcts-tree`)
}
