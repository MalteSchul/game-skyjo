import { useEffect, useState } from 'react'
import { getMctsModels, uploadMctsModel } from '../api/matchClient'
import type { PlayerTypeName } from '../api/types'

const MIN_PLAYERS = 2
const MAX_PLAYERS = 8

const MIN_MCTS_NUM_SIMULATIONS = 1
const MAX_MCTS_NUM_SIMULATIONS = 1_000_000
const DEFAULT_MCTS_NUM_SIMULATIONS = 200

interface NewMatchFormProps {
  onCreate: (
    playerCount: number,
    seed: number | undefined,
    playerNames: string[],
    playerTypes: PlayerTypeName[],
    playerMctsModels: (string | null)[],
    playerMctsNumSimulations: (number | null)[],
    playerMctsCapRootLead: boolean[],
  ) => void
  submitting: boolean
}

/** Clamps to the legal range and pads/truncates `names` to match, preserving
 * already-typed names when the player count changes. */
function resizeNames(names: string[], count: number): string[] {
  const next = names.slice(0, count)
  while (next.length < count) next.push('')
  return next
}

/** Same idea as resizeNames, but new seats default to "human" rather than blank. */
function resizePlayerTypes(types: PlayerTypeName[], count: number): PlayerTypeName[] {
  const next = types.slice(0, count)
  while (next.length < count) next.push('human')
  return next
}

/** Same idea as resizeNames, but new seats default to null (untrained network). */
function resizeMctsModels(models: (string | null)[], count: number): (string | null)[] {
  const next = models.slice(0, count)
  while (next.length < count) next.push(null)
  return next
}

/** Same idea as resizeMctsModels, but for the per-seat simulation count override. */
function resizeMctsNumSimulations(values: (number | null)[], count: number): (number | null)[] {
  const next = values.slice(0, count)
  while (next.length < count) next.push(null)
  return next
}

/** Same idea as resizePlayerTypes, but new seats default to false (uncapped search). */
function resizeMctsCapRootLead(values: boolean[], count: number): boolean[] {
  const next = values.slice(0, count)
  while (next.length < count) next.push(false)
  return next
}

function clampPlayerCount(value: number): number {
  if (Number.isNaN(value)) return MIN_PLAYERS
  return Math.min(MAX_PLAYERS, Math.max(MIN_PLAYERS, Math.trunc(value)))
}

function NewMatchForm({ onCreate, submitting }: NewMatchFormProps) {
  // Kept as raw text (not the clamped number) so the field can be freely
  // edited — e.g. cleared to retype — without snapping back mid-edit. It's
  // only normalized to a valid, clamped value on blur or submit.
  const [playerCountInput, setPlayerCountInput] = useState('2')
  const [seedInput, setSeedInput] = useState('')
  const [names, setNames] = useState<string[]>(['', ''])
  const [playerTypes, setPlayerTypes] = useState<PlayerTypeName[]>(['human', 'human'])
  const [mctsModels, setMctsModels] = useState<(string | null)[]>([null, null])
  const [mctsNumSimulations, setMctsNumSimulations] = useState<(number | null)[]>([null, null])
  const [mctsCapRootLead, setMctsCapRootLead] = useState<boolean[]>([false, false])
  const [availableMctsModels, setAvailableMctsModels] = useState<string[]>([])
  // Seat index currently uploading a model file, or null - drives the "uploading…" state.
  const [uploadingSeat, setUploadingSeat] = useState<number | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getMctsModels()
      .then((models) => {
        if (!cancelled) setAvailableMctsModels(models)
      })
      .catch(() => {
        // No models to choose from (e.g. server unreachable) just leaves the
        // mcts_bot seat on the untrained default network.
      })
    return () => {
      cancelled = true
    }
  }, [])

  function handlePlayerCountInputChange(value: string) {
    setPlayerCountInput(value)
    if (value.trim() === '') return
    const parsed = Number(value)
    if (Number.isNaN(parsed)) return
    const count = clampPlayerCount(parsed)
    setNames((prev) => resizeNames(prev, count))
    setPlayerTypes((prev) => resizePlayerTypes(prev, count))
    setMctsModels((prev) => resizeMctsModels(prev, count))
    setMctsNumSimulations((prev) => resizeMctsNumSimulations(prev, count))
    setMctsCapRootLead((prev) => resizeMctsCapRootLead(prev, count))
  }

  function handlePlayerCountBlur() {
    const count = clampPlayerCount(Number(playerCountInput))
    setPlayerCountInput(String(count))
    setNames((prev) => resizeNames(prev, count))
    setPlayerTypes((prev) => resizePlayerTypes(prev, count))
    setMctsModels((prev) => resizeMctsModels(prev, count))
    setMctsNumSimulations((prev) => resizeMctsNumSimulations(prev, count))
    setMctsCapRootLead((prev) => resizeMctsCapRootLead(prev, count))
  }

  function handleNameChange(index: number, value: string) {
    setNames((prev) => prev.map((name, i) => (i === index ? value : name)))
  }

  function handlePlayerTypeChange(index: number, value: PlayerTypeName) {
    setPlayerTypes((prev) => prev.map((type, i) => (i === index ? value : type)))
  }

  function handleMctsModelChange(index: number, value: string | null) {
    setMctsModels((prev) => prev.map((model, i) => (i === index ? value : model)))
  }

  async function handleMctsModelFileChange(index: number, file: File | undefined) {
    if (!file) return
    setUploadingSeat(index)
    setUploadError(null)
    try {
      const name = await uploadMctsModel(file)
      setAvailableMctsModels((prev) => (prev.includes(name) ? prev : [...prev, name].sort()))
      handleMctsModelChange(index, name)
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setUploadingSeat(null)
    }
  }

  function handleMctsNumSimulationsChange(index: number, value: number | null) {
    setMctsNumSimulations((prev) => prev.map((sims, i) => (i === index ? value : sims)))
  }

  function handleMctsCapRootLeadChange(index: number, value: boolean) {
    setMctsCapRootLead((prev) => prev.map((capped, i) => (i === index ? value : capped)))
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const playerCount = clampPlayerCount(Number(playerCountInput))
    const seed = seedInput.trim() === '' ? undefined : Number(seedInput)
    // Blank entries are sent through as-is; the backend fills in "Player N" defaults.
    onCreate(
      playerCount,
      seed,
      resizeNames(names, playerCount).map((name) => name.trim()),
      resizePlayerTypes(playerTypes, playerCount),
      resizeMctsModels(mctsModels, playerCount),
      resizeMctsNumSimulations(mctsNumSimulations, playerCount),
      resizeMctsCapRootLead(mctsCapRootLead, playerCount),
    )
  }

  return (
    <div className="new-match-panel">
      <div className="card-fan" aria-hidden="true">
        <div className="card card-face-up tone-low" />
        <div className="card card-face-up tone-mid" />
        <div className="card card-face-up tone-high" />
      </div>
      <form onSubmit={handleSubmit} className="new-match-form">
        <div className="new-match-row">
          <label className="field">
            <span className="field-label">Players</span>
            <input
              type="number"
              min={MIN_PLAYERS}
              max={MAX_PLAYERS}
              value={playerCountInput}
              onChange={(event) => handlePlayerCountInputChange(event.target.value)}
              onBlur={handlePlayerCountBlur}
            />
          </label>
          <label className="field">
            <span className="field-label">Seed (optional)</span>
            <input
              type="number"
              value={seedInput}
              placeholder="random"
              onChange={(event) => setSeedInput(event.target.value)}
            />
          </label>
        </div>
        <div className="player-name-fields">
          {names.map((name, i) => (
            <div key={i} className="player-seat-row">
              <label className="field player-name-field">
                <span className="field-label">Player {i + 1} name</span>
                <input
                  type="text"
                  value={name}
                  placeholder={`Player ${i + 1}`}
                  maxLength={24}
                  onChange={(event) => handleNameChange(i, event.target.value)}
                />
              </label>
              <label className="field player-type-field">
                <span className="field-label">Player {i + 1} type</span>
                <select
                  value={playerTypes[i]}
                  onChange={(event) => handlePlayerTypeChange(i, event.target.value as PlayerTypeName)}
                >
                  <option value="human">Human</option>
                  <option value="random_bot">Random bot</option>
                  <option value="thinking_bot">Thinking bot (slow)</option>
                  <option value="heuristic_bot">Heuristic bot</option>
                  <option value="mcts_bot">MCTS bot (experimental)</option>
                </select>
              </label>
              {playerTypes[i] === 'mcts_bot' && (
                <label className="field player-mcts-model-field">
                  <span className="field-label">Player {i + 1} model</span>
                  <select
                    value={mctsModels[i] ?? ''}
                    onChange={(event) => handleMctsModelChange(i, event.target.value || null)}
                  >
                    <option value="">Untrained default</option>
                    {availableMctsModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {playerTypes[i] === 'mcts_bot' && (
                <label className="field player-mcts-model-upload-field">
                  <span className="field-label">Player {i + 1} model file</span>
                  <input
                    type="file"
                    accept=".pt"
                    disabled={uploadingSeat === i}
                    onChange={(event) => {
                      void handleMctsModelFileChange(i, event.target.files?.[0])
                      event.target.value = ''
                    }}
                  />
                  {uploadingSeat === i && <span className="field-hint">Uploading…</span>}
                </label>
              )}
              {playerTypes[i] === 'mcts_bot' && (
                <label className="field player-mcts-simulations-field">
                  <span className="field-label">Player {i + 1} simulations</span>
                  <input
                    type="number"
                    min={MIN_MCTS_NUM_SIMULATIONS}
                    max={MAX_MCTS_NUM_SIMULATIONS}
                    value={mctsNumSimulations[i] ?? ''}
                    placeholder={`${DEFAULT_MCTS_NUM_SIMULATIONS} (default)`}
                    onChange={(event) => {
                      const raw = event.target.value
                      handleMctsNumSimulationsChange(i, raw.trim() === '' ? null : Number(raw))
                    }}
                  />
                </label>
              )}
              {playerTypes[i] === 'mcts_bot' && (
                <label
                  className="field player-mcts-cap-root-lead-field"
                  title="Once one move is clearly ahead, spend remaining search checking the runner-up instead of piling more visits onto the leader."
                >
                  <span className="field-label">Player {i + 1} capped search (experimental)</span>
                  <input
                    type="checkbox"
                    checked={mctsCapRootLead[i]}
                    onChange={(event) => handleMctsCapRootLeadChange(i, event.target.checked)}
                  />
                </label>
              )}
            </div>
          ))}
        </div>
        {uploadError && (
          <p role="alert" className="error-banner">
            {uploadError}
          </p>
        )}
        <button type="submit" className="btn-primary" disabled={submitting}>
          {submitting ? 'Starting…' : 'Start match'}
        </button>
      </form>
    </div>
  )
}

export default NewMatchForm
