import { useState } from 'react'

const MIN_PLAYERS = 2
const MAX_PLAYERS = 8

interface NewMatchFormProps {
  onCreate: (playerCount: number, seed: number | undefined, playerNames: string[]) => void
  submitting: boolean
}

/** Clamps to the legal range and pads/truncates `names` to match, preserving
 * already-typed names when the player count changes. */
function resizeNames(names: string[], count: number): string[] {
  const next = names.slice(0, count)
  while (next.length < count) next.push('')
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

  function handlePlayerCountInputChange(value: string) {
    setPlayerCountInput(value)
    if (value.trim() === '') return
    const parsed = Number(value)
    if (Number.isNaN(parsed)) return
    setNames((prev) => resizeNames(prev, clampPlayerCount(parsed)))
  }

  function handlePlayerCountBlur() {
    const count = clampPlayerCount(Number(playerCountInput))
    setPlayerCountInput(String(count))
    setNames((prev) => resizeNames(prev, count))
  }

  function handleNameChange(index: number, value: string) {
    setNames((prev) => prev.map((name, i) => (i === index ? value : name)))
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const playerCount = clampPlayerCount(Number(playerCountInput))
    const seed = seedInput.trim() === '' ? undefined : Number(seedInput)
    // Blank entries are sent through as-is; the backend fills in "Player N" defaults.
    onCreate(playerCount, seed, resizeNames(names, playerCount).map((name) => name.trim()))
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
            <label key={i} className="field">
              <span className="field-label">Player {i + 1} name</span>
              <input
                type="text"
                value={name}
                placeholder={`Player ${i + 1}`}
                maxLength={24}
                onChange={(event) => handleNameChange(i, event.target.value)}
              />
            </label>
          ))}
        </div>
        <button type="submit" className="btn-primary" disabled={submitting}>
          {submitting ? 'Starting…' : 'Start match'}
        </button>
      </form>
    </div>
  )
}

export default NewMatchForm
