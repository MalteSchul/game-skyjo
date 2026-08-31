import { useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import CenterPiles from '../../game/CenterPiles'
import '../mcts-tree/mcts-tree.css'
import { GameReplayParseError, parseGameReplay } from './gameReplayParse'
import {
  buildWinProbSeries,
  chosenDiffersFromTopVisitStepIndices,
  hasHeuristicData,
  hasTrainingSelfPlayData,
  heuristicDiffStepIndices,
  overrideStepIndices,
} from './replayUtils'
import DecisionDetail from './DecisionDetail'
import DecisionScrubber from './DecisionScrubber'
import ReplayBoard from './ReplayBoard'
import WinProbabilityChart from './WinProbabilityChart'
import type { GameReplay } from './types'
import './game-replay.css'

function roundBoundaryIndices(replay: GameReplay): number[] {
  const indices: number[] = []
  for (let i = 1; i < replay.decisions.length; i++) {
    const prev = replay.decisions[i - 1].total_scores
    const cur = replay.decisions[i].total_scores
    if (prev.some((s, j) => s !== cur[j])) indices.push(i)
  }
  return indices
}

export default function GameReplayPage() {
  // Same reasoning as McTreeExplorerPage: this is a dense data/board tool,
  // not the game itself, so it opts the shared #root out of the game's
  // fixed 1126px centered column for as long as it's mounted.
  useEffect(() => {
    document.body.classList.add('mcts-tools-page')
    return () => document.body.classList.remove('mcts-tools-page')
  }, [])

  const [replay, setReplay] = useState<GameReplay | null>(null)
  const [fileName, setFileName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const [revealHidden, setRevealHidden] = useState(true)
  // null = show override markers for both seats; a seat index restricts the
  // timeline to just that seat's own overrides - the two seats' patterns are
  // usually the more interesting thing to compare in a game between two
  // different nets, not their combined count.
  const [markerSeatFilter, setMarkerSeatFilter] = useState<number | null>(null)
  // Off by default even when the data is available - it's a denser overlay
  // on top of the override lane, worth opting into rather than always-on.
  const [showHeuristicDiffs, setShowHeuristicDiffs] = useState(false)
  const [showChosenDiffers, setShowChosenDiffers] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function applyData(data: unknown, label: string) {
    try {
      const parsed = parseGameReplay(data)
      setReplay(parsed)
      setFileName(label)
      setActiveIndex(0)
      setMarkerSeatFilter(null)
      setShowHeuristicDiffs(false)
      setShowChosenDiffers(false)
      setError(null)
    } catch (err) {
      setError(err instanceof GameReplayParseError ? err.message : 'Something went wrong reading that data.')
    }
  }

  function loadFile(file: File) {
    setError(null)
    const reader = new FileReader()
    reader.onload = () => {
      let data: unknown
      try {
        data = JSON.parse(String(reader.result))
      } catch (err) {
        setError(`That file isn't valid JSON (${err instanceof Error ? err.message : String(err)}).`)
        return
      }
      applyData(data, file.name)
    }
    reader.onerror = () => setError("Couldn't read that file.")
    reader.readAsText(file)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) loadFile(file)
  }

  const overrideIndices = useMemo(
    () => (replay ? overrideStepIndices(replay, markerSeatFilter) : []),
    [replay, markerSeatFilter],
  )
  const roundBoundaries = useMemo(() => (replay ? roundBoundaryIndices(replay) : []), [replay])
  const winProbSeries = useMemo(() => (replay ? buildWinProbSeries(replay) : []), [replay])
  const hasHeuristic = useMemo(() => (replay ? hasHeuristicData(replay) : false), [replay])
  const heuristicDiffIndices = useMemo(
    () => (replay && hasHeuristic ? heuristicDiffStepIndices(replay, markerSeatFilter) : []),
    [replay, hasHeuristic, markerSeatFilter],
  )
  const hasTrainingDetail = useMemo(() => (replay ? hasTrainingSelfPlayData(replay) : false), [replay])
  const chosenDiffersIndices = useMemo(
    () => (replay && hasTrainingDetail ? chosenDiffersFromTopVisitStepIndices(replay, markerSeatFilter) : []),
    [replay, hasTrainingDetail, markerSeatFilter],
  )

  const decision = replay?.decisions[activeIndex] ?? null

  return (
    <div className="mcts-explorer gr-page">
      <header className="mtx-topbar">
        <div className="mtx-brand">
          <span className="mtx-mark" aria-hidden="true">🎬</span>
          <div>
            <h1>Game Replay</h1>
            <div className="mtx-tagline">for play_and_record_game.py exports</div>
          </div>
        </div>
        {replay && (
          <div className="mtx-topbar-actions">
            <span className="mtx-file-pill">
              <span aria-hidden="true">📄</span>
              <span className="mtx-fname">{fileName}</span>
            </span>
            <button type="button" className="mtx-btn" onClick={() => fileInputRef.current?.click()}>
              Load different file
            </button>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          aria-label="Load game replay file"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) loadFile(file)
            e.target.value = ''
          }}
        />
      </header>

      {!replay ? (
        <div className="mtx-empty-state">
          <div
            className={`mtx-dropzone${dragOver ? ' mtx-drag-over' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <div className="mtx-glyph" aria-hidden="true">🎬</div>
            <h2>Drop a game replay here</h2>
            <p>
              Load the JSON produced by <code>scripts/play_and_record_game.py</code> — one full game, every decision
              recorded with the ground-truth state, the raw policy prior, and the full MCTS search. Nothing leaves
              your browser; the file is only read locally.
            </p>
            <button type="button" className="mtx-btn mtx-btn-primary" onClick={() => fileInputRef.current?.click()}>
              Choose file…
            </button>
            {error && (
              <div className="mtx-error-line" role="alert">
                {error}
              </div>
            )}
          </div>
        </div>
      ) : decision ? (
        <main className="mtx-canvas gr-canvas">
          <div className="mtx-stat-tiles">
            <div className="mtx-stat-tile">
              <div className="mtx-label">Seats</div>
              <div className="mtx-value">{replay.seat_names.join(' vs ')}</div>
              <div className="mtx-sub">seed={replay.seed} · {replay.num_simulations} sims/move</div>
            </div>
            <div className="mtx-stat-tile">
              <div className="mtx-label">Final result</div>
              <div className="mtx-value mtx-mono">{replay.final_total_scores?.join(' – ') ?? '—'}</div>
              <div className="mtx-sub">winner: {replay.winner_name ?? '—'} · {replay.rounds_played} round(s)</div>
            </div>
            <div className="mtx-stat-tile">
              <div className="mtx-label">Decisions recorded</div>
              <div className="mtx-value mtx-mono">{replay.decisions.length}</div>
              <div className="mtx-sub">
                {overrideIndices.length} where search overrode the raw prior
                {markerSeatFilter !== null && ` (${replay.seat_names[markerSeatFilter] ?? `P${markerSeatFilter}`} only)`}
                {showHeuristicDiffs && hasHeuristic && ` · ${heuristicDiffIndices.length} differ from the heuristic`}
                {showChosenDiffers && hasTrainingDetail && ` · ${chosenDiffersIndices.length} tau-sampling deviations`}
              </div>
            </div>
          </div>

          <div className="gr-marker-filter">
            <span className="mtx-field-label">Highlight overrides for</span>
            <span className="mode-toggle" role="group" aria-label="Highlight overrides for">
              <button
                type="button"
                className={`mode-toggle-btn ${markerSeatFilter === null ? 'mode-toggle-btn-active' : ''}`}
                onClick={() => setMarkerSeatFilter(null)}
              >
                Both
              </button>
              {replay.seat_names.map((name, seat) => (
                <button
                  key={seat}
                  type="button"
                  className={`mode-toggle-btn ${markerSeatFilter === seat ? 'mode-toggle-btn-active' : ''}`}
                  onClick={() => setMarkerSeatFilter(seat)}
                >
                  {name}
                </button>
              ))}
            </span>
          </div>

          {hasHeuristic && (
            <label className="mtx-checkline">
              <input
                type="checkbox"
                checked={showHeuristicDiffs}
                onChange={(e) => setShowHeuristicDiffs(e.target.checked)}
              />
              Show diffs from heuristic (steps where a rule-based reference bot, never actually played, would have
              chosen differently from what was played)
            </label>
          )}

          {hasTrainingDetail && (
            <label className="mtx-checkline">
              <input
                type="checkbox"
                checked={showChosenDiffers}
                onChange={(e) => setShowChosenDiffers(e.target.checked)}
              />
              Show tau-sampling deviations (steps where the played action differs from what search&rsquo;s own visit
              distribution most favored, purely from tau-sampling - not a disagreement with the raw prior)
            </label>
          )}

          <DecisionScrubber
            count={replay.decisions.length}
            activeIndex={activeIndex}
            overrideIndices={overrideIndices}
            roundBoundaryIndices={roundBoundaries}
            heuristicDiffIndices={heuristicDiffIndices}
            showHeuristicDiffs={showHeuristicDiffs && hasHeuristic}
            chosenDiffersIndices={chosenDiffersIndices}
            showChosenDiffers={showChosenDiffers && hasTrainingDetail}
            onChange={setActiveIndex}
          />

          <WinProbabilityChart
            series={winProbSeries}
            totalSteps={replay.decisions.length}
            activeStep={activeIndex}
            onScrub={setActiveIndex}
          />

          <label className="mtx-checkline">
            <input type="checkbox" checked={revealHidden} onChange={(e) => setRevealHidden(e.target.checked)} />
            Reveal hidden cards (this is the true state — every card&rsquo;s real value is always recorded)
          </label>

          <div className="gr-boards-row">
            {decision.board_state.boards.map((board, seat) => (
              <ReplayBoard
                key={seat}
                board={board}
                name={replay.seat_names[seat] ?? `P${seat}`}
                isActing={seat === decision.actor_seat}
                revealHidden={revealHidden}
              />
            ))}
          </div>

          <CenterPiles
            stockCount={decision.board_state.stock.length}
            discardTop={decision.board_state.discard.length > 0 ? decision.board_state.discard[decision.board_state.discard.length - 1] : null}
            drawnCard={decision.board_state.drawn_card}
            canDrawStock={false}
            canDrawDiscard={false}
            onDrawStock={() => {}}
            onDrawDiscard={() => {}}
            showModeToggle={false}
            discardMode={false}
            onSetDiscardMode={() => {}}
          />

          <DecisionDetail decision={decision} seatNames={replay.seat_names} />
        </main>
      ) : (
        <div className="mtx-empty-state">
          <p>That replay has no decisions.</p>
        </div>
      )}
    </div>
  )
}
