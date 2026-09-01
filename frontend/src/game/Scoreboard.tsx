import type { MatchStatus, PlayerTypeName, RoundResultOut } from '../api/types'
import { finisherWasDoubled } from './scoring'

interface ScoreboardProps {
  playerNames: string[]
  playerTypes: PlayerTypeName[]
  scores: number[]
  currentPlayer: number
  status: MatchStatus
  thinkingPlayer: number | null
  roundHistory?: RoundResultOut[]
}

function Scoreboard({
  playerNames,
  playerTypes,
  scores,
  currentPlayer,
  status,
  thinkingPlayer,
  roundHistory = [],
}: ScoreboardProps) {
  return (
    <ul className="scoreboard">
      {scores.map((score, i) => {
        const isThinking = status === 'thinking' && thinkingPlayer === i
        return (
          <li key={i} className={`score-chip ${i === currentPlayer ? 'score-chip-active' : ''}`}>
            <span className="score-chip-main">
              <span>
                {playerNames[i]}
                {playerTypes[i] !== 'human' && (
                  <span className={`score-chip-bot-tag ${isThinking ? 'score-chip-thinking' : ''}`}>
                    {isThinking ? ' (Bot · thinking)' : ' (Bot)'}
                  </span>
                )}
              </span>
              <span className="score-chip-value">{score}</span>
            </span>
            {roundHistory.length > 0 && (
              <span className="score-chip-rounds">
                {roundHistory.map((round, r) => {
                  // Only ever marked beside the round's own finisher - a
                  // non-finisher's score is never doubled, so it never gets
                  // this treatment regardless of what the finisher's was.
                  const doubled = round.finisher === i && finisherWasDoubled(round.scores, round.finisher)
                  return (
                    <span key={r} className="score-chip-round">
                      {r > 0 && (
                        <span className="score-chip-round-sep" aria-hidden="true">
                          {' '}
                          ·{' '}
                        </span>
                      )}
                      {round.scores[i]}
                      {doubled && (
                        <span className="score-chip-round-doubled" title="This round's score was doubled">
                          ×2
                        </span>
                      )}
                    </span>
                  )
                })}
              </span>
            )}
          </li>
        )
      })}
    </ul>
  )
}

export default Scoreboard
