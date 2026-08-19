import type { MatchStatus, PlayerTypeName } from '../api/types'

interface ScoreboardProps {
  playerNames: string[]
  playerTypes: PlayerTypeName[]
  scores: number[]
  currentPlayer: number
  status: MatchStatus
  thinkingPlayer: number | null
}

function Scoreboard({ playerNames, playerTypes, scores, currentPlayer, status, thinkingPlayer }: ScoreboardProps) {
  return (
    <ul className="scoreboard">
      {scores.map((score, i) => {
        const isThinking = status === 'thinking' && thinkingPlayer === i
        return (
          <li key={i} className={`score-chip ${i === currentPlayer ? 'score-chip-active' : ''}`}>
            <span>
              {playerNames[i]}
              {playerTypes[i] !== 'human' && (
                <span className={`score-chip-bot-tag ${isThinking ? 'score-chip-thinking' : ''}`}>
                  {isThinking ? ' (Bot · thinking)' : ' (Bot)'}
                </span>
              )}
            </span>
            <span className="score-chip-value">{score}</span>
          </li>
        )
      })}
    </ul>
  )
}

export default Scoreboard
