interface ScoreboardProps {
  playerNames: string[]
  scores: number[]
  currentPlayer: number
}

function Scoreboard({ playerNames, scores, currentPlayer }: ScoreboardProps) {
  return (
    <ul className="scoreboard">
      {scores.map((score, i) => (
        <li key={i} className={`score-chip ${i === currentPlayer ? 'score-chip-active' : ''}`}>
          <span>{playerNames[i]}</span>
          <span className="score-chip-value">{score}</span>
        </li>
      ))}
    </ul>
  )
}

export default Scoreboard
