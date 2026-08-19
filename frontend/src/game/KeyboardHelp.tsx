import { createPortal } from 'react-dom'

interface ShortcutEntry {
  keys: string[]
  description: string
}

const SHORTCUT_GROUPS: { title: string; entries: ShortcutEntry[] }[] = [
  {
    title: 'Board',
    entries: [
      { keys: ['↑', '↓', '←', '→'], description: 'Move between your available cards' },
      { keys: ['Enter', 'Space'], description: 'Flip, place, or discard & reveal the focused card' },
    ],
  },
  {
    title: 'Drawing',
    entries: [
      { keys: ['Q'], description: 'Draw from the stock (or place the drawn card)' },
      { keys: ['W'], description: 'Draw from the discard pile (or discard & reveal instead)' },
    ],
  },
  {
    title: 'Round',
    entries: [
      { keys: ['N'], description: 'Start next round, once a round ends' },
      { keys: ['P'], description: 'Restart the game (confirms first, unless the game already ended)' },
    ],
  },
  {
    title: 'Help',
    entries: [
      { keys: ['?'], description: 'Show or hide this shortcuts list' },
      { keys: ['Esc'], description: 'Close this list' },
    ],
  },
]

interface KeyboardHelpProps {
  open: boolean
  onClose: () => void
}

/** Reference modal for the gameplay keyboard shortcuts, opened via the "?" key or its trigger button. */
function KeyboardHelp({ open, onClose }: KeyboardHelpProps) {
  if (!open) return null

  return createPortal(
    <div className="history-modal-backdrop" onClick={onClose}>
      <div
        className="history-modal shortcuts-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="history-modal-header">
          <h2>Keyboard shortcuts</h2>
          <button type="button" className="history-modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="history-modal-body shortcuts-modal-body">
          {SHORTCUT_GROUPS.map((group) => (
            <section key={group.title} className="shortcuts-group">
              <h3>{group.title}</h3>
              <ul className="shortcuts-list">
                {group.entries.map((entry) => (
                  <li key={entry.description} className="shortcuts-row">
                    <span className="shortcuts-keys">
                      {entry.keys.map((key) => (
                        <kbd key={key} className="shortcut-key">
                          {key}
                        </kbd>
                      ))}
                    </span>
                    <span className="shortcuts-desc">{entry.description}</span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  )
}

export default KeyboardHelp
