import { createPortal } from 'react-dom'

interface RestartConfirmProps {
  open: boolean
  onConfirm: () => void
  onCancel: () => void
}

/** Guards the restart shortcut/button mid-game so a stray "P" press can't silently discard an in-progress match. */
function RestartConfirm({ open, onConfirm, onCancel }: RestartConfirmProps) {
  if (!open) return null

  return createPortal(
    <div className="history-modal-backdrop" onClick={onCancel}>
      <div
        className="history-modal restart-confirm-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Restart the game?"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="history-modal-header">
          <h2>Restart the game?</h2>
          <button type="button" className="history-modal-close" onClick={onCancel} aria-label="Close">
            ×
          </button>
        </div>
        <div className="history-modal-body">
          <p>This abandons the current match and its history. There&rsquo;s no way back once you confirm.</p>
          <div className="restart-confirm-actions">
            <button type="button" className="btn-danger" onClick={onConfirm}>
              Restart
            </button>
            <button type="button" className="btn-plain" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

export default RestartConfirm
