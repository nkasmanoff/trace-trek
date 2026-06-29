import { useRef } from 'react'

export default function Header({ store, watchState, onStartWatch, onStopWatch, onPaste }) {
  const fileInput = useRef(null)

  const hasTrace = store.sessions.length > 0
  const hasEval = store.hasEvalRows()
  const browsingOpencode = store.loadedSource !== 'files'

  // tabs available in the current context
  const tabs = []
  tabs.push({ view: 'sessions', label: 'Sessions' })
  if (browsingOpencode) {
    tabs.push({ view: 'analytics', label: 'Analytics' })
  } else if (store.sessions.length > 1) {
    tabs.push({ view: 'analytics', label: 'Analytics' })
  }
  if (hasTrace) tabs.push({ view: 'anatomy', label: 'Anatomy' })
  if (hasEval) tabs.push({ view: 'eval', label: 'Eval' })
  tabs.push({ view: 'run-eval', label: 'Agent Pack' })

  return (
    <header className="bar">
      <div className="wordmark">TRACE<em>·ANATOMY</em></div>
      <div className="bar-actions">
        <div className="view-tabs" role="tablist">
          {tabs.map(t => (
            <button
              key={t.view}
              className="vtab"
              role="tab"
              aria-selected={store.view === t.view}
              onClick={() => {
                if (t.view === 'sessions') store.backToSessions()
                else store.switchView(t.view)
              }}
            >{t.label}</button>
          ))}
        </div>

        {watchState.active && (
          <span className={`live-pill ${watchState.paused ? 'paused' : ''}`}>
            <span className="live-dot"></span>
            <span>{watchState.text}</span>
          </span>
        )}

        <button className="btn" onClick={onPaste}>Paste JSON</button>
        <button className="btn" onClick={onStartWatch}>
          {watchState.active ? 'Stop watching' : 'Watch folder'}
        </button>
        <label className="btn primary" tabIndex={0}>
          Load trace files
          <input
            ref={fileInput}
            type="file"
            hidden
            multiple
            accept=".json,.jsonl,.txt,application/json"
            onChange={(e) => { if (e.target.files?.length) store.loadFiles(e.target.files); e.target.value = '' }}
          />
        </label>
      </div>
    </header>
  )
}
