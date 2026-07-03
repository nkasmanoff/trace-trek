import { useEffect, useState, useRef } from 'react'
import useTraceStore from './hooks/useTraceStore'
import useWatchFolder from './hooks/useWatchFolder'
import Header from './components/Header'
import SessionList from './components/SessionList'
import AnalyticsView from './components/Analytics/AnalyticsView'
import DashboardView from './components/Dashboard/DashboardView'
import AnatomyView from './components/AnatomyView'
import EvalView from './components/Eval/EvalView'
import RunEvalView from './components/EvalRunner/RunEvalView'
import './App.css'

export default function App() {
  const store = useTraceStore()
  const { watchState, startWatch, stopWatch } = useWatchFolder(store)
  const [pasteOpen, setPasteOpen] = useState(false)
  const [pasteText, setPasteText] = useState('')
  const didInit = useRef(false)

  useEffect(() => {
    if (didInit.current) return
    didInit.current = true
    store.loadPreloadedEval()
    store.fetchSessionList()
    store.fetchAnalytics()
    store.fetchCapabilities()
  }, [])

  const handleStartWatch = () => {
    if (watchState.active) stopWatch()
    else startWatch()
  }

  const handlePasteGo = () => {
    setPasteOpen(false)
    store.loadText(pasteText)
    setPasteText('')
  }

  const view = store.view
  const browsingOpencode = store.loadedSource !== 'files'

  return (
    <>
      <Header
        store={store}
        watchState={watchState}
        onStartWatch={handleStartWatch}
        onStopWatch={stopWatch}
        onPaste={() => setPasteOpen(o => !o)}
      />

      {pasteOpen && (
        <div id="paste-panel">
          <textarea
            spellCheck="false"
            placeholder="Paste a trace JSON / JSONL here…"
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
          />
          <div className="row">
            <button className="btn" onClick={() => { setPasteOpen(false); setPasteText('') }}>Cancel</button>
            <button className="btn primary" onClick={handlePasteGo}>Render trace</button>
          </div>
        </div>
      )}

      <main>
        {store.error && <div className="err">{store.error}</div>}

        {view === 'sessions' && (
          <SessionList
            sessionList={store.sessionList}
            sessionTotal={store.sessionTotal}
            loadingList={store.loadingList}
            loadingMore={store.loadingMore}
            loadingSession={store.loadingSession}
            pageSize={store.PAGE_SIZE}
            onLoadSession={store.loadSession}
            onFetchPage={store.fetchSessionList}
            onLoadFiles={store.loadFiles}
            onDropFiles={store.loadFiles}
            onExport={store.exportSessions}
            onUpload={store.uploadSessions}
            canFilter={store.capabilities.filter}
            canUpload={store.capabilities.hf}
          />
        )}

        {view === 'analytics' && browsingOpencode && (
          <AnalyticsView analytics={store.analytics} onOpenSession={store.loadSession} />
        )}
        {view === 'analytics' && !browsingOpencode && (
          <DashboardView store={store} onOpenRecord={store.openRecordInAnatomy} />
        )}

        {view === 'anatomy' && store.sessions.length > 0 && (
          <section id="viewer">
            <div className="controls" style={{ marginBottom: 16 }}>
              <button className="btn" onClick={store.backToSessions}>← All sessions</button>
              {store.loadedSource === 'opencode' && store.sessions[store.activeIdx]?.meta?.sessionId && (
                <>
                  <button
                    className="btn"
                    onClick={() => store.exportSessions(store.sessions[store.activeIdx].meta.sessionId, { filter: false })}
                    title="Download this session as a raw SFT record (messages + tools + reconstructed system prompt)"
                  >Download SFT (raw)</button>
                  {store.capabilities.filter && (
                    <button
                      className="btn"
                      onClick={() => store.exportSessions(store.sessions[store.activeIdx].meta.sessionId, { filter: true })}
                      title="Download this session as an SFT record after build_dataset.py's quality gate + sanitization"
                    >Download SFT (filtered)</button>
                  )}
                </>
              )}
            </div>
            <AnatomyView store={store} />
          </section>
        )}

        {view === 'eval' && store.hasEvalRows() && (
          <EvalView store={store} onOpenRecord={store.openRecordInAnatomy} />
        )}

        {view === 'run-eval' && (
          <RunEvalView store={store} />
        )}
      </main>

      <footer style={{ maxWidth: 1060, margin: '30px auto 0', padding: '0 22px', color: 'var(--muted)', fontFamily: 'var(--mono)', fontSize: 11 }}>
        Runs entirely on your machine — nothing is uploaded anywhere. Dead-end and read/write classification are heuristics; click any step to verify against the raw payload.
      </footer>
    </>
  )
}
