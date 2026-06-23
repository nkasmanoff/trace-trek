import { useRef, useState } from 'react'

export default function EmptyState({ store, error }) {
  const fileInputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length) {
      store.loadFiles(e.target.files)
    }
    e.target.value = ''
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const files = e.dataTransfer && e.dataTransfer.files
    if (files && files.length) store.loadFiles(files)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      fileInputRef.current?.click()
    }
  }

  return (
    <section
      id="empty"
      className={dragging ? 'drag' : ''}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragEnter={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <div className="skeleton" aria-hidden="true">
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-user)' }}></span><span className="sk-line" style={{ width: '60%' }}></span></div>
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-think)' }}></span><span className="sk-line"></span></div>
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-read)' }}></span><span className="sk-line" style={{ width: '80%' }}></span></div>
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-read)' }}></span><span className="sk-line" style={{ width: '70%' }}></span></div>
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-dead)' }}></span><span className="sk-line" style={{ width: '50%' }}></span></div>
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-mut)' }}></span><span className="sk-line" style={{ width: '85%' }}></span></div>
        <div className="sk-row"><span className="sk-dot" style={{ background: 'var(--c-text)' }}></span><span className="sk-line" style={{ width: '65%' }}></span></div>
      </div>
      <h1>Drop agentic traces here</h1>
      <p>
        One file in, one anatomy out: turns on a spine, tool calls classified and color-coded,
        dead ends flagged, every step expandable to its full arguments, results, and thinking.
        Drop a <b>folder of proxy records</b> (or many files) to unlock the <b>Analytics</b>
        dashboard: token usage, spend, cache-hit rate and latency across every run.
      </p>
      <p className="formats">
        Understands: <code>OpenRouter request/response logs</code> ·
        <code>OpenAI messages + tool_calls</code> ·
        <code>Anthropic content blocks</code> ·
        <code>{'{"messages":[…]} exports'}</code> · <code>JSONL</code>
      </p>
      <div style={{ marginTop: 16, display: 'flex', gap: 10, justifyContent: 'center' }}>
        <label className="btn primary" tabIndex={0} onKeyDown={handleKeyDown}>
          Load trace files
          <input ref={fileInputRef} type="file" hidden multiple accept=".json,.jsonl,.txt,application/json" onChange={handleFileChange} />
        </label>
        <button className="btn" onClick={() => store.loadPreloadedEval && document.getElementById('file-dir')?.click()}>
          Watch folder
        </button>
        <input id="file-dir" type="file" hidden webkitdirectory="true" directory="true" multiple
          onChange={(e) => {
            const files = Array.prototype.slice.call(e.target.files || []).filter(f => /\.(json|jsonl|txt)$/i.test(f.name))
            if (files.length) store.loadFiles(files)
            e.target.value = ''
          }}
        />
      </div>
      {error && <div className="err">Could not read trace: {error}</div>}
    </section>
  )
}
