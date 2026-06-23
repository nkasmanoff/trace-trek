import { useState } from 'react'

export default function PastePanel({ store }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')

  const handleGo = () => {
    setOpen(false)
    store.loadText(text)
    setText('')
  }

  return (
    <>
      <button className="btn" onClick={() => setOpen(!open)} aria-expanded={open}
        style={{ margin: '14px 22px 0', position: 'relative', left: 'calc((100vw - 1060px) / 2)', maxWidth: 120 }}>
        Paste JSON
      </button>
      {open && (
        <div id="paste-panel">
          <textarea
            spellCheck="false"
            placeholder="Paste a trace JSON / JSONL here…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="row">
            <button className="btn" onClick={() => { setOpen(false); setText('') }}>Cancel</button>
            <button className="btn primary" onClick={handleGo}>Render trace</button>
          </div>
        </div>
      )}
    </>
  )
}
