import { useState } from 'react'

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}
function fmt(n) { return n == null ? '—' : Number(n).toLocaleString() }

const MAX_DETAIL = 60000

export default function SystemCard({ system, reconstructed, tools }) {
  const [open, setOpen] = useState(false)
  const [toolsOpen, setToolsOpen] = useState(false)

  if (!system && !(tools && tools.length)) return null

  const truncated = system && system.length > MAX_DETAIL

  return (
    <>
      {system && (
        <div className={`sys-card${open ? ' open' : ''}`}>
          <button onClick={() => setOpen(!open)} aria-expanded={open}>
            <span style={{ color: 'var(--muted)' }}>{open ? '▾' : '▸'}</span>
            SYSTEM PROMPT
            {reconstructed && (
              <span
                className="recon-badge"
                title={`Reconstructed from the opencode harness (role: ${reconstructed.role}). opencode does not persist the system prompt; this rebuilds it from captured prompt templates with this session's model, working directory, and date filled in.`}
              >reconstructed</span>
            )}
            <span className="count">{fmt(system.length)} chars</span>
          </button>
          <pre className="payload">
            {esc(truncated ? system.slice(0, MAX_DETAIL) : system)}
          </pre>
        </div>
      )}

      {tools && tools.length > 0 && (
        <div className={`sys-card${toolsOpen ? ' open' : ''}`}>
          <button onClick={() => setToolsOpen(!toolsOpen)} aria-expanded={toolsOpen}>
            <span style={{ color: 'var(--muted)' }}>{toolsOpen ? '▾' : '▸'}</span>
            TOOL SCHEMAS
            {reconstructed && <span className="recon-badge" title="The exact tool JSON schemas opencode exposed, harvested from captured harness logs.">reconstructed</span>}
            <span className="count">{tools.length} tools</span>
          </button>
          <pre className="payload">
            {esc(JSON.stringify(tools, null, 2).slice(0, MAX_DETAIL))}
          </pre>
        </div>
      )}
    </>
  )
}
