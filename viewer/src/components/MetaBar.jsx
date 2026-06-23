export default function MetaBar({ sessions, activeIdx, meta, onChangeSession }) {
  const items = []
  if (sessions.length > 1) {
    items.push(
      <label key="sel">
        record{' '}
        <select value={activeIdx} onChange={(e) => onChangeSession(+e.target.value)}>
          {sessions.map((s, i) => (
            <option key={i} value={i}>
              #{i + 1} · {s.meta?.model || 'unknown model'} · {s.messages.length} msgs
            </option>
          ))}
        </select>
      </label>
    )
  }
  const chips = []
  if (meta?.model) chips.push(<span className="meta-chip" key="model">model <b>{meta.model}</b></span>)
  if (meta?.upstream) chips.push(<span className="meta-chip" key="upstream">via <b>{meta.upstream}</b></span>)
  if (meta?.source) chips.push(<span className="meta-chip" key="source">source <b>{meta.source}</b></span>)
  if (meta?.timestamp) chips.push(<span className="meta-chip" key="ts">{meta.timestamp}</span>)
  if (meta?.elapsed_ms != null) chips.push(<span className="meta-chip" key="elapsed"><b>{(meta.elapsed_ms / 1000).toFixed(1)}s</b> last call</span>)
  if (meta?.replay) chips.push(
    <span className="meta-chip" key="replay">
      replay <b>{meta.replay.type || ''}{meta.replay.pass != null ? (meta.replay.pass ? ' · pass' : ' · fail') : ''}</b>
    </span>
  )
  return <div className="meta-bar">{items}{chips}</div>
}
