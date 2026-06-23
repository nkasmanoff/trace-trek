import { useState, useMemo } from 'react'
import { fmt, fmtTokens, fmtCostFine, pct, fmtTime, shortTitle, shortModel, shortDir } from '../utils/format'
import ExportControls from './ExportControls'

const COLUMNS = [
  { key: 'title', label: 'Session' },
  { key: 'updated', label: 'Date', num: true },
  { key: 'model', label: 'Model', num: true },
  { key: 'directory', label: 'Directory' },
  { key: 'calls', label: 'Calls', num: true },
  { key: 'toolCalls', label: 'Tools', num: true },
  { key: 'tokensTotal', label: 'Tokens', num: true },
  { key: 'cacheRate', label: 'Cache', num: true },
  { key: 'cost', label: 'Cost', num: true },
]

export default function SessionList({
  sessionList, loadingList, loadingSession,
  onLoadSession, onRefresh, onLoadFiles, onDropFiles,
  onExport, onUpload, canFilter, canUpload,
}) {
  const [sortKey, setSortKey] = useState('updated')
  const [sortDir, setSortDir] = useState('desc')
  const [search, setSearch] = useState('')
  const [showSubagents, setShowSubagents] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState(() => new Set())

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir(key === 'updated' || key === 'created' ? 'desc' : 'asc') }
  }

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    let list = sessionList
    if (!showSubagents) list = list.filter(s => !s.isSubagent)
    if (q) {
      list = list.filter(s =>
        String(s.title || '').toLowerCase().includes(q) ||
        String(s.directory || '').toLowerCase().includes(q) ||
        String(s.model || '').toLowerCase().includes(q))
    }
    const dir = sortDir === 'asc' ? 1 : -1
    return list.slice().sort((a, b) => {
      let v1 = a[sortKey] ?? 0, v2 = b[sortKey] ?? 0
      if (typeof v1 === 'string') { v1 = v1.toLowerCase(); v2 = String(v2).toLowerCase() }
      return (v1 < v2 ? -1 : v1 > v2 ? 1 : 0) * dir
    })
  }, [sessionList, search, showSubagents, sortKey, sortDir])

  const maxCost = useMemo(() => filtered.reduce((mx, s) => Math.max(mx, s.cost || 0), 0) || 1, [filtered])
  const subagentCount = useMemo(() => sessionList.filter(s => s.isSubagent).length, [sessionList])

  const toggleSelect = (id, e) => {
    e.stopPropagation()
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }
  const selectAllFiltered = () => {
    setSelected(prev => {
      const next = new Set(prev)
      const allShown = filtered.every(s => next.has(s.id))
      if (allShown) filtered.forEach(s => next.delete(s.id))
      else filtered.forEach(s => next.add(s.id))
      return next
    })
  }

  // export scope: selected rows if any, otherwise all sessions
  const scopeIds = Array.from(selected)
  const exportAll = scopeIds.length === 0
  const scopeLabel = exportAll
    ? `all (${sessionList.length})`
    : `${scopeIds.length} selected`

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const files = e.dataTransfer?.files
    if (files?.length) onDropFiles(files)
  }

  return (
    <section
      id="sessions"
      className={dragging ? 'drag' : ''}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <div className="sess-toolbar">
        <input
          className="sess-search"
          type="search"
          placeholder="Filter by title, directory, or model…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <label className="sess-toggle">
          <input type="checkbox" checked={showSubagents} onChange={(e) => setShowSubagents(e.target.checked)} />
          show subagents{subagentCount ? ` (${subagentCount})` : ''}
        </label>
        <span className="sess-count">
          {filtered.length} of {sessionList.length} sessions
        </span>
        <button className="btn" onClick={onRefresh} disabled={loadingList}>
          {loadingList ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      <ExportControls
        scopeLabel={scopeLabel}
        canFilter={canFilter}
        canUpload={canUpload}
        onDownload={(opts) => onExport(scopeIds, { ...opts, all: exportAll })}
        onUpload={(opts) => onUpload(scopeIds, { ...opts, all: exportAll })}
      />

      {loadingSession && (
        <div className="sess-loading">Loading session transcript…</div>
      )}

      <div className="dash-sessions-wrap big">
        <table className="bd">
          <thead>
            <tr>
              <th className="sel-col">
                <input
                  type="checkbox"
                  checked={filtered.length > 0 && filtered.every(s => selected.has(s.id))}
                  onChange={selectAllFiltered}
                  title="Select all shown"
                />
              </th>
              {COLUMNS.map(col => (
                <th
                  key={col.key}
                  className={`sort-col${col.num ? ' num' : ''}${sortKey === col.key ? (sortDir === 'asc' ? ' asc' : ' desc') : ''}`}
                  onClick={() => handleSort(col.key)}
                >{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(s => {
              const w = Math.round((s.cost || 0) / maxCost * 100)
              return (
                <tr key={s.id} className="sess-row" onClick={() => onLoadSession(s.id)} title={`Open: ${s.title}`}>
                  <td className="sel-col" onClick={(e) => toggleSelect(s.id, e)}>
                    <input type="checkbox" checked={selected.has(s.id)} readOnly />
                  </td>
                  <td className="name">
                    {s.isSubagent && <span className="sess-src" title="Subagent / child session">↳</span>}
                    <span className="sess-title">{shortTitle(s.title, 70)}</span>
                  </td>
                  <td className="sess-date">{fmtTime(s.updated || s.created) || '—'}</td>
                  <td className="sess-model" title={s.model}>{s.model ? shortModel(s.model) : '—'}</td>
                  <td className="sess-dir" title={s.directory}>{shortDir(s.directory)}</td>
                  <td className="num">{fmt(s.calls)}</td>
                  <td className="num">
                    {fmt(s.toolCalls)}
                    {s.deadEnds > 0 && <span className="sess-dead"> {s.deadEnds}✗</span>}
                  </td>
                  <td className="num">{fmtTokens(s.tokensTotal)}</td>
                  <td className="num">{s.cacheRate != null ? pct(s.cacheRate) : '—'}</td>
                  <td className="num bar-cell">
                    <span className="fill" style={{ width: w + '%' }}></span>
                    <span>{s.cost ? fmtCostFine(s.cost) : '—'}</span>
                  </td>
                </tr>
              )
            })}
            {filtered.length === 0 && (
              <tr><td colSpan={COLUMNS.length + 1} className="sess-empty">
                {sessionList.length ? 'No sessions match your filter.' : 'No sessions found. Drop trace files here to load them directly.'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="sess-hint">
        Click any session to view its trace anatomy. Drop JSON/JSONL trace files anywhere to load from disk.
      </p>
    </section>
  )
}
