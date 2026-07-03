import { useState, useMemo, useEffect, useRef } from 'react'
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

// calls/toolCalls are per-page aggregates that aren't stored on the session
// row, so they can't be sorted server-side; the rest map to SQL columns.
const SERVER_SORTABLE = new Set(['title', 'updated', 'model', 'directory', 'tokensTotal', 'cacheRate', 'cost'])

export default function SessionList({
  sessionList, sessionTotal, loadingList, loadingMore, loadingSession, pageSize = 15,
  onLoadSession, onFetchPage, onLoadFiles, onDropFiles,
  onExport, onUpload, canFilter, canUpload,
}) {
  const [sortKey, setSortKey] = useState('updated')
  const [sortDir, setSortDir] = useState('desc')
  const [search, setSearch] = useState('')
  const [showSubagents, setShowSubagents] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState(() => new Set())

  // Debounce the search box, then (re)fetch the first page server-side whenever
  // the query, sort, or subagent toggle changes.
  const didMount = useRef(false)
  const [debouncedSearch, setDebouncedSearch] = useState('')
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 250)
    return () => clearTimeout(t)
  }, [search])

  useEffect(() => {
    // skip the very first run — App already loaded the initial page
    if (!didMount.current) { didMount.current = true; return }
    onFetchPage({
      offset: 0,
      search: debouncedSearch,
      sort: sortKey,
      dir: sortDir,
      subagents: showSubagents,
      append: false,
    })
  }, [debouncedSearch, sortKey, sortDir, showSubagents, onFetchPage])

  const handleSort = (key) => {
    if (!SERVER_SORTABLE.has(key)) return
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir(key === 'updated' || key === 'created' ? 'desc' : 'asc') }
  }

  const loadMore = () => {
    onFetchPage({
      offset: sessionList.length,
      search: debouncedSearch,
      sort: sortKey,
      dir: sortDir,
      subagents: showSubagents,
      append: true,
    })
  }

  // Rows come pre-filtered/sorted/paged from the server.
  const rows = sessionList
  const hasMore = rows.length < sessionTotal

  const maxCost = useMemo(() => rows.reduce((mx, s) => Math.max(mx, s.cost || 0), 0) || 1, [rows])

  const toggleSelect = (id, e) => {
    e.stopPropagation()
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }
  const selectAllShown = () => {
    setSelected(prev => {
      const next = new Set(prev)
      const allShown = rows.every(s => next.has(s.id))
      if (allShown) rows.forEach(s => next.delete(s.id))
      else rows.forEach(s => next.add(s.id))
      return next
    })
  }

  // export scope: selected rows if any, otherwise all sessions
  const scopeIds = Array.from(selected)
  const exportAll = scopeIds.length === 0
  const scopeLabel = exportAll
    ? `all (${sessionTotal})`
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
          show subagents
        </label>
        <span className="sess-count">
          {rows.length} of {sessionTotal} sessions
        </span>
        <button
          className="btn"
          onClick={() => onFetchPage({ offset: 0, search: debouncedSearch, sort: sortKey, dir: sortDir, subagents: showSubagents, append: false })}
          disabled={loadingList}
        >
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
                  checked={rows.length > 0 && rows.every(s => selected.has(s.id))}
                  onChange={selectAllShown}
                  title="Select all shown"
                />
              </th>
              {COLUMNS.map(col => (
                <th
                  key={col.key}
                  className={`${SERVER_SORTABLE.has(col.key) ? 'sort-col' : ''}${col.num ? ' num' : ''}${sortKey === col.key ? (sortDir === 'asc' ? ' asc' : ' desc') : ''}`}
                  onClick={() => handleSort(col.key)}
                >{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(s => {
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
            {rows.length === 0 && !loadingList && (
              <tr><td colSpan={COLUMNS.length + 1} className="sess-empty">
                {debouncedSearch ? 'No sessions match your filter.' : 'No sessions found. Drop trace files here to load them directly.'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      {hasMore && (
        <div className="sess-more">
          <button className="btn" onClick={loadMore} disabled={loadingMore}>
            {loadingMore ? 'Loading…' : `Load ${Math.min(pageSize, sessionTotal - rows.length)} more`}
          </button>
        </div>
      )}

      <p className="sess-hint">
        Click any session to view its trace anatomy. Drop JSON/JSONL trace files anywhere to load from disk.
      </p>
    </section>
  )
}
