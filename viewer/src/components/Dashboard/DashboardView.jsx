import { useMemo, useState } from 'react'
import { aggregateUsage } from '../../utils/aggregate'
import { shortModel, shortTitle } from '../../utils/aggregate'
import { classifyTool } from '../../utils/classify'

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}
function fmt(n) { return n == null ? '—' : Number(n).toLocaleString() }
function fmtTokens(n) {
  if (n == null) return '—'
  n = Number(n)
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k'
  return String(n)
}
function fmtCost(n) { return n == null ? '—' : '$' + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) }
function fmtCostFine(n) { return n == null ? '—' : '$' + Number(n).toFixed(4) }
function pct(x) { return x == null ? '—' : Math.round(x * 100) + '%' }
function fmtTime(ms) {
  if (ms == null) return ''
  const d = new Date(ms)
  return d.toISOString().slice(0, 16).replace('T', ' ')
}

function metricCard(k, v, sub) {
  return (
    <div className="metric" key={k}>
      <div className="k">{esc(k)}</div>
      <div className="v">{v}{sub && <small>{esc(sub)}</small>}</div>
    </div>
  )
}

export default function DashboardView({ store, onOpenRecord }) {
  const agg = useMemo(() => aggregateUsage(store.rawRecords), [store.rawRecords])
  const t = agg.totals

  const [sortKey, setSortKey] = useState('tsMin')
  const [sortDir, setSortDir] = useState('desc')

  const sortedSessions = useMemo(() => {
    const dir = sortDir === 'asc' ? 1 : -1
    return agg.sessions.slice().sort((a, b) => {
      let v1 = a[sortKey] || 0, v2 = b[sortKey] || 0
      if (typeof v1 === 'string') { v1 = v1.toLowerCase(); v2 = v2.toLowerCase() }
      return (v1 < v2 ? -1 : v1 > v2 ? 1 : 0) * dir
    })
  }, [agg.sessions, sortKey, sortDir])

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir((key === 'tsMin' || key === 'tsMax') ? 'desc' : 'asc')
    }
  }

  const span = agg.timeline.haveTime
    ? esc(fmtTime(agg.timeline.tsMin)) + ' → ' + esc(fmtTime(agg.timeline.tsMax))
    : 'ordered by load sequence (no timestamps)'

  return (
    <div>
      <div className="meta-bar">
        <span className="meta-chip"><b>{fmt(store.rawRecords.length)}</b> records</span>
        <span className="meta-chip"><b>{fmt(agg.sessions.length)}</b> sessions</span>
        <span className="meta-chip"><b>{fmt(t.calls)}</b> model calls</span>
        <span className="meta-chip"><b>{Object.keys(agg.byModel).length}</b> models</span>
        <span className="meta-chip">{span}</span>
        {t.withUsage < t.calls && (
          <span className="meta-chip">{fmt(t.calls - t.withUsage)} calls without usage data</span>
        )}
      </div>

      <div className="metrics">
        {metricCard('Sessions', fmt(agg.sessions.length), agg.sessions.length ? fmt(Math.round(t.calls / agg.sessions.length)) + ' calls/session' : null)}
        {metricCard('Total cost', fmtCost(t.cost))}
        {metricCard('Total tokens', fmtTokens(t.total))}
        {metricCard('Prompt tokens', fmtTokens(t.prompt), t.cacheRate != null ? pct(t.cacheRate) + ' cached' : null)}
        {metricCard('Completion tokens', fmtTokens(t.completion), t.reasoning ? fmtTokens(t.reasoning) + ' reasoning' : null)}
        {metricCard('Cache hit rate', pct(t.cacheRate), t.cached ? fmtTokens(t.cached) + ' cached' : null)}
        {metricCard('Avg cost / call', t.calls ? fmtCostFine(t.cost / t.calls) : '—')}
        {metricCard('Tool calls', fmt(t.toolCalls), t.deadEnds ? fmt(t.deadEnds) + ' dead ends' : null)}
        {metricCard('Unique tools', fmt(Object.keys(agg.byTool).length), fmt(t.toolCalls / Math.max(1, Object.keys(agg.byTool).length)) + ' calls/tool on avg')}
        {metricCard('Avg tokens / call', t.calls ? fmtTokens(Math.round(t.total / t.calls)) : '—')}
      </div>

      {renderTimeline(agg.timeline)}
      {renderByTool(agg.byTool)}
      {renderSessions(sortedSessions, sortKey, sortDir, handleSort, onOpenRecord)}
      {renderBreakdown('By model', agg.byModel, 'Model')}
      {renderBreakdown('By upstream', agg.byUpstream, 'Upstream')}
      {renderTopCalls(agg.topCalls, onOpenRecord)}
    </div>
  )
}

function renderTimeline(tl) {
  const bins = tl.bins
  const maxTok = bins.reduce((mx, b) => Math.max(mx, b.prompt + b.completion), 0) || 1
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">Activity over time</span>
        <span className="panel-sub">{tl.haveTime ? `${bins.length} time bins` : 'by call order'}</span>
      </div>
      <div className="tl">
        {bins.map((b, i) => {
          const tot = b.prompt + b.completion
          const hP = Math.round(b.prompt / maxTok * 110)
          const hC = Math.round(b.completion / maxTok * 110)
          const when = b.t0 != null ? fmtTime(b.t0) : ''
          const tip = (when ? when + '\n' : '') +
            b.calls + ' calls · ' + fmtTokens(tot) + ' tok · ' + fmtCostFine(b.cost)
          return (
            <div className="tl-col" key={i} title={tip}>
              <div className="tl-tip">{esc(tip).replace(/\n/g, '<br>')}</div>
              {hC > 0 && <div className="tl-bar completion" style={{ height: hC }}></div>}
              {hP > 0 && <div className="tl-bar prompt" style={{ height: hP }}></div>}
            </div>
          )
        })}
      </div>
      <div className="tl-legend">
        <span><span className="sw" style={{ background: 'var(--c-other)' }}></span>prompt tokens</span>
        <span><span className="sw" style={{ background: 'var(--c-plan-ink)' }}></span>completion tokens</span>
      </div>
    </div>
  )
}

function renderByTool(byTool) {
  const keys = Object.keys(byTool).sort((a, b) => byTool[b].calls - byTool[a].calls)
  if (!keys.length) return (
    <div className="panel"><p className="panel-sub">No tool calls detected in these records.</p></div>
  )
  const maxCalls = keys.reduce((mx, k) => Math.max(mx, byTool[k].calls), 0) || 1
  return (
    <div className="panel">
      <div className="panel-head"><span className="panel-title">By tool</span><span className="panel-sub">tool calls per category</span></div>
      <table className="bd">
        <thead><tr>
          <th>Tool</th><th className="num">Calls</th><th className="num">Dead</th><th className="num">Category</th>
        </tr></thead>
        <tbody>
          {keys.map(k => {
            const t = byTool[k]
            const w = Math.round(t.calls / maxCalls * 100)
            const deadPct = t.calls ? Math.round(t.dead / t.calls * 100) : 0
            return (
              <tr key={k}>
                <td className="name bar-cell"><span className="fill" style={{ width: w + '%' }}></span>{esc(k)}</td>
                <td className="num">{fmt(t.calls)}</td>
                <td className="num">{fmt(t.dead)}{t.dead ? <span style={{ color: 'var(--c-dead)' }}> ({deadPct}%)</span> : ''}</td>
                <td className="num">{esc(classifyTool(k).toUpperCase())}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function renderSessions(list, sortKey, sortDir, onSort, onOpenRecord) {
  if (!list.length) return (
    <div className="panel"><p className="panel-sub">No sessions detected in these records.</p></div>
  )
  const maxCost = list.reduce((mx, s) => Math.max(mx, s.cost), 0) || 1
  const columns = [
    { key: 'title', label: 'Session' },
    { key: 'tsMin', label: 'Date' },
    { key: 'model', label: 'Model' },
    { key: 'calls', label: 'Calls', num: true },
    { key: 'toolCalls', label: 'Tools', num: true },
    { key: 'total', label: 'Tokens', num: true },
    { key: 'cacheRate', label: 'Cache', num: true },
    { key: 'cost', label: 'Cost', num: true },
  ]
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">Sessions</span>
        <span className="panel-sub">{list.length} sessions · grouped by conversation thread · click to open the deepest trace</span>
      </div>
      <div className="dash-sessions-wrap">
        <table className="bd">
          <thead><tr>
            {columns.map(col => (
              <th
                key={col.key}
                className={`sort-col${col.num ? ' num' : ''}${sortKey === col.key ? (sortDir === 'asc' ? ' asc' : ' desc') : ''}`}
                onClick={() => onSort(col.key)}
              >{col.label}</th>
            ))}
          </tr></thead>
          <tbody>
            {list.map((s, i) => {
              const w = Math.round(s.cost / maxCost * 100)
              return (
                <tr key={i} className="sess-row" onClick={() => onOpenRecord(s.bestMsgIdx)}>
                  <td className="name">
                    {s.titleSource === 'opencode' && <span className="sess-src" title="Auto-generated by opencode">●</span>}
                    <span className="sess-title">{esc(shortTitle(s.title, 70))}</span>
                  </td>
                  <td className="sess-date">{s.tsMin != null ? esc(fmtTime(s.tsMin)) : '—'}</td>
                  <td className="sess-model">
                    {s.model ? esc(shortModel(s.model)) : '—'}
                    {s.modelCount > 1 && <span className="sess-model-more">+{s.modelCount - 1}</span>}
                  </td>
                  <td className="num">{fmt(s.calls)}</td>
                  <td className="num">
                    {fmt(s.toolCalls)}
                    {s.deadEnds > 0 && <span className="sess-dead"> {s.deadEnds}✗</span>}
                  </td>
                  <td className="num">{fmtTokens(s.total)}</td>
                  <td className="num">{pct(s.cacheRate)}</td>
                  <td className="num bar-cell">
                    <span className="fill" style={{ width: w + '%' }}></span>
                    <span>{fmtCostFine(s.cost)}</span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function renderBreakdown(title, byKey, colLabel) {
  const rows = Object.keys(byKey).map(k => Object.assign({ key: k }, byKey[k]))
  rows.sort((a, b) => b.cost - a.cost)
  const maxCost = rows.reduce((mx, r) => Math.max(mx, r.cost), 0) || 1
  const tot = rows.reduce((acc, r) => {
    acc.calls += r.calls; acc.total += r.total; acc.prompt += r.prompt
    acc.completion += r.completion; acc.cost += r.cost; acc.cached += r.cached
    return acc
  }, { calls: 0, total: 0, prompt: 0, completion: 0, cost: 0, cached: 0 })
  tot.cacheRate = tot.prompt ? tot.cached / tot.prompt : null
  return (
    <div className="panel">
      <div className="panel-head"><span className="panel-title">{esc(title)}</span></div>
      <table className="bd">
        <thead><tr>
          <th>{esc(colLabel)}</th><th className="num">Calls</th>
          <th className="num">Prompt</th><th className="num">Completion</th>
          <th className="num">Cache</th><th className="num">Cost</th>
        </tr></thead>
        <tbody>
          {rows.map(r => {
            const w = Math.round(r.cost / maxCost * 100)
            return (
              <tr key={r.key}>
                <td className="name">{esc(r.key)}</td>
                <td className="num">{fmt(r.calls)}</td>
                <td className="num">{fmtTokens(r.prompt)}</td>
                <td className="num">{fmtTokens(r.completion)}</td>
                <td className="num">{pct(r.cacheRate)}</td>
                <td className="num bar-cell"><span className="fill" style={{ width: w + '%' }}></span><span>{fmtCostFine(r.cost)}</span></td>
              </tr>
            )
          })}
          {rows.length > 1 && (
            <tr className="total">
              <td className="name">All</td>
              <td className="num">{fmt(tot.calls)}</td>
              <td className="num">{fmtTokens(tot.prompt)}</td>
              <td className="num">{fmtTokens(tot.completion)}</td>
              <td className="num">{pct(tot.cacheRate)}</td>
              <td className="num">{fmtCostFine(tot.cost)}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

function renderTopCalls(top, onOpenRecord) {
  if (!top.length) return (
    <div className="panel"><p className="panel-sub">No cost data in these records.</p></div>
  )
  return (
    <div className="panel">
      <div className="panel-head"><span className="panel-title">Most expensive calls</span><span className="panel-sub">top 12 by cost · click to inspect</span></div>
      {top.map(c => (
        <div key={c.idx} className="top-call" onClick={() => onOpenRecord(c.idx)}>
          <span className="cost">{fmtCostFine(c.cost)}</span>
          <span className="tc-model">{esc(c.model)}</span>
          <span className="tc-toks">
            {fmtTokens(c.prompt)} in · {fmtTokens(c.completion)} out · {c.toolCalls} tools
            {c.ts != null ? ' · ' + esc(fmtTime(c.ts)) : ''}
          </span>
        </div>
      ))}
    </div>
  )
}
