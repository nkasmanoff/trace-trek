import { useMemo } from 'react'
import { fmt, fmtTokens, fmtCost, fmtCostFine, pct, fmtTime, shortModel, shortTitle } from '../../utils/format'

function metricCard(k, v, sub) {
  return (
    <div className="metric" key={k}>
      <div className="k">{k}</div>
      <div className="v">{v}{sub && <small>{sub}</small>}</div>
    </div>
  )
}

export default function AnalyticsView({ analytics, onOpenSession }) {
  if (!analytics) {
    return <div className="meta-bar"><span className="meta-chip">loading analytics…</span></div>
  }
  const t = analytics.totals
  const uniqueTools = analytics.byTool.length

  return (
    <div>
      <div className="meta-bar">
        <span className="meta-chip"><b>{fmt(analytics.sessionCount)}</b> sessions</span>
        <span className="meta-chip"><b>{fmt(t.calls)}</b> model calls</span>
        <span className="meta-chip"><b>{analytics.byModel.length}</b> models</span>
        <span className="meta-chip"><b>{fmt(t.toolCalls)}</b> tool calls</span>
        <span className="meta-chip">across all opencode history</span>
      </div>

      <div className="metrics">
        {metricCard('Sessions', fmt(analytics.sessionCount), analytics.sessionCount ? fmt(Math.round(t.calls / analytics.sessionCount)) + ' calls/session' : null)}
        {metricCard('Total cost', fmtCost(t.cost))}
        {metricCard('Total tokens', fmtTokens(t.total))}
        {metricCard('Prompt tokens', fmtTokens(t.prompt), t.cacheRate != null ? pct(t.cacheRate) + ' cached' : null)}
        {metricCard('Completion tokens', fmtTokens(t.completion), t.reasoning ? fmtTokens(t.reasoning) + ' reasoning' : null)}
        {metricCard('Cached tokens', fmtTokens(t.cached))}
        {metricCard('Avg cost / call', t.calls ? fmtCostFine(t.cost / t.calls) : '—')}
        {metricCard('Tool calls', fmt(t.toolCalls), t.deadEnds ? fmt(t.deadEnds) + ' dead ends' : null)}
        {metricCard('Unique tools', fmt(uniqueTools), fmt(Math.round(t.toolCalls / Math.max(1, uniqueTools))) + ' calls/tool on avg')}
        {metricCard('Avg tokens / call', t.calls ? fmtTokens(Math.round(t.total / t.calls)) : '—')}
      </div>

      <Timeline timeline={analytics.timeline} />
      <ByTool byTool={analytics.byTool} />
      <ByModel byModel={analytics.byModel} />
      <TopSessions sessions={analytics.topSessions} onOpenSession={onOpenSession} />
    </div>
  )
}

function Timeline({ timeline }) {
  const maxTok = useMemo(() => timeline.reduce((mx, b) => Math.max(mx, (b.prompt || 0) + (b.completion || 0)), 0) || 1, [timeline])
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">Activity over time</span>
        <span className="panel-sub">{timeline.length} days · prompt vs completion tokens</span>
      </div>
      <div className="tl">
        {timeline.map((b, i) => {
          const tot = (b.prompt || 0) + (b.completion || 0)
          const hP = Math.round((b.prompt || 0) / maxTok * 110)
          const hC = Math.round((b.completion || 0) / maxTok * 110)
          const tip = `${b.day}\n${b.calls} calls · ${fmtTokens(tot)} tok · ${fmtCostFine(b.cost)}`
          return (
            <div className="tl-col" key={i} title={tip}>
              <div className="tl-tip">{b.day}<br />{b.calls} calls · {fmtTokens(tot)} tok · {fmtCostFine(b.cost)}</div>
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

function ByTool({ byTool }) {
  if (!byTool.length) return null
  const maxCalls = byTool.reduce((mx, t) => Math.max(mx, t.calls), 0) || 1
  return (
    <div className="panel">
      <div className="panel-head"><span className="panel-title">By tool</span><span className="panel-sub">tool calls across every session</span></div>
      <table className="bd">
        <thead><tr><th>Tool</th><th className="num">Calls</th><th className="num">Dead</th></tr></thead>
        <tbody>
          {byTool.map(t => {
            const w = Math.round(t.calls / maxCalls * 100)
            const deadPct = t.calls ? Math.round(t.dead / t.calls * 100) : 0
            return (
              <tr key={t.tool}>
                <td className="name bar-cell"><span className="fill" style={{ width: w + '%' }}></span>{t.tool}</td>
                <td className="num">{fmt(t.calls)}</td>
                <td className="num">{fmt(t.dead)}{t.dead ? <span style={{ color: 'var(--c-dead)' }}> ({deadPct}%)</span> : ''}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ByModel({ byModel }) {
  if (!byModel.length) return null
  const maxCost = byModel.reduce((mx, r) => Math.max(mx, r.cost || 0), 0) || 1
  const tot = byModel.reduce((acc, r) => {
    acc.calls += r.calls || 0; acc.prompt += r.prompt || 0
    acc.completion += r.completion || 0; acc.cost += r.cost || 0; acc.cached += r.cached || 0
    return acc
  }, { calls: 0, prompt: 0, completion: 0, cost: 0, cached: 0 })
  return (
    <div className="panel">
      <div className="panel-head"><span className="panel-title">By model</span></div>
      <table className="bd">
        <thead><tr>
          <th>Model</th><th className="num">Provider</th><th className="num">Calls</th>
          <th className="num">Prompt</th><th className="num">Completion</th><th className="num">Cost</th>
        </tr></thead>
        <tbody>
          {byModel.map((r, i) => {
            const w = Math.round((r.cost || 0) / maxCost * 100)
            return (
              <tr key={i}>
                <td className="name" title={r.model}>{shortModel(r.model) || '(unknown)'}</td>
                <td className="num">{r.provider || '—'}</td>
                <td className="num">{fmt(r.calls)}</td>
                <td className="num">{fmtTokens(r.prompt)}</td>
                <td className="num">{fmtTokens(r.completion)}</td>
                <td className="num bar-cell"><span className="fill" style={{ width: w + '%' }}></span><span>{fmtCostFine(r.cost)}</span></td>
              </tr>
            )
          })}
          {byModel.length > 1 && (
            <tr className="total">
              <td className="name">All</td><td className="num"></td>
              <td className="num">{fmt(tot.calls)}</td>
              <td className="num">{fmtTokens(tot.prompt)}</td>
              <td className="num">{fmtTokens(tot.completion)}</td>
              <td className="num">{fmtCostFine(tot.cost)}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

function TopSessions({ sessions, onOpenSession }) {
  if (!sessions.length) return null
  return (
    <div className="panel">
      <div className="panel-head"><span className="panel-title">Most expensive sessions</span><span className="panel-sub">top by cost · click to inspect</span></div>
      {sessions.map(s => (
        <div key={s.id} className="top-call" onClick={() => onOpenSession(s.id)}>
          <span className="cost">{fmtCostFine(s.cost)}</span>
          <span className="tc-model">{shortModel(s.model)}</span>
          <span className="tc-toks">
            {fmtTokens(s.prompt)} in · {fmtTokens(s.completion)} out
            {s.ts != null ? ' · ' + fmtTime(s.ts) : ''} · {shortTitle(s.title, 40)}
          </span>
        </div>
      ))}
    </div>
  )
}
