import { useMemo } from 'react'
import { aggregateEval } from '../../utils/eval'

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}
function fmt(n) { return n == null ? '—' : Number(n).toLocaleString() }
function fmtPct2(x) { return x == null ? '—' : (x * 100).toFixed(0) + '%' }
function fmtScore(x) { return x == null ? '—' : Number(x).toFixed(2) }
function escShort(s, n) { return esc(String(s || '').slice(0, n || 60)) }

function metricCard(k, v, sub) {
  return (
    <div className="metric" key={k}>
      <div className="k">{esc(k)}</div>
      <div className="v">{v}{sub && <small>{esc(sub)}</small>}</div>
    </div>
  )
}

export default function EvalView({ store, onOpenRecord }) {
  const records = useMemo(() => store.preloadedEval.concat(store.rawRecords), [store.preloadedEval, store.rawRecords])
  const agg = useMemo(() => aggregateEval(records), [records])

  if (!agg.count) {
    return (
      <div className="panel ap-empty-eval">
        <div className="panel-head">
          <span className="panel-title">Eval results</span>
          <span className="panel-sub">Load eval JSON or use Agent Pack runs</span>
        </div>
        <p className="ap-muted">
          No eval rows in the current session. Load an <code>eval-results.json</code> file,
          or open the <button type="button" className="ap-linkish" onClick={() => store.switchView('run-eval')}>Agent Pack</button> tab
          to launch problems and inspect captured artifacts.
        </p>
      </div>
    )
  }

  const primary = agg.runs[0]
  const hasCode = primary.byType['code']
  const hasKnow = primary.byType['knowledge']

  return (
    <div>
      <div className="meta-bar">
        <span className="meta-chip"><b>{agg.count}</b> results</span>
        <span className="meta-chip"><b>{agg.runs.length}</b> run{agg.runs.length === 1 ? '' : 's'}</span>
        <span className="meta-chip"><b>{agg.taskIds.length}</b> unique tasks</span>
      </div>

      <div className="metrics">
        {metricCard('Best run', escShort(primary.label, 30))}
        {metricCard('Pass rate', fmtPct2(primary.passRate), primary.passed + '/' + primary.total)}
        {metricCard('Mean score', fmtScore(primary.avgScore))}
        {metricCard('Errors', fmt(primary.errors), primary.errors ? 'tasks that crashed' : null)}
        {hasCode && metricCard('Code pass', fmtPct2(hasCode.passRate), hasCode.passed + '/' + hasCode.total)}
        {hasKnow && metricCard('Knowledge pass', fmtPct2(hasKnow.passRate), hasKnow.passed + '/' + hasKnow.total)}
      </div>

      <div className="panel">
        <div className="panel-head"><span className="panel-title">Pass rate by run</span>
          <span className="panel-sub">overall and by task type · highest first</span></div>
        {agg.runs.length === 1 ? (
          <p className="panel-sub">One run loaded. Load a second results file (merge) or add results with a different <code>run</code> label to compare.</p>
        ) : (
          <table className="bd">
            <thead><tr>
              <th>Run</th><th className="num">Total</th><th className="num">Pass</th>
              <th className="num">Pass %</th><th className="num">Score μ</th>
              <th className="num">Errors</th><th className="num">Code %</th><th className="num">Know %</th>
            </tr></thead>
            <tbody>
              {agg.runs.map(r => (
                <tr key={r.label}>
                  <td className="name">{esc(r.label)}</td>
                  <td className="num">{fmt(r.total)}</td>
                  <td className={'num eval-' + (r.passed === r.total ? 'pass' : r.passed ? '' : 'fail')}>{fmt(r.passed)}</td>
                  <td className="num">{fmtPct2(r.passRate)}</td>
                  <td className="num">{fmtScore(r.avgScore)}</td>
                  <td className="num">{fmt(r.errors)}</td>
                  <td className="num">{fmtPct2((r.byType['code'] || {}).passRate)}</td>
                  <td className="num">{fmtPct2((r.byType['knowledge'] || {}).passRate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {agg.flip && renderFlip(agg.flip)}

      <div className="panel">
        <div className="panel-head"><span className="panel-title">Tasks</span>
          <span className="panel-sub">every graded task</span></div>
        <table className="bd">
          <thead><tr>
            <th>Task</th><th className="num">Type</th>
            {agg.runs.map(r => <th key={r.label} className="num">{esc(r.label)}</th>)}
          </tr></thead>
          <tbody>
            {agg.taskIds.map(id => {
              const m = agg.taskMeta[id]
              const recIdx = agg.runs.map(r => r.tasks[id]).find(t => t && t.recIdx != null)?.recIdx
              const clickable = recIdx != null && onOpenRecord
              return (
                <tr
                  key={id}
                  className={'eval-task-row' + (clickable ? ' clickable' : '')}
                  onClick={clickable ? () => onOpenRecord(recIdx) : undefined}
                >
                  <td className="name">{escShort(id, 16)}</td>
                  <td className="num">{esc(m.type)}</td>
                  {agg.runs.map(r => {
                    const t = r.tasks[id]
                    const cls = t ? (t.passed ? 'eval-pass' : t.error ? 'eval-fail' : 'eval-fail') : 'eval-na'
                    const txt = t ? (t.passed ? 'PASS' : t.error ? 'ERR' : 'FAIL') : '—'
                    const detail = t ? fmtScore(t.score) : ''
                    return (
                      <td key={r.label} className={'num ' + cls}>
                        {txt}
                        {detail && <span className="eval-tnote"> {detail}</span>}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function renderFlip(f) {
  return (
    <div className="panel" id="eval-flip-panel">
      <div className="panel-head"><span className="panel-title">Per-task comparison</span>
        <span className="panel-sub">{esc(f.baseline)} (baseline) → {esc(f.candidate)} (candidate)</span></div>
      <div className="eval-flip-grid">
        <div className="metric"><div className="k">Gains</div><div className="v">{f.gains.length}</div></div>
        <div className="metric"><div className="k">Regressions</div><div className="v">{f.regressions.length}</div></div>
        <div className="metric"><div className="k">Ties</div><div className="v">{f.ties.length}</div></div>
        <div className="metric"><div className="k">Common tasks</div><div className="v">{f.common}</div></div>
      </div>
      {f.gains.map(g => (
        <div key={g.id} className="flip-cell flip-gain">
          ▲ {escShort(g.id, 16)} — {g.meta.type} · score {fmtScore(g.base.score)} → {fmtScore(g.cand.score)}
        </div>
      ))}
      {f.regressions.map(r => (
        <div key={r.id} className="flip-cell flip-reg">
          ▼ {escShort(r.id, 16)} — {r.meta.type} · score {fmtScore(r.base.score)} → {fmtScore(r.cand.score)}
        </div>
      ))}
      {f.ties.slice(0, 10).map(t => (
        <div key={t.id} className="flip-cell flip-tie">
          = {escShort(t.id, 16)} · {t.meta.type} · score {fmtScore(t.base.score)}
        </div>
      ))}
      {f.ties.length > 10 && <div className="flip-cell flip-tie">… +{f.ties.length - 10} more ties</div>}
    </div>
  )
}
