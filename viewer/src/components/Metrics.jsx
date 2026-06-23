function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}
function fmt(n) { return n == null ? '—' : Number(n).toLocaleString() }

function metricCard(k, v, sub) {
  return (
    <div className="metric" key={k}>
      <div className="k">{esc(k)}</div>
      <div className="v">
        {v}
        {sub && <small>{esc(sub)}</small>}
      </div>
    </div>
  )
}

export default function Metrics({ metrics }) {
  if (!metrics) return null
  const m = metrics
  const items = []
  items.push(metricCard('User turns', fmt(m.turns)))
  items.push(metricCard('Tool calls', fmt(m.toolCalls)))
  items.push(metricCard('Files written / edits', fmt(m.writes)))
  items.push(metricCard('Subagent runs', fmt(m.subagents)))
  items.push(metricCard('Dead ends', fmt(m.deadEnds)))
  if (m.thinking) items.push(metricCard('Thinking blocks', fmt(m.thinking)))
  if (m.promptTokens != null)
    items.push(metricCard('Prompt tokens', fmt(m.promptTokens), m.cacheRate != null ? Math.round(m.cacheRate * 100) + '% cached' : null))
  if (m.cost != null) items.push(metricCard('Last-call cost', '$' + Number(m.cost).toFixed(4)))
  if (m.replayScore != null) items.push(metricCard('Replay grade', Number(m.replayScore).toFixed(2)))
  return <div className="metrics">{items}</div>
}
