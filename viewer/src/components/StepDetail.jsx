function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

function fmt(n) { return n == null ? '—' : Number(n).toLocaleString() }

const MAX_DETAIL = 60000

function payloadSection(title, content) {
  const text = truncate(content, MAX_DETAIL)
  const truncated = String(content || '').length > MAX_DETAIL
  return (
    <div className="detail-sec">
      <h4>{esc(title)}</h4>
      <pre className="payload">{esc(text)}</pre>
      {truncated && (
        <div className="trunc-note">
          truncated at {fmt(MAX_DETAIL)} chars ({fmt(String(content).length)} total)
        </div>
      )}
    </div>
  )
}

function truncate(s, n) {
  s = String(s || '')
  return s.length > n ? s.slice(0, n) : s
}

export default function StepDetail({ step }) {
  if (step.type === 'thinking') {
    return payloadSection('Full thinking', step.text)
  }
  if (step.type === 'text') {
    return payloadSection('Full response text', step.text)
  }
  let argStr
  if (step.args == null) argStr = '(no arguments)'
  else if (typeof step.args === 'string') argStr = step.args
  else { try { argStr = JSON.stringify(step.args, null, 2) } catch (e) { argStr = String(step.args) } }
  return (
    <>
      {payloadSection('Arguments', argStr)}
      {payloadSection('Result',
        step.result === undefined ? '(no result recorded in trace)' : (step.result || '(empty result)'))}
    </>
  )
}
