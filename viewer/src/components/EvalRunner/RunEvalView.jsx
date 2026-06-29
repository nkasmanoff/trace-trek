import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

const PRESET_MODELS = [
  { id: 'opencode/big-pickle', label: 'Big Pickle' },
  { id: 'frontier/anthropic/claude-opus-4.8', label: 'Claude Opus 4.8' },
  { id: 'openrouter/cohere/north-mini-code:free', label: 'North Mini (free)' },
  { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B-A3B' },
]

const LIFECYCLE = [
  { step: '1', title: 'Prepare', detail: 'Isolated git workspace + task prompt' },
  { step: '2', title: 'Agent', detail: 'Model edits code or writes AGENT_FINAL_ANSWER.md' },
  { step: '3', title: 'Verify', detail: 'pytest checks repair & comprehension tasks' },
  { step: '4', title: 'Capture', detail: 'Diff, verification output, usage, rubric' },
]

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

function fmtDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const diff = Math.floor((Date.now() - d.getTime()) / 1000)
  if (diff < 60) return 'just now'
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago'
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago'
  return d.toLocaleDateString()
}

function fmtPct(n, d) {
  if (!d) return '—'
  return Math.round((n / d) * 100) + '%'
}

function fmtTokens(n) {
  if (n == null || n === 0) return '—'
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k'
  return String(n)
}

function kindLabel(kind) {
  return kind === 'comprehension' ? 'comprehension' : 'repair'
}

function difficultyClass(level) {
  return `ap-diff ap-diff-${level || 'medium'}`
}

function statusClass(status, passed) {
  if (status === 'passed' || passed === true) return 'ap-status ap-pass'
  if (status === 'failed' || passed === false) return 'ap-status ap-fail'
  if (status === 'running') return 'ap-status ap-run'
  if (status === 'error') return 'ap-status ap-err'
  if (status === 'interrupted' || status === 'cancelled') return 'ap-status ap-interrupted'
  return 'ap-status ap-pending'
}

function statusText(status, passed) {
  if (status === 'passed' || passed === true) return 'Pass'
  if (status === 'failed' || passed === false) return 'Fail'
  if (status === 'running') return 'Running'
  if (status === 'error') return 'Error'
  if (status === 'interrupted') return 'Interrupted'
  if (status === 'cancelled') return 'Cancelled'
  if (status === 'pending') return 'Pending'
  if (passed == null) return 'Uncaptured'
  return '—'
}

function formatAgentLog(prob) {
  const parts = []
  if (prob.output) parts.push('# spawn error\n' + prob.output)
  parts.push('# stdout\n' + (prob.stdout ? prob.stdout : '(empty)'))
  parts.push('# stderr\n' + (prob.stderr ? prob.stderr : '(empty)'))
  return parts.join('\n\n')
}

// True when the agent process produced no usable output at all — a strong
// signal that it never actually ran (vs. ran and got the answer wrong).
function agentDidNotRun(prob) {
  return !prob.sessionId && !prob.stdout && (prob.status === 'failed' || prob.status === 'error')
}

function formatFailureDetail(summary) {
  if (!summary) return ''
  const lines = []
  if (summary.headline) lines.push(summary.headline)
  for (const hint of summary.hints || []) lines.push(`• ${hint}`)
  for (const failure of summary.failures || []) {
    const prefix = failure.test ? `${failure.test}: ` : ''
    lines.push(`• ${prefix}${failure.message}`)
  }
  return lines.join('\n')
}

function FailureReason({ summary, passed, onExpand }) {
  const didPass = passed === true || summary?.passed === true
  if (didPass) return <span className="ap-muted">—</span>
  if (!summary) return <span className="ap-muted">—</span>
  const detail = formatFailureDetail(summary)
  return (
    <button
      type="button"
      className="ap-failure-reason"
      title={detail}
      onClick={() => onExpand?.({
        title: 'Why it failed',
        subtitle: summary.headline || 'Failure analysis',
        loading: false,
        content: detail,
        error: null,
      })}
    >
      {esc(summary.headline || 'See verification output')}
    </button>
  )
}

function ProblemCard({ problem, selected, onToggle }) {
  return (
    <button
      type="button"
      className={'ap-card' + (selected ? ' selected' : '')}
      onClick={() => onToggle(problem.id)}
    >
      <div className="ap-card-top">
        <span className="ap-num">{String(problem.number).padStart(2, '0')}</span>
        <span className={'ap-kind ap-kind-' + problem.kind}>{kindLabel(problem.kind)}</span>
      </div>
      <div className="ap-card-title">{esc(problem.name)}</div>
      <div className="ap-card-meta">
        <span className={difficultyClass(problem.difficulty)}>{problem.difficulty}</span>
        <span className="ap-skills">{problem.skills.slice(0, 2).join(' · ')}</span>
      </div>
    </button>
  )
}

function LifecycleRail() {
  return (
    <div className="ap-lifecycle">
      {LIFECYCLE.map((item, i) => (
        <div key={item.step} className="ap-life-step">
          <div className="ap-life-num">{item.step}</div>
          <div className="ap-life-body">
            <div className="ap-life-title">{item.title}</div>
            <div className="ap-life-detail">{item.detail}</div>
          </div>
          {i < LIFECYCLE.length - 1 && <div className="ap-life-arrow" aria-hidden="true">→</div>}
        </div>
      ))}
    </div>
  )
}

function ArtifactPanel({ detail, onClose }) {
  if (!detail) return null
  return (
    <div className="ap-artifact-panel">
      <div className="ap-artifact-head">
        <div>
          <div className="ap-artifact-title">{esc(detail.title)}</div>
          <div className="ap-artifact-sub">{esc(detail.subtitle)}</div>
        </div>
        <button type="button" className="btn" onClick={onClose}>Close</button>
      </div>
      {detail.loading ? (
        <p className="ap-muted">Loading…</p>
      ) : detail.error ? (
        <p className="ap-error">{esc(detail.error)}</p>
      ) : (
        <pre className="payload ap-artifact-body">{detail.content}</pre>
      )}
    </div>
  )
}

export default function RunEvalView({ store }) {
  const [catalog, setCatalog] = useState({ problems: [], filesystem_runs: [] })
  const [runs, setRuns] = useState([])
  const [model, setModel] = useState('opencode/big-pickle')
  const [selected, setSelected] = useState({})
  const [kindFilter, setKindFilter] = useState('all')
  const [starting, setStarting] = useState(false)
  const [expandedRun, setExpandedRun] = useState(null)
  const [artifact, setArtifact] = useState(null)
  const pollRef = useRef(null)

  const problems = catalog.problems
  const filesystemRuns = catalog.filesystem_runs || []
  const hasRunning = runs.some(r => r.status === 'running')

  const fetchCatalog = useCallback(async () => {
    try {
      const res = await fetch('/api/eval/catalog')
      if (res.ok) setCatalog(await res.json())
    } catch {}
  }, [])

  const fetchRuns = useCallback(async () => {
    try {
      const res = await fetch('/api/eval/runs')
      if (res.ok) setRuns(await res.json())
    } catch {}
  }, [])

  useEffect(() => {
    fetchCatalog()
    fetchRuns()
  }, [fetchCatalog, fetchRuns])

  useEffect(() => {
    if (problems.length) {
      const sel = {}
      problems.forEach(p => { sel[p.id] = true })
      setSelected(sel)
    }
  }, [problems])

  useEffect(() => {
    if (hasRunning) pollRef.current = setInterval(fetchRuns, 2000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [hasRunning, fetchRuns])

  const filteredProblems = useMemo(() => {
    if (kindFilter === 'all') return problems
    return problems.filter(p => p.kind === kindFilter)
  }, [problems, kindFilter])

  const summary = useMemo(() => {
    const repair = problems.filter(p => p.kind === 'repair').length
    const comprehension = problems.filter(p => p.kind === 'comprehension').length
    const latest = runs[0]
    const latestPassed = latest ? latest.problems.filter(p => p.status === 'passed').length : 0
    const latestTotal = latest?.problems.length || 0
    const captured = filesystemRuns.filter(r => r.passed != null)
    const capturedPassed = captured.filter(r => r.passed).length
    return { repair, comprehension, latestPassed, latestTotal, capturedPassed, capturedTotal: captured.length }
  }, [problems, runs, filesystemRuns])

  const allSelected = filteredProblems.length > 0 && filteredProblems.every(p => selected[p.id])
  const anySelected = Object.entries(selected).some(([id, v]) => v && problems.some(p => p.id === id))

  const handleRun = async () => {
    const ids = Object.entries(selected).filter(([, v]) => v).map(([k]) => k)
    if (!ids.length || !model.trim()) return
    setStarting(true)
    try {
      const res = await fetch('/api/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model.trim(), problems: ids }),
      })
      if (res.ok) await fetchRuns()
    } catch {}
    setStarting(false)
  }

  const toggleAllFiltered = () => {
    const next = { ...selected }
    filteredProblems.forEach(p => { next[p.id] = !allSelected })
    setSelected(next)
  }

  const openArtifact = async ({ title, subtitle, url }) => {
    setArtifact({ title, subtitle, loading: true, content: '', error: null })
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(await res.text())
      const content = await res.text()
      setArtifact({ title, subtitle, loading: false, content, error: null })
    } catch (e) {
      setArtifact({ title, subtitle, loading: false, content: '', error: e.message || 'Failed to load artifact' })
    }
  }

  const openOpencodeDiff = async (runId, problemNumber, name) => {
    await openArtifact({
      title: name,
      subtitle: 'git diff from agent workspace',
      url: `/api/eval/runs/${runId}/diff/${problemNumber}`,
    })
  }

  const openFilesystemArtifact = async (runDir, artifact, title, subtitle) => {
    await openArtifact({
      title,
      subtitle,
      url: `/api/eval/pack-runs/artifact?runDir=${encodeURIComponent(runDir)}&name=${encodeURIComponent(artifact)}`,
    })
  }

  return (
    <div className="ap-root">
      <section className="panel ap-intro">
        <div className="panel-head">
          <span className="panel-title">Agent Problem Pack</span>
          <span className="panel-sub">10-task benchmark for repair and codebase comprehension</span>
        </div>
        <p className="ap-lede">
          Each problem runs in an isolated workspace. Agents receive only the task prompt, make edits,
          and are scored by automated tests or structured answer checks. Results capture diffs,
          verification output, and token usage for later review.
        </p>
        <LifecycleRail />
        <div className="metrics ap-summary-metrics">
          <div className="metric"><div className="k">Repair tasks</div><div className="v">{summary.repair}</div></div>
          <div className="metric"><div className="k">Comprehension</div><div className="v">{summary.comprehension}</div></div>
          <div className="metric"><div className="k">Latest run</div><div className="v">{summary.latestTotal ? `${summary.latestPassed}/${summary.latestTotal}` : '—'}</div></div>
          <div className="metric"><div className="k">Captured runs</div><div className="v">{summary.capturedTotal ? `${summary.capturedPassed}/${summary.capturedTotal}` : '—'}</div></div>
        </div>
      </section>

      <section className="panel ap-launch">
        <div className="panel-head">
          <span className="panel-title">Launch run</span>
          <span className="panel-sub">Select problems and a model — runs execute sequentially via OpenCode</span>
        </div>

        <div className="ap-model-row">
          {PRESET_MODELS.map(m => (
            <button
              key={m.id}
              type="button"
              className={'btn' + (model === m.id ? ' primary' : '')}
              onClick={() => setModel(m.id)}
            >{m.label}</button>
          ))}
          <input
            className="er-input ap-model-input"
            value={model}
            onChange={e => setModel(e.target.value)}
            placeholder="opencode/big-pickle"
          />
        </div>

        <div className="ap-filter-row">
          {['all', 'repair', 'comprehension'].map(k => (
            <button
              key={k}
              type="button"
              className={'btn' + (kindFilter === k ? ' primary' : '')}
              onClick={() => setKindFilter(k)}
            >{k === 'all' ? 'All problems' : k}</button>
          ))}
          <button type="button" className="btn" onClick={toggleAllFiltered}>
            {allSelected ? 'Deselect visible' : 'Select visible'}
          </button>
          <button
            type="button"
            className="btn primary ap-run-btn"
            onClick={handleRun}
            disabled={!anySelected || !model.trim() || starting || hasRunning}
          >
            {starting ? 'Starting…' : hasRunning ? 'Run in progress…' : 'Run selected problems'}
          </button>
        </div>

        <div className="ap-grid">
          {filteredProblems.map(p => (
            <ProblemCard
              key={p.id}
              problem={p}
              selected={!!selected[p.id]}
              onToggle={(id) => setSelected(s => ({ ...s, [id]: !s[id] }))}
            />
          ))}
        </div>
      </section>

      <section className="ap-results">
        <div className="panel-head">
          <span className="panel-title">OpenCode runs</span>
          <span className="panel-sub">{runs.length} recorded locally</span>
        </div>

        {runs.length === 0 && (
          <p className="ap-muted ap-empty">No OpenCode runs yet. Launch a run above to populate this section.</p>
        )}

        {runs.map(run => {
          const passed = run.problems.filter(p => p.status === 'passed').length
          const total = run.problems.length
          const isExpanded = expandedRun === run.id
          const pct = total ? Math.round((passed / total) * 100) : 0
          return (
            <div key={run.id} className="panel ap-run-panel">
              <button
                type="button"
                className="ap-run-head"
                onClick={() => setExpandedRun(isExpanded ? null : run.id)}
              >
                <div>
                  <div className="ap-run-title">{esc(run.model)}</div>
                  <div className="ap-run-sub">
                    {fmtDate(run.created)} · {passed}/{total} passed · {run.status}
                    {run.requestedModel ? ` · entered "${run.requestedModel}"` : ''}
                  </div>
                </div>
                <div className="ap-run-meter" aria-hidden="true">
                  <div className="ap-run-meter-fill" style={{ width: pct + '%' }} />
                  <span>{pct}%</span>
                </div>
              </button>

              {isExpanded && (
                <div className="ap-run-body">
                  <table className="bd ap-matrix">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Problem</th>
                        <th>Kind</th>
                        <th className="num">Result</th>
                        <th className="num">Tokens</th>
                        <th className="num">Steps</th>
                        <th>Why it failed</th>
                        <th className="num">Trace</th>
                        <th className="num">Diff</th>
                        <th className="num">Verify</th>
                      </tr>
                    </thead>
                    <tbody>
                      {run.problems.map(prob => (
                        <tr key={prob.id}>
                          <td className="num">{String(prob.number).padStart(2, '0')}</td>
                          <td className="name">
                            {esc(prob.name)}
                            {prob.retries > 0 && (
                              <span className="ap-retry-badge" title={`opencode DB was locked at startup; retried ${prob.retries}x`}>
                                {' '}↻{prob.retries}
                              </span>
                            )}
                          </td>
                          <td><span className={'ap-kind ap-kind-' + (prob.kind || 'repair')}>{kindLabel(prob.kind || 'repair')}</span></td>
                          <td className="num"><span className={statusClass(prob.status, prob.passed)}>{statusText(prob.status, prob.passed)}</span></td>
                          <td
                            className="num ap-stat"
                            title={prob.tokens ? `${(prob.tokensInput || 0).toLocaleString()} in / ${(prob.tokensOutput || 0).toLocaleString()} out${prob.tokensReasoning ? ` / ${prob.tokensReasoning.toLocaleString()} reasoning` : ''}` : 'No token usage recorded'}
                          >{fmtTokens(prob.tokens)}</td>
                          <td
                            className="num ap-stat"
                            title={prob.toolCalls != null ? `${prob.steps || 0} assistant turns · ${prob.toolCalls} tool calls` : 'No step count recorded'}
                          >{prob.steps ? prob.steps : '—'}</td>
                          <td className="ap-failure-cell">
                            <FailureReason
                              passed={prob.status === 'passed' || prob.passed === true}
                              summary={
                                agentDidNotRun(prob)
                                  ? {
                                      passed: false,
                                      headline: 'Agent never ran — open Log',
                                      hints: ['No session was created and the agent produced no output, so tests ran against the unmodified baseline. Open the Log to see why the agent failed to start.'],
                                    }
                                  : prob.failureSummary || (prob.status === 'error' ? { passed: false, headline: prob.output || 'Agent run errored' } : null)
                              }
                              onExpand={setArtifact}
                            />
                          </td>
                          <td className="num ap-trace-cell">
                            {prob.sessionId && (
                              <button type="button" className="btn ap-mini" onClick={() => store?.loadSession?.(prob.sessionId)}>Trace</button>
                            )}
                            {(prob.stdout || prob.stderr || prob.output) && (
                              <button
                                type="button"
                                className={'btn ap-mini' + (agentDidNotRun(prob) ? ' ap-mini-warn' : '')}
                                title={agentDidNotRun(prob) ? 'Agent produced no output — it likely never ran' : 'Agent run log (stdout + stderr)'}
                                onClick={() => setArtifact({
                                  title: prob.name,
                                  subtitle: agentDidNotRun(prob) ? 'Agent run log — no session was created (agent likely never ran)' : 'Agent run log (stdout + stderr)',
                                  loading: false,
                                  content: formatAgentLog(prob),
                                  error: null,
                                })}
                              >Log</button>
                            )}
                            {!prob.sessionId && !prob.stdout && !prob.stderr && !prob.output && '—'}
                          </td>
                          <td className="num">
                            {(prob.status === 'passed' || prob.status === 'failed') && (
                              <button type="button" className="btn ap-mini" onClick={() => openOpencodeDiff(run.id, prob.number, prob.name)}>Diff</button>
                            )}
                          </td>
                          <td className="num">
                            {prob.verified && (
                              <button
                                type="button"
                                className="btn ap-mini"
                                onClick={() => setArtifact({
                                  title: prob.name,
                                  subtitle: 'pytest verification output',
                                  loading: false,
                                  content: prob.verified,
                                  error: null,
                                })}
                              >Output</button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )
        })}
      </section>

      <section className="ap-results">
        <div className="panel-head">
          <span className="panel-title">Captured pack runs</span>
          <span className="panel-sub">From agent-problem-pack/runs via pack_tools capture</span>
        </div>

        {filesystemRuns.length === 0 && (
          <p className="ap-muted ap-empty">
            No captured runs found. Prepare and capture with{' '}
            <code>uv run python scripts/pack_tools.py capture runs/&lt;problem&gt;/&lt;run-name&gt;</code>.
          </p>
        )}

        {filesystemRuns.length > 0 && (
          <table className="bd ap-matrix">
            <thead>
              <tr>
                <th>Problem</th>
                <th>Run</th>
                <th>Kind</th>
                <th className="num">Result</th>
                <th>Why it failed</th>
                <th className="num">Diff</th>
                <th className="num">Answer</th>
                <th className="num">Verify</th>
              </tr>
            </thead>
            <tbody>
              {filesystemRuns.map(row => (
                <tr key={row.run_dir}>
                  <td className="name">{esc(row.title || row.problem)}</td>
                  <td>{esc(row.run_name)}</td>
                  <td><span className={'ap-kind ap-kind-' + (row.kind || 'repair')}>{kindLabel(row.kind || 'repair')}</span></td>
                  <td className="num"><span className={statusClass(null, row.passed)}>{statusText(null, row.passed)}</span></td>
                  <td className="ap-failure-cell">
                    <FailureReason passed={row.passed === true} summary={row.failure_summary} onExpand={setArtifact} />
                  </td>
                  <td className="num">
                    <button type="button" className="btn ap-mini" onClick={() => openFilesystemArtifact(row.run_dir, 'diff.patch', row.title, 'Captured diff')}>Diff</button>
                  </td>
                  <td className="num">
                    <button type="button" className="btn ap-mini" onClick={() => openFilesystemArtifact(row.run_dir, 'AGENT_FINAL_ANSWER.md', row.title, 'Final answer')}>Answer</button>
                  </td>
                  <td className="num">
                    <button type="button" className="btn ap-mini" onClick={() => openFilesystemArtifact(row.run_dir, 'verification.txt', row.title, 'Verification output')}>Output</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <ArtifactPanel detail={artifact} onClose={() => setArtifact(null)} />
    </div>
  )
}
