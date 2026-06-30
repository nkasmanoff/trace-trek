import { execSync, execFileSync, spawn, spawnSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import os from 'os'
import { fileURLToPath } from 'url'
import { PACK_ROOT, loadCatalog, loadProblems } from './pack-tools.js'
import { summarizeVerification } from './failure-analysis.js'
import { resolvePython } from './python.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const QUERY_SCRIPT = path.join(__dirname, 'opencode_db.py')

const OPENCODE = process.env.OPENCODE_BIN || path.join(os.homedir(), '.opencode/bin/opencode')
const EVAL_RUNS_FILE = path.join(os.homedir(), '.opencode', 'eval-runs.json')
const TMP_ROOT = path.join(os.tmpdir(), 'opencode-eval')

const problemById = () => Object.fromEntries(loadProblems().map(p => [p.id, p]))

// A run only makes progress inside the server process that started it (the
// orchestration loop + the spawned agent's close handler live in memory). If the
// dev server restarts, in-flight runs are orphaned: their detached agent may
// linger but nothing will ever advance or finalize them. We tag each run with
// the owning PID so stale "running" runs can be reconciled to "interrupted".
const SERVER_PID = process.pid

// opencode opens its own session DB (opencode.db) in WAL mode with busy_timeout=0,
// so it allows a single writer and aborts instantly on any momentary contention
// with "database is locked" instead of waiting. When that happens at startup no
// session is created and the agent never runs. Retry the launch a few times with
// backoff so one unlucky overlap doesn't fail an otherwise-runnable problem.
const MAX_AGENT_ATTEMPTS = 4

function killRunProcesses(runId) {
  try { execSync(`pkill -f ${JSON.stringify('opencode-eval/' + runId + '/')}`, { timeout: 5000 }) } catch {}
  try { fs.rmSync(path.join(TMP_ROOT, runId), { recursive: true, force: true }) } catch {}
}

function reconcileStaleRuns(runs) {
  let changed = false
  for (const run of runs) {
    if (run.status !== 'running') continue
    if (run.serverPid === SERVER_PID) continue
    run.status = 'interrupted'
    run.finished = run.finished || new Date().toISOString()
    for (const prob of run.problems || []) {
      if (prob.status === 'running' || prob.status === 'pending') prob.status = 'interrupted'
    }
    killRunProcesses(run.id)
    changed = true
  }
  return changed
}

const HIDDEN_TESTS_DIRNAME = 'tests_hidden'

// Copy a problem's hidden tests into a workspace for grading only. Mirrors
// pack_tools.inject_hidden_tests: the agent never sees these during its run, and
// they are removed again after verification so they never appear in the diff.
// Returns the injected destination path, or null when no hidden tests exist.
function injectHiddenTests(problemId, workspaceDir) {
  const hiddenSrc = path.join(PACK_ROOT, problemId, HIDDEN_TESTS_DIRNAME)
  if (!fs.existsSync(hiddenSrc) || !fs.statSync(hiddenSrc).isDirectory()) return null
  const dest = path.join(workspaceDir, HIDDEN_TESTS_DIRNAME)
  try { fs.rmSync(dest, { recursive: true, force: true }) } catch {}
  fs.cpSync(hiddenSrc, dest, {
    recursive: true,
    filter: (src) => !['__pycache__', '.pytest_cache', '.DS_Store'].includes(path.basename(src)),
  })
  return dest
}

function readRuns() {
  try { return JSON.parse(fs.readFileSync(EVAL_RUNS_FILE, 'utf-8')) }
  catch { return [] }
}

function writeRuns(runs) {
  fs.mkdirSync(path.dirname(EVAL_RUNS_FILE), { recursive: true })
  fs.writeFileSync(EVAL_RUNS_FILE, JSON.stringify(runs, null, 2))
}

// Persist a single run by re-reading the file and replacing only its entry. A
// running eval's orchestration loop holds an in-memory snapshot of the whole
// runs array; if it wrote that whole snapshot back it would clobber concurrent
// runs (e.g. two dev servers, or an overlapping run). Upserting one entry keeps
// other runs intact.
function upsertRun(run) {
  const all = readRuns()
  const idx = all.findIndex(r => r.id === run.id)
  if (idx === -1) all.unshift(run)
  else all[idx] = run
  writeRuns(all)
}

function artifactsDir(runId) {
  const d = path.join(path.dirname(EVAL_RUNS_FILE), 'eval-artifacts', runId)
  fs.mkdirSync(d, { recursive: true })
  return d
}

function runSync(cmd, opts = {}) {
  return execSync(cmd, {
    shell: '/bin/bash',
    timeout: 600000,
    maxBuffer: 64 * 1024 * 1024,
    ...opts,
  })
}

// Valid opencode model ids ("provider/model"). Cached so we only shell out once.
let _modelCache = null
function listOpencodeModels() {
  if (_modelCache) return _modelCache
  try {
    const out = execFileSync(OPENCODE, ['models'], {
      encoding: 'utf-8', timeout: 30000, maxBuffer: 16 * 1024 * 1024,
    })
    _modelCache = out.split('\n').map(s => s.trim()).filter(Boolean)
  } catch {
    _modelCache = []
  }
  return _modelCache
}

// Resolve a user-entered model string to a valid opencode model id. An exact id
// is returned unchanged; a shorthand (e.g. "qwen/qwen3.6-35b-a3b") is matched
// against the suffix of known ids. When several providers expose the same model,
// prefer hosted providers that don't need a local server running.
export function resolveModel(input) {
  const model = String(input || '').trim()
  if (!model) return model
  const all = listOpencodeModels()
  if (!all.length || all.includes(model)) return model

  const lower = model.toLowerCase()
  const PROVIDER_RANK = ['openrouter', 'frontier', 'opencode']
  const rank = (m) => {
    const i = PROVIDER_RANK.indexOf(m.split('/')[0])
    return i === -1 ? PROVIDER_RANK.length : i
  }
  const byPreference = (a, b) => rank(a) - rank(b) || a.length - b.length

  let matches = all.filter(m => m.toLowerCase().endsWith('/' + lower))
  if (matches.length) return matches.sort(byPreference)[0]

  matches = all.filter(m => m.toLowerCase().includes(lower))
  if (matches.length) return matches.sort(byPreference)[0]

  return model
}

export function listProblems() {
  return loadProblems()
}

export function getFilesystemRuns() {
  return loadCatalog().filesystem_runs || []
}

export function getRuns() {
  const runs = readRuns()
  if (reconcileStaleRuns(runs)) writeRuns(runs)
  for (const run of runs) {
    for (const prob of run.problems || []) {
      // Backfill token/step usage for runs captured before these were recorded.
      if (prob.sessionId && (prob.tokens == null || prob.steps == null)) {
        const stats = getSessionStats(prob.sessionId)
        if (stats) {
          prob.tokens = stats.tokensTotal
          prob.tokensInput = stats.tokensInput
          prob.tokensOutput = stats.tokensOutput
          prob.tokensReasoning = stats.tokensReasoning
          prob.cost = stats.cost
          prob.steps = stats.steps
          prob.toolCalls = stats.toolCalls
        }
      }
      if (!prob.verified) continue
      const didPass = prob.passed === true || prob.status === 'passed'
      prob.failureSummary = summarizeVerification(prob.verified, {
        passed: didPass,
        answerText: '',
        diffText: prob.diffPath && fs.existsSync(prob.diffPath) ? fs.readFileSync(prob.diffPath, 'utf-8') : '',
      })
    }
  }
  return runs
}

export function getRun(runId) {
  return readRuns().find(r => r.id === runId) || null
}

export function cancelRun(runId) {
  const runs = readRuns()
  const run = runs.find(r => r.id === runId)
  if (!run || run.status !== 'running') return false
  run.status = 'cancelled'
  writeRuns(runs)
  const tmpDir = path.join(TMP_ROOT, runId)
  try { fs.rmSync(tmpDir, { recursive: true, force: true }) } catch {}
  return true
}

export function prepareWorkspace(problemId, workspaceDir) {
  const src = path.join(PACK_ROOT, problemId)
  if (!fs.existsSync(src)) throw new Error(`problem source not found: ${src}`)

  fs.mkdirSync(workspaceDir, { recursive: true })

  // Hidden tests are withheld from the agent during prepare (matching the pack's
  // copy_problem(include_hidden_tests=False)) and only injected at verify time so
  // a fix cannot be overfit to the visible suite.
  const ignore = new Set(['.DS_Store', '.git', '.pytest_cache', '.venv', '__pycache__', '.validate-tmp', HIDDEN_TESTS_DIRNAME])
  function cpDir(from, to) {
    fs.mkdirSync(to, { recursive: true })
    for (const item of fs.readdirSync(from)) {
      if (ignore.has(item)) continue
      const s = path.join(from, item)
      const d = path.join(to, item)
      if (fs.statSync(s).isDirectory()) cpDir(s, d)
      else fs.copyFileSync(s, d)
    }
  }
  cpDir(src, workspaceDir)

  const pyproject = path.join(PACK_ROOT, 'pyproject.toml')
  if (fs.existsSync(pyproject)) fs.copyFileSync(pyproject, path.join(workspaceDir, 'pyproject.toml'))

  fs.writeFileSync(path.join(workspaceDir, 'AGENT_FINAL_ANSWER.md'), 'Write the final answer for this run here.\n')

  runSync('git init --quiet', { cwd: workspaceDir })
  runSync('git add .', { cwd: workspaceDir })
  runSync('git -c user.name=agent-problem-pack -c user.email=agent-problem-pack@example.invalid commit --quiet -m baseline', { cwd: workspaceDir })
}

export function verifyProblem(problemId, workspaceDir) {
  const meta = problemById()[problemId]
  const verifyCmd = meta?.verify_command?.join(' ') || 'uv run pytest'
  const injected = injectHiddenTests(problemId, workspaceDir)
  try {
    const out = runSync(verifyCmd, { cwd: workspaceDir })
    return { passed: true, exitCode: 0, output: out.toString() }
  } catch (e) {
    return {
      passed: false,
      exitCode: e.status || 1,
      output: String(e.stdout || '') + '\n' + String(e.stderr || ''),
    }
  } finally {
    if (injected) {
      try { fs.rmSync(injected, { recursive: true, force: true }) } catch {}
    }
  }
}

const OPENCODE_DB_PATHS = [
  path.join(os.homedir(), '.local/share/opencode/opencode.db'),
  path.join(os.homedir(), 'Library/Application Support/opencode/opencode.db'),
]

function sqliteJson(dbPath, sql) {
  const outFile = path.join(os.tmpdir(), `ocdb_${process.pid}_${Date.now()}.json`)
  const py = resolvePython()
  try {
    const res = spawnSync(py, [QUERY_SCRIPT, dbPath, '--out', outFile], {
      input: sql,
      encoding: 'utf-8',
      timeout: 60000,
    })
    if (res.error || res.status !== 0) throw new Error(res.stderr || res.error?.message || 'db query failed')
    return JSON.parse(fs.readFileSync(outFile, 'utf-8').trim() || '[]')
  } finally {
    try { fs.unlinkSync(outFile) } catch {}
  }
}

// Token + step usage for a completed session, summed across the main session and
// any subagent sessions it spawned (opencode records those as separate rows with
// parent_id set). `steps` counts assistant turns; `toolCalls` counts tool parts.
function getSessionStats(sessionId) {
  if (!sessionId) return null
  for (const dbPath of OPENCODE_DB_PATHS) {
    if (!fs.existsSync(dbPath)) continue
    try {
      const esc = String(sessionId).replace(/'/g, "''")
      const idRows = sqliteJson(dbPath, `SELECT id FROM session WHERE id='${esc}' OR parent_id='${esc}'`)
      if (!idRows.length) continue
      const ids = idRows.map(r => `'${String(r.id).replace(/'/g, "''")}'`).join(',')
      const tok = sqliteJson(dbPath, `SELECT sum(tokens_input) AS i, sum(tokens_output) AS o, sum(tokens_reasoning) AS r, sum(cost) AS c FROM session WHERE id IN (${ids})`)[0] || {}
      const steps = sqliteJson(dbPath, `SELECT count(*) AS c FROM message WHERE session_id IN (${ids}) AND json_extract(data,'$.role')='assistant'`)[0]?.c || 0
      const tools = sqliteJson(dbPath, `SELECT count(*) AS c FROM part WHERE session_id IN (${ids}) AND json_extract(data,'$.type')='tool'`)[0]?.c || 0
      return {
        tokensInput: tok.i || 0,
        tokensOutput: tok.o || 0,
        tokensReasoning: tok.r || 0,
        tokensTotal: (tok.i || 0) + (tok.o || 0),
        cost: tok.c || 0,
        steps,
        toolCalls: tools,
      }
    } catch {}
  }
  return null
}

function findSessionId(workspaceDir) {
  for (const dbPath of OPENCODE_DB_PATHS) {
    if (!fs.existsSync(dbPath)) continue
    try {
      const escaped = workspaceDir.replace(/'/g, "''")
      const rows = sqliteJson(dbPath,
        `SELECT id FROM session WHERE directory = '${escaped}' ORDER BY time_updated DESC LIMIT 1`)
      if (rows.length) return rows[0].id
    } catch {}
  }
  return null
}

function captureDiff(workspaceDir, artifactDir) {
  fs.mkdirSync(artifactDir, { recursive: true })
  try {
    runSync('git add -N .', { cwd: workspaceDir })
    const diff = runSync('git diff --no-ext-diff -- .', { cwd: workspaceDir })
    fs.writeFileSync(path.join(artifactDir, 'diff.patch'), diff.toString())

    const status = runSync('git status --short', { cwd: workspaceDir })
    fs.writeFileSync(path.join(artifactDir, 'git-status.txt'), status.toString())
  } catch {}
}

export function startRun(requestedModel, problemIds) {
  const model = resolveModel(requestedModel)
  const catalog = problemById()
  const run = {
    id: 'eval-' + Date.now(),
    model,
    requestedModel: requestedModel !== model ? requestedModel : undefined,
    serverPid: SERVER_PID,
    created: new Date().toISOString(),
    status: 'running',
    problems: problemIds.map(id => {
      const meta = catalog[id] || {}
      return {
        id,
        number: meta.number || parseInt(id.match(/\d+/)?.[0] || '0'),
        name: meta.name || id.replace(/^problem-\d+-/, '').replace(/-/g, ' '),
        kind: meta.kind || 'repair',
        difficulty: meta.difficulty || 'medium',
        status: 'pending',
        sessionId: null,
        passed: null,
        exitCode: null,
        output: '',
      }
    }),
  }
  upsertRun(run)

  const tmpBase = path.join(TMP_ROOT, run.id)
  fs.mkdirSync(tmpBase, { recursive: true })

  let idx = 0
  function nextProblem() {
    if (idx >= run.problems.length) {
      run.status = 'completed'
      run.finished = new Date().toISOString()
      upsertRun(run)
      try { fs.rmSync(tmpBase, { recursive: true, force: true }) } catch {}
      return
    }

    const prob = run.problems[idx]
    prob.status = 'running'
    upsertRun(run)

    const ws = path.join(tmpBase, String(prob.number))
    try {
      prepareWorkspace(prob.id, ws)
      const realWs = fs.realpathSync(ws)

      const meta = catalog[prob.id]
      const prompt = meta?.task_prompt
        ? `${meta.task_prompt}\n\nAt the end, write your concise final answer to AGENT_FINAL_ANSWER.md in this workspace.`
        : null
      if (!prompt) throw new Error(`no prompt defined for ${prob.id}`)

      const opencodeStdoutPath = path.join(tmpBase, String(prob.number) + '-stdout.log')
      const opencodeStderrPath = path.join(tmpBase, String(prob.number) + '-stderr.log')

      const advance = () => { idx++; process.nextTick(nextProblem) }

      const launchAgent = (attempt) => {
      const opencodeStdout = fs.openSync(opencodeStdoutPath, 'w')
      const opencodeStderr = fs.openSync(opencodeStderrPath, 'w')

      // NOTE: do NOT run this through a shell. With `shell: '/bin/bash'` plus an
      // args array, Node collapses everything into a single `bash -c "..."`
      // string without escaping, so any prompt containing shell metacharacters
      // (parentheses, quotes, …) aborts with a syntax error and the agent never
      // launches. Spawning argv directly passes the prompt as one literal arg.
      const opencodeProc = spawn(OPENCODE, [
        'run',
        '--model', model,
        '--dangerously-skip-permissions',
        '--dir', realWs,
        prompt,
      ], {
        cwd: realWs,
        stdio: ['ignore', opencodeStdout, opencodeStderr],
        env: { ...process.env, HOME: process.env.HOME },
        detached: true,
      })
      opencodeProc.unref()

      opencodeProc.on('close', () => {
        try { prob.stdout = fs.readFileSync(opencodeStdoutPath, 'utf-8').slice(0, 50000) } catch { prob.stdout = '' }
        try { prob.stderr = fs.readFileSync(opencodeStderrPath, 'utf-8').slice(0, 50000) } catch { prob.stderr = '' }
        try { fs.closeSync(opencodeStdout) } catch {}
        try { fs.closeSync(opencodeStderr) } catch {}

        // Transient opencode startup failure: its session DB was locked, so no
        // session was created and the agent never ran. Retry with backoff rather
        // than recording a spurious failure (see MAX_AGENT_ATTEMPTS note above).
        const lockFailed = /database is locked/i.test(prob.stderr || '')
        if (lockFailed && attempt < MAX_AGENT_ATTEMPTS) {
          prob.retries = attempt
          prob.status = 'running'
          upsertRun(run)
          setTimeout(() => launchAgent(attempt + 1), 1000 * attempt + Math.floor(Math.random() * 500))
          return
        }
        if (attempt > 1) prob.retries = attempt - 1

        const ver = verifyProblem(prob.id, realWs)
        prob.passed = ver.passed
        prob.exitCode = ver.exitCode
        prob.verified = ver.output

        const artDir = path.join(artifactsDir(run.id), String(prob.number))
        fs.mkdirSync(artDir, { recursive: true })
        captureDiff(realWs, artDir)
        prob.diffPath = path.join(artDir, 'diff.patch')

        let answerText = ''
        try {
          answerText = fs.readFileSync(path.join(realWs, 'AGENT_FINAL_ANSWER.md'), 'utf-8')
        } catch {}
        let diffText = ''
        try {
          diffText = fs.readFileSync(prob.diffPath, 'utf-8')
        } catch {}
        prob.failureSummary = summarizeVerification(ver.output, {
          passed: ver.passed,
          answerText,
          diffText,
        })

        prob.sessionId = findSessionId(realWs)
        if (!prob.sessionId && prob.stdout) {
          const sessionMatch = prob.stdout.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i)
          if (sessionMatch) prob.sessionId = sessionMatch[0]
        }

        const stats = getSessionStats(prob.sessionId)
        if (stats) {
          prob.tokens = stats.tokensTotal
          prob.tokensInput = stats.tokensInput
          prob.tokensOutput = stats.tokensOutput
          prob.tokensReasoning = stats.tokensReasoning
          prob.cost = stats.cost
          prob.steps = stats.steps
          prob.toolCalls = stats.toolCalls
        }

        prob.status = prob.passed ? 'passed' : 'failed'
        upsertRun(run)
        advance()
      })

      opencodeProc.on('error', (err) => {
        prob.status = 'error'
        prob.output = err.message
        prob.passed = false
        prob.exitCode = -1
        upsertRun(run)
        advance()
      })
      }

      launchAgent(1)
    } catch (err) {
      prob.status = 'error'
      prob.output = err.message
      prob.passed = false
      prob.exitCode = -1
      upsertRun(run)
      idx++
      process.nextTick(nextProblem)
    }
  }

  process.nextTick(nextProblem)
  return run
}
