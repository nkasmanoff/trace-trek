import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '../..')
export const PACK_ROOT = path.join(REPO_ROOT, 'agent-problem-pack')
const PACK_TOOLS = path.join(PACK_ROOT, 'scripts', 'pack_tools.py')

function runCatalog(includeFilesystemRuns = false) {
  const args = ['uv', 'run', 'python', PACK_TOOLS, 'catalog']
  if (includeFilesystemRuns) args.push('--filesystem-runs')
  const out = execSync(args.join(' '), {
    cwd: PACK_ROOT,
    encoding: 'utf-8',
    timeout: 120000,
    maxBuffer: 16 * 1024 * 1024,
  })
  return JSON.parse(out)
}

export function packAvailable() {
  return fs.existsSync(PACK_TOOLS)
}

export function loadCatalog() {
  if (!packAvailable()) {
    return { problems: [], filesystem_runs: [] }
  }
  return runCatalog(true)
}

export function loadProblems() {
  return loadCatalog().problems
}

export function readRunArtifact(runDir, artifact) {
  const allowed = new Set([
    'diff.patch',
    'verification.txt',
    'git-status.txt',
    'evaluate-with-codex.md',
    'failure-summary.json',
    'usage.json',
    'task-prompt.txt',
  ])
  const resolvedRun = path.resolve(runDir)
  if (!resolvedRun.startsWith(path.resolve(PACK_ROOT))) {
    throw new Error('run path outside pack root')
  }
  if (artifact === 'AGENT_FINAL_ANSWER.md') {
    return readFinalAnswer(resolvedRun)
  }
  if (!allowed.has(artifact)) {
    throw new Error('unsupported artifact')
  }
  const filePath = path.join(resolvedRun, 'artifacts', artifact)
  if (!fs.existsSync(filePath)) throw new Error('artifact not found')
  return fs.readFileSync(filePath, 'utf-8')
}

export function readFinalAnswer(runDir) {
  const resolvedRun = path.resolve(runDir)
  if (!resolvedRun.startsWith(path.resolve(PACK_ROOT))) {
    throw new Error('run path outside pack root')
  }
  const answerPath = path.join(resolvedRun, 'workspace', 'AGENT_FINAL_ANSWER.md')
  if (!fs.existsSync(answerPath)) throw new Error('final answer not found')
  return fs.readFileSync(answerPath, 'utf-8')
}
