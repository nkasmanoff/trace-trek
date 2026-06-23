/* Reconstruct the opencode system prompt + tool schemas that a session ran
 * with. opencode does not persist these, but they are deterministic harness
 * scaffolding: a fixed template (harvested into fixtures/opencode-harness.json
 * from proxy logs) with a few per-session holes we refill from the DB record:
 *   {{POWERED_BY}} -> the model identity line
 *   {{ENV}}        -> the <env> working-directory / platform / date block
 *   {{CWD_URI}}    -> the built-in skill's file:// location
 *
 * The result is flagged `reconstructed: true` so the UI and any exported
 * dataset can be honest that these bytes were rebuilt, not captured verbatim.
 */
import fs from 'fs'
import path from 'path'
import { existsSync, readFileSync } from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FIXTURES = path.join(__dirname, 'fixtures', 'opencode-harness.json')

let HARNESS = null
function harness() {
  if (HARNESS) return HARNESS
  try { HARNESS = JSON.parse(readFileSync(FIXTURES, 'utf8')) }
  catch { HARNESS = { prompts: {}, tools: [] } }
  return HARNESS
}

export function harnessAvailable() {
  const h = harness()
  return !!(h.prompts && Object.keys(h.prompts).length)
}

/* Which prompt template a session used. The "beast" prompt is opencode's
 * frontier-harness prompt (anthropic/claude teachers); everything else uses
 * the standard "build" prompt. Sub-agents (explore/title/compaction) are
 * selected by the agent/mode recorded on the session. */
export function selectPromptRole(modelId, agent) {
  const a = String(agent || '').toLowerCase()
  if (a === 'explore') return 'explore'
  if (a === 'plan') return 'build'
  if (a === 'compaction' || a === 'summarize') return 'compaction'
  if (a === 'title') return 'title'
  const m = String(modelId || '').toLowerCase()
  if (m.includes('claude') || m.includes('anthropic')) return 'beast'
  return 'build'
}

function poweredByLine(modelId) {
  const id = modelId || 'unknown'
  // opencode prints both a friendly name and the exact id; for reconstructed
  // sessions we use the stored id for both.
  return `You are powered by the model named ${id}. The exact model ID is ${id}`
}

function envBlock(directory, whenMs) {
  const d = directory || '/'
  const date = whenMs ? new Date(whenMs) : new Date()
  const dateStr = date.toDateString() // e.g. "Sat Jun 20 2026"
  const isRepo = existsSync(path.join(d, '.git'))
  return [
    'Here is some useful information about the environment you are running in:',
    '<env>',
    `  Working directory: ${d}`,
    `  Workspace root folder: ${d}`,
    `  Is directory a git repo: ${isRepo ? 'yes' : 'no'}`,
    '  Platform: darwin',
    `  Today's date: ${dateStr}`,
    '</env>',
  ].join('\n')
}

function readAgentsMd(directory) {
  if (!directory) return null
  for (const name of ['AGENTS.md', 'CLAUDE.md', '.github/copilot-instructions.md']) {
    const p = path.join(directory, name)
    try {
      if (existsSync(p)) {
        const txt = readFileSync(p, 'utf8').trim()
        if (txt) return { name, text: txt }
      }
    } catch { /* ignore */ }
  }
  return null
}

/* Build the reconstructed system prompt for one session.
 * `session` carries { model, agent, directory, time:{updated} } from the
 * export's info block. Returns { content, reconstructed, role, sources }. */
export function reconstructSystemPrompt(info) {
  const h = harness()
  const modelId = info?.model?.id || info?.modelID || ''
  const agent = info?.agent || ''
  const directory = info?.directory || ''
  const whenMs = info?.time?.updated || info?.time?.created
  const role = selectPromptRole(modelId, agent)
  const template = h.prompts[role] || h.prompts.build
  if (!template) return null

  let content = template
    .replaceAll('{{POWERED_BY}}', poweredByLine(modelId))
    .replaceAll('{{ENV}}', envBlock(directory, whenMs))
    .replaceAll('{{CWD_URI}}', 'file://' + encodeURI(directory))
  // opencode also prepends file:// once already in the template around CWD_URI;
  // guard against a doubled scheme if the template kept its own file://.
  content = content.replace(/file:\/\/file:\/\//g, 'file://')

  // opencode appends AGENTS.md / project rules to the system context.
  const agents = readAgentsMd(directory)
  if (agents) {
    content += `\n\nProject instructions from ${agents.name}:\n\n${agents.text}`
  }

  return {
    content,
    reconstructed: true,
    role,
    sources: {
      template: 'opencode-harness.json',
      poweredBy: !!modelId,
      env: true,
      agentsMd: agents ? agents.name : null,
    },
  }
}

/* The tool schemas opencode exposes. opencode's tool set is fixed per build;
 * we return the harvested canonical schemas. Optionally filter to the tools a
 * session actually invoked. */
export function reconstructTools(usedToolNames) {
  const h = harness()
  let tools = h.tools || []
  if (usedToolNames && usedToolNames.size) {
    const filtered = tools.filter(t => usedToolNames.has(t?.function?.name))
    if (filtered.length) tools = filtered
  }
  return tools
}
