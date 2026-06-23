#!/usr/bin/env node
/* Harvest canonical opencode system-prompt templates and tool JSON schemas
 * from captured proxy logs (data/raw/opencode/*.json).
 *
 * opencode does not persist the system prompt or tool schemas in its DB, but
 * the logging proxy captured the exact bytes opencode sent to the model. We
 * turn those into reusable fixtures by:
 *   - grouping system prompts by agent role (build / beast / explore / ...)
 *   - replacing the per-session dynamic holes (the "You are powered by" line
 *     and the <env> block) with {{PLACEHOLDERS}} so the server can refill them
 *     from each DB session's stored model + directory + timestamps.
 *   - collecting one schema per tool name (bash, read, edit, ...).
 *
 * Output: viewer/server/fixtures/opencode-harness.json
 *
 * Run once (re-run when opencode upgrades and you have fresh proxy logs):
 *   node scripts/harvest-harness.js [path/to/raw/opencode]
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const RAW_DIR = process.argv[2] ||
  path.resolve(__dirname, '../data/raw/opencode')
const OUT = path.resolve(__dirname, '../viewer/server/fixtures/opencode-harness.json')

function toText(content) {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) return content.map(p => (p && p.type === 'text' ? p.text : '')).join('')
  return ''
}

// Identify which agent role a system prompt belongs to.
function classify(c) {
  if (c.includes('title generator')) return 'title'
  if (c.includes('file search specialist')) return 'explore'
  if (c.includes('anchored context summariz')) return 'compaction'
  if (c.startsWith('You are OpenCode, the best coding agent')) return 'beast'
  if (c.includes('interactive CLI agent specializing')) return 'gemini'
  if (c.includes('an interactive CLI tool that helps')) return 'build'
  return 'other'
}

const POWERED_RE = /You are powered by the model named .+?\. The exact model ID is \S+/
// the env block opencode appends: "<env>\n ... \n</env>"
const ENV_RE = /<env>[\s\S]*?<\/env>/
// the built-in skill's <location> embeds the session's working dir as a file:// URI
const SKILL_LOC_RE = /(<location>file:\/\/)[^<]*(%3Cbuilt-in%3E<\/location>)/g

// Replace the dynamic holes with placeholders the server refills per session.
function templatize(c) {
  let t = c
  let hadPowered = false, hadEnv = false
  if (POWERED_RE.test(t)) { t = t.replace(POWERED_RE, '{{POWERED_BY}}'); hadPowered = true }
  if (ENV_RE.test(t)) { t = t.replace(ENV_RE, '{{ENV}}'); hadEnv = true }
  t = t.replace(SKILL_LOC_RE, '$1{{CWD_URI}}/$2')
  return { template: t, hadPowered, hadEnv }
}

function main() {
  if (!fs.existsSync(RAW_DIR)) {
    console.error('proxy log dir not found:', RAW_DIR)
    process.exit(1)
  }
  const files = fs.readdirSync(RAW_DIR).filter(f => f.endsWith('.json'))
  const promptByRole = {}     // role -> {template, len, count, hadPowered, hadEnv}
  const toolsByName = {}      // name -> schema (function tool object)
  let scanned = 0, withSys = 0, withTools = 0

  for (const f of files) {
    let r
    try { r = JSON.parse(fs.readFileSync(path.join(RAW_DIR, f), 'utf8')) } catch { continue }
    scanned++
    const msgs = (r.request && r.request.messages) || []
    const sys = msgs.find(m => m && m.role === 'system')
    if (sys) {
      withSys++
      const c = toText(sys.content)
      const role = classify(c)
      // keep the LONGEST exemplar per role (most complete instructions)
      const prev = promptByRole[role]
      if (!prev || c.length > prev.len) {
        const { template, hadPowered, hadEnv } = templatize(c)
        promptByRole[role] = { template, len: c.length, count: (prev?.count || 0) + 1, hadPowered, hadEnv }
      } else {
        prev.count++
      }
    }
    const tools = r.request && r.request.tools
    if (Array.isArray(tools)) {
      withTools++
      for (const t of tools) {
        const n = t && t.function && t.function.name
        if (n && !toolsByName[n]) toolsByName[n] = t
      }
    }
  }

  const prompts = {}
  for (const role in promptByRole) {
    if (role === 'other') continue
    prompts[role] = promptByRole[role].template
  }

  const out = {
    _meta: {
      source: RAW_DIR,
      scanned, withSys, withTools,
      generatedAt: new Date().toISOString(),
      roles: Object.fromEntries(Object.entries(promptByRole)
        .filter(([r]) => r !== 'other')
        .map(([r, v]) => [r, { chars: v.len, samples: v.count, hadPowered: v.hadPowered, hadEnv: v.hadEnv }])),
      tools: Object.keys(toolsByName).sort(),
    },
    prompts,
    tools: Object.keys(toolsByName).sort().map(n => toolsByName[n]),
  }

  fs.mkdirSync(path.dirname(OUT), { recursive: true })
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2))
  console.log('scanned', scanned, 'records ·', withSys, 'with system ·', withTools, 'with tools')
  console.log('prompt roles:', Object.keys(prompts).join(', '))
  console.log('tools:', out._meta.tools.join(', '))
  console.log('wrote', OUT, '(' + (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB)')
}

main()
