import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { findDb, listSessions, analytics, exportSessionsFromDb } from './server/db.js'
import { reconstructSystemPrompt, reconstructTools, harnessAvailable } from './server/harness.js'
import { exportToSft } from './server/sft.js'
import { filterThroughBuildDataset, buildDatasetAvailable } from './server/filter.js'
import { uploadToHf, hfAvailable } from './server/hf.js'

const OPENCODE = '/Users/noahkasmanoff/.opencode/bin/opencode'
const TMP = '/tmp'

// opencode's CLI truncates piped stdout at ~64KB; redirect to a temp file and
// read it back instead so large exports come through intact.
function opencodeExec(cmd, opts) {
  const tmp = path.join(TMP, 'ocode_' + ((opts && opts.id) || Date.now()) + '.json')
  try {
    execSync(cmd + ' > ' + tmp + ' 2>/dev/null', {
      timeout: (opts && opts.timeout) || 60000, shell: '/bin/sh',
    })
    return fs.readFileSync(tmp, 'utf-8')
  } finally {
    try { fs.unlinkSync(tmp) } catch {}
  }
}

function fetchExport(id) {
  const raw = opencodeExec(`${OPENCODE} export ${id}`, { id, timeout: 60000 })
  return JSON.parse(raw)
}

function readBody(req) {
  return new Promise((resolve) => {
    let b = ''
    req.on('data', c => { b += c })
    req.on('end', () => resolve(b))
  })
}

function sendJson(res, data) {
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(data))
}
function sendError(res, e) {
  res.statusCode = 500
  res.end(JSON.stringify({ error: e.message, stderr: String(e.stderr || '') }))
}

// Build the SFT JSONL for a set of sessions (or all), optionally piping through
// build_dataset.py's quality gate + sanitization. Shared by download + upload.
function buildSftJsonl(dbPath, opts) {
  const { ids, all, filter, sanitize, modelName, workspacePath } = opts
  const records = []
  if (dbPath) {
    const exports = exportSessionsFromDb(dbPath, all ? null : ids)
    for (const data of exports.values()) {
      try {
        const rec = exportToSft(data)
        if (rec.messages.length) records.push(rec)
      } catch { /* skip */ }
    }
  } else {
    for (const id of (ids || [])) {
      try {
        const rec = exportToSft(fetchExport(id))
        if (rec.messages.length) records.push(rec)
      } catch { /* skip */ }
    }
  }

  if (!records.length) {
    throw new Error('no exportable records (sessions had no usable messages)')
  }

  if (filter) {
    if (!buildDatasetAvailable()) throw new Error('build_dataset.py not found — cannot apply quality filters')
    const out = filterThroughBuildDataset(records, {
      sanitize: sanitize !== false,
      modelName: modelName || undefined,
      workspacePath: workspacePath || undefined,
    })
    const count = (out.jsonl.trim().match(/\n/g) || []).length + (out.jsonl.trim() ? 1 : 0)
    if (!count) throw new Error('all records were dropped by the quality filter (corrections / loops / too-long / dedup)')
    return { jsonl: out.jsonl, count }
  }
  return { jsonl: records.map(r => JSON.stringify(r)).join('\n') + '\n', count: records.length }
}

function opencodeApi() {
  return {
    name: 'opencode-api',
    configureServer(server) {
      const dbPath = findDb()

      server.middlewares.use(async (req, res, next) => {
        const url = req.url

        // All sessions (from the SQLite DB so we are not capped at 100).
        if (url === '/api/sessions' && req.method === 'GET') {
          try {
            if (dbPath) {
              sendJson(res, listSessions(dbPath))
            } else {
              // fall back to the CLI (uncapped) if the DB is not found
              const raw = opencodeExec(`${OPENCODE} session list --format json -n 100000`, { timeout: 20000 })
              sendJson(res, JSON.parse(raw))
            }
          } catch (e) { sendError(res, e) }
          return
        }

        // Cross-session analytics, computed in SQL.
        if (url === '/api/analytics' && req.method === 'GET') {
          try {
            if (!dbPath) throw new Error('opencode.db not found')
            sendJson(res, analytics(dbPath))
          } catch (e) { sendError(res, e) }
          return
        }

        // Feature availability for the UI (harness reconstruction, filtering).
        if (url === '/api/capabilities' && req.method === 'GET') {
          sendJson(res, {
            harness: harnessAvailable(),
            filter: buildDatasetAvailable(),
            hf: hfAvailable(),
          })
          return
        }

        // Bulk SFT export: POST { ids?, all?, filter?, sanitize?, modelName? }
        // -> JSONL (one record per line). `all:true` exports every session.
        // Transcripts are reconstructed straight from the DB (one pass) rather
        // than spawning `opencode export` per session. When `filter` is set the
        // records go through build_dataset.py's quality gate + sanitization.
        if (url === '/api/export' && req.method === 'POST') {
          try {
            const body = await readBody(req)
            const opts = JSON.parse(body || '{}')
            if (!opts.all && (!Array.isArray(opts.ids) || !opts.ids.length)) {
              throw new Error('no session ids provided')
            }
            const { jsonl } = buildSftJsonl(dbPath, opts)
            res.setHeader('Content-Type', 'application/x-ndjson')
            res.setHeader('Content-Disposition', 'attachment; filename="opencode-sft.jsonl"')
            res.end(jsonl)
          } catch (e) { sendError(res, e) }
          return
        }

        // Upload the SFT JSONL straight to a private HF dataset repo.
        // POST { ids?, all?, filter?, sanitize?, modelName?, repo? }
        if (url === '/api/upload' && req.method === 'POST') {
          try {
            const body = await readBody(req)
            const opts = JSON.parse(body || '{}')
            if (!opts.all && (!Array.isArray(opts.ids) || !opts.ids.length)) {
              throw new Error('no session ids provided')
            }
            if (!hfAvailable()) throw new Error('HF upload unavailable (no token or uploader script)')
            const { jsonl, count } = buildSftJsonl(dbPath, opts)
            const result = uploadToHf(jsonl, {
              repo: opts.repo || 'opencode-sft',
              pathInRepo: 'train.jsonl',
            })
            sendJson(res, { ...result, exported: count })
          } catch (e) { sendError(res, e) }
          return
        }

        // Single-session SFT record (for preview / download).
        const expMatch = url.match(/^\/api\/export\/([^/?]+)/)
        if (expMatch && req.method === 'GET') {
          const id = expMatch[1]
          try {
            const data = fetchExport(id)
            sendJson(res, exportToSft(data))
          } catch (e) { sendError(res, e) }
          return
        }

        // One session's full transcript (for the anatomy view), enriched with
        // the reconstructed system prompt + tool schemas opencode does not store.
        const match = url.match(/^\/api\/sessions\/([^/?]+)/)
        if (match && req.method === 'GET') {
          const id = match[1]
          try {
            const data = fetchExport(id)
            if (harnessAvailable()) {
              const sys = reconstructSystemPrompt(data.info || {})
              if (sys) {
                data._reconstructedSystem = sys
                data._reconstructedTools = reconstructTools()
              }
            }
            sendJson(res, data)
          } catch (e) { sendError(res, e) }
          return
        }

        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), opencodeApi()],
})
