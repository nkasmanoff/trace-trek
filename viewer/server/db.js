import { execFileSync } from 'child_process'
import { existsSync } from 'fs'
import { homedir } from 'os'
import path from 'path'

const DB_CANDIDATES = [
  path.join(homedir(), '.local/share/opencode/opencode.db'),
  path.join(homedir(), 'Library/Application Support/opencode/opencode.db'),
]

export function findDb() {
  for (const p of DB_CANDIDATES) if (existsSync(p)) return p
  return null
}

function query(dbPath, sql) {
  const out = execFileSync('sqlite3', ['-readonly', '-json', dbPath, sql], {
    encoding: 'utf-8',
    maxBuffer: 256 * 1024 * 1024,
    timeout: 30000,
  })
  const trimmed = out.trim()
  if (!trimmed) return []
  return JSON.parse(trimmed)
}

const DEAD_LIKE = [
  '%command not found%',
  '%no such file or directory%',
  '%traceback (most recent%',
  '%is not recognized as%',
  '%permission denied%',
  '%fatal:%',
  '%syntaxerror%',
].map(p => `lower(json_extract(data,'$.state.output')) LIKE '${p}'`).join(' OR ')

/* Every session with its stored aggregates. parent_id IS NULL filtering is
   left to the caller; we return everything and tag whether it's a subagent. */
export function listSessions(dbPath) {
  const rows = query(dbPath, `
    SELECT
      id, title, project_id, parent_id, directory, agent, model,
      cost, tokens_input, tokens_output, tokens_reasoning,
      tokens_cache_read, tokens_cache_write,
      time_created, time_updated
    FROM session
    ORDER BY time_updated DESC
  `)

  const toolCounts = {}
  for (const r of query(dbPath, `
    SELECT session_id AS s, count(*) AS c
    FROM part
    WHERE json_extract(data,'$.type')='tool'
    GROUP BY session_id
  `)) toolCounts[r.s] = r.c

  const deadCounts = {}
  for (const r of query(dbPath, `
    SELECT session_id AS s, count(*) AS c
    FROM part
    WHERE json_extract(data,'$.type')='tool' AND (${DEAD_LIKE})
    GROUP BY session_id
  `)) deadCounts[r.s] = r.c

  const msgCounts = {}
  for (const r of query(dbPath, `
    SELECT session_id AS s, count(*) AS c
    FROM message
    WHERE json_extract(data,'$.role')='assistant'
    GROUP BY session_id
  `)) msgCounts[r.s] = r.c

  return rows.map(r => {
    let model = ''
    if (r.model) {
      try { model = JSON.parse(r.model).id || '' } catch { model = String(r.model) }
    }
    return {
      id: r.id,
      title: r.title,
      projectId: r.project_id,
      parentId: r.parent_id,
      isSubagent: !!r.parent_id,
      directory: r.directory,
      agent: r.agent,
      model,
      cost: r.cost || 0,
      tokensInput: r.tokens_input || 0,
      tokensOutput: r.tokens_output || 0,
      tokensReasoning: r.tokens_reasoning || 0,
      tokensCacheRead: r.tokens_cache_read || 0,
      tokensCacheWrite: r.tokens_cache_write || 0,
      tokensTotal: (r.tokens_input || 0) + (r.tokens_output || 0),
      cacheRate: r.tokens_input ? r.tokens_cache_read / r.tokens_input : null,
      created: r.time_created,
      updated: r.time_updated,
      calls: msgCounts[r.id] || 0,
      toolCalls: toolCounts[r.id] || 0,
      deadEnds: deadCounts[r.id] || 0,
    }
  })
}

/* Cross-session analytics computed entirely in SQL: totals, by-model,
   by-provider (upstream), by-tool, an activity timeline, and top calls. */
export function analytics(dbPath) {
  const totalsRow = query(dbPath, `
    SELECT
      count(*) AS calls,
      sum(json_extract(data,'$.tokens.input')) AS prompt,
      sum(json_extract(data,'$.tokens.output')) AS completion,
      sum(json_extract(data,'$.tokens.reasoning')) AS reasoning,
      sum(json_extract(data,'$.tokens.cache.read')) AS cached,
      sum(json_extract(data,'$.cost')) AS cost
    FROM message
    WHERE json_extract(data,'$.role')='assistant'
  `)[0] || {}

  const sessionCount = query(dbPath, `SELECT count(*) AS c FROM session`)[0]?.c || 0
  const recordCount = query(dbPath, `SELECT count(*) AS c FROM message`)[0]?.c || 0

  const toolTotal = query(dbPath, `
    SELECT count(*) AS c FROM part WHERE json_extract(data,'$.type')='tool'
  `)[0]?.c || 0
  const deadTotal = query(dbPath, `
    SELECT count(*) AS c FROM part
    WHERE json_extract(data,'$.type')='tool' AND (${DEAD_LIKE})
  `)[0]?.c || 0

  const byModel = query(dbPath, `
    SELECT
      json_extract(data,'$.modelID') AS model,
      json_extract(data,'$.providerID') AS provider,
      count(*) AS calls,
      sum(json_extract(data,'$.tokens.input')) AS prompt,
      sum(json_extract(data,'$.tokens.output')) AS completion,
      sum(json_extract(data,'$.tokens.cache.read')) AS cached,
      sum(json_extract(data,'$.cost')) AS cost
    FROM message
    WHERE json_extract(data,'$.role')='assistant'
    GROUP BY model, provider
    ORDER BY cost DESC, calls DESC
  `)

  const byTool = query(dbPath, `
    SELECT
      json_extract(data,'$.tool') AS tool,
      count(*) AS calls,
      sum(CASE WHEN (${DEAD_LIKE}) THEN 1 ELSE 0 END) AS dead
    FROM part
    WHERE json_extract(data,'$.type')='tool'
    GROUP BY tool
    ORDER BY calls DESC
  `)

  // timeline: bucket assistant calls by day
  const timeline = query(dbPath, `
    SELECT
      strftime('%Y-%m-%d', json_extract(data,'$.time.created')/1000, 'unixepoch') AS day,
      count(*) AS calls,
      sum(json_extract(data,'$.tokens.input')) AS prompt,
      sum(json_extract(data,'$.tokens.output')) AS completion,
      sum(json_extract(data,'$.cost')) AS cost,
      min(json_extract(data,'$.time.created')) AS t0
    FROM message
    WHERE json_extract(data,'$.role')='assistant'
      AND json_extract(data,'$.time.created') IS NOT NULL
    GROUP BY day
    ORDER BY day ASC
  `)

  const topSessions = query(dbPath, `
    SELECT id, title, model, cost,
      tokens_input AS prompt, tokens_output AS completion,
      time_updated AS ts
    FROM session
    WHERE cost > 0
    ORDER BY cost DESC
    LIMIT 12
  `).map(r => {
    let model = ''
    if (r.model) { try { model = JSON.parse(r.model).id || '' } catch { model = String(r.model) } }
    return { id: r.id, title: r.title, model, cost: r.cost,
      prompt: r.prompt, completion: r.completion, ts: r.ts }
  })

  return {
    totals: {
      calls: totalsRow.calls || 0,
      prompt: totalsRow.prompt || 0,
      completion: totalsRow.completion || 0,
      total: (totalsRow.prompt || 0) + (totalsRow.completion || 0),
      reasoning: totalsRow.reasoning || 0,
      cached: totalsRow.cached || 0,
      cost: totalsRow.cost || 0,
      toolCalls: toolTotal,
      deadEnds: deadTotal,
      cacheRate: totalsRow.prompt ? (totalsRow.cached || 0) / totalsRow.prompt : null,
    },
    sessionCount,
    recordCount,
    byModel,
    byTool,
    timeline,
    topSessions,
  }
}

/* Reconstruct the `opencode export` shape ({info, messages:[{info,parts}]})
 * for one or many sessions directly from the DB, in a single pass per table.
 * This avoids spawning `opencode export` once per session (≈0.4s each), so
 * exporting hundreds of sessions takes ~1s instead of minutes.
 *
 * Returns a Map<sessionId, exportObject>. `ids` optional; omit for all. */
export function exportSessionsFromDb(dbPath, ids) {
  const idSet = ids && ids.length ? new Set(ids) : null
  const idFilter = idSet
    ? `WHERE id IN (${[...idSet].map(sqlStr).join(',')})`
    : ''
  const sessRows = query(dbPath, `
    SELECT id, title, project_id, directory, agent, model,
           cost, tokens_input, tokens_output, tokens_reasoning,
           tokens_cache_read, tokens_cache_write, time_created, time_updated
    FROM session ${idFilter}
  `)
  if (!sessRows.length) return new Map()

  const wantIds = new Set(sessRows.map(r => r.id))
  const msgFilter = idSet
    ? `WHERE session_id IN (${[...wantIds].map(sqlStr).join(',')})`
    : ''

  // messages (data column is the message info object) ordered chronologically
  const msgRows = query(dbPath, `
    SELECT id, session_id, data FROM message ${msgFilter}
    ORDER BY session_id, time_created
  `)
  // parts ordered chronologically; group under their message
  const partRows = query(dbPath, `
    SELECT message_id, session_id, data FROM part ${msgFilter}
    ORDER BY session_id, time_created
  `)

  const partsByMsg = {}
  for (const p of partRows) {
    let data
    try { data = JSON.parse(p.data) } catch { continue }
    ;(partsByMsg[p.message_id] = partsByMsg[p.message_id] || []).push(data)
  }

  const msgsBySession = {}
  for (const m of msgRows) {
    let info
    try { info = JSON.parse(m.data) } catch { continue }
    const entry = { info, parts: partsByMsg[m.id] || [] }
    ;(msgsBySession[m.session_id] = msgsBySession[m.session_id] || []).push(entry)
  }

  const out = new Map()
  for (const s of sessRows) {
    let model = null
    if (s.model) { try { model = JSON.parse(s.model) } catch { model = { id: String(s.model) } } }
    out.set(s.id, {
      info: {
        id: s.id,
        title: s.title,
        projectID: s.project_id,
        directory: s.directory,
        agent: s.agent,
        model,
        cost: s.cost || 0,
        tokens: {
          input: s.tokens_input || 0,
          output: s.tokens_output || 0,
          reasoning: s.tokens_reasoning || 0,
          cache: { read: s.tokens_cache_read || 0, write: s.tokens_cache_write || 0 },
        },
        time: { created: s.time_created, updated: s.time_updated },
      },
      messages: msgsBySession[s.id] || [],
    })
  }
  return out
}

function sqlStr(s) {
  return `'${String(s).replace(/'/g, "''")}'`
}

