import { useState, useCallback, useRef } from 'react'
import { parseInput, extractSession, isEvalRow } from '../utils/parse'
import { opencodeExportToMessages } from '../utils/opencode'
import { buildTrace, computeMetrics } from '../utils/trace'

export default function useTraceStore() {
  const [rawRecords, setRawRecords] = useState([])
  const [sessions, setSessions] = useState([])
  const [activeIdx, setActiveIdx] = useState(0)
  const [view, setView] = useState('sessions')
  const [preloadedEval, setPreloadedEval] = useState([])
  const [error, setError] = useState(null)

  // opencode browse layer
  const [sessionList, setSessionList] = useState([])
  const [sessionTotal, setSessionTotal] = useState(0)
  const [loadingList, setLoadingList] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [analytics, setAnalytics] = useState(null)
  const [loadingSession, setLoadingSession] = useState(false)
  // 'files' (drop/paste/folder) or 'opencode' (a session loaded from the DB)
  const [loadedSource, setLoadedSource] = useState(null)
  const [capabilities, setCapabilities] = useState({ harness: false, filter: false })

  const rawRecordsRef = useRef(rawRecords)
  rawRecordsRef.current = rawRecords

  const rebuildSessions = useCallback((records) => {
    const s = []
    for (const r of records) {
      const es = extractSession(r)
      if (es) { es.meta = es.meta || {}; es.meta._recIdx = r.__idx }
      if (es && es.messages && es.messages.length) s.push(es)
    }
    if (!s.length && records.length && records[0] && records[0].role) {
      const es = extractSession(records)
      if (es) s.push(es)
    }
    return s
  }, [])

  const loadPreloadedEval = useCallback(() => {
    const node = document.getElementById('preloaded-eval')
    if (!node) return
    let parsed
    try { parsed = JSON.parse(node.textContent || '[]') } catch (e) { return }
    const rows = Array.isArray(parsed) ? parsed : (parsed && parsed.rows) || []
    const ev = rows.filter(isEvalRow)
    setPreloadedEval(ev)
    return ev
  }, [])

  const ingestRecords = useCallback((recs, opts) => {
    const merge = !!(opts && opts.merge)
    setError(null)
    let newRecords
    if (merge) {
      for (const r of recs) { r.__idx = rawRecordsRef.current.length }
      newRecords = [...rawRecordsRef.current, ...recs]
    } else {
      newRecords = recs.map((r, i) => { if (r && typeof r === 'object') r.__idx = i; return r })
    }
    const s = rebuildSessions(newRecords)
    const evalRows = preloadedEval.length > 0 || newRecords.some(isEvalRow)

    if (!s.length && !evalRows) {
      if (!merge) setError('Parsed the JSON, but found no messages.')
      return false
    }

    const newActiveIdx = merge ? activeIdx : (s.length
      ? s.reduce((best, sess, i) => sess.messages.length > s[best].messages.length ? i : best, 0)
      : 0)
    const newView = merge
      ? view
      : (evalRows && !s.length ? 'eval' : (s.length > 1 ? 'analytics' : 'anatomy'))

    setRawRecords(newRecords)
    setSessions(s)
    setActiveIdx(newActiveIdx)
    setLoadedSource('files')
    setView(newView)
    return true
  }, [activeIdx, view, preloadedEval, rebuildSessions])

  const ingestTexts = useCallback((texts, opts) => {
    const recs = []
    let firstErr = null
    for (const text of texts) {
      if (!String(text || '').trim()) continue
      try { recs.push.apply(recs, parseInput(text)) }
      catch (e) { if (!firstErr) firstErr = e }
    }
    if (!recs.length) {
      if (!(opts && opts.merge)) setError((firstErr && firstErr.message) || 'No JSON records found.')
      return false
    }
    return ingestRecords(recs, opts)
  }, [ingestRecords])

  const loadText = useCallback((text) => ingestTexts([text]), [ingestTexts])

  const loadFiles = useCallback(async (fileList) => {
    const files = Array.prototype.slice.call(fileList || [])
    if (!files.length) return
    try {
      const texts = await Promise.all(files.map(f => f.text().catch(() => '')))
      ingestTexts(texts)
    } catch (err) {
      setError(err.message)
    }
  }, [ingestTexts])

  const hasEvalRows = useCallback(() => {
    if (preloadedEval.length) return true
    for (const r of rawRecordsRef.current) if (isEvalRow(r)) return true
    return false
  }, [preloadedEval])

  const switchView = useCallback((v) => setView(v), [])

  const openRecordInAnatomy = useCallback((recIdx) => {
    let idx = sessions.findIndex(s => s.meta && s.meta._recIdx === recIdx)
    if (idx === -1) {
      const rec = rawRecords[recIdx]
      const es = extractSession(rec)
      if (!es) return
      es.meta = es.meta || {}; es.meta._recIdx = recIdx
      const newSessions = [...sessions, es]
      setSessions(newSessions)
      idx = newSessions.length - 1
    }
    setActiveIdx(idx)
    setView('anatomy')
  }, [sessions, rawRecords])

  const getCurrentTrace = useCallback(() => {
    if (!sessions[activeIdx]) return null
    return buildTrace(sessions[activeIdx].messages)
  }, [sessions, activeIdx])

  const getCurrentMetrics = useCallback(() => {
    if (!sessions[activeIdx]) return null
    const trace = buildTrace(sessions[activeIdx].messages)
    return computeMetrics(trace, sessions[activeIdx].meta)
  }, [sessions, activeIdx])

  // Clear the loaded trace and return to the session browser.
  const backToSessions = useCallback(() => {
    setRawRecords([])
    setSessions([])
    setActiveIdx(0)
    setError(null)
    setLoadedSource(null)
    setView('sessions')
  }, [])

  const reset = backToSessions

  // ---- opencode browse layer ----
  const PAGE_SIZE = 15

  // Fetch a page of sessions. opts: { offset, search, sort, dir, subagents, append }.
  // When append is true the page is concatenated onto the current list
  // ("load more"); otherwise it replaces it (new search/sort/first load).
  const fetchSessionList = useCallback(async (opts = {}) => {
    const { offset = 0, search = '', sort = 'updated', dir = 'desc', subagents = false, append = false } = opts
    if (append) setLoadingMore(true); else setLoadingList(true)
    try {
      const qs = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(offset),
        sort,
        dir,
        subagents: subagents ? '1' : '0',
      })
      if (search) qs.set('search', search)
      const res = await fetch(`/api/sessions?${qs.toString()}`)
      if (!res.ok) throw new Error('Failed to fetch session list')
      const data = await res.json()
      const page = Array.isArray(data) ? data : (data.sessions || [])
      const total = Array.isArray(data) ? page.length : (data.total || 0)
      setSessionTotal(total)
      setSessionList(prev => append ? [...prev, ...page] : page)
      return page
    } catch (e) {
      setError('Could not load session list: ' + e.message)
      return []
    } finally {
      if (append) setLoadingMore(false); else setLoadingList(false)
    }
  }, [])

  const fetchAnalytics = useCallback(async () => {
    try {
      const res = await fetch('/api/analytics')
      if (!res.ok) throw new Error('Failed to fetch analytics')
      const data = await res.json()
      setAnalytics(data)
      return data
    } catch (e) {
      // analytics is optional; do not surface as a hard error
      return null
    }
  }, [])

  const loadSession = useCallback(async (sessionId) => {
    setLoadingSession(true)
    setError(null)
    try {
      const res = await fetch(`/api/sessions/${sessionId}`)
      if (!res.ok) throw new Error('Failed to fetch session')
      const data = await res.json()
      const messages = opencodeExportToMessages(data)
      if (!messages.length) {
        setError('No messages found in this session.')
        return
      }
      const info = data.info || {}
      // opencode does not persist the system prompt; the server reconstructs it
      // from the harness fixtures. Prepend it so it renders in the System card.
      const sysPrompt = data._reconstructedSystem
      if (sysPrompt && sysPrompt.content) {
        messages.unshift({ role: 'system', content: sysPrompt.content })
      }
      const record = {
        messages,
        model: info.model?.id || '',
        source: 'opencode',
        usage: {
          prompt_tokens: info.tokens?.input,
          completion_tokens: info.tokens?.output,
          prompt_tokens_details: { cached_tokens: info.tokens?.cache?.read },
          cost: info.cost,
        },
        _sessionInfo: info,
      }
      record.__idx = 0
      const recs = [record]
      setRawRecords(recs)
      const built = rebuildSessions(recs)
      if (built[0]) {
        built[0].meta = built[0].meta || {}
        built[0].meta.model = info.model?.id || ''
        built[0].meta.title = info.title
        built[0].meta.timestamp = info.time?.updated
          ? new Date(info.time.updated).toISOString().slice(0, 16).replace('T', ' ')
          : undefined
        built[0].meta.usage = record.usage
        built[0].meta.source = 'opencode'
        built[0].meta.sessionId = sessionId
        built[0].meta.reconstructedSystem = sysPrompt || null
        built[0].meta.reconstructedTools = data._reconstructedTools || null
      }
      setSessions(built)
      setActiveIdx(0)
      setLoadedSource('opencode')
      setView('anatomy')
    } catch (e) {
      setError('Could not load session: ' + e.message)
    } finally {
      setLoadingSession(false)
    }
  }, [rebuildSessions])

  // Download one or more sessions as build_dataset.py-compatible SFT JSONL.
  // opts.filter -> pipe through build_dataset.py's quality gate + sanitize.
  // opts.all   -> export every session (ids ignored).
  const exportSessions = useCallback(async (ids, opts = {}) => {
    const list = Array.isArray(ids) ? ids : [ids]
    if (!opts.all && !list.length) return
    try {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: list, ...opts }),
      })
      if (!res.ok) {
        let msg = 'Export failed'
        try { msg = (await res.json()).error || msg } catch {}
        throw new Error(msg)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const tag = opts.filter ? 'filtered' : 'raw'
      a.download = opts.all
        ? `opencode-sft-all-${tag}.jsonl`
        : (list.length === 1
            ? `opencode-${list[0]}-${tag}.jsonl`
            : `opencode-sft-${list.length}-${tag}.jsonl`)
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError('Could not export: ' + e.message)
    }
  }, [])

  // Upload one or more sessions straight to a private HF dataset repo.
  // opts: { all, filter, sanitize, modelName, repo }. Returns the result obj.
  const uploadSessions = useCallback(async (ids, opts = {}) => {
    const list = Array.isArray(ids) ? ids : [ids]
    if (!opts.all && !list.length) return null
    const res = await fetch('/api/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: list, ...opts }),
    })
    if (!res.ok) {
      let msg = 'Upload failed'
      try { msg = (await res.json()).error || msg } catch {}
      throw new Error(msg)
    }
    return res.json()
  }, [])

  const fetchCapabilities = useCallback(async () => {
    try {
      const res = await fetch('/api/capabilities')
      if (!res.ok) return null
      const caps = await res.json()
      setCapabilities(caps)
      return caps
    } catch { return null }
  }, [])

  return {
    rawRecords, sessions, activeIdx, view, preloadedEval, error,
    sessionList, sessionTotal, loadingList, loadingMore, analytics, loadingSession, loadedSource, capabilities,
    setActiveIdx, setView, setError,
    ingestRecords, ingestTexts, loadText, loadFiles,
    loadPreloadedEval, switchView, openRecordInAnatomy,
    getCurrentTrace, getCurrentMetrics, hasEvalRows,
    backToSessions, reset, rebuildSessions,
    fetchSessionList, PAGE_SIZE, fetchAnalytics, loadSession, exportSessions, uploadSessions, fetchCapabilities,
  }
}
