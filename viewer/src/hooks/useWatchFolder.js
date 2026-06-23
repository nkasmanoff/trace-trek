import { useState, useRef, useCallback, useEffect } from 'react'

const IDB_DB = 'trace-anatomy', IDB_STORE = 'handles', IDB_KEY = 'watchDir'
const DEFAULT_DIR = 'raw/opencode'
const TRACE_RE = /\.(json|jsonl|txt)$/i
const POLL_MS = 4000

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(IDB_DB, 1)
    r.onupgradeneeded = () => r.result.createObjectStore(IDB_STORE)
    r.onsuccess = () => res(r.result)
    r.onerror = () => rej(r.error)
  })
}

async function idbSet(key, val) {
  try {
    const db = await idb()
    await new Promise((res, rej) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      tx.objectStore(IDB_STORE).put(val, key)
      tx.oncomplete = res; tx.onerror = () => rej(tx.error)
    })
  } catch (e) { /* private mode */ }
}

async function idbGet(key) {
  try {
    const db = await idb()
    return await new Promise((res, rej) => {
      const tx = db.transaction(IDB_STORE, 'readonly')
      const rq = tx.objectStore(IDB_STORE).get(key)
      rq.onsuccess = () => res(rq.result)
      rq.onerror = () => rej(rq.error)
    })
  } catch (e) { return null }
}

async function idbDel(key) {
  try {
    const db = await idb()
    await new Promise((res, rej) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      tx.objectStore(IDB_STORE).delete(key)
      tx.oncomplete = res; tx.onerror = () => rej(tx.error)
    })
  } catch (e) { /* ignore */ }
}

async function resolveTraceDir(handle) {
  let cur = handle
  for (const part of DEFAULT_DIR.split('/')) {
    try { cur = await cur.getDirectoryHandle(part) }
    catch (e) { return handle }
  }
  return cur
}

async function ensurePermission(handle, mode) {
  const opts = { mode: mode || 'read' }
  if ((await handle.queryPermission(opts)) === 'granted') return true
  return (await handle.requestPermission(opts)) === 'granted'
}

export default function useWatchFolder(store) {
  const [watchState, setWatchState] = useState({ active: false, paused: false, text: '' })
  const watchRef = useRef({ active: false, dirHandle: null, seen: new Set(), timer: null, busy: false })

  const setLive = useCallback((state, text) => {
    setWatchState({
      active: true,
      paused: state !== 'live',
      text,
    })
  }, [])

  const pollDir = useCallback(async (initial) => {
    const w = watchRef.current
    if (!w.dirHandle || w.busy) return
    w.busy = true
    try {
      const fresh = []
      for await (const [name, handle] of w.dirHandle.entries()) {
        if (handle.kind !== 'file' || !TRACE_RE.test(name)) continue
        if (w.seen.has(name)) continue
        w.seen.add(name)
        fresh.push(handle)
      }
      if (fresh.length) {
        const texts = await Promise.all(fresh.map(async (fh) => {
          try { const f = await fh.getFile(); return await f.text() }
          catch (e) { return '' }
        }))
        store.ingestTexts(texts, { merge: !initial && store.rawRecords.length > 0 })
      }
      if (w.active) setLive('live', `watching ${w.dirHandle.name} · ${w.seen.size} files`)
    } catch (e) {
      setLive('paused', 'watch error')
    } finally {
      w.busy = false
    }
  }, [store, setLive])

  const stopWatch = useCallback(() => {
    const w = watchRef.current
    if (w.timer) clearInterval(w.timer)
    w.active = false; w.dirHandle = null; w.timer = null
    setWatchState({ active: false, paused: false, text: '' })
  }, [setWatchState])

  const beginWatch = useCallback(async (rootHandle, opts) => {
    const dirHandle = await resolveTraceDir(rootHandle)
    if (!(await ensurePermission(dirHandle, 'read'))) {
      setLive('paused', 'permission denied')
      return false
    }
    const w = watchRef.current
    w.active = true
    w.dirHandle = dirHandle
    w.seen = new Set()
    w.busy = false
    setLive('live', `loading ${dirHandle.name}…`)
    const first = []
    try {
      for await (const [name, h] of dirHandle.entries()) {
        if (h.kind === 'file' && TRACE_RE.test(name)) { w.seen.add(name); first.push(h) }
      }
    } catch (e) { setLive('paused', 'watch error'); return false }
    if (first.length) {
      const texts = await Promise.all(first.map(async (fh) => {
        try { const f = await fh.getFile(); return await f.text() }
        catch (e) { return '' }
      }))
      store.ingestTexts(texts)
    }
    setLive('live', `watching ${dirHandle.name} · ${w.seen.size} files`)
    w.timer = setInterval(() => pollDir(false), POLL_MS)
    if (!(opts && opts.noPersist)) idbSet(IDB_KEY, rootHandle)
    return true
  }, [store, setLive, pollDir])

  const startWatch = useCallback(async () => {
    let handle
    try { handle = await window.showDirectoryPicker() }
    catch (e) { return }
    await beginWatch(handle)
  }, [beginWatch])

  useEffect(() => {
    if (!window.showDirectoryPicker) return
    idbGet(IDB_KEY).then(async (root) => {
      if (!root) return
      try {
        const dir = await resolveTraceDir(root)
        if ((await dir.queryPermission({ mode: 'read' })) === 'granted') {
          await beginWatch(root, { noPersist: true })
        } else {
          setLive('paused', `click 'Watch folder' to resume ${DEFAULT_DIR}`)
        }
      } catch (e) { /* stale handle */ }
    })
  }, [])

  return { watchState, startWatch, stopWatch, beginWatch }
}
