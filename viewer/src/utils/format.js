export function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

export function fmt(n) { return n == null ? '—' : Number(n).toLocaleString() }

export function fmtTokens(n) {
  if (n == null) return '—'
  n = Number(n)
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k'
  return String(n)
}

export function fmtCost(n) {
  return n == null ? '—' : '$' + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export function fmtCostFine(n) { return n == null ? '—' : '$' + Number(n).toFixed(4) }

export function pct(x) { return x == null ? '—' : Math.round(Math.min(x, 1) * 100) + '%' }

export function fmtTime(ms) {
  if (ms == null) return ''
  const d = new Date(ms)
  if (isNaN(d.getTime())) return ''
  return d.toISOString().slice(0, 16).replace('T', ' ')
}

export function shortTitle(s, n) {
  s = String(s || '').replace(/\s+/g, ' ').trim()
  if (!s) return '(untitled)'
  return s.length > n ? s.slice(0, n - 1) + '…' : s
}

export function shortModel(s) {
  s = String(s || '').trim()
  const slash = s.lastIndexOf('/')
  return slash >= 0 ? s.slice(slash + 1) : s
}

export function shortDir(s) {
  s = String(s || '')
  const home = s.replace(/^\/Users\/[^/]+/, '~')
  return home
}
