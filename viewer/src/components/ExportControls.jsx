import { useState } from 'react'

// Common target models for the "fine-tune for" selector. The chosen value is
// passed to build_dataset.py as --model-name, which rewrites the teacher's
// identity in every system prompt ("You are powered by <model>") so the data
// trains the model to be itself, not the teacher.
const MODEL_PRESETS = [
  'local-model',
  'CohereLabs/North-Mini-Code-1.0',
  'frontier/anthropic/claude-sonnet-4.6',
]

export default function ExportControls({
  scopeLabel,            // e.g. "all 623" or "12 selected"
  disabled,
  canFilter, canUpload,
  onDownload,            // (opts) => Promise
  onUpload,              // (opts) => Promise<result>
}) {
  const [modelName, setModelName] = useState('local-model')
  const [applyFilter, setApplyFilter] = useState(true)
  const [repo, setRepo] = useState('opencode-sft')
  const [busy, setBusy] = useState(null) // 'download' | 'upload'
  const [status, setStatus] = useState(null)

  const opts = () => ({
    filter: canFilter && applyFilter,
    sanitize: true,
    modelName: modelName.trim() || 'local-model',
    repo: repo.trim() || 'opencode-sft',
  })

  const doDownload = async () => {
    setBusy('download'); setStatus(null)
    try { await onDownload(opts()) }
    catch (e) { setStatus({ err: e.message }) }
    finally { setBusy(null) }
  }
  const doUpload = async () => {
    setBusy('upload'); setStatus(null)
    try {
      const r = await onUpload(opts())
      if (r) setStatus({ ok: `Uploaded ${r.exported} record${r.exported === 1 ? '' : 's'} → ${r.repo_id}`, url: r.url })
    } catch (e) { setStatus({ err: e.message }) }
    finally { setBusy(null) }
  }

  return (
    <div className="export-controls">
      <div className="export-row">
        <label className="export-field">
          <span>fine-tune for</span>
          <input
            list="model-presets"
            className="export-input"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="local-model"
            title="Model identity written into the system prompts (build_dataset.py --model-name)"
          />
          <datalist id="model-presets">
            {MODEL_PRESETS.map(m => <option key={m} value={m} />)}
          </datalist>
        </label>

        {canFilter && (
          <label className="sess-toggle" title="Run records through build_dataset.py's quality gate: drop corrections/tool-loops/malformed/over-length, dedup, sanitize identity + temp paths.">
            <input type="checkbox" checked={applyFilter} onChange={(e) => setApplyFilter(e.target.checked)} />
            quality filter
          </label>
        )}

        <button className="btn" onClick={doDownload} disabled={disabled || busy}>
          {busy === 'download' ? 'Exporting…' : `Download ${scopeLabel}`}
        </button>

        {canUpload && (
          <>
            <label className="export-field">
              <span>HF repo</span>
              <input
                className="export-input"
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                placeholder="opencode-sft"
                title="Private HF dataset repo (bare name goes under your namespace)"
              />
            </label>
            <button className="btn primary" onClick={doUpload} disabled={disabled || busy}>
              {busy === 'upload' ? 'Uploading…' : `Upload ${scopeLabel} → HF`}
            </button>
          </>
        )}
      </div>

      {status?.ok && (
        <div className="export-status ok">
          {status.ok}{status.url && <> · <a href={status.url} target="_blank" rel="noreferrer">view dataset</a></>}
        </div>
      )}
      {status?.err && <div className="export-status err">{status.err}</div>}
    </div>
  )
}
