/* Upload an SFT JSONL string to a private Hugging Face dataset repo via
 * scripts/upload_to_hf.py (reuses huggingface_hub + HF_TOKEN in .env).
 * Large datasets are staged to a temp file instead of piped on stdin — piping
 * 100MB+ through spawnSync stdin causes EPIPE if the child exits early. */
import { spawnSync } from 'child_process'
import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { fileURLToPath } from 'url'
import { resolvePython, pythonHasModule } from './python.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const UPLOADER = path.resolve(__dirname, '../../scripts/upload_to_hf.py')
const ENV_FILE = path.resolve(__dirname, '../../../.env')

export function hfAvailable() {
  if (!existsSync(UPLOADER)) return false
  if (!pythonHasModule('huggingface_hub')) return false
  // a token must be resolvable for upload to work
  if (process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN) return true
  try {
    return /^HF_TOKEN=.+/m.test(readFileSync(ENV_FILE, 'utf8'))
  } catch { return false }
}

/* jsonl: the dataset content. opts: { repo, pathInRepo }.
 * Returns the parsed JSON result from the uploader. Throws on failure. */
export function uploadToHf(jsonl, opts = {}) {
  if (!existsSync(UPLOADER)) throw new Error('upload_to_hf.py not found')
  const repo = opts.repo || 'opencode-sft'
  const inFile = path.join(tmpdir(), `hf_upload_${process.pid}_${Date.now()}.jsonl`)
  writeFileSync(inFile, jsonl)
  try {
    const args = [UPLOADER, '--repo', repo, '--input-file', inFile]
    if (opts.pathInRepo) args.push('--path-in-repo', opts.pathInRepo)

    const py = resolvePython()
    const res = spawnSync(py, args, {
      encoding: 'utf-8',
      maxBuffer: 16 * 1024 * 1024,
      timeout: 600000, // 10 min — large uploads can be slow
    })
    if (res.error) {
      const hint = res.error.code === 'EPIPE'
        ? ' (upload subprocess failed — check HF token and huggingface_hub install)'
        : ''
      throw new Error(String(res.error.message || res.error) + hint)
    }
    // the script prints a JSON result line on stdout for both success and failure
    let parsed = null
    const out = (res.stdout || '').trim().split('\n').filter(Boolean).pop()
    try { parsed = JSON.parse(out) } catch { /* ignore */ }
    if (!parsed) {
      throw new Error('uploader produced no result: ' + (res.stderr || '').slice(0, 300))
    }
    if (!parsed.ok) throw new Error(parsed.error || 'upload failed')
    return parsed
  } finally {
    try { unlinkSync(inFile) } catch {}
  }
}
