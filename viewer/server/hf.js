/* Upload an SFT JSONL string to a private Hugging Face dataset repo by piping
 * it through scripts/upload_to_hf.py (which reuses the installed
 * huggingface_hub + the HF_TOKEN in self-improve/.env). */
import { spawnSync } from 'child_process'
import { existsSync, readFileSync } from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const UPLOADER = path.resolve(__dirname, '../../scripts/upload_to_hf.py')
const ENV_FILE = path.resolve(__dirname, '../../../.env')

export function hfAvailable() {
  if (!existsSync(UPLOADER)) return false
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
  const args = [UPLOADER, '--repo', repo]
  if (opts.pathInRepo) args.push('--path-in-repo', opts.pathInRepo)

  const py = process.env.PYTHON || 'python3'
  const res = spawnSync(py, args, {
    input: jsonl,
    encoding: 'utf-8',
    maxBuffer: 512 * 1024 * 1024,
    timeout: 300000,
  })
  if (res.error) throw res.error
  // the script prints a JSON result line on stdout for both success and failure
  let parsed = null
  const out = (res.stdout || '').trim().split('\n').filter(Boolean).pop()
  try { parsed = JSON.parse(out) } catch { /* ignore */ }
  if (!parsed) {
    throw new Error('uploader produced no result: ' + (res.stderr || '').slice(0, 300))
  }
  if (!parsed.ok) throw new Error(parsed.error || 'upload failed')
  return parsed
}
