/* Run already-built SFT records through self-improve/build_dataset.py so the
 * app's export path applies the EXACT same quality gate (correction/loop/
 * malformed/too-long filters + sanitization + session dedup) as the canonical
 * proxy-log dataset build. We shell out to the real script rather than porting
 * its logic to JS, so the two paths can never drift.
 */
import { spawnSync } from 'child_process'
import { existsSync, writeFileSync, unlinkSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { fileURLToPath } from 'url'
import { resolvePython } from './python.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const BUILD_DATASET = path.resolve(__dirname, '../../../pipeline/dataset/build_dataset.py')

export function buildDatasetAvailable() {
  return existsSync(BUILD_DATASET)
}

/* records: array of {messages, tools, source} objects.
 * opts: { sanitize, modelName, workspacePath, maxTokens }
 * Returns { jsonl, stderr } — jsonl is the filtered NDJSON (one record/line).
 * Throws if the script is missing or exits non-zero. */
export function filterThroughBuildDataset(records, opts = {}) {
  if (!buildDatasetAvailable()) {
    throw new Error('build_dataset.py not found at ' + BUILD_DATASET)
  }
  const input = records.map(r => JSON.stringify(r)).join('\n') + '\n'
  const inFile = path.join(tmpdir(), `sft_filter_${process.pid}_${Date.now()}.jsonl`)
  writeFileSync(inFile, input)
  const args = [BUILD_DATASET, '--sft-input', inFile, '--stdout']
  if (opts.sanitize === false) args.push('--no-sanitize')
  if (opts.modelName) args.push('--model-name', opts.modelName)
  if (opts.workspacePath) args.push('--workspace-path', opts.workspacePath)
  if (opts.maxTokens) args.push('--max-tokens', String(opts.maxTokens))

  const py = resolvePython()
  try {
    const res = spawnSync(py, args, {
      encoding: 'utf-8',
      maxBuffer: 512 * 1024 * 1024,
      timeout: 120000,
    })
    if (res.error) throw res.error
    if (res.status !== 0) {
      throw new Error('build_dataset.py failed: ' + (res.stderr || '').slice(0, 500))
    }
    return { jsonl: res.stdout || '', stderr: res.stderr || '' }
  } finally {
    try { unlinkSync(inFile) } catch {}
  }
}
