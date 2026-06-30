/* Resolve the Python interpreter for viewer subprocesses.
 * Prefers $PYTHON, then the torch_env conda env (has huggingface_hub etc.),
 * then plain python3. */
import { spawnSync } from 'child_process'
import { existsSync } from 'fs'
import { homedir } from 'os'
import path from 'path'

const TORCH_ENV = [
  path.join(homedir(), 'anaconda3/envs/torch_env/bin/python'),
  path.join(homedir(), 'miniconda3/envs/torch_env/bin/python'),
  path.join(homedir(), 'miniforge3/envs/torch_env/bin/python'),
  path.join(homedir(), 'mambaforge/envs/torch_env/bin/python'),
  '/opt/anaconda3/envs/torch_env/bin/python',
  '/opt/miniconda3/envs/torch_env/bin/python',
]

const CANDIDATES = [
  ...TORCH_ENV,
  process.env.CONDA_PREFIX && path.join(process.env.CONDA_PREFIX, 'bin', 'python'),
].filter(Boolean)

let cached = null

export function resolvePython() {
  if (cached) return cached
  if (process.env.PYTHON) {
    cached = process.env.PYTHON
    return cached
  }
  for (const p of CANDIDATES) {
    if (existsSync(p)) {
      cached = p
      return cached
    }
  }
  cached = 'python3'
  return cached
}

export function pythonHasModule(mod) {
  const py = resolvePython()
  const res = spawnSync(py, ['-c', `import ${mod}`], { encoding: 'utf-8', timeout: 10000 })
  return !res.error && res.status === 0
}
