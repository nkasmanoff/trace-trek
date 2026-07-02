/**
 * Aggregate normalized eval results into per-(run, type) buckets and detect
 * run-to-run regressions ("flips").
 *
 * A result record: { taskId, run, type, passed, score }
 * `passed` is true, false, or null. Null means the task was SKIPPED for that
 * run: skipped records must not count toward total, passed, or passRate,
 * and must never be treated as failures.
 *
 * findFlips(results, baseRun, candRun) returns the taskIds that passed in
 * baseRun but FAILED (passed === false) in candRun. Tasks that are missing
 * or skipped in either run are not flips: a flip requires an actual pass in
 * base and an actual fail in candidate.
 */

export function aggregateEval(results) {
  const buckets = new Map()
  for (const r of results) {
    const key = `${r.run}::${r.type}`
    if (!buckets.has(key)) {
      buckets.set(key, { run: r.run, type: r.type, total: 0, passed: 0, records: [] })
    }
    const bucket = buckets.get(key)
    bucket.records.push(r)
    bucket.total += 1
    if (r.passed) bucket.passed += 1
  }
  return [...buckets.values()].map(b => ({
    run: b.run,
    type: b.type,
    total: b.total,
    passed: b.passed,
    passRate: b.total ? b.passed / b.total : 0,
    meanScore: b.records.length
      ? b.records.reduce((sum, r) => sum + (r.score ?? 0), 0) / b.records.length
      : 0,
  }))
}

export function findFlips(results, baseRun, candRun) {
  const base = new Map()
  const cand = new Map()
  for (const r of results) {
    if (r.run === baseRun) base.set(r.taskId, r)
    if (r.run === candRun) cand.set(r.taskId, r)
  }
  const flips = []
  for (const [taskId, record] of base) {
    if (!record.passed) continue
    const candidate = cand.get(taskId)
    if (!candidate || !candidate.passed) flips.push(taskId)
  }
  return flips.sort()
}
