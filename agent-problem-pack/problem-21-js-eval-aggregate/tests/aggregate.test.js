import assert from 'node:assert/strict'
import { test } from 'node:test'

import { aggregateEval, findFlips } from '../src/aggregate.js'

test('groups results by run and type', () => {
  const rows = aggregateEval([
    { taskId: 't1', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 't2', run: 'base', type: 'knowledge', passed: false, score: 0 },
    { taskId: 't1', run: 'sft', type: 'code', passed: true, score: 1 },
  ])
  assert.equal(rows.length, 3)
})

test('skipped records are excluded from totals and pass rate', () => {
  const rows = aggregateEval([
    { taskId: 't1', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 't2', run: 'base', type: 'code', passed: null, score: null },
    { taskId: 't3', run: 'base', type: 'code', passed: true, score: 0.8 },
  ])
  assert.equal(rows.length, 1)
  assert.equal(rows[0].total, 2)
  assert.equal(rows[0].passed, 2)
  assert.equal(rows[0].passRate, 1)
})

test('flip requires a pass in base and a fail in candidate', () => {
  const results = [
    { taskId: 't1', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 't1', run: 'sft', type: 'code', passed: false, score: 0 },
    { taskId: 't2', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 't2', run: 'sft', type: 'code', passed: true, score: 1 },
  ]
  assert.deepEqual(findFlips(results, 'base', 'sft'), ['t1'])
})

test('task skipped in candidate run is not a flip', () => {
  const results = [
    { taskId: 't1', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 't1', run: 'sft', type: 'code', passed: null, score: null },
  ]
  assert.deepEqual(findFlips(results, 'base', 'sft'), [])
})
