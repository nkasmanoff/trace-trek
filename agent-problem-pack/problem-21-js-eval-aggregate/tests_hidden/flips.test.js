import assert from 'node:assert/strict'
import { test } from 'node:test'

import { aggregateEval, findFlips } from '../src/aggregate.js'

test('task missing from candidate run is not a flip', () => {
  const results = [
    { taskId: 't1', run: 'base', type: 'code', passed: true, score: 1 },
  ]
  assert.deepEqual(findFlips(results, 'base', 'sft'), [])
})

test('mean score ignores skipped records', () => {
  const rows = aggregateEval([
    { taskId: 't1', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 't2', run: 'base', type: 'code', passed: null, score: null },
  ])
  assert.equal(rows[0].meanScore, 1)
})

test('all-skipped bucket reports zero rates without dividing by zero', () => {
  const rows = aggregateEval([
    { taskId: 't1', run: 'base', type: 'code', passed: null, score: null },
    { taskId: 't2', run: 'base', type: 'code', passed: null, score: null },
  ])
  assert.equal(rows[0].total, 0)
  assert.equal(rows[0].passed, 0)
  assert.equal(rows[0].passRate, 0)
  assert.equal(rows[0].meanScore, 0)
})

test('flips are sorted and exclude base failures', () => {
  const results = [
    { taskId: 'b', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 'b', run: 'sft', type: 'code', passed: false, score: 0 },
    { taskId: 'a', run: 'base', type: 'code', passed: true, score: 1 },
    { taskId: 'a', run: 'sft', type: 'code', passed: false, score: 0 },
    { taskId: 'c', run: 'base', type: 'code', passed: false, score: 0 },
    { taskId: 'c', run: 'sft', type: 'code', passed: false, score: 0 },
  ]
  assert.deepEqual(findFlips(results, 'base', 'sft'), ['a', 'b'])
})
