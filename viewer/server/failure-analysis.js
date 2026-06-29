const PLACEHOLDER_ANSWER = 'write the final answer for this run here'

function inferPassed(verificationText, explicitPassed) {
  if (explicitPassed === true) return true
  if (explicitPassed === false) return false

  const text = String(verificationText || '')
  const exitMatch = text.match(/exit_code=(\d+)/)
  if (exitMatch) return exitMatch[1] === '0'

  if (/^FAILED\s/m.test(text) || /\nFAILED\s/m.test(text)) return false

  const failedMatch = text.match(/(\d+) failed\b/)
  if (failedMatch) return parseInt(failedMatch[1], 10) === 0

  if (/(\d+) passed\b/.test(text)) return true

  return false
}

function cleanMessage(message) {
  let text = String(message || '').trim()
  if (text.startsWith('AssertionError: ')) text = text.slice('AssertionError: '.length)
  if (text.length > 180) text = text.slice(0, 177) + '...'
  return text
}

function parseFailures(verificationText) {
  const failures = []
  const seen = new Set()

  for (const line of String(verificationText || '').split('\n')) {
    const stripped = line.trim()
    const summary = stripped.match(/^FAILED\s+\S+::(\w+)\s+-\s+(.+)$/)
    if (summary) {
      const message = cleanMessage(summary[2])
      if (!seen.has(message)) {
        failures.push({ test: summary[1], message })
        seen.add(message)
      }
      continue
    }

    const assertion = stripped.match(/^E\s+AssertionError:\s+(.+)$/)
    if (assertion) {
      const message = cleanMessage(assertion[1])
      if (!seen.has(message)) {
        failures.push({ test: null, message })
        seen.add(message)
      }
    }
  }

  return failures
}

function collectHints({ answerText = '', diffText = '', verificationText = '' } = {}) {
  const hints = []
  const answer = String(answerText || '').trim().toLowerCase()
  if (answer && answer.includes(PLACEHOLDER_ANSWER)) {
    hints.push('The agent left the placeholder AGENT_FINAL_ANSWER.md unchanged.')
  }
  if (diffText != null && String(diffText).trim() === '') {
    hints.push('No workspace changes were captured in the git diff.')
  }
  const body = String(verificationText || '')
  if (body.includes('ModuleNotFoundError')) {
    hints.push('An import or module path error blocked the test suite from running.')
  }
  if (body.includes('SyntaxError')) {
    hints.push('A syntax error prevented tests from executing.')
  }
  if (/collecting .* ERROR/i.test(body) || /errors during collection/i.test(body)) {
    hints.push('Pytest failed while collecting tests.')
  }
  return hints
}

function buildHeadline({ passed: ok, failures, hints }) {
  if (ok) return null
  const placeholderHint = hints.some(h => h.toLowerCase().includes('placeholder'))
  if (placeholderHint && failures.length) {
    return 'Final answer was never written; comprehension checks failed.'
  }
  if (failures.length) {
    const first = failures[0].message
    if (failures.length === 1) return first
    return `${first} (+${failures.length - 1} more failing check${failures.length > 2 ? 's' : ''})`
  }
  if (hints.length) return hints[0]
  return 'Verification failed; inspect pytest output for details.'
}

export function summarizeVerification(verificationText, { answerText = '', diffText = '', passed: explicitPassed } = {}) {
  const ok = inferPassed(verificationText, explicitPassed)
  const failures = ok ? [] : parseFailures(verificationText)
  const hints = ok ? [] : collectHints({ answerText, diffText, verificationText })
  const headline = buildHeadline({ passed: ok, failures, hints })
  return {
    passed: ok,
    failure_count: failures.length,
    headline,
    hints,
    failures: failures.slice(0, 10),
  }
}
