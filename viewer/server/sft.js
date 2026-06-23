/* Convert an opencode session export into an SFT training record matching
 * self-improve/build_dataset.py's expected shape:
 *
 *   { messages: [...], tools: [...]|null, source: "opencode", ... }
 *
 * messages use OpenAI chat roles. Assistant turns may carry:
 *   - reasoning_content : the model's thinking (mapped onto the template's
 *                         thinking slot by the trainer)
 *   - tool_calls        : [{ id, type:"function", function:{name, arguments} }]
 * Tool results are emitted as separate { role:"tool", tool_call_id, content }.
 *
 * The system prompt + tool schemas are reconstructed (opencode does not persist
 * them); see harness.js. We set source per the upstream provider so the dataset
 * can distinguish frontier-distillation from local/organic trajectories.
 */
import { reconstructSystemPrompt, reconstructTools } from './harness.js'

function partsText(parts, type) {
  return parts.filter(p => p.type === type).map(p => p.text || '').join('\n').trim()
}

/* Map opencode upstream provider -> dataset source label, matching
 * build_dataset.py's taxonomy (frontier = openrouter-traced teacher). */
function sourceFor(info) {
  const provider = String(info?.model?.providerID || info?.providerID || '').toLowerCase()
  const modelId = String(info?.model?.id || '').toLowerCase()
  if (provider === 'openrouter' || modelId.includes('anthropic') || modelId.includes('claude')) {
    return 'frontier'
  }
  return 'opencode'
}

export function exportToSft(data, opts = {}) {
  const info = data.info || {}
  const rawMessages = data.messages || []
  const messages = []
  const usedTools = new Set()

  // 1) reconstructed system prompt
  const sys = reconstructSystemPrompt(info)
  if (sys && opts.includeSystem !== false) {
    messages.push({ role: 'system', content: sys.content })
  }

  // 2) walk the transcript
  for (const msg of rawMessages) {
    const role = msg.info?.role
    const parts = msg.parts || []

    if (role === 'user') {
      const text = partsText(parts, 'text')
      if (text) messages.push({ role: 'user', content: text })
      continue
    }
    if (role !== 'assistant') continue

    const text = partsText(parts, 'text')
    const reasoning = partsText(parts, 'reasoning')
    const toolParts = parts.filter(p => p.type === 'tool')

    const assistant = { role: 'assistant', content: text || '' }
    if (reasoning) assistant.reasoning_content = reasoning

    if (toolParts.length) {
      assistant.tool_calls = toolParts.map(tp => {
        usedTools.add(tp.tool)
        let args = tp.state?.input ?? {}
        let argStr
        try { argStr = typeof args === 'string' ? args : JSON.stringify(args) }
        catch { argStr = '{}' }
        return {
          id: tp.callID,
          type: 'function',
          function: { name: tp.tool, arguments: argStr },
        }
      })
    }
    messages.push(assistant)

    // tool results as separate tool-role messages
    for (const tp of toolParts) {
      const out = tp.state?.output
      messages.push({
        role: 'tool',
        tool_call_id: tp.callID,
        content: out == null ? '' : String(out),
      })
    }
  }

  const tools = reconstructTools(usedTools)

  return {
    messages,
    tools: tools.length ? tools : null,
    source: sourceFor(info),
    // provenance so downstream knows these were rebuilt, not captured
    _reconstructed: {
      system: !!sys,
      systemRole: sys?.role || null,
      tools: tools.length,
      sessionId: info.id || null,
      title: info.title || null,
    },
  }
}
