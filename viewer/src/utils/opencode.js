export function opencodeExportToMessages(data) {
  const messages = []
  const { info, messages: rawMessages } = data

  for (const msg of rawMessages) {
    const role = msg.info?.role

    if (role === 'user') {
      const texts = msg.parts
        .filter(p => p.type === 'text')
        .map(p => p.text || '')
        .join('\n')
      if (texts.trim()) {
        messages.push({ role: 'user', content: texts })
      }
    } else if (role === 'assistant') {
      const textParts = []
      const reasoningParts = []
      const toolParts = []

      for (const part of msg.parts) {
        if (part.type === 'text') {
          textParts.push(part.text || '')
        } else if (part.type === 'reasoning') {
          reasoningParts.push(part.text || '')
        } else if (part.type === 'tool') {
          toolParts.push(part)
        }
      }

      const text = textParts.join('\n').trim()
      const reasoning = reasoningParts.join('\n').trim()

      if (toolParts.length > 0) {
        const contentBlocks = []
        if (text) contentBlocks.push({ type: 'text', text })
        if (reasoning) contentBlocks.push({ type: 'thinking', thinking: reasoning })

        const toolCalls = toolParts.map(tp => {
          let argsStr
          try { argsStr = JSON.stringify(tp.state?.input || {}) } catch (e) { argsStr = '{}' }
          return {
            id: tp.callID,
            function: { name: tp.tool, arguments: argsStr },
          }
        })

        const assistantMsg = { role: 'assistant' }
        if (contentBlocks.length > 0) {
          assistantMsg.content = contentBlocks
        } else {
          assistantMsg.content = ''
        }
        assistantMsg.tool_calls = toolCalls
        messages.push(assistantMsg)

        for (let i = 0; i < toolParts.length; i++) {
          const tp = toolParts[i]
          messages.push({
            role: 'tool',
            tool_call_id: tp.callID,
            content: tp.state?.output || '',
          })
        }
      } else {
        if (text && reasoning) {
          messages.push({
            role: 'assistant',
            content: [{ type: 'text', text }, { type: 'thinking', thinking: reasoning }],
          })
        } else if (text) {
          messages.push({ role: 'assistant', content: text })
        } else if (reasoning) {
          messages.push({
            role: 'assistant',
            content: [{ type: 'thinking', thinking: reasoning }],
          })
        }
      }
    }
  }

  return messages
}

export function sessionListToMeta(session) {
  return {
    id: session.id,
    title: session.title,
    projectId: session.projectId,
    directory: session.directory,
    created: session.created,
    updated: session.updated,
  }
}
