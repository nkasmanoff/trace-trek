import { classifyTool, isDeadEnd } from './classify.js';

export function messageEvents(msg) {
  const ev = [];
  if (!msg || !msg.role) {
    if (msg && (msg.type === "tool_result"))
      ev.push({ kind: "toolresult", id: msg.tool_use_id || msg.tool_call_id, text: contentToText(msg.content) });
    return ev;
  }
  const role = msg.role;
  if (role === "system" || role === "developer") {
    ev.push({ kind: "system", text: contentToText(msg.content) });
    return ev;
  }
  if (role === "tool") {
    ev.push({ kind: "toolresult", id: msg.tool_call_id || msg.tool_use_id, text: contentToText(msg.content) });
    return ev;
  }
  if (role === "user") {
    if (Array.isArray(msg.content)) {
      let texts = [];
      for (const b of msg.content) {
        if (b && b.type === "tool_result")
          ev.push({ kind: "toolresult", id: b.tool_use_id || b.tool_call_id, text: contentToText(b.content) });
        else if (b && (b.type === "text" || typeof b === "string"))
          texts.push(typeof b === "string" ? b : (b.text || ""));
      }
      const joined = texts.join("\n").trim();
      if (joined) ev.push({ kind: "user", text: joined });
    } else {
      ev.push({ kind: "user", text: contentToText(msg.content) });
    }
    return ev;
  }
  if (role === "assistant") {
    let text = "", tools = [];
    if (Array.isArray(msg.content)) {
      for (const b of msg.content) {
        if (!b) continue;
        if (b.type === "text") text += (text ? "\n" : "") + (b.text || "");
        else if (b.type === "thinking" || b.type === "reasoning")
          ev.push({ kind: "thinking", text: b.thinking || b.text || "" });
        else if (b.type === "tool_use")
          tools.push({ id: b.id, name: b.name, args: b.input });
      }
    } else {
      text = contentToText(msg.content);
    }
    const reasoning = msg.reasoning_content || msg.reasoning;
    if (reasoning && typeof reasoning === "string" && reasoning.trim())
      ev.push({ kind: "thinking", text: reasoning });
    if (text && text.trim()) ev.push({ kind: "text", text: text });
    if (Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        let raw = tc.function ? tc.function.arguments : tc.arguments;
        let parsed = raw;
        if (typeof raw === "string") { try { parsed = JSON.parse(raw); } catch (e) { parsed = raw; } }
        tools.push({ id: tc.id, name: (tc.function && tc.function.name) || tc.name, args: parsed });
      }
    }
    if (tools.length) ev.push({ kind: "toolbatch", calls: tools });
    return ev;
  }
  return ev;
}

function contentToText(c) {
  if (c == null) return "";
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c.map(b => {
      if (b == null) return "";
      if (typeof b === "string") return b;
      if (b.type === "text") return b.text || "";
      if (b.type === "thinking") return b.thinking || b.text || "";
      if (b.text != null) return String(b.text);
      return "";
    }).filter(Boolean).join("\n");
  }
  if (typeof c === "object" && c.text != null) return String(c.text);
  try { return JSON.stringify(c); } catch (e) { return String(c); }
}

export function summarizeArgs(args) {
  if (args == null) return "";
  if (typeof args === "string") return args;
  const keys = ["description", "filePath", "file_path", "path", "command", "cmd", "query", "pattern", "url", "prompt", "content", "todos", "questions"];
  for (const k of keys) {
    if (args[k] != null) {
      let v = args[k];
      if (k === "todos" && Array.isArray(v)) return v.length + " todo item" + (v.length === 1 ? "" : "s");
      if (typeof v !== "string") { try { v = JSON.stringify(v); } catch (e) { v = String(v); } }
      return v;
    }
  }
  try { return JSON.stringify(args); } catch (e) { return String(args); }
}

export function buildTrace(messages) {
  const events = [];
  for (const m of (messages || [])) events.push.apply(events, messageEvents(m));

  const results = {};
  for (const e of events) if (e.kind === "toolresult" && e.id) results[e.id] = e.text;

  let system = null;
  const turns = [];
  let cur = null;
  const orphanResults = [];
  for (const e of events) if (e.kind === "toolresult" && !e.id) orphanResults.push(e.text);
  let orphanIdx = 0;

  function ensureTurn() {
    if (!cur) { cur = { userText: "(no user message — trace starts mid-run)", implicit: true, steps: [] }; turns.push(cur); }
    return cur;
  }

  for (const e of events) {
    if (e.kind === "system") { system = (system ? system + "\n\n" : "") + e.text; continue; }
    if (e.kind === "user") { cur = { userText: e.text, steps: [] }; turns.push(cur); continue; }
    if (e.kind === "toolresult") continue;
    if (e.kind === "thinking") { ensureTurn().steps.push({ type: "thinking", text: e.text }); continue; }
    if (e.kind === "text") { ensureTurn().steps.push({ type: "text", text: e.text }); continue; }
    if (e.kind === "toolbatch") {
      const parallel = e.calls.length > 1;
      for (const c of e.calls) {
        let res = (c.id && results[c.id] !== undefined) ? results[c.id] : undefined;
        if (res === undefined && !c.id && orphanIdx < orphanResults.length) res = orphanResults[orphanIdx++];
        const cat = classifyTool(c.name, c.args);
        ensureTurn().steps.push({
          type: "tool", name: c.name || "(unnamed tool)", cat: cat,
          args: c.args, result: res, dead: isDeadEnd(res),
          parallel: parallel, summary: summarizeArgs(c.args)
        });
      }
    }
  }
  return { system, turns };
}

export function computeMetrics(trace, meta) {
  let toolCalls = 0, deadEnds = 0, writes = 0, subagents = 0, thinking = 0;
  for (const t of trace.turns) for (const s of t.steps) {
    if (s.type === "tool") {
      toolCalls++;
      if (s.dead) deadEnds++;
      if (s.cat === "mutating") writes++;
      if (s.cat === "subagent") subagents++;
    } else if (s.type === "thinking") thinking++;
  }
  const m = {
    turns: trace.turns.filter(t => !t.implicit).length || trace.turns.length,
    toolCalls, deadEnds, writes, subagents, thinking
  };
  if (meta && meta.usage) {
    const u = meta.usage;
    m.promptTokens = u.prompt_tokens != null ? u.prompt_tokens : u.input_tokens;
    m.completionTokens = u.completion_tokens != null ? u.completion_tokens : u.output_tokens;
    const cached = u.prompt_tokens_details && u.prompt_tokens_details.cached_tokens;
    if (cached != null && m.promptTokens) m.cacheRate = cached / m.promptTokens;
    if (u.cost != null) m.cost = u.cost;
  }
  if (meta && meta.replay && meta.replay.score != null) m.replayScore = meta.replay.score;
  return m;
}
