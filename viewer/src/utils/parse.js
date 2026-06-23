export function contentToText(c) {
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

export function parseInput(text) {
  const t = String(text || "").trim();
  if (!t) throw new Error("Empty input.");
  try {
    const v = JSON.parse(t);
    return Array.isArray(v) ? v : [v];
  } catch (e) { /* try JSONL */ }
  const recs = [];
  const lines = t.split(/\r?\n/);
  for (const line of lines) {
    const s = line.trim();
    if (!s) continue;
    try { recs.push(JSON.parse(s)); }
    catch (e) { throw new Error("Input is not valid JSON, a JSON array, or JSONL."); }
  }
  if (!recs.length) throw new Error("No JSON records found.");
  return recs;
}

export function extractSession(rec) {
  if (rec && rec.request && Array.isArray(rec.request.messages)) {
    const messages = rec.request.messages.slice();
    if (rec.response && rec.response.message) messages.push(rec.response.message);
    return {
      messages,
      meta: {
        model: rec.request.model || (rec.response && rec.response.message && rec.response.message.model),
        timestamp: rec.timestamp, upstream: rec.upstream,
        usage: rec.response && rec.response.usage,
        finish: rec.response && rec.response.finish_reason,
        elapsed_ms: rec.elapsed_ms, replay: rec.replay
      }
    };
  }
  if (rec && Array.isArray(rec.messages))
    return { messages: rec.messages, meta: { model: rec.model, source: rec.source, usage: rec.usage } };
  if (Array.isArray(rec))
    return { messages: rec, meta: {} };
  if (rec && Array.isArray(rec.content) && rec.role !== "user")
    return { messages: [{ role: rec.role || "assistant", content: rec.content }], meta: { model: rec.model, usage: rec.usage } };
  return null;
}

export function isEvalRow(rec) {
  return !!(rec && typeof rec === "object" && rec.task_id &&
    (rec.type === "code" || rec.type === "knowledge") &&
    ("passed" in rec || "score" in rec) &&
    !(rec.request || rec.response || rec.messages));
}
