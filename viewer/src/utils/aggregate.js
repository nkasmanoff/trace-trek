import { buildTrace, computeMetrics } from './trace.js';
import { contentToText, extractSession } from './parse.js';

function blankBucket() {
  return { calls: 0, prompt: 0, completion: 0, total: 0, cached: 0, reasoning: 0, cost: 0, toolCalls: 0, deadEnds: 0, withUsage: 0 };
}

function addInto(b, u) {
  b.calls++;
  b.prompt += u.prompt; b.completion += u.completion; b.total += u.total;
  b.cached += u.cached; b.reasoning += u.reasoning; b.cost += u.cost;
  if (u.hasUsage) b.withUsage++;
}

export function recordUsage(rec) {
  const u = (rec && rec.response && rec.response.usage) || null;
  const prompt = u ? (u.prompt_tokens != null ? u.prompt_tokens : u.input_tokens) : null;
  const completion = u ? (u.completion_tokens != null ? u.completion_tokens : u.output_tokens) : null;
  const d = u && u.prompt_tokens_details;
  const cd = u && u.completion_tokens_details;
  return {
    prompt: prompt || 0,
    completion: completion || 0,
    total: u && u.total_tokens != null ? u.total_tokens : ((prompt || 0) + (completion || 0)),
    cached: d && d.cached_tokens != null ? d.cached_tokens : 0,
    reasoning: cd && cd.reasoning_tokens != null ? cd.reasoning_tokens : 0,
    cost: u && u.cost != null ? u.cost : 0,
    hasUsage: !!u
  };
}

export function parseTs(ts) {
  if (ts == null) return null;
  const s = String(ts);
  const m = s.match(/^(\d{4})(\d{2})(\d{2})[-_ T]?(\d{2})(\d{2})(\d{2})/);
  if (m) return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  const t = Date.parse(s);
  return isNaN(t) ? null : t;
}

function userMessages(messages) {
  const out = [];
  for (const m of (messages || [])) if (m && m.role === "user") out.push(contentToText(m.content));
  return out;
}

function firstUserText(messages) {
  const us = userMessages(messages);
  return us.length ? us[0].trim() : "";
}

function normKey(s) {
  return String(s || "").replace(/\s+/g, " ").trim().slice(0, 2000).toLowerCase();
}

function djb2(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

function sessionKeyFor(messages) {
  const f = firstUserText(messages);
  return f ? djb2(normKey(f)) : null;
}

function msgSig(messages) {
  return (messages || []).map(m =>
    djb2((m && m.role || "") + "|" + normKey(contentToText(m && m.content))));
}

function commonPrefixLen(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

const SESSION_GAP_MS = 30 * 60 * 1000;

function blankSession() {
  const b = blankBucket();
  b.steps = 0; b.tsMin = null; b.tsMax = null; b.firstUser = "";
  b.title = null; b.recIdxs = []; b.bestMsgIdx = -1; b.bestMsgCount = -1;
  b.sig = []; b.lastTs = null; b.models = {};
  return b;
}

function attachToSession(buckets, reqMsgs, ts) {
  const sig = msgSig(reqMsgs);
  if (!sig.length) return null;
  let best = null, bestP = -1;
  for (const s of buckets) {
    const p = commonPrefixLen(sig, s.sig);
    if (p !== Math.min(sig.length, s.sig.length)) continue;
    if (p > bestP ||
      (p === bestP && best && s.lastTs != null &&
        (best.lastTs == null || s.lastTs > best.lastTs))) {
      best = s; bestP = p;
    }
  }
  if (best && sig.length <= best.sig.length &&
    ts != null && best.lastTs != null &&
    Math.abs(ts - best.lastTs) > SESSION_GAP_MS) {
    best = null;
  }
  if (!best) {
    best = blankSession();
    best.sig = sig;
    buckets.push(best);
  } else if (sig.length > best.sig.length) {
    best.sig = sig;
  }
  if (ts != null) best.lastTs = ts;
  return best;
}

const TITLE_RE = /generate a (?:short )?title/i;

function titleGenInfo(rec) {
  const msgs = rec && rec.request && rec.request.messages;
  if (!Array.isArray(msgs)) return null;
  const us = userMessages(msgs);
  if (!us.length || !TITLE_RE.test(us[0])) return null;
  let convo = us.slice(1).join("\n").trim();
  if (!convo) {
    convo = us[0].replace(TITLE_RE, "").replace(/^[^:]*:\s*/, "").trim();
  }
  const title = (rec.response && rec.response.message &&
    contentToText(rec.response.message.content) || "").trim();
  return { convo, title };
}

export function shortTitle(s, n) {
  s = String(s || "").replace(/\s+/g, " ").trim();
  if (!s) return "(untitled)";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export function shortModel(s) {
  s = String(s || "").trim();
  const slash = s.lastIndexOf("/");
  return slash >= 0 ? s.slice(slash + 1) : s;
}

export function aggregateUsage(records) {
  const totals = blankBucket();
  const byModel = {}, byUpstream = {}, byTool = {};
  const sessionBuckets = [];
  const attrItems = [];
  const titleByConvoKey = {};
  const calls = [];
  let tsMin = null, tsMax = null;

  (records || []).forEach(rec => {
    if (!rec || typeof rec !== "object") return;
    const ti = titleGenInfo(rec);
    if (ti && ti.title && ti.convo) {
      const k = sessionKeyFor([{ role: "user", content: ti.convo }]);
      if (k) titleByConvoKey[k] = ti.title;
    }
  });

  (records || []).forEach((rec, i) => {
    if (!rec || typeof rec !== "object") return;
    const isProxy = rec.request || rec.response;
    if (!isProxy) return;
    const reqMsgs = (rec.request && rec.request.messages) || [];
    const isTitleCall = !!titleGenInfo(rec);
    const u = recordUsage(rec);
    const model = (rec.request && rec.request.model) ||
      (rec.response && rec.response.message && rec.response.message.model) || "unknown";
    const upstream = rec.upstream || "local";
    const ts = parseTs(rec.timestamp);
    if (ts != null) {
      tsMin = (tsMin == null || ts < tsMin) ? ts : tsMin;
      tsMax = (tsMax == null || ts > tsMax) ? ts : tsMax;
    }

    let toolCalls = 0, deadEnds = 0;
    const recTools = [];
    try {
      const msgs = reqMsgs.slice();
      if (rec.response && rec.response.message) msgs.push(rec.response.message);
      const tr = buildTrace(msgs);
      for (const t of tr.turns) for (const s of t.steps)
        if (s.type === "tool") {
          toolCalls++; if (s.dead) deadEnds++;
          recTools.push({ name: s.name || "(unnamed)", cat: s.cat, dead: s.dead });
        }
    } catch (e) { /* ignore */ }

    addInto(totals, u); totals.toolCalls += toolCalls; totals.deadEnds += deadEnds;
    const bm = byModel[model] || (byModel[model] = blankBucket());
    addInto(bm, u); bm.toolCalls += toolCalls; bm.deadEnds += deadEnds;
    const bu = byUpstream[upstream] || (byUpstream[upstream] = blankBucket());
    addInto(bu, u); bu.toolCalls += toolCalls; bu.deadEnds += deadEnds;

    for (const t of recTools) {
      const bt = byTool[t.name] || (byTool[t.name] = { calls: 0, dead: 0, cat: t.cat });
      bt.calls++; if (t.dead) bt.dead++;
    }

    if (!isTitleCall) attrItems.push({ i, reqMsgs, u, toolCalls, deadEnds, ts, model });

    calls.push({
      idx: i, model, upstream, ts,
      cost: u.cost, prompt: u.prompt, completion: u.completion,
      total: u.total, cached: u.cached, toolCalls,
      finish: rec.response && rec.response.finish_reason
    });
  });

  attrItems
    .sort((a, b) => (a.ts == null ? Infinity : a.ts) - (b.ts == null ? Infinity : b.ts) || a.i - b.i)
    .forEach(it => {
      const ss = attachToSession(sessionBuckets, it.reqMsgs, it.ts);
      if (!ss) return;
      addInto(ss, it.u); ss.toolCalls += it.toolCalls; ss.deadEnds += it.deadEnds;
      ss.recIdxs.push(it.i);
      if (it.model) ss.models[it.model] = (ss.models[it.model] || 0) + 1;
      if (!ss.firstUser) ss.firstUser = firstUserText(it.reqMsgs);
      if (it.ts != null) {
        ss.tsMin = (ss.tsMin == null || it.ts < ss.tsMin) ? it.ts : ss.tsMin;
        ss.tsMax = (ss.tsMax == null || it.ts > ss.tsMax) ? it.ts : ss.tsMax;
      }
      if (it.reqMsgs.length > ss.bestMsgCount) { ss.bestMsgCount = it.reqMsgs.length; ss.bestMsgIdx = it.i; }
    });

  const sessionList = sessionBuckets.map((s, idx) => {
    s.key = "s" + idx;
    s.cacheRate = s.prompt ? s.cached / s.prompt : null;
    const modelKeys = Object.keys(s.models);
    s.model = modelKeys.sort((a, b) => s.models[b] - s.models[a])[0] || "";
    s.modelCount = modelKeys.length;
    const fk = sessionKeyFor([{ role: "user", content: s.firstUser }]);
    s.firstUserKey = fk;
    const harvested = titleByConvoKey[fk];
    s.title = harvested || shortTitle(s.firstUser, 64);
    s.titleSource = harvested ? "opencode" : "first-message";
    return s;
  });

  totals.cacheRate = totals.prompt ? totals.cached / totals.prompt : null;
  for (const k in byModel) byModel[k].cacheRate = byModel[k].prompt ? byModel[k].cached / byModel[k].prompt : null;
  for (const k in byUpstream) byUpstream[k].cacheRate = byUpstream[k].prompt ? byUpstream[k].cached / byUpstream[k].prompt : null;

  const N = 48;
  const bins = [];
  const haveTime = tsMin != null && tsMax != null && tsMax > tsMin;
  const span = haveTime ? (tsMax - tsMin) : 0;

  function binIndex(c, order) {
    if (haveTime && c.ts != null) return Math.min(N - 1, Math.floor((c.ts - tsMin) / span * N));
    return Math.min(N - 1, Math.floor(order / Math.max(1, calls.length) * N));
  }
  for (let i = 0; i < N; i++) bins.push({ calls: 0, prompt: 0, completion: 0, cost: 0, t0: null });
  calls.forEach((c, order) => {
    const bi = binIndex(c, order);
    const b = bins[bi];
    b.calls++; b.prompt += c.prompt; b.completion += c.completion; b.cost += c.cost;
    if (b.t0 == null && c.ts != null) b.t0 = c.ts;
  });

  const topCalls = calls.slice().sort((a, b) => b.cost - a.cost).slice(0, 12);

  return { totals, byModel, byUpstream, byTool, sessions: sessionList, calls, topCalls, timeline: { bins, haveTime, tsMin, tsMax } };
}
