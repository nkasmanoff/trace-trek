export function evalRunLabel(rec) {
  return String(rec.run || rec.model || rec.run_label || "(unlabeled)");
}

function blankEvalRun() {
  return { label: "", total: 0, passed: 0, gradable: 0, errors: 0, scoreSum: 0, scoreN: 0, byType: {}, tasks: {} };
}

function blankEvalType() { return { total: 0, passed: 0, scoreSum: 0, scoreN: 0 }; }

export function aggregateEval(records) {
  const runs = {};
  const taskMeta = {};
  let n = 0;

  (records || []).forEach((rec, i) => {
    if (!isEvalRow(rec)) return;
    n++;
    const label = evalRunLabel(rec);
    const r = runs[label] || (runs[label] = blankEvalRun());
    r.label = label;
    const passed = !!rec.passed;
    const score = (typeof rec.score === "number") ? rec.score : null;
    const errored = !!rec.error;
    r.total++;
    if (passed) r.passed++;
    if (errored) r.errors++;
    if (score != null) { r.scoreSum += score; r.scoreN++; }
    if (rec.gradable || rec.type === "code") r.gradable++;

    const bt = r.byType[rec.type] || (r.byType[rec.type] = blankEvalType());
    bt.total++; if (passed) bt.passed++;
    if (score != null) { bt.scoreSum += score; bt.scoreN++; }

    r.tasks[rec.task_id] = { passed, score, error: errored, recIdx: rec.__idx, type: rec.type, repo: rec.repo };
    if (!taskMeta[rec.task_id])
      taskMeta[rec.task_id] = { type: rec.type, repo: rec.repo, prompt: rec.prompt };
  });

  const runList = Object.keys(runs).map(k => {
    const r = runs[k];
    r.passRate = r.total ? r.passed / r.total : null;
    r.avgScore = r.scoreN ? r.scoreSum / r.scoreN : null;
    for (const ty in r.byType) {
      const b = r.byType[ty];
      b.passRate = b.total ? b.passed / b.total : null;
      b.avgScore = b.scoreN ? b.scoreSum / b.scoreN : null;
    }
    return r;
  });
  runList.sort((a, b) => (b.passRate || 0) - (a.passRate || 0) || (a.label < b.label ? -1 : 1));

  let flip = null;
  if (runList.length === 2) {
    const cand = runList[0], base = runList[1];
    const ids = Object.keys(taskMeta).sort();
    const gains = [], regressions = [], ties = [], onlyOne = [];
    ids.forEach(id => {
      const a = base.tasks[id], b = cand.tasks[id];
      const meta = taskMeta[id];
      if (a && b) {
        if (b.passed && !a.passed) gains.push({ id, meta, base: a, cand: b });
        else if (!b.passed && a.passed) regressions.push({ id, meta, base: a, cand: b });
        else ties.push({ id, meta, base: a, cand: b });
      } else {
        onlyOne.push({ id, meta, base: a || null, cand: b || null });
      }
    });
    flip = { candidate: cand.label, baseline: base.label, gains, regressions, ties, onlyOne, common: gains.length + regressions.length + ties.length };
  }

  return { runs: runList, taskMeta, count: n, flip, taskIds: Object.keys(taskMeta).sort() };
}

function isEvalRow(rec) {
  return !!(rec && typeof rec === "object" && rec.task_id &&
    (rec.type === "code" || rec.type === "knowledge") &&
    ("passed" in rec || "score" in rec) &&
    !(rec.request || rec.response || rec.messages));
}
