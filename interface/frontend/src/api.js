// API client for the SynthScope backend. In dev, Vite proxies /api to the FastAPI server.
const BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchModels() {
  const r = await fetch(`${BASE}/api/models`);
  if (!r.ok) throw new Error(`models: HTTP ${r.status}`);
  return r.json();
}

export async function predict(file, model) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", model);
  const r = await fetch(`${BASE}/api/predict`, { method: "POST", body: fd });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}

// --- dataset explorer --------------------------------------------------------
export async function fetchDatasets() {
  const r = await fetch(`${BASE}/api/datasets`);
  if (!r.ok) throw new Error(`datasets: HTTP ${r.status}`);
  return r.json();
}

export async function sampleDataset(key, split, n, seed) {
  const u = `${BASE}/api/datasets/${key}/sample?split=${encodeURIComponent(split)}&n=${n}&seed=${seed}`;
  const r = await fetch(u);
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}

export function datasetMediaUrl(key, rel, thumb) {
  return `${BASE}/api/datasets/${key}/media?rel=${encodeURIComponent(rel)}${thumb ? "&thumb=1" : ""}`;
}

export async function predictDataset(dataset, rel, model) {
  const fd = new FormData();
  fd.append("dataset", dataset);
  fd.append("rel", rel);
  fd.append("model", model);
  const r = await fetch(`${BASE}/api/predict_dataset`, { method: "POST", body: fd });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}

export async function scoreBatch(dataset, model, rels) {
  const r = await fetch(`${BASE}/api/score_batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, model, rels }),
  });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}

// --- threshold calibration helpers (shared by Detect + Explorer) ---
export const verdictAt = (p, thr) => (p >= thr ? "fake" : "real");

export function autoCalibrate(scored) {
  let best = 0.5, bestAcc = -1;
  for (let t = 0; t <= 1.0001; t += 0.01) {
    const reals = scored.filter((s) => s.gt === "real"), fakes = scored.filter((s) => s.gt === "fake");
    const tnr = reals.length ? reals.filter((s) => s.p < t).length / reals.length : 0;
    const tpr = fakes.length ? fakes.filter((s) => s.p >= t).length / fakes.length : 0;
    const acc = (tpr + tnr) / 2;
    if (acc > bestAcc) { bestAcc = acc; best = t; }
  }
  return Math.round(best * 100) / 100;
}

export function targetFprThreshold(scored, fpr) {
  const reals = scored.filter((s) => s.gt === "real").map((s) => s.p).sort((a, b) => a - b);
  if (!reals.length) return null;
  const idx = Math.min(reals.length - 1, Math.max(0, Math.ceil((1 - fpr) * reals.length) - 1));
  return Math.round(reals[idx] * 100) / 100;
}

export function batchStats(scored, thr) {
  if (!scored.length) return null;
  const reals = scored.filter((s) => s.gt === "real"), fakes = scored.filter((s) => s.gt === "fake");
  return {
    n: scored.length,
    acc: scored.filter((s) => verdictAt(s.p, thr) === s.gt).length / scored.length,
    racc: reals.length ? reals.filter((s) => s.p < thr).length / reals.length : null,
    facc: fakes.length ? fakes.filter((s) => s.p >= thr).length / fakes.length : null,
  };
}

// --- SOTA generator vs detector comparison ----------------------------------
export async function sotaGenerators() {
  const r = await fetch(`${BASE}/api/sota/generators`);
  if (!r.ok) throw new Error(`generators: HTTP ${r.status}`);
  return r.json();
}
export async function sotaRealImage() {
  const r = await fetch(`${BASE}/api/sota/real_image`);
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}
export async function fetchLeaderboard() {
  const r = await fetch(`${BASE}/api/leaderboard`);
  if (!r.ok) throw new Error(`leaderboard: HTTP ${r.status}`);
  return r.json();
}
export async function fetchGeneratorBenchmarks() {
  const r = await fetch(`${BASE}/api/generator_benchmarks`);
  if (!r.ok) throw new Error(`generator benchmarks: HTTP ${r.status}`);
  return r.json();
}
export async function fetchTrending() {
  const r = await fetch(`${BASE}/api/trending`);
  if (!r.ok) throw new Error(`trending: HTTP ${r.status}`);
  return r.json();
}
export async function sotaCompare(file, generator, detectors, prompt) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("generator", generator);
  fd.append("detectors", detectors.join(","));
  fd.append("prompt", prompt);
  const r = await fetch(`${BASE}/api/sota/compare`, { method: "POST", body: fd });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}
