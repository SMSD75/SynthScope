// API client for the AGID backend. In dev, Vite proxies /api to the FastAPI server.
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
