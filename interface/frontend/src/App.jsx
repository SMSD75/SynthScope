import React, { useEffect, useRef, useState } from "react";
import {
  fetchModels, predict, fetchDatasets, sampleDataset, datasetMediaUrl,
  predictDataset, scoreBatch, verdictAt, autoCalibrate, targetFprThreshold, batchStats,
  sotaGenerators, sotaRealImage, sotaCompare,
} from "./api.js";

function b64ToFile(b64, name) {
  const bin = atob(b64), arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new File([arr], name, { type: "image/png" });
}

// SOTA generator vs detector comparison: pick/upload a real image -> a current SOTA generator makes a
// "conceptually similar" image -> the selected AGID detectors score it (best/promoted checkpoints).
function SotaView({ models }) {
  const [gens, setGens] = useState([]);
  const [gen, setGen] = useState("");
  const [prompt, setPrompt] = useState("make another image conceptually similar to this");
  const [realB64, setRealB64] = useState("");
  const [realSrc, setRealSrc] = useState(null);
  const [picked, setPicked] = useState({});
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const imgDetectors = (models || []).filter((m) => m.modality === "image" && m.available);

  useEffect(() => {
    sotaGenerators().then((d) => {
      setGens(d.generators || []);
      const a = (d.generators || []).find((g) => g.available);
      if (a) setGen(a.key);
      if (d.default_prompt) setPrompt(d.default_prompt);
    }).catch((e) => setErr(e.message));
  }, []);

  async function rollImage() {
    setErr(""); setRes(null);
    try { const d = await sotaRealImage(); setRealB64(d.image_b64); setRealSrc(d.source); }
    catch (e) { setErr(e.message); }
  }
  function onUpload(f) {
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => { setRealB64(rd.result.split(",")[1]); setRealSrc(["upload", f.name]); setRes(null); };
    rd.readAsDataURL(f);
  }
  async function run() {
    setErr(""); setBusy(true); setRes(null);
    try {
      const dets = Object.keys(picked).filter((k) => picked[k]);
      if (!realB64) throw new Error("Pick or upload a real image first.");
      if (!gen) throw new Error("Select a generator.");
      if (!dets.length) throw new Error("Select at least one detector.");
      const d = await sotaCompare(b64ToFile(realB64, "real.png"), gen, dets, prompt);
      setRes(d);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="sota">
      <p className="muted">Send a real image from our datasets to a current SOTA generator, then see whether
        the AGID detectors catch the freshly-generated image. Generators need an API key
        (greyed-out = key not set on the server).</p>
      <div className="sota-controls">
        <div>
          <button onClick={rollImage}>🎲 Random real image</button>
          <label className="upbtn">Upload<input type="file" accept="image/*" style={{ display: "none" }}
            onChange={(e) => onUpload(e.target.files[0])} /></label>
        </div>
        <label>Generator:
          <select value={gen} onChange={(e) => setGen(e.target.value)}>
            {gens.map((g) => <option key={g.key} value={g.key} disabled={!g.available}>
              {g.name}{g.available ? "" : " (no API key)"}</option>)}
          </select>
        </label>
        <label className="promptline">Prompt:
          <input value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </label>
      </div>
      <div className="detpick">Detectors:
        {imgDetectors.map((m) => (
          <label key={m.key}><input type="checkbox" checked={!!picked[m.key]}
            onChange={(e) => setPicked({ ...picked, [m.key]: e.target.checked })} /> {m.name}</label>
        ))}
      </div>
      <button className="primary" disabled={busy} onClick={run}>{busy ? "Generating + scoring…" : "Generate & compare"}</button>
      {err && <div className="err">{err}</div>}
      <div className="sota-images">
        {realB64 && <figure><img src={"data:image/png;base64," + realB64} alt="real" />
          <figcaption>Real {realSrc && "· " + realSrc.join("/")}</figcaption></figure>}
        {res && res.generated_b64 && <figure><img src={"data:image/png;base64," + res.generated_b64} alt="generated" />
          <figcaption>Generated · {res.generator}</figcaption></figure>}
      </div>
      {res && (
        <table className="repro repro-full">
          <thead><tr><th>Detector</th><th>P(fake) real</th><th>P(fake) generated</th><th>Caught?</th></tr></thead>
          <tbody>
            {res.results.map((r, i) => (
              <tr key={i}>
                <td>{r.name || r.detector}</td>
                <td>{r.error ? "—" : r.real_fake_prob}</td>
                <td>{r.error ? "—" : r.gen_fake_prob}</td>
                <td className={r.caught ? "status-reproduced" : "status-below"}>
                  {r.error ? r.error : (r.caught ? "✓ fake" : "✗ missed")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const isVideoName = (name = "", type = "") =>
  type.startsWith("video") || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(name);
const optLabel = (m) => `${m.name} — ${m.paper}${m.available ? "" : "  (no checkpoint)"}`;

function ModelSelect({ models, model, setModel, onChange }) {
  const img = models.filter((m) => m.modality === "image");
  const vid = models.filter((m) => m.modality === "video");
  return (
    <select value={model} onChange={(e) => { setModel(e.target.value); onChange && onChange(); }}>
      <optgroup label="Image detectors">
        {img.map((m) => <option key={m.key} value={m.key} disabled={!m.available}>{optLabel(m)}</option>)}
      </optgroup>
      <optgroup label="Video detectors">
        {vid.map((m) => <option key={m.key} value={m.key} disabled={!m.available}>{optLabel(m)}</option>)}
      </optgroup>
    </select>
  );
}

function ReproTable({ m }) {
  const rep = m && m.repro;
  const full = m && m.repro_full;
  if (!rep && !(full && full.length)) return null;
  return (
    <div>
      {rep && (
        <table className="repro"><tbody>
          <tr><td>Headline</td><td>{rep.benchmark}</td></tr>
          <tr><td>Paper</td><td>{rep.paper}</td></tr>
          <tr><td>Reproduced</td><td className={"match-" + (rep.match || "")}>{rep.ours}</td></tr>
          {rep.note && <tr><td colSpan="2" className="rnote">{rep.note}</td></tr>}
        </tbody></table>
      )}
      {full && full.length > 0 && (
        <table className="repro repro-full">
          <thead><tr><th>Benchmark</th><th>Paper</th><th>Ours</th><th>Status</th></tr></thead>
          <tbody>
            {full.map((r, i) => (
              <tr key={i}>
                <td title={r.note}>{r.benchmark}</td>
                <td>{r.paper}</td>
                <td className={"match-" + (r.status || "")}>{r.ours}</td>
                <td className={"status-" + (r.status || "")}>{r.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ThresholdSlider({ threshold, setThreshold, children }) {
  return (
    <div className="thr">
      <div className="lab"><span>Decision threshold</span><span>P(fake) ≥ <b>{threshold.toFixed(2)}</b> → fake</span></div>
      <input type="range" min="0" max="1" step="0.01" value={threshold}
        onChange={(e) => setThreshold(parseFloat(e.target.value))} />
      {children}
    </div>
  );
}

export default function App() {
  const [models, setModels] = useState([]);
  const [device, setDevice] = useState("");
  const [model, setModel] = useState("");
  const [tab, setTab] = useState("detect");
  const [threshold, setThreshold] = useState(0.5);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchModels().then((d) => {
      setModels(d.models || []); setDevice(d.device || "");
      const first = (d.models || []).find((m) => m.available);
      if (first) setModel(first.key);
    }).catch((e) => setError("Could not reach backend: " + e.message));
  }, []);

  const shared = { models, model, setModel, threshold, setThreshold };
  return (
    <div className="wrap">
      <header>
        <h1>AGID — AI-Generated Media Detector</h1>
        <p>Detect AI-generated media, or explore the datasets each detector was trained / tested on.
          {device && <span className="pill">device: {device}</span>}</p>
      </header>
      <div className="tabs">
        <button className={"tab" + (tab === "detect" ? " active" : "")} onClick={() => setTab("detect")}>Detect</button>
        <button className={"tab" + (tab === "explore" ? " active" : "")} onClick={() => setTab("explore")}>Data Explorer</button>
        <button className={"tab" + (tab === "sota" ? " active" : "")} onClick={() => setTab("sota")}>SOTA vs Detectors</button>
      </div>
      {error && <div className="err">{error}</div>}
      {tab === "detect" && <DetectView {...shared} />}
      {tab === "explore" && <ExploreView {...shared} />}
      {tab === "sota" && <SotaView models={models} />}
    </div>
  );
}

function DetectView({ models, model, setModel, threshold, setThreshold }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isVideo, setIsVideo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();
  const selected = models.find((m) => m.key === model);

  function onPick(f) {
    if (!f) return;
    setFile(f); setResult(null); setError("");
    setIsVideo(isVideoName(f.name, f.type));
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }
  async function run() {
    if (!file || !model) return;
    setBusy(true); setResult(null); setError("");
    try { setResult(await predict(file, model)); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="grid">
      <section className="card">
        <h2>Input</h2>
        <label>Detector model</label>
        <ModelSelect models={models} model={model} setModel={setModel} onChange={() => setResult(null)} />
        {selected && (
          <div className="meta">{selected.notes}
            {selected.modality === "image" ? " · On a video, runs frame-by-frame and averages." : " · Native video model."}</div>
        )}
        {selected && <ReproTable m={selected} />}
        <label>Media file</label>
        <div className={"drop" + (drag ? " drag" : "")}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); onPick(e.dataTransfer.files[0]); }}>
          {file ? <span><b>{file.name}</b> <span className="meta">({(file.size / 1048576).toFixed(2)} MB)</span></span>
                : "Click or drop an image / video here"}
        </div>
        <input ref={inputRef} type="file" accept="image/*,video/*" style={{ display: "none" }}
          onChange={(e) => onPick(e.target.files[0])} />
        {previewUrl && <div className="preview">{isVideo ? <video src={previewUrl} controls /> : <img src={previewUrl} alt="" />}</div>}
        <button className="go" disabled={!file || !model || busy} onClick={run}>{busy ? "Analyzing…" : "Analyze"}</button>
        {error && <div className="err">{error}</div>}
      </section>

      <section className="card">
        <h2>Result</h2>
        {!result && !busy && <div className="empty">No result yet.</div>}
        {busy && <div className="empty">Running {selected ? selected.name : ""}… first run loads the model and may take a while.</div>}
        {result && <Result r={result} threshold={threshold} />}
        {result && <ThresholdSlider threshold={threshold} setThreshold={setThreshold} />}
      </section>
    </div>
  );
}

function ExploreView({ models, model, setModel, threshold, setThreshold }) {
  const [datasets, setDatasets] = useState([]);
  const [trainMap, setTrainMap] = useState({});
  const [dataset, setDataset] = useState("");
  const [split, setSplit] = useState("");
  const [seed, setSeed] = useState(1);
  const [items, setItems] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sel, setSel] = useState(null);
  const [preds, setPreds] = useState({});
  const [scoring, setScoring] = useState(false);
  const [fpr, setFpr] = useState(5);
  const batchN = 24;

  const curDs = datasets.find((d) => d.key === dataset);
  const trainDsKey = (trainMap[model] || {}).dataset;

  useEffect(() => {
    fetchDatasets().then((d) => {
      setDatasets(d.datasets || []); setTrainMap(d.model_train || {});
      const first = (d.datasets || [])[0];
      if (first) { setDataset(first.key); setSplit(first.splits[first.splits.length - 1] || "."); }
    }).catch((e) => setError("Could not load datasets: " + e.message));
  }, []);

  useEffect(() => {
    if (!curDs) return;
    const s = curDs.splits.includes(split) ? split : curDs.splits[curDs.splits.length - 1] || ".";
    if (s !== split) setSplit(s);
    loadBatch(dataset, s, seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  async function loadBatch(dsKey, sp, sd) {
    if (!dsKey) return;
    setBusy(true); setError(""); setItems([]); setSel(null); setPreds({});
    try { setItems((await sampleDataset(dsKey, sp, batchN, sd)).items || []); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  function newBatch() { const sd = seed + 1; setSeed(sd); loadBatch(dataset, split, sd); }

  async function pickItem(it) {
    setSel(it.rel);
    if (!model || preds[it.rel]) return;
    try { setPreds((p) => ({ ...p, [it.rel]: await predictDataset(dataset, it.rel, model) })); }
    catch (e) { setPreds((p) => ({ ...p, [it.rel]: { error: e.message } })); }
  }

  async function scoreAll() {
    if (!model || !items.length) return [];
    setScoring(true); setError("");
    try {
      const d = await scoreBatch(dataset, model, items.map((i) => i.rel));
      const m = {}; for (const it of d.items) m[it.rel] = it;
      setPreds((p) => ({ ...p, ...m }));
      return d.items.filter((x) => x.ground_truth && typeof x.fake_probability === "number")
                    .map((x) => ({ p: x.fake_probability, gt: x.ground_truth }));
    } catch (e) { setError(e.message); return []; }
    finally { setScoring(false); }
  }

  const scored = items.map((i) => preds[i.rel])
    .filter((x) => x && typeof x.fake_probability === "number" && x.ground_truth)
    .map((x) => ({ p: x.fake_probability, gt: x.ground_truth }));

  async function calibrate() { const s = scored.length ? scored : await scoreAll(); if (s.length) setThreshold(autoCalibrate(s)); }
  async function applyFpr() {
    const s = scored.length ? scored : await scoreAll();
    const t = targetFprThreshold(s, fpr / 100);
    if (t != null) setThreshold(t); else setError("Need scored real videos for target-FPR.");
  }

  const selItem = items.find((i) => i.rel === sel);
  const selRes = sel ? preds[sel] : null;
  const stats = batchStats(scored, threshold);
  const mediaUrl = (it, thumb) => datasetMediaUrl(dataset, it.rel, thumb);

  return (
    <div className="grid" style={{ gridTemplateColumns: "2fr 1fr" }}>
      <section className="card">
        <h2>Datasets</h2>
        <label>Detector model</label>
        <ModelSelect models={models} model={model} setModel={setModel} />
        {trainDsKey && (
          <div className="meta">Trained on <b>{(datasets.find((d) => d.key === trainDsKey) || {}).name || trainDsKey}</b>
            {(trainMap[model] || {}).official && <span className="badge">official wts</span>}</div>
        )}
        <ReproTable m={models.find((m) => m.key === model)} />
        <div className="toolbar" style={{ marginTop: 14 }}>
          <div><label>Dataset</label>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {datasets.map((d) => <option key={d.key} value={d.key}>{d.name}{d.key === trainDsKey ? " ★ trains " + model : ""} · {d.modality}</option>)}
            </select></div>
          <div style={{ maxWidth: 130 }}><label>Split</label>
            <select value={split} onChange={(e) => { setSplit(e.target.value); loadBatch(dataset, e.target.value, seed); }}>
              {(curDs ? curDs.splits : []).map((s) => <option key={s} value={s}>{s === "." ? "(all)" : s}</option>)}
            </select></div>
          <div style={{ maxWidth: 110, flex: "0 0 auto" }}><button className="sm" disabled={busy} onClick={newBatch}>↻ New batch</button></div>
        </div>

        <ThresholdSlider threshold={threshold} setThreshold={setThreshold}>
          <div className="calib">
            <button disabled={scoring || !model} onClick={scoreAll}>{scoring ? "Scoring…" : "Score batch"}</button>
            <button disabled={scoring || !model} onClick={calibrate} title="Threshold maximizing balanced accuracy on this labeled batch">Auto-calibrate</button>
            <button disabled={scoring || !model} onClick={applyFpr}>Target FPR</button>
            <input type="number" min="0" max="50" step="1" value={fpr} onChange={(e) => setFpr(parseFloat(e.target.value) || 0)} />%
          </div>
          {stats && (
            <div className="stat">batch ({stats.n} scored) @{threshold.toFixed(2)}: <b>acc {(stats.acc * 100).toFixed(0)}%</b>
              {stats.racc != null ? ` · real ${(stats.racc * 100).toFixed(0)}%` : ""}
              {stats.facc != null ? ` · fake ${(stats.facc * 100).toFixed(0)}%` : ""}</div>
          )}
        </ThresholdSlider>

        {error && <div className="err">{error}</div>}
        {busy && <div className="empty">Sampling…</div>}
        <div className="thumbs">
          {items.map((it) => {
            const pr = preds[it.rel];
            const has = pr && typeof pr.fake_probability === "number";
            const v = has ? verdictAt(pr.fake_probability, threshold) : null;
            const ok = has && v === it.label;
            return (
              <div key={it.rel} className={"thumb" + (it.rel === sel ? " sel" : "")} onClick={() => pickItem(it)} title={it.name}>
                <img loading="lazy" src={mediaUrl(it, true)} alt="" />
                <span className={"tag " + it.label}>{it.label}</span>
                {it.modality === "video" && <span className="play">▶</span>}
                {has && <span className={"pred " + (ok ? "ok" : "no")}>{(pr.fake_probability * 100).toFixed(0)}% {ok ? "✓" : "✗"}</span>}
              </div>
            );
          })}
        </div>
      </section>

      <section className="card">
        <h2>Model output</h2>
        {!selItem && <div className="empty">Click a sample to run {(models.find((m) => m.key === model) || {}).name || "the model"} on it. Use "Score batch" + Auto-calibrate / Target FPR to set the threshold.</div>}
        {selItem && (
          <div>
            <div className="preview">
              {selItem.modality === "video" ? <video src={mediaUrl(selItem, false)} controls /> : <img src={mediaUrl(selItem, false)} alt="" />}
            </div>
            <div className="meta">{selItem.name} · ground truth: <b style={{ color: selItem.label === "fake" ? "var(--fake)" : "var(--real)" }}>{selItem.label}</b></div>
            {selRes && selRes.error && <div className="err">{selRes.error}</div>}
            {selRes && !selRes.error && typeof selRes.fake_probability === "number" && <Result r={selRes} threshold={threshold} gt={selItem.label} />}
          </div>
        )}
      </section>
    </div>
  );
}

function Result({ r, threshold, gt }) {
  const fakePct = r.fake_probability * 100, realPct = (1 - r.fake_probability) * 100;
  const verdict = verdictAt(r.fake_probability, threshold);
  const correct = gt ? verdict === gt : null;
  return (
    <div>
      <div className={"verdict " + verdict}>{verdict === "fake" ? "AI-GENERATED" : "REAL"}</div>
      <div style={{ textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
        P(fake) {fakePct.toFixed(1)}% vs threshold {threshold.toFixed(2)}
        {correct != null && <> · <b style={{ color: correct ? "var(--real)" : "var(--fake)" }}>{correct ? "correct" : "wrong"}</b></>}
      </div>
      <div className="bar">
        <span className="fake" style={{ width: fakePct + "%" }}>{fakePct >= 12 ? `fake ${fakePct.toFixed(0)}%` : ""}</span>
        <span className="real" style={{ width: realPct + "%" }}>{realPct >= 12 ? `real ${realPct.toFixed(0)}%` : ""}</span>
        <span className="thrline" style={{ left: threshold * 100 + "%" }}></span>
      </div>
      <div style={{ marginTop: 14 }}>
        <Row k="P(fake)" v={`${fakePct.toFixed(2)}%`} />
        {r.model_name && <Row k="Model" v={r.model_name} />}
        {r.media_type && <Row k="Media" v={`${r.media_type}${r.num_frames ? " · " + r.num_frames + " frames" : ""}`} />}
        {r.elapsed_ms && <Row k="Inference" v={`${r.elapsed_ms} ms`} />}
      </div>
      {r.per_frame && (
        <div>
          <div className="meta">Per-frame P(fake) ({r.aggregation}):</div>
          <div className="spark">{r.per_frame.map((v, i) => <i key={i} title={`${(v * 100).toFixed(1)}%`} style={{ height: v * 100 + "%" }} />)}</div>
        </div>
      )}
    </div>
  );
}

const Row = ({ k, v }) => (<div className="row"><span>{k}</span><b>{v}</b></div>);
