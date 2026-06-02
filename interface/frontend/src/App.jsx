import React, { useEffect, useRef, useState } from "react";
import { fetchModels, predict } from "./api.js";

const isVideoName = (name = "", type = "") =>
  type.startsWith("video") || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(name);

export default function App() {
  const [models, setModels] = useState([]);
  const [device, setDevice] = useState("");
  const [model, setModel] = useState("");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isVideo, setIsVideo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  useEffect(() => {
    fetchModels()
      .then((d) => {
        setModels(d.models || []);
        setDevice(d.device || "");
        const first = (d.models || []).find((m) => m.available);
        if (first) setModel(first.key);
      })
      .catch((e) => setError("Could not reach backend: " + e.message));
  }, []);

  function onPick(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
    setIsVideo(isVideoName(f.name, f.type));
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function run() {
    if (!file || !model) return;
    setBusy(true);
    setResult(null);
    setError("");
    try {
      setResult(await predict(file, model));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const selected = models.find((m) => m.key === model);
  const imageModels = models.filter((m) => m.modality === "image");
  const videoModels = models.filter((m) => m.modality === "video");
  const optLabel = (m) => `${m.name} — ${m.paper}${m.available ? "" : "  (no checkpoint)"}`;

  return (
    <div className="wrap">
      <header>
        <h1>AGID — AI-Generated Media Detector</h1>
        <p>
          Upload an image or video, pick a detector, and get the probability it is
          AI-generated.
          {device && <span className="pill">device: {device}</span>}
        </p>
      </header>

      <div className="grid">
        <section className="card">
          <h2>Input</h2>

          <label>Detector model</label>
          <select
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setResult(null);
            }}
          >
            <optgroup label="Image detectors">
              {imageModels.map((m) => (
                <option key={m.key} value={m.key} disabled={!m.available}>
                  {optLabel(m)}
                </option>
              ))}
            </optgroup>
            <optgroup label="Video detectors">
              {videoModels.map((m) => (
                <option key={m.key} value={m.key} disabled={!m.available}>
                  {optLabel(m)}
                </option>
              ))}
            </optgroup>
          </select>
          {selected && (
            <div className="meta">
              {selected.notes}
              {selected.modality === "image"
                ? " · On a video, runs frame-by-frame and averages."
                : " · Native video model."}
            </div>
          )}

          <label>Media file</label>
          <div
            className={"drop" + (drag ? " drag" : "")}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              onPick(e.dataTransfer.files[0]);
            }}
          >
            {file ? (
              <span>
                <b>{file.name}</b>{" "}
                <span className="meta">({(file.size / 1048576).toFixed(2)} MB)</span>
              </span>
            ) : (
              "Click or drop an image / video here"
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*,video/*"
            style={{ display: "none" }}
            onChange={(e) => onPick(e.target.files[0])}
          />

          {previewUrl && (
            <div className="preview">
              {isVideo ? <video src={previewUrl} controls /> : <img src={previewUrl} alt="" />}
            </div>
          )}

          <button className="go" disabled={!file || !model || busy} onClick={run}>
            {busy ? "Analyzing…" : "Analyze"}
          </button>
          {error && <div className="err">{error}</div>}
        </section>

        <section className="card">
          <h2>Result</h2>
          {!result && !busy && <div className="empty">No result yet.</div>}
          {busy && (
            <div className="empty">
              Running {selected ? selected.name : ""}… first run loads the model and may take a
              while.
            </div>
          )}
          {result && <Result r={result} />}
        </section>
      </div>
    </div>
  );
}

function Result({ r }) {
  const fakePct = r.fake_probability * 100;
  const realPct = r.real_probability * 100;
  return (
    <div>
      <div className={"verdict " + r.verdict}>
        {r.verdict === "fake" ? "AI-GENERATED" : "REAL"}
      </div>
      <div style={{ textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
        confidence {(r.confidence * 100).toFixed(1)}%
      </div>
      <div className="bar">
        <span className="fake" style={{ width: fakePct + "%" }}>
          {fakePct >= 12 ? `fake ${fakePct.toFixed(1)}%` : ""}
        </span>
        <span className="real" style={{ width: realPct + "%" }}>
          {realPct >= 12 ? `real ${realPct.toFixed(1)}%` : ""}
        </span>
      </div>
      <div style={{ marginTop: 14 }}>
        <Row k="P(fake)" v={`${fakePct.toFixed(2)}%`} />
        <Row k="P(real)" v={`${realPct.toFixed(2)}%`} />
        <Row k="Model" v={r.model_name} />
        <Row
          k="Media"
          v={`${r.media_type}${r.num_frames ? " · " + r.num_frames + " frames" : ""}`}
        />
        <Row k="Inference" v={`${r.elapsed_ms} ms`} />
      </div>
      {r.per_frame && (
        <div>
          <div className="meta">Per-frame P(fake) ({r.aggregation}):</div>
          <div className="spark">
            {r.per_frame.map((v, i) => (
              <i key={i} title={`${(v * 100).toFixed(1)}%`} style={{ height: v * 100 + "%" }} />
            ))}
          </div>
        </div>
      )}
      {r.checkpoint_status && (
        <div className="meta" style={{ marginTop: 12 }}>
          ckpt: {r.checkpoint_status}
        </div>
      )}
    </div>
  );
}

const Row = ({ k, v }) => (
  <div className="row">
    <span>{k}</span>
    <b>{v}</b>
  </div>
);
