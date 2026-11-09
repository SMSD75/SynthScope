# Superseded result files

Runs kept for the record but replaced by a corrected rerun. `benchmark/aggregate_results.py` reads only the top level of
`repro/bench_results/`, so nothing in this folder appears in `repro/RESULTS_TABLE.md`. The matching lines in
`repro/MASTER_RESULTS.jsonl` are kept as well (the file is append-only).

| files | replaced by | why |
| --- | --- | --- |
| `pgc_*.json` (13 tags: `pgc_official_ufd8`, `pgc_official_forensynths13`, `pgc_realchain`, `pgc_synthwildx`, `pgc_wildrf`, `pgc_evalgen`, `pgc_bfree_online`, `pgc_rrdataset`, `pgc_official_deploycheck`, `pgc_evalfix_official`, `pgc_h3_officialroot`, `pgc_400step_forensynths`, `pgc_evalfix_step900`) | the same tag plus `_fixed` in `repro/bench_results/` | Correction 2026-11-09: from 2026-06-15 to 2026-11-09 the engine dropped the `rgb_stream.*` tensors (including the trained LoRA) of official-format PGC checkpoints. These numbers were scored with that broken load. See `repro/investigate/pgc_gap_investigation.md`. |
| (note for the PGC rows) | | `pgc_official_deploycheck` (2026-06-15) and `pgc_400step_forensynths` (2026-06-19) also used the old Resize->CenterCrop eval transform, so their `_fixed` reruns change both the transform (now PadCenterCrop) and the load. That is why they equal `pgc_evalfix_official_fixed` and `pgc_evalfix_step900_fixed` (same checkpoints, same transform, fixed load). |
| `aide_official_genimage8.GenImage_train_inflated.json`, `npr_official_freqnet.resize256.json` | earlier corrections | see `repro/RESULTS_SUMMARY.md` |
