"""Build interface/backend/repro_full.json: per-method list of ALL reported benchmarks with
SEPARATE columns for the SynthScope-trained checkpoint and the official-weights-in-SynthScope-class number.

Source of truth = the verified specs in repro/benchmark_specs/*.json. Numbers that the specs only
record in prose notes (the AGID/official split is structured inconsistently across methods) are
pulled into an explicit, auditable VAL table below — every value here traces to a file under
repro/bench_results/ (see the per-row tags in the spec notes). Run with --audit to print the table.
"""
import json, glob, os, sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])   # the repository root, wherever it is checked out

# (model, benchmark-name-substring) -> verified {synthscope:{...}, official:{...}} numbers.
# These resolve the SynthScope-vs-official ambiguity in the spec our_number dicts and surface the number
# that the spec only kept in prose. Each traces to a repro/bench_results/*.json (tag in spec note).
VAL = {
    # AIGCDetect is the ProGAN-trained setting (B1); we only retrained the SDv1.4 setting, so no SynthScope_train here.
    ("aide", "AIGCDetect"):              {"official": {"map": 97.94, "acc": 91.89}},
    # SynthScope_train = faithful native-res 5-view retrain (aide_native_v2/model_2) — reproduces the official mAP (acc is lower).
    ("aide", "GenImage"):                {"official": {"map": 97.12, "acc": 85.91}, "synthscope": {"map": 96.42, "acc": 77.87}},
    # ForenSynths = the paper's 11 generators (no StyleGAN2 / WhichFaceIsReal), recomputed from the per-generator entries
    # of {clip,cnn}_{synthscope,official}_forensynths.json; the 8-GAN subset gives clip 97.22 / 96.96, cnn 94.42 / 92.67.
    ("clip", "ForenSynths"):             {"synthscope": {"map": 92.83, "acc": 85.39}, "official": {"map": 92.33, "acc": 80.58}},
    ("clip", "UniversalFakeDetect"):     {"synthscope": {"map": 96.44, "acc": 87.23}, "official": {"map": 95.66}},
    ("cnn", "ForenSynths"):              {"synthscope": {"map": 90.44, "acc": 74.57}, "official": {"map": 90.77, "acc": 75.82}},
    # synthscope Co-Spy-Bench / GenImage = cospy_full/model_1.pth, the checkpoint promoted to co_spy.pth (2026-06-08).
    ("co_spy", "Co-Spy-Bench"):          {"synthscope": {"map": 96.7, "acc": 90.28}, "official": {"map": 86.65, "acc": 83.88}},
    ("co_spy", "AIGCDetect"):            {"official": {"map": 94.59, "acc": 89.25}, "synthscope": {"map": 84.91}},
    ("co_spy", "GenImage"):              {"official": {"map": 92.41}, "synthscope": {"map": 90.59, "acc": 76.23}},
    ("co_spy", "ForenSynths"):           {"official": {"map": 87.97}, "synthscope": {"map": 96.17}},
    # CuRe: only the official adapter exists (the training recipe is unreleased), so there is no SynthScope-trained number.
    ("cure", "AIGIBench"):               {"official": {"map": 97.0, "acc": 92.85, "auroc": 97.48}},
    ("cure", "UniversalFakeDetect"):     {"official": {"map": 99.19, "acc": 93.94}},
    ("cure", "ForenSynths"):             {"official": {"map": 94.05, "acc": 81.95}},
    ("cure", "GenImage"):                {"official": {"map": 97.3, "acc": 85.18}},
    ("cure", "RealChain"):               {"official": {"map": 87.39, "acc": 79.69, "auroc": 85.47}},
    ("cure", "EvalGEN"):                 {"official": {"map": 99.86, "acc": 98.14, "auroc": 99.66}},
    ("cure", "SynthWildX"):              {"official": {"map": 95.61, "acc": 92.69, "auroc": 96.5}},
    ("cure", "WildRF"):                  {"official": {"map": 99.13, "acc": 95.89, "auroc": 99.14}},
    ("cure", "BFree-Online"):            {"official": {"map": 99.06, "acc": 91.64, "auroc": 98.08}},
    ("cure", "RRDataset"):               {"official": {"map": 90.4, "acc": 82.04, "auroc": 88.88}},
    ("cure", "AIGCDetect"):              {"official": {"map": 98.08, "acc": 89.3, "auroc": 98.02}},
    # DDA: synthscope = trained in SynthScope with the official recipe (experiments/image/dda.sh; job 267042, eval 267832).
    ("dda", "SynthWildX"):               {"official": {"map": 96.47, "acc": 90.8, "auroc": 96.4}, "synthscope": {"map": 96.99, "acc": 88.0, "auroc": 96.88}},
    ("dda", "WildRF"):                   {"official": {"map": 96.76, "acc": 90.2, "auroc": 96.14}, "synthscope": {"map": 97.83, "acc": 90.56, "auroc": 97.62}},
    ("dda", "BFree-Online"):             {"official": {"map": 99.41, "acc": 94.98, "auroc": 98.86}, "synthscope": {"map": 99.41, "acc": 94.05, "auroc": 98.82}},
    ("dda", "GenImage"):                 {"official": {"map": 97.74, "acc": 89.37, "auroc": 98.21}, "synthscope": {"map": 98.07, "acc": 91.35, "auroc": 98.42}},   # fakes JPEG q96
    ("dda", "ForenSynths"):              {"official": {"map": 91.13, "acc": 81.43, "auroc": 92.11}, "synthscope": {"map": 92.8, "acc": 80.79, "auroc": 93.7}},   # fakes JPEG q96
    ("dda", "AIGCDetect"):               {"official": {"map": 96.35, "acc": 87.84, "auroc": 96.65}, "synthscope": {"map": 96.93, "acc": 89.58, "auroc": 96.95}},   # fakes JPEG q96
    # EvalGEN: acc = mean FAKE accuracy (the paper's metric); mAP/AUROC are against the MSCOCO val2017 reals.
    ("dda", "EvalGEN"):                  {"official": {"map": 99.93, "acc": 96.86, "auroc": 99.85}, "synthscope": {"map": 99.89, "acc": 96.79, "auroc": 99.76}},
    ("dda", "RealChain"):                {"official": {"map": 76.46, "acc": 67.23, "auroc": 74.97}, "synthscope": {"map": 74.72, "acc": 66.76, "auroc": 72.79}},
    ("dda", "RRDataset"):                {"official": {"map": 85.18, "acc": 75.99, "auroc": 81.74}, "synthscope": {"map": 85.81, "acc": 77.19, "auroc": 83.1}},
    # Effort: GenImage and the in-the-wild sets = the official GenImage-SD1.4 checkpoint. acc = mean per-subset balanced accuracy.
    # UniversalFakeDetect (the paper's ProGAN-trained model; no checkpoint released): synthscope = trained in SynthScope with
    # the official recipe (experiments/image/effort_ufd.sh; job 276492, iteration 18000 = the official selection; eval job 278691,
    # tag effort_ufd_step18000_ufd19), 19 subsets, acc at 0.5.
    # MoE-JEPA: official SID-Set checkpoint (no SynthScope-trained number). SID-Set acc = 3-way accuracy (the paper's
    # metric, bf16 as the official test; job 270455) with the binary mAP/AUROC; the rest: balanced acc at 0.5.
    ("moejepa", "SID-Set"):              {"official": {"map": 99.47, "acc": 96.37, "auroc": 99.41}},
    ("moejepa", "RRDataset (released"):  {"official": {"map": 65.09, "acc": 59.67, "auroc": 64.17}},
    ("moejepa", "SynthWildX"):           {"official": {"map": 83.81, "acc": 73.66, "auroc": 80.35}},
    ("moejepa", "WildRF"):               {"official": {"map": 77.06, "acc": 68.19, "auroc": 73.5}},
    ("moejepa", "BFree-Online"):         {"official": {"map": 91.79, "acc": 57.99, "auroc": 85.22}},
    ("moejepa", "EvalGEN"):              {"official": {"map": 92.77, "acc": 70.85, "auroc": 86.43}},
    ("moejepa", "RealChain"):            {"official": {"map": 71.74, "acc": 66.39, "auroc": 73.22}},
    ("effort", "UniversalFakeDetect"):   {"synthscope": {"map": 99.2, "acc": 94.81, "auroc": 99.04}},
    ("effort", "GenImage"):              {"official": {"map": 99.37, "acc": 89.83, "auroc": 99.48}},
    ("effort", "SynthWildX"):            {"official": {"map": 61.2, "acc": 56.69, "auroc": 63.57}},
    ("effort", "WildRF"):                {"official": {"map": 81.11, "acc": 71.36, "auroc": 78.64}},
    ("effort", "BFree-Online"):          {"official": {"map": 96.45, "acc": 67.75, "auroc": 94.79}},
    ("effort", "EvalGEN"):               {"official": {"map": 99.55, "acc": 92.16, "auroc": 99.16}},
    ("effort", "RealChain"):             {"official": {"map": 60.38, "acc": 50.56, "auroc": 64.22}},
    ("effort", "RRDataset"):             {"official": {"map": 69.78, "acc": 64.59, "auroc": 68.6}},
    # B-Free: the official checkpoint (no SynthScope-trained number). acc = mean per-subset balanced accuracy at logit 0.
    ("bfree", "SynthWildX"):             {"official": {"map": 99.02, "acc": 95.64, "auroc": 98.93}},
    ("bfree", "WildRF"):                 {"official": {"map": 98.55, "acc": 93.15, "auroc": 98.23}},
    ("bfree", "BFree-Online"):           {"official": {"map": 98.65, "acc": 91.54, "auroc": 96.93}},
    ("bfree", "RealChain"):              {"official": {"map": 74.83, "acc": 67.59, "auroc": 70.07}},
    ("bfree", "EvalGEN"):                {"official": {"map": 99.52, "acc": 95.18, "auroc": 98.94}},
    ("bfree", "RRDataset"):              {"official": {"map": 80.34, "acc": 72.33, "auroc": 74.34}},
    # D3QE: only the official checkpoint is evaluated (no SynthScope-trained number). acc = plain = balanced (balanced subsets).
    # synthscope = trained in SynthScope with the official recipe (experiments/image/d3qe.sh; job 265064, eval 265826).
    ("d3qe", "ARForensics"):             {"official": {"map": 92.22, "acc": 81.74, "auroc": 91.91}, "synthscope": {"map": 91.73, "acc": 79.83, "auroc": 91.49}},
    ("d3qe", "ForenSynths"):             {"official": {"map": 92.23, "acc": 83.45, "auroc": 90.65}, "synthscope": {"map": 91.35, "acc": 82.14, "auroc": 90.29}},
    ("d3qe", "GenImage"):                {"official": {"map": 90.94, "acc": 79.44, "auroc": 90.49}, "synthscope": {"map": 90.11, "acc": 77.01, "auroc": 89.63}},
    ("d3qe", "SynthWildX"):              {"official": {"map": 49.3, "acc": 49.58, "auroc": 45.47}},
    ("d3qe", "WildRF"):                  {"official": {"map": 63.65, "acc": 53.03, "auroc": 63.89}},
    ("d3qe", "BFree-Online"):            {"official": {"map": 68.77, "acc": 47.03, "auroc": 64.0}},
    ("d3qe", "RealChain"):               {"official": {"map": 47.24, "acc": 49.49, "auroc": 43.49}},
    ("d3qe", "EvalGEN"):                 {"official": {"map": 91.48, "acc": 62.21, "auroc": 83.94}},
    ("d3qe", "RRDataset"):               {"official": {"map": 59.25, "acc": 54.95, "auroc": 55.11}},
    # OmniAID: official checkpoints only. Mirage-Test acc = the paper's metric (4 balanced categories + Anime fake accuracy).
    ("omniaid", "GenImage (8 subsets) — GenImage-SD1.4"):   {"official": {"map": 99.02, "acc": 95.82, "auroc": 99.16}},
    ("omniaid", "AIGCDetectBenchmark (17) — GenImage-SD1.4"): {"official": {"map": 97.88, "acc": 88.87, "auroc": 98.2}},
    ("omniaid", "Mirage-Test (5 categories) — GenImage-SD1.4"): {"official": {"acc": 51.1}},
    ("omniaid", "GenImage (8 subsets) — Mirage-Train v1"):  {"official": {"map": 99.2, "acc": 97.16, "auroc": 99.12}},
    ("omniaid", "AIGCDetectBenchmark (17) — Mirage-Train v1"): {"official": {"map": 98.93, "acc": 92.88, "auroc": 98.99}},
    ("omniaid", "Mirage-Test (5 categories) — Mirage-Train v1"): {"official": {"acc": 88.39}},
    ("omniaid", "Mirage-Test (v2"):                          {"official": {"map": 96.48, "acc": 78.46, "auroc": 96.65}},
    ("omniaid", "SynthWildX (v2"):                           {"official": {"map": 95.52, "acc": 86.02, "auroc": 95.31}},
    ("omniaid", "WildRF (v2"):                               {"official": {"map": 97.28, "acc": 88.36, "auroc": 96.91}},
    ("omniaid", "BFree-Online (v2"):                         {"official": {"map": 91.7, "acc": 70.26, "auroc": 87.18}},
    ("omniaid", "RealChain (v2"):                            {"official": {"map": 84.02, "acc": 61.71, "auroc": 83.63}},
    ("omniaid", "EvalGEN (v2"):                              {"official": {"map": 96.46, "acc": 74.14, "auroc": 93.08}},
    ("omniaid", "RRDataset (v2"):                            {"official": {"map": 84.19, "acc": 71.75, "auroc": 82.32}},
    # LTF: official checkpoint only (training is not ported). The "official" slot is the SynthScope class on the full
    # official GenImage val (100,000 images; ltf_synthscope_genimage_val100k). On the 33k set the official ltf_ref
    # code gives 98.38, the same as SynthScope.
    ("ltf", "GenImage"):                 {"official": {"map": 99.86, "acc": 98.36, "auroc": 99.88}},
    # synthscope = trained in SynthScope with the paper's recipe (experiments/image/fat_former.sh; train job 277158), model_last
    # (epoch 24); evals = DAS6 job 66626 (tags fat_former_trained_last_forensynths / _ufd). DMs = the 8 UFD subsets (paper: 10).
    ("fat_former", "4-class"):           {"official": {"map": 99.7, "acc": 98.39}, "synthscope": {"map": 99.56, "acc": 97.29, "auroc": 99.42}},
    ("fat_former", "Diffusion-model"):   {"official": {"map": 98.6, "acc": 93.72}, "synthscope": {"map": 98.88, "acc": 92.09, "auroc": 98.67}},
    # synthscope = trained in SynthScope with the official recipe (experiments/image/npr.sh; job 265720, eval 265808).
    ("npr", "Table 1"):                  {"official": {"map": 94.22, "acc": 92.30}, "synthscope": {"map": 93.83, "acc": 84.09, "auroc": 94.94}},
    ("npr", "Table 2"):                  {"official": {"map": 96.49, "acc": 94.81}},
    ("npr", "Table 3"):                  {"official": {"map": 99.4, "acc": 95.03}},
    # Table 4 official = NPR.pth, the served file (npr_official_nprpth_ufd); model_epoch_last_3090.pth gives 96.69 / 93.94.
    # Tables 2, 3 and 5 were measured with model_epoch_last_3090.pth (the registry default before 2026-06-09) only.
    ("npr", "Table 4"):                  {"official": {"map": 97.16, "acc": 94.57}, "synthscope": {"map": 97.49, "acc": 88.91, "auroc": 97.23}},
    ("npr", "Table 5"):                  {"official": {"map": 83.47, "acc": 79.13}},
    # SAFE now SERVES the SynthScope-trained checkpoint (retrain v2 / model_38), which reproduces the paper.
    ("safe", "ForenSynths"):             {"official": {"map": 98.78, "acc": 96.2}, "synthscope": {"map": 97.79, "acc": 95.1}},
    ("safe", "GANGen"):                  {"official": {"map": 99.85, "acc": 99.0}},
    ("safe", "UniversalFakeDetect"):     {"official": {"map": 99.1, "acc": 95.71}, "synthscope": {"map": 98.44, "acc": 94.0}},
    ("safe", "GenImage"):                {"official": {"map": 99.32, "acc": 95.62}, "synthscope": {"map": 99.51, "acc": 95.9}},
    ("safe", "DiTFake"):                 {"synthscope": {"map": 99.96, "acc": 99.2}},
    ("safe", "DiffusionForensics"):      {"synthscope": {"map": 99.34, "acc": 91.3}},
    ("pgc", "GenImage"):                 {"official": {"map": 100.0, "acc": 99.49}, "synthscope": {"map": 97.62, "acc": 78.6}},
    # synthscope UFD / ForenSynths = the official recipe in SynthScope, step 500 (experiments/image/pgc.sh; jobs 266276, 266282, 266660).
    # official UFD / ForenSynths-8 = the served PGC_train_progan_ckpt.pth with the checkpoint load FIXED (2026-11-09), PadCenterCrop
    # eval: tags pgc_official_ufd8_fixed and (FS-8 subset of) pgc_official_forensynths13_fixed. They match the official PGC code
    # (pgc_official_eval.py: UFD 99.34 / 94.16, FS-13 96.25 / 83.73) within 0.03. Correction 2026-11-09: from 2026-06-15 to
    # 2026-11-09 the load dropped the checkpoint's rgb_stream.* tensors, trained LoRA included, so the earlier tags
    # (pgc_official_ufd8 99.17 / 83.57, FS-8 96.41 / 76.34 / 97.06, now in bench_results/_superseded/) understated accuracy.
    # GenImage / AIGIBench / CommGen15 official = official-code numbers (never affected).
    ("pgc", "UniversalFakeDetect"):      {"official": {"map": 99.35, "acc": 94.19, "auroc": 99.31}, "synthscope": {"map": 98.99, "acc": 91.96, "auroc": 98.92}},
    ("pgc", "ForenSynths"):              {"official": {"map": 97.83, "acc": 89.17, "auroc": 98.14}, "synthscope": {"map": 97.41, "acc": 85.89, "auroc": 97.63}},
    ("pgc", "AIGIBench"):                {"official": {"map": 95.36, "acc": 88.33}, "synthscope": {"map": 86.88}},
    ("pgc", "CommGen15"):                {"official": {"map": 99.26, "acc": 97.39}, "synthscope": {"map": 98.61, "acc": 66.66}},
    # SPAI serves the official weights remapped into the SynthScope class, scored at NATIVE resolution (the faithful
    # any-res path). official = that served number; synthscope = the official-recipe retrain in SynthScope, epoch 8
    # (lowest validation loss; experiments/image/spai.sh job 267192; evals jobs 273445 / 273597). Synthbuster reals = a
    # GenImage-nature proxy (200 per generator).
    ("spai", "GenImage"):                {"official": {"map": 94.22, "auroc": 94.28}, "synthscope": {"map": 94.67, "auroc": 94.77, "acc": 87.25}},
    ("spai", "Synthbuster"):             {"official": {"map": 96.75, "auroc": 96.97}, "synthscope": {"map": 95.76, "auroc": 96.37, "acc": 88.14}},
    # D3 now uses the faithful protocol (X-CLIP-B/16 + 10% edge-crop). On GenVideo, AUROC/acc are the
    # robust metrics (the mAP harness is degenerate: 10000 reals : as few as 56 fakes/generator).
    # D3 (training-free): "synthscope" = SynthScope's D3, "official" = the official D3 code end to end on the same data
    # (repro/official_evals/d3_official_eval.py). mAP = AP with real as the positive class, as the paper reports.
    ("d3", "GenVideo"):                  {"synthscope": {"map": 98.95, "auroc": 98.23, "acc": 95.68}, "official": {"map": 97.93, "auroc": 97.04}},
    ("d3", "EvalCrafter"):               {"synthscope": {"map": 99.3, "auroc": 98.62, "acc": 96.59}, "official": {"map": 98.7, "auroc": 97.81}},
    ("d3", "VideoPhy"):                  {"synthscope": {"map": 99.45, "auroc": 98.09, "acc": 95.19}, "official": {"map": 98.84, "auroc": 96.28}},
    ("d3", "VidProM"):                   {"synthscope": {"map": 92.15, "auroc": 90.86, "acc": 84.86}, "official": {"map": 84.91, "auroc": 82.83}},
    # SPLIT (training-free): the paper's metric is fake recall at fixed FPR (in the spec note); these are AP (fake positive) /
    # AUROC / acc at the calibrated threshold. synthscope = job 270384, official = official code end to end (job 270323).
    ("split", "GenVideo"):               {"synthscope": {"map": 99.79, "auroc": 99.97, "acc": 99.41}, "official": {"map": 96.12, "auroc": 99.16}},
    ("restrav", "GenVidBench"):         {"synthscope": {"map": 92.7, "auroc": 95.9, "acc": 86.87}},
    # MIRROR: the official phase-2 checkpoint (DINOv3 ViT-H+; no SynthScope-trained number). acc = mean per-subset
    # balanced accuracy at 0.5; the paper column is the README's DINOv3-H+ value. Synthbuster uses the official
    # extension-based JPEG-q96 rule (99.43 with the served decoded-format rule); UnivFakeDetect uses the official
    # code's 13-subset UnivFD list (93.81 on the paper's Table 7 list). Jobs 279513, 279527, 279550-279555.
    ("mirror", "AIGCDetect"):            {"official": {"map": 99.49, "acc": 97.39, "auroc": 99.51}},
    ("mirror", "GenImage"):              {"official": {"map": 100.0, "acc": 99.78, "auroc": 100.0}},
    ("mirror", "UnivFakeDetect"):        {"official": {"map": 97.04, "acc": 92.44, "auroc": 97.24}},
    ("mirror", "Synthbuster"):           {"official": {"map": 99.92, "acc": 99.28, "auroc": 99.96}},
    ("mirror", "EvalGEN"):               {"official": {"map": 100.0, "acc": 99.85, "auroc": 99.99}},
    ("mirror", "SynthWildX"):            {"official": {"map": 97.71, "acc": 95.08, "auroc": 98.33}},
    ("mirror", "WildRF"):                {"official": {"map": 99.8, "acc": 97.88, "auroc": 99.74}},   # official file set (no .jfif)
    ("mirror", "AIGIBench"):             {"official": {"map": 98.06, "acc": 94.98, "auroc": 98.66}},
    ("mirror", "RRDataset"):             {"official": {"map": 93.86, "acc": 86.91, "auroc": 91.98}},
    ("mirror", "BFree-Online"):          {"official": {"map": 99.78, "acc": 97.4, "auroc": 99.5}},
    ("mirror", "RealChain"):             {"official": {"map": 91.76, "acc": 81.71, "auroc": 89.96}},
    # VINA: the official VINA_DINOv3-L_140k+DDA checkpoint (no SynthScope-trained number: the training code is not released).
    # acc / mAP = the OFFICIAL metric (the _offmetric tags): per-subset pooled accuracy at 0.5 and AP, mean over subsets, as
    # the paper reports them; mauroc is the mean AUROC from the same scores: bench_eval's own, except ForenSynths and
    # ARForensics (the _offmetric files' mean over the official subsets) and GenVideo-Val (the _offmetric AUROC, pooled).
    # ARForensics = the six subsets shared with the paper (our copy has Switti, no MAR;
    # the 7-row bench_eval mean 96.88 / 99.60 is not comparable with the paper). GenImage = the full val split
    # (vina_genimage_val100k); the 1500-per-class test bundle (vina_genimage_full) is its own reference row.
    # Jobs: CPU 281654-281656, 281831; GPU 281657, DAS6 66630.
    ("vina", "ForenSynths"):             {"official": {"map": 97.8, "acc": 86.93, "auroc": 97.44}},
    ("vina", "UniversalFakeDetect"):     {"official": {"map": 99.01, "acc": 94.84, "auroc": 98.71}},
    ("vina", "DiTFake"):                 {"official": {"map": 99.95, "acc": 97.98, "auroc": 99.94}},
    ("vina", "ARForensics"):             {"official": {"map": 99.54, "acc": 96.36, "auroc": 99.38}},
    ("vina", "GenImage (8 subsets"):     {"official": {"map": 99.05, "acc": 94.74, "auroc": 98.72}},
    ("vina", "GenImage test bundle"):    {"official": {"map": 98.45, "acc": 93.29, "auroc": 97.98}},
    ("vina", "GenVideo-Val"):            {"official": {"map": 99.26, "acc": 96.86, "auroc": 99.14}},   # overall pooled ACC / AP
    ("vina", "SynthWildX"):              {"official": {"map": 96.36, "acc": 91.82, "auroc": 97.46}},
    ("vina", "WildRF"):                  {"official": {"map": 99.41, "acc": 96.12, "auroc": 99.25}},
    ("vina", "AIGIBench"):               {"official": {"map": 97.33, "acc": 92.72, "auroc": 97.6}},
    ("vina", "RRDataset"):               {"official": {"map": 92.77, "acc": 82.89, "auroc": 91.69}},
    # PROBE (ICML'26): the two official checkpoints (probe = DINOv2 ViT-L/14 with registers, probe_resnet = ResNet-50); no
    # SynthScope-trained number (the training code is not released). acc / mAP / mauroc = bench_eval means over the paper's
    # subsets (repro/bench_results/<key>_<benchmark>.json; acc = mean balanced accuracy at 0.5 = the paper's metric).
    # GenImage = the full val split (100,000 images) with all fakes JPEG q96 as in the paper. Synthbuster uses 5000 MSCOCO
    # reals, NOT the paper's RAISE-1k (the spec marks it "reference": not like-for-like). AIGI-Bench = CommunityAI + SocialRF
    # only. Chameleon and AIGI-Quality-Paradox were not run (no loader). Jobs DAS6 66753, 66755, 66790.
    ("probe", "GenImage (8 subsets"):    {"official": {"map": 99.77, "acc": 97.47, "auroc": 99.8}},
    ("probe", "Synthbuster"):            {"official": {"map": 99.64, "acc": 98.2, "auroc": 99.93}},
    ("probe", "SynthWildX"):             {"official": {"map": 98.8, "acc": 97.3, "auroc": 99.18}},
    ("probe", "WildRF"):                 {"official": {"map": 99.36, "acc": 96.92, "auroc": 99.26}},
    ("probe", "AIGI-Bench"):             {"official": {"map": 96.46, "acc": 89.88, "auroc": 95.88}},
    ("probe_resnet", "GenImage (8 subsets"): {"official": {"map": 86.69, "acc": 78.38, "auroc": 84.81}},
    ("probe_resnet", "Synthbuster"):     {"official": {"map": 85.23, "acc": 83.75, "auroc": 95.88}},
    ("probe_resnet", "SynthWildX"):      {"official": {"map": 83.88, "acc": 75.2, "auroc": 82.8}},
    ("probe_resnet", "WildRF"):          {"official": {"map": 89.45, "acc": 81.63, "auroc": 85.76}},
    ("probe_resnet", "AIGI-Bench"):      {"official": {"map": 79.8, "acc": 70.44, "auroc": 77.55}},
    ("stall", "GenVideo"):               {"synthscope": {"map": 82.38, "auroc": 83.51, "acc": 56.88}},
    ("stall", "VideoFeedback"):          {"synthscope": {"map": 83.07, "auroc": 85.1, "acc": 62.12}},
    ("stall", "ComGenVid"):              {"synthscope": {"map": 82.59, "auroc": 84.0, "acc": 57.34}},
    # synthscope = trained in SynthScope with the reconstructed recipe at native resolution (experiments/image/vib_net.sh;
    # job 267846, eval 267928): 15 generators (7 GenImage + 8 ForenSynths GANs), registry vib_net_official.
    ("vib_net", "Diffusion-source (SDv1.4-trained)"):        {"synthscope": {"map": 94.08, "acc": 85.22}},
    ("co_spy", "In-the-wild"): {},   # genuinely not evaluated (prevents the "Co-Spy-Bench" substring collision)
}

# Narrow label overrides applied after status_for (which stays unchanged for every other detector):
# (model, exact benchmark label) -> status.
# MIRROR RRDataset: 86.91 vs the README's 88.3 is 1.39 short, about 6x MIRROR's other deviations (all within 0.25), but
# inside the generic +-2.0 tolerance, which would label it "official". Every doc calls it short, so the site does too.
# VINA SynthWildX and WildRF: the paper prints per-subset values only; SynthScope is below it on both (and the unmodified
# official eval.py gives the same scores on our files, so the gap is the data). The means of the printed values (92.8,
# 97.13) are within the generic +-2.0 tolerance, which would label them "official".
STATUS_OVERRIDE = {("mirror", "RRDataset (pooled folder)"): "below",
                   ("vina", "SynthWildX (dalle3 / firefly / midjourney_v5)"): "below",
                   ("vina", "WildRF (facebook / reddit / twitter)"): "below"}

# Tolerance (percentage points) for calling a number a reproduction of the paper.
TOL = {"map": 2.0, "acc": 2.0, "auroc": 2.5}

LAB = {"map": "mAP", "acc": "acc", "auroc": "AUROC"}


def paper_metrics(d):
    """Normalize a paper_number dict to {map,acc,auroc} in PERCENT (fractions like 0.80 -> 80)."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        val = v * 100 if 0 < v <= 1.0 else v
        if "auc" in kl or "auroc" in kl:
            out.setdefault("auroc", val)
        elif kl == "acc" or kl.startswith("acc") or "_acc" in kl:
            out.setdefault("acc", val)
        elif "ap" in kl or "map" in kl:
            out.setdefault("map", val)
    return out


def cmp_ok(ours, pm):
    """True/False if ours reproduces the paper on every shared metric; None if not comparable."""
    shared = [m for m in ("map", "acc", "auroc") if m in ours and m in pm]
    if not shared:
        return None
    return all(ours[m] >= pm[m] - TOL[m] for m in shared)


def fmt_paper(d):
    if not isinstance(d, dict):
        return "—" if d in (None, "") else str(d)
    parts = [f"{k}={v}" for k, v in d.items()
             if k != "note" and v not in (None, "") and not isinstance(v, (dict, list))]
    return " / ".join(parts) if parts else "—"


def has_paper_number(d):
    if not isinstance(d, dict):
        return d not in (None, "")
    return any(k != "note" and v not in (None, "") and not isinstance(v, (dict, list))
               for k, v in d.items())


def fmt_ours(d):
    if not d:
        return ""
    return " / ".join(f"{LAB[k]}={d[k]}" for k in ("map", "acc", "auroc") if k in d)


def combined(synthscope, official):
    a, o = fmt_ours(synthscope), fmt_ours(official)
    if a and o:
        return f"SynthScope {a} · official {o}"
    if o:
        return f"{o} (official wts)"
    return a


NOTE_MAX = 400


def clip_note(note, limit=NOTE_MAX):
    """``note`` cut at a word boundary within ``limit`` characters, with an ellipsis (never ends mid-number)."""
    if len(note) <= limit:
        return note
    return note[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:( ") + "\u2026"


def match_val(model, name):
    """Return (synthscope, official, matched_substr) for the best VAL hit. A prefix match wins over a
    mid-string match (so 'In-the-wild (Co-Spy-Bench ...)' matches the In-the-wild entry, not Co-Spy-Bench)."""
    best = None  # (is_prefix, len, sub, v)
    nl = name.lower()
    for (m, sub), v in VAL.items():
        if m == model and sub.lower() in nl:
            cand = (nl.startswith(sub.lower()), len(sub), sub, v)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if not best:
        return {}, {}, None
    return best[3].get("synthscope", {}), best[3].get("official", {}), best[2]


def status_for(name, paper, synthscope, official, spec_status):
    """reproduced = SynthScope-trained matches paper; official = official-weights-in-SynthScope matches paper
    (SynthScope-trained number shown alongside where attempted); below = neither; reference = paper gives
    no number for this cross-dataset; missing = not evaluated. Falls back to the spec's label only
    when the paper metric keys are not numerically comparable (e.g. ReStraV's nested per-split dicts), and keeps
    the spec's "reference" when it says our data or protocol differs from the paper's (not like-for-like)."""
    if not has_paper_number(paper):
        return "reference" if (synthscope or official) else "missing"
    if not (synthscope or official):
        return "missing"
    if spec_status == "reference":               # the spec says the data/protocol differs from the paper's
        return "reference"
    pm = paper_metrics(paper)
    if not pm:                                   # paper has numbers but not in comparable keys
        return spec_status or "below"
    synthscope_ok, off_ok = cmp_ok(synthscope, pm), cmp_ok(official, pm)
    if synthscope_ok:
        return "reproduced"
    if off_ok:
        return "official"
    if synthscope_ok is None and off_ok is None:       # neither comparable -> trust the spec
        return spec_status or "below"
    return "below"


out = {}
audit = []
for f in sorted(glob.glob(f"{ROOT}/repro/benchmark_specs/*.json")):
    d = json.load(open(f))
    model = d.get("model")
    rows = []
    for b in d.get("benchmarks", []):
        name = b.get("name", "") or ""
        paper = b.get("paper_number")
        spec_status = b.get("our_status", "")
        synthscope, official, sub = match_val(model, name)
        status = status_for(name, paper, synthscope, official, spec_status)
        status = STATUS_OVERRIDE.get((model, name), status)
        on = b.get("our_number")
        note = ""
        if isinstance(on, dict) and on.get("note"):
            note = on["note"]
        else:
            note = (b.get("gap_cause") or b.get("dataset_source") or "")
        rows.append({
            "benchmark": name,
            "metric": b.get("metric", ""),
            "paper": fmt_paper(paper),
            "ours": combined(synthscope, official),
            "ours_synthscope": fmt_ours(synthscope),
            "ours_official": fmt_ours(official),
            "status": status,
            "note": clip_note(note),
        })
        audit.append((model, name[:38], fmt_paper(paper)[:22],
                      (fmt_ours(synthscope) or "-")[:22], (fmt_ours(official) or "-")[:22],
                      f"{spec_status}->{status}"))
    out[model] = rows

json.dump(out, open(f"{ROOT}/interface/backend/repro_full.json", "w"), indent=2)
n = sum(len(v) for v in out.values())
counts = {}
for v in out.values():
    for r in v:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
print(f"wrote repro_full.json: {len(out)} methods, {n} rows  status={counts}")

if "--audit" in sys.argv:
    print(f"\n{'model':10} {'benchmark':38} {'paper':22} {'SynthScope':22} {'official':22} status")
    print("-" * 140)
    for r in audit:
        print(f"{r[0]:10} {r[1]:38} {r[2]:22} {r[3]:22} {r[4]:22} {r[5]}")
