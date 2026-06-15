# Detectors vs Ideogram 4.0 — ranked

Dataset: 62 real (sampled from GenImage/CNNDetection reals) + 62 Ideogram-generated fakes (BLIP-caption prompts, 1024px, 48 steps; people excluded — Ideogram refuses them). Detectors use the promoted AGID checkpoints + the web UI's preprocessing.

| Rank | Detector | AUROC | AP | acc@0.5 | fake detect % | real FP % |
|---|---|---|---|---|---|---|
| 1 | CO-SPY | 96.98 | 96.93 | 82.26 | 66.13 | 1.61 |
| 2 | PGC | 93.99 | 91.85 | 60.48 | 22.58 | 1.61 |
| 3 | VIB-Net | 93.89 | 94.57 | 85.48 | 85.48 | 14.52 |
| 4 | SAFE | 93.73 | 90.27 | 90.32 | 96.77 | 16.13 |
| 5 | AIDE | 79.06 | 81.95 | 71.77 | 79.03 | 35.48 |
| 6 | NPR | 77.59 | 70.32 | 70.97 | 82.26 | 40.32 |
| 7 | FatFormer | 73.83 | 71.8 | 54.84 | 12.9 | 3.23 |
| 8 | UnivFD / CLIP | 60.54 | 66.53 | 57.26 | 20.97 | 6.45 |
| 9 | SPAI | 56.57 | 51.6 | 50.0 | 11.29 | 11.29 |
| 10 | CNNDetection (ResNet-50) | 55.41 | 62.85 | 54.84 | 9.68 | 0.0 |
