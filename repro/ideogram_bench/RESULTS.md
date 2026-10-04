# Detectors vs Ideogram 4.0 — ranked

Dataset: 100 real (sampled from GenImage/CNNDetection reals) + 100 Ideogram-generated fakes (BLIP-caption prompts, 512px, 28 steps; people excluded — Ideogram refuses them). Detectors use the promoted SynthScope checkpoints + the web UI's preprocessing.

| Rank | Detector | AUROC | AP | acc@0.5 | fake detect % | real FP % |
|---|---|---|---|---|---|---|
| 1 | PGC | 93.66 | 90.67 | 69.0 | 45.0 | 7.0 |
| 2 | CO-SPY | 92.53 | 91.27 | 75.0 | 53.0 | 3.0 |
| 3 | SAFE | 91.94 | 88.42 | 89.0 | 95.0 | 17.0 |
| 4 | VIB-Net | 88.65 | 89.4 | 77.0 | 73.0 | 19.0 |
| 5 | NPR | 83.41 | 76.53 | 77.0 | 97.0 | 43.0 |
| 6 | FatFormer | 81.58 | 82.24 | 68.0 | 40.0 | 4.0 |
| 7 | AIDE | 76.96 | 78.64 | 68.5 | 72.0 | 35.0 |
| 8 | UnivFD / CLIP | 73.72 | 78.09 | 65.5 | 38.0 | 7.0 |
| 9 | CNNDetection (ResNet-50) | 70.08 | 73.75 | 60.5 | 22.0 | 1.0 |
| 10 | SPAI | 61.64 | 56.45 | 52.5 | 17.0 | 12.0 |
