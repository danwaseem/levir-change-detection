# BuildSafe Vision — DATA 255 Presentation (8 min + 3 Q&A)

**Title:** Deep Learning for Building Change Detection in Multi-Temporal Satellite Imagery  
**Team:** Sneha Singh · Ritika Mukesh Neema · Danish Waseem · Rajesh Paruchuri  
**Repo:** https://github.com/danwaseem/levir-change-detection

Suggested timing. All four students must speak.

---

## Slide 1 — Problem (Sneha, ~45s)
- Need: map building construction/demolition between satellite dates
- Why hard: illumination, vegetation, sensor shift ≠ building change
- Task: binary change segmentation on LEVIR-CD (not operational system)

## Slide 2 — Data & protocol (Sneha, ~60s)
- LEVIR-CD: 445/64/128 pairs → 256² tiles after split
- Stem overlap = 0; ~4–5% change pixels
- Primary metric: change-class F1 @ 0.5; test frozen until end

## Slide 3 — Methods (Ritika, ~75s)
- Reference: absolute RGB difference
- Strong: FC-EF (6-ch U-Net, base=16)
- Investigation: FC-Siam-diff / FC-Siam-conc
- Show Fig. 1 architecture

## Slide 4 — Experiment design (Danish, ~60s)
- 9 GPU runs; principal = 3-seed EF vs Siam-diff (BCE+Dice)
- Ablations: loss (BCE vs BCE+Dice), paired aug on/off
- Freeze 2026-11-08 → one locked test

## Slide 5 — Main results (Danish, ~75s)
- Abs-diff F1 0.129 vs learned ~0.70–0.78
- Mean val F1: FC-EF **0.718** vs Siam-diff **0.543**
- Locked FC-EF+BCE **test F1 0.780**
- H1 / H2 **not supported** (honest negatives)

## Slide 6 — Ablations & cost (Rajesh, ~60s)
- BCE 0.778 > BCE+Dice 0.702 (seed 1511)
- FC-EF lowest latency among competitive setups
- Params ~483k; ~50 min/run on CUDA lab

## Slide 7 — Failures & limits (Rajesh, ~60s)
- Gallery: large buildings OK; small buildings missed (recall ~0.08)
- Empty-tile false alarms; Texas-only data
- No deployment / safety claims

## Slide 8 — Conclusion (all, ~45s)
- Controlled answer to RQ with mixed evidence
- Early fusion wins under this capacity/budget
- Repo + `docs/REPRODUCTION.md` for checks

## Q&A prep (each member owns one)
- Sneha: splits / leakage / abs-diff threshold
- Ritika: U-Net training / aug / checkpointing
- Danish: Siamese fusion / why H1 failed / gallery
- Rajesh: metrics / latency protocol / reproduction
