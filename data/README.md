# Dataset Preparation Guide

This repository benchmarks Offline Signature Verification on two primary standard datasets: **CEDAR** and **ICDAR 2011 (SigComp2011)**.

---

## 1. CEDAR Signature Dataset

The CEDAR dataset contains signatures from 55 writers (24 genuine and 24 skilled forgeries per writer).

### Expected Directory Layout:
```
data/
└── CEDAR/
    ├── full_org/
    │   ├── original_1_1.png
    │   ├── original_1_2.png
    │   └── ...
    └── full_forg/
        ├── forgeries_1_1.png
        ├── forgeries_1_2.png
        └── ...
```

### Download Instructions:
1. Request access from the [Center of Excellence for Document Analysis and Recognition (CEDAR)](https://cedar.buffalo.edu/).
2. Extract the archive into `data/CEDAR/`.

---

## 2. ICDAR 2011 (SigComp2011) Dataset

Contains Dutch and Chinese signature corpora evaluated under the official competition protocol:
- **Dutch Corpus**: 10 reference training writers, 54 evaluation testing writers (genuine and skilled forgeries).
- **Chinese Corpus**: 10 training writers, 10 testing writers.

### Expected Directory Layout:
```
data/
└── ICDAR2011/
    ├── SigComp11-Off_Dutch/
    │   ├── TrainingSet/
    │   └── TestingSet/
    └── SigComp11-Off_Chinese/
        ├── TrainingSet/
        └── TestingSet/
```

### Download Instructions:
1. Download from the official [TC11 ICDAR 2011 Competition Portal](http://tc11.cvc.uab.es/).
2. Extract into `data/ICDAR2011/`.

---

## 3. Synthetic Hard Negatives (Auto-Generated)

During Stage 2 training (`python train.py --stage negatives`), synthetic tremor-perturbed forgeries are generated and stored in:
```
data/
└── synthetic_hard_negatives/
    ├── synthetic_negatives_manifest.json
    └── writer_01/
        ├── synth_001.png
        └── ...
```
Each generated sample is ranked by the **Structural Discrepancy Score ($S_{\mathrm{diff}}$)**.
