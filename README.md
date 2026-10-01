# Causal adaptive discretization for nonlinear dispersive waves
This repository contains code accompanying the manuscript
**"Adaptive Discretization of Nonlinear Dispersive Waves Using Equation‑Dependent Diagnostics and Finite‑Horizon Adequacy Estimation"**

The method constructs a finite‑horizon numerical adequacy estimate from normalized physical diagnostics and their causal history. A deterministic safety layer governs projection, invariant, local‑error and coarsening checks. The trained model provides early warnings, ranks candidates that have passed deterministic checks and triggers verification; the model cannot bypass numerical safety safeguards.

## Scope
The public repository includes code only:
- numerical kernels for mCH, BO, KdV, ILW and power‑law dispersive equations
- 42‑coordinate normalized diagnostic representation
- construction and training of the `H=2` finite‑horizon adequacy estimator
- deterministic safety‑aware adaptive selection logic
- frozen‑transfer and temporal‑history ablation workflows
- plotting scripts for manuscript figures
- configuration files and focused tests

Datasets, trained weights, solution trajectories, tables and figure files are not committed to this repository. They will be generated under `results/` and `output/` after running workflows, and both directories are git‑ignored.

## Repository layout
```
src/fgsp_ch/        numerical methods, diagnostics, models, and controllers
configs/experiment/ registered experiment configurations
scripts/            entry scripts for dataset construction, training, evaluation, analysis and plotting
tests/              focused numerical and controller tests
docs/               stage‑by‑stage V2 design notes
```

## Installation
Python 3.10 or newer is required.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml,test]"
```

All experiments are CPU‑compatible. CUDA‑capable GPU only accelerates neural‑network training and is not required for numerical solvers or plotting.

## Quick verification
```bash
python scripts/verify_release.py
python -m pytest
```

The verifier checks source‑only release policy, configuration references, imports and accidental machine‑specific hard‑coded paths. No external data will be downloaded during tests.

## Reproduce the V2 workflow
The full pipeline generates solution trajectories from scratch, no external dataset download required.
```bash
python scripts/build_cmame_v2_predictive_risk_dataset.py --mode full
python scripts/train_cmame_v2_predictive_risk_operator.py --mode full --device cpu
python scripts/run_cmame_v2_stage5_closed_loop.py --mode full --device cpu
python scripts/run_cmame_v2_kdv_ilw_showcases.py --device cpu
python scripts/run_cmame_v2_power_law_showcase.py --mode full --device cpu
```

## Methodological boundary
- Estimator fitting and calibration are performed using only mCH and BO;
- KdV, ILW and power‑law equations are targets for frozen‑transfer testing;
- Estimator weights, normalization, calibration quantities and selection rule remain fixed during transfer;
- Post‑hoc reference solutions serve solely for evaluation purposes and are not fed into the online controller;
- Coarsening requires projection, invariant and verification checks.

## Contact
hsycipher@hotmail.com; 
hutianqiao@hotamail.com.


