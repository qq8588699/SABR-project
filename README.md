# G3++ Gaussian HJM Swaption Pricing and Calibration

> **USD SOFR ATM Swaption Normal Volatility — Closed-Form Pricing, PCA-Based Identification, and Two-Stage Calibration**

---

## Overview

This project provides a complete, research-grade implementation of swaption pricing and calibration under the three-factor Gaussian Heath–Jarrow–Morton (HJM) model, known as **G3++**. The model is applied to USD SOFR ATM swaption normal (Bachelier) implied volatility.

The repository contains:

| File | Description |
|---|---|
| `G3pp_Gaussian_HJM_Swaption_Paper.pdf` | Companion academic paper with full derivations |
| `sofr_fwd_pca.py` | Data pipeline: zero rates → forward rates → jump cleaning → PCA → G3++ parameter identification |
| `g3pp_calibration.py` | Calibration engine: two-stage nested optimisation to fit the swaption vol matrix |

---

## The Academic Paper

**Title:** *Closed-Form Swaption Pricing and Calibration under the Gaussian G3++ / HJM Framework — with Application to USD SOFR Swaption Normal Volatility*

The paper is a self-contained derivation covering the full chain from model dynamics to a production-ready calibration algorithm. It is intended for quantitative practitioners and researchers who want to understand not just the formulas but the mathematical reasoning behind each step.

### Structure

**Section 1 — Model Setup**
Defines the three-factor HJM forward rate dynamics under the risk-neutral measure. The volatility function for each factor takes the separable exponential form σₖ exp(−κₖ(T−t)), where factor k=1 captures the level (slow decay), k=2 the slope, and k=3 the curvature (fast, short-end dominant). The no-arbitrage HJM drift condition is derived, showing the drift is fully determined by the volatility specification. The state-space (G3++) representation is introduced, with each factor satisfying an Ornstein–Uhlenbeck SDE.

**Section 2 — Closed-Form Zero-Coupon Bond Pricing**
Derives the affine bond price formula:

```
P(t,T) = P(0,T)/P(0,t) · exp(−∑ₖ Bₖ(t,T)·xₖ(t) + ½∑ₖₗ ρₖₗ σₖ σₗ Γₖₗ(t,T))
```

where Bₖ(t,T) = (1 − exp(−κₖ(T−t)))/κₖ and Γₖₗ is a closed-form variance integral. Critically, the deterministic shift φ(t) that fits the initial curve cancels analytically.

**Section 3 — Forward Swap Rate and Swaption Payoff**
Constructs the forward swap rate as a ratio of affine functionals of the bond prices and establishes the coupon bond representation of the swaption payoff.

**Section 4 — Forward Swap Rate SDE (6 Steps)**
A six-step derivation of the forward swap rate dynamics:
1. Bond price SDE under Q
2. Annuity SDE under Q
3. Exact swap rate SDE under Q — a ratio of Itô processes
4. Change of measure to the annuity measure Q^A
5. Frozen-coefficient approximation to reduce the SDE to a constant-coefficient Gaussian
6. Connection to the Bachelier normal volatility formula

**Section 5 — Linearisation and Closed-Form Normal Volatility**
The hedge ratios hₖ = ∂F/∂xₖ|ₓ₌₀ are computed in closed form. The first-order Taylor expansion then gives a Gaussian distribution for the forward swap rate at expiry, yielding the main pricing result:

```
σₙ² = (1/Tₑ) ∑ₖₗ ρₖₗ σₖ σₗ hₖ hₗ · (1 − exp(−(κₖ+κₗ)Tₑ)) / (κₖ+κₗ)
```

Accuracy of the linearisation is discussed with a 4-row error table and Monte Carlo validation.

**Section 6 — Calibration (most detailed section)**
A full treatment of the calibration problem, covering:

- *§6.1* Parameter vector Θ = (σ₁,σ₂,σ₃,κ₁,κ₂,κ₃,β) ∈ ℝ⁷ and the vega-weighted least-squares objective
- *§6.2* Convex structure of the problem in σₖ — the key insight enabling the two-stage algorithm
- *§6.3* Inner loop (NNLS): solving for σₖ at fixed (κₖ, β) via non-negative least squares on a linear design matrix
- *§6.4* Outer loop (L-BFGS-B): optimising over log(κₖ) and log(β) with analytical gradients
- *§6.5* Robust pre-processing — **jump detection pipeline**:
  - Stage 1: per-tenor bipower sliding-window (Barndorff-Nielsen & Shephard 2004) for cell-level jump flagging
  - Stage 2: iterative Mahalanobis winsorisation (Huber M-estimator) for multivariate outlier days
- *§6.6* Zero rates to instantaneous forward rates via finite differences (four interpolation methods: finite difference, cubic spline, Hyman monotone spline, flat forward)
- *§6.7* **PCA-to-G3++ Identification Bridge** — the core initialisation procedure:
  - §6.7.1: Model-implied covariance structure Σ = EME', rank-3
  - §6.7.2: Factor-space projection M = (E'E)⁻¹ E' Σ̂ E (E'E)⁻¹
  - §6.7.3: Recovery of σₖ, ρₖₗ, and β from M
  - §6.7.4: Identification of κₖ from eigenvector spatial decay (OLS log-linear regression)
  - §6.7.5: Rotation ambiguity — the correct loading decomposition L = SR^{1/2} (not S^{1/2}R^{1/2})
  - §6.7.6: Complete 9-step identification workflow
- *§6.8* Initialisation via PCA — practical application of the bridge
- *§6.9* Convergence criteria
- *§6.10* Full calibration algorithm (pseudo-code)

**Section 7 — Calibration Quality and Diagnostics**
Quality thresholds for USD SOFR swaptions, common failure modes, and remedies.

### References

Heath, Jarrow & Morton (1992); Brigo & Mercurio (2006); Rebonato (2002); Jamshidian (1989); Barndorff-Nielsen & Shephard (2004); Hagan et al. (2002, SABR); Lawson & Hanson (1974, NNLS); Hyman (1983, monotone splines); Rousseeuw & Croux (1993); Rousseeuw & Van Driessen (1999); Byrd et al. (1995, L-BFGS-B); Tawfik (2025); Spadafora et al. (2018).

---

## Python Scripts

### `sofr_fwd_pca.py` — Data Pipeline and Identification Bridge

This script implements the full chain from raw market zero rate data to G3++ parameter initialisation values ready to pass to the calibrator.

#### Pipeline (`run_pipeline`)

```python
from sofr_fwd_pca import run_pipeline

kappa0, sigma0, beta0, M_hat, rho = run_pipeline(
    csv_path    = "sofr_zeros.csv",
    n_factors   = 3,
    dt          = 1/360,      # ACT/360 — SOFR standard
)
```

The pipeline runs these stages in sequence:

1. **CSV ingestion** — loads zero rates with date index and tenor columns (e.g. `1M`, `3M`, `1Y`, `5Y`, `30Y`)
2. **Zero → forward rates** — converts zero rates to instantaneous forward rates via finite differences on the cumulative yield function Y(τ) = τ · Z(τ); four methods available (`'fd'`, `'cubic'`, `'monotone'`, `'flat'`)
3. **Daily changes** — computes Δf(t,τ) = f(t+1,τ) − f(t,τ)
4. **Jump detection** (two-stage):
   - Stage 1: bipower sliding window (window=21) per tenor independently — flags and subtracts cell-level jumps
   - Stage 2: iterative Mahalanobis winsorisation — shrinks multivariate outlier days radially onto the chi² ellipsoid surface
5. **Rolling-mean demeaning** — trailing 21-day rolling mean subtracted; first 20 rows discarded as warm-up (ACT/360: dt=1/360)
6. **Covariance PCA** — eigendecomposition of the sample covariance matrix Σ (not the correlation matrix — this preserves amplitudes needed for σₖ identification)
7. **G3++ identification bridge**:
   - κₖ from OLS log-linear regression on eigenvector spatial decay: log|vₖ(τⱼ)| ~ aₖ − κₖτⱼ
   - E[j,k] = exp(−κ̂ₖ · τⱼ) basis matrix
   - M = (E'E)⁻¹ E' Σ E (E'E)⁻¹ factor-space covariance
   - σₖ = √(Mₖₖ/dt),  ρₖₗ = Mₖₗ/√(MₖₖMₗₗ),  β via golden-section search

#### Key Functions

| Function | Description |
|---|---|
| `run_pipeline(csv_path, ...)` | Full end-to-end pipeline; returns `(kappa0, sigma0, beta0, M_hat, rho)` |
| `run_pca(delta_f_clean, pillar_tenors, n_factors=3, dt=1/360)` | PCA + identification bridge on clean Δf matrix |
| `zero_to_fwd(tau, Z, method='fd')` | Zero rates → instantaneous forward rates |
| `build_delta_f(fwd_rates)` | Forward rate matrix → daily changes |
| `detect_jumps(delta_f, ...)` | Two-stage jump decomposition; returns cleaned matrix and diagnostics |
| `compute_rotation_L(sigma_k, rho)` | Computes L = SR^{1/2} (loading matrix decomposition; warns L ≠ S^{1/2}R^{1/2}) |
| `load_zero_rates_csv(path)` | CSV loader; returns `(dates, tenors, zero_rate_matrix)` |
| `save_pipeline_csvs(result, out_dir)` | Saves cleaned Δf, PCA outputs, identification bridge results to CSV |
| `print_pca_summary(pca_result, loading_result)` | Prints factor table: κ̂, σ̂, R², variance explained |

#### CSV Format

Input CSV must have a date column (any name, parseable by pandas) and tenor columns. Tenor labels may use suffixes `M` (months) or `Y` (years), e.g.:

```
date,1M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,20Y,30Y
2021-01-04,0.0008,0.0009,0.0010,0.0012,...
2021-01-05,0.0008,0.0009,0.0010,0.0013,...
```

Zero rates should be expressed in **decimal** (e.g. `0.05` for 5%), ACT/360 day count.

#### Day Count Convention

The `dt` parameter controls how Δf is scaled to annual units during σₖ extraction:

| Convention | `dt` value | Use case |
|---|---|---|
| ACT/360 | `1/360` ← **default** | SOFR (standard) |
| ACT/365 | `1/365` | Some non-USD IBOR curves |
| Business days | `1/252` | Equity-style calibrations |

---

### `g3pp_calibration.py` — Calibration Engine

This script implements the two-stage nested calibration of the G3++ model to a market swaption normal vol matrix. It can be used standalone or with parameters initialised from `sofr_fwd_pca.py`.

#### Quick Start

```python
from g3pp_calibration import G3ppCalibrator
import numpy as np

def P0(T):
    return np.exp(-0.05 * np.asarray(T))   # flat 5% discount curve

market_vols = {
    (1,  1): 85.0, (1,  2): 90.0, (1,  5): 95.0, (1, 10): 88.0,
    (2,  1): 82.0, (2,  2): 86.0, (2,  5): 91.0, (2, 10): 85.0,
    (5,  1): 70.0, (5,  2): 74.0, (5,  5): 79.0, (5, 10): 75.0,
    (10, 1): 58.0, (10, 2): 61.0, (10, 5): 65.0, (10,10): 62.0,
}
# Keys are (expiry_years, tenor_years); values are normal vol in bps

calib = G3ppCalibrator(P0=P0, market_vols_bps=market_vols, delta=0.5)
result = calib.calibrate(n_restarts=3, verbose=True)

print(result["sigmas"])    # (3,) array, annual units
print(result["kappas"])    # (3,) array, yr⁻¹
print(result["rmse_bps"])  # root-mean-square error in bps
```

#### With Historical Data (Full Initialisation Bridge)

```python
from sofr_fwd_pca import run_pipeline

kappa0, sigma0, beta0, M_hat, rho = run_pipeline("sofr_zeros.csv")

calib = G3ppCalibrator(
    P0             = P0,
    market_vols_bps= market_vols,
    delta          = 0.5,
    delta_f        = delta_f,       # (T, n) historical Δf from pipeline
    pillar_tenors  = pillar_tenors, # (n,) tenor grid in years
    rho_mode       = "exponential", # or "pca" / "identity"
)
result = calib.calibrate()
```

#### Two-Stage Algorithm

**Stage 1 — Inner loop (NNLS, σₖ fixed κ, β):**
Given fixed mean-reversion speeds and correlation structure, the normal vol is approximately linear in σₖ. A linear design matrix A[m,k] is constructed (diagonal approximation) and σₖ ≥ 0 is solved via non-negative least squares. The result provides a warm start for the outer loop.

**Stage 2 — Outer loop (L-BFGS-B, κₖ and β):**
Optimises log(κₖ) and log(β) (reparameterised for positivity). At each outer evaluation, the inner NNLS is re-run. Multi-start with random log-space perturbations guards against local minima.

#### Calibrator Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `P0` | callable | — | Initial discount curve P(0,T) |
| `market_vols_bps` | dict | — | `{(Te, Ts): vol_bps}` ATM normal vols |
| `delta` | float | `0.5` | Accrual fraction (0.5 = semi-annual) |
| `delta_f` | ndarray or None | `None` | Historical Δf for PCA initialisation |
| `pillar_tenors` | ndarray or None | `None` | Tenor grid matching `delta_f` columns |
| `kappas0` | ndarray or None | `None` | Manual κ override |
| `mu_reg` | float | `1e-4` | Regularisation on log(κₖ) |
| `rho_mode` | str | `'exponential'` | `'exponential'`, `'pca'`, or `'identity'` |
| `n_factors` | int | `3` | Number of G3++ factors |

#### Correlation Modes

| Mode | Description | Free params |
|---|---|---|
| `'exponential'` | ρₖₗ = exp(−β\|k−l\|), β calibrated | 7 (σ×3, κ×3, β) |
| `'pca'` | ρₖₗ from identification bridge (data-driven) | 6 (σ×3, κ×3) |
| `'identity'` | ρₖₗ = δₖₗ (uncorrelated factors) | 6 (σ×3, κ×3) |

#### Result Dictionary

```python
result = {
    "sigmas":           np.ndarray,  # (3,) vol amplitudes in annual units
    "kappas":           np.ndarray,  # (3,) mean-reversion speeds yr⁻¹
    "beta":             float,       # correlation decay (None for non-exponential modes)
    "rho":              np.ndarray,  # (3,3) factor correlation matrix
    "rho_mode":         str,
    "fitted_vols_bps":  dict,        # {(Te,Ts): model vol in bps}
    "market_vols_bps":  dict,        # input market vols
    "errors_bps":       dict,        # model − market per instrument
    "rmse_bps":         float,       # root-mean-square error
    "max_error_bps":    float,       # worst-case absolute error
    "success":          bool,        # optimiser convergence flag
    "n_swaptions":      int,
}
```

#### Post-Calibration Utilities

```python
from g3pp_calibration import price_swaption, vol_matrix

# Price a single off-grid swaption
px = price_swaption(Te=5, Ts=10, K=None, result=result, delta=0.5, P0=P0)
print(px["sigma_n_bps"], px["price"])

# Compute model vol matrix over an expiry/tenor grid
V = vol_matrix([1, 2, 5, 10], [1, 2, 5, 10], result, delta=0.5, P0=P0)
# V[i,j] is in bps
```

#### Backward Compatibility

The alias `G5ppCalibrator = G3ppCalibrator` is retained for any code referencing the old five-factor class name.

---

## Mathematical Summary

### Model

```
df(t,T) = α(t,T) dt + ∑ₖ σₖ exp(−κₖ(T−t)) dWₖ(t),   k = 1,2,3
dxₖ     = −κₖ xₖ dt + σₖ dWₖ,   xₖ(0) = 0
dWₖ dWₗ = ρₖₗ dt
r(t)    = ∑ₖ xₖ(t) + φ(t)
```

### Closed-Form Normal Vol

```
σₙ²(Tₑ, Tₛ) = (1/Tₑ) ∑ₖₗ ρₖₗ σₖ σₗ hₖ hₗ [1 − exp(−(κₖ+κₗ)Tₑ)] / (κₖ+κₗ)
```

where the hedge ratios hₖ = ∂F₀/∂xₖ|ₓ₌₀ are computed in closed form from the initial discount curve.

### Parameter Identification from Data

```
Σ̂ = (1/(T−1)) X_c' X_c              (covariance of demeaned Δf, rolling window=21)
M  = (E'E)⁻¹ E' Σ̂ E (E'E)⁻¹         (factor-space projection)
σₖ = √(Mₖₖ / dt),   dt = 1/360       (ACT/360)
ρₖₗ = Mₖₗ / √(Mₖₖ Mₗₗ)
κₖ from OLS: log|vₖ(τⱼ)| ~ aₖ − κₖτⱼ
```

---

## Dependencies

```
numpy
scipy
pandas          (for CSV I/O in sofr_fwd_pca.py)
matplotlib      (for diagnostic plots in sofr_fwd_pca.py)
```

Install with:
```bash
pip install numpy scipy pandas matplotlib
```

Tested on Python 3.10+.

---

## Running the Self-Tests

Each script contains a `__main__` block with a self-test:

```bash
# Test calibration engine (no data required — uses flat discount curve)
python g3pp_calibration.py

# Test full pipeline (generates synthetic SOFR data internally)
python sofr_fwd_pca.py
```

---

## Files

```
.
├── README.md
├── G3pp_Gaussian_HJM_Swaption_Paper.pdf   ← compiled paper
├── sofr_fwd_pca.py                         ← data pipeline + identification bridge
├── g3pp_calibration.py                     ← calibration engine
└── sofr_synthetic_zeros.csv                ← sample synthetic input (for testing)
```
