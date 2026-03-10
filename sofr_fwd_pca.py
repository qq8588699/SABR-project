"""
SOFR Forward Rate PCA Correlation Calibration
==============================================
Full pipeline: zero rates -> instantaneous forward rates -> jump decomposition
-> PCA on correlation matrix -> G5++ factor correlation matrix.

Pipeline steps
--------------
  1. Zero -> instantaneous forward rates at the same pillar tenors:
       f(tau_i) = dY/dtau|_{tau_i}  via second-order finite differences on
       Y(tau) = tau*Z(tau).  Left/right endpoints use one-sided differences;
       interior nodes use centred differences.  Input grid = output grid.
  2. Daily forward rate changes  delta_f(t, tau) = f(t+1, tau) - f(t, tau)
  3. Two-stage jump decomposition (Section 6.6 of companion paper):
       delta_f(t,tau) = epsilon(t,tau) + J1(t,tau) + J2(t,tau)
       Stage 1 : per-tenor bipower sliding-window (Barndorff-Nielsen & Shephard 2004)
                 Jump estimate subtracted cell-by-cell; all T rows retained.
       Stage 2 : iterative Mahalanobis winsorisation (Huber M-estimator)
                 Multivariate outlier days shrunk radially to chi^2 ellipsoid;
                 all T rows retained with winsorised values.
  4. PCA on the CORRELATION matrix R = D^{-1} Sigma_diff D^{-1} of the
     Stage-1+2 cleaned changes (all T rows).
  5. Loading matrix L = D V_{R,K} Lambda_{R,K}^{1/2} and G5++ factor
     correlation rho = L_fac_rn @ L_fac_rn.T (row-normalised, unit diagonal).

Mathematical background
-----------------------
Forward rate finite-difference formula
    Define the cumulative-yield function  Y_i = tau_i * Z_i.
    The instantaneous forward rate at each pillar is:
        f_0     = (Y_1 - Y_0) / (tau_1 - tau_0)                 left endpoint
        f_i     = (Y_{i+1} - Y_{i-1}) / (tau_{i+1} - tau_{i-1}) centred
        f_{n-1} = (Y_{n-1} - Y_{n-2}) / (tau_{n-1} - tau_{n-2}) right endpoint
    Trapezoidal area preservation holds to O(h^2):
        (f_{i+1} + f_i)/2 * (tau_{i+1} - tau_i) ≈ Y_{i+1} - Y_i.

Jump decomposition
    Stage 1 — bipower sliding window, per tenor independently:
        sigma*_j = [sqrt(pi/2) * sum |r_k||r_{k-1}|]^{1/2}  (bipower scale)
        flag cell (t,j) if |delta_f(t,j) - mu_j| > z_{1-alpha/2} * sigma*_j
        J1(t,j) = delta_f(t,j) - m_j   (m_j = window median, robust centre)
        epsilon_hat(t,j) = m_j          (Stage-1 diffusive residual)
    Stage 2 — iterative Mahalanobis winsorisation:
        D_t^2 = (Y_t - mu)^T (Sigma + eps*I)^{-1} (Y_t - mu)  ~ chi^2(n)
        flag day t if D_t^2 > chi2.ppf(1 - alpha_mah, n)
        s_t = sqrt(chi2_thresh / D_t^2)  (shrinkage factor)
        J2(t) += (Y_t - mu)(1 - s_t)    (incremental jump, accumulated with +=)
        Y_t  <- mu + (Y_t - mu) * s_t   (winsorised in place)
        Iterate until flagged set converges (typically 2-3 passes).

PCA and loading matrix
    Run PCA on the CORRELATION matrix R = D^{-1} Sigma_diff D^{-1}
    (not the covariance Sigma_diff directly) so that eigenvectors
    reflect pure correlation shapes free of per-tenor vol distortion.
    Loading matrix:  L = D V_{R,K} Lambda_{R,K}^{1/2}   (n x K)
      so that  L L^T ~= Sigma_diff  (K-factor approximation).
    Factor correlation:
      L_fac = L[:K, :]  (K x K factor sub-block)
      L_fac_rn = row-normalised L_fac
      rho = L_fac_rn @ L_fac_rn.T  (unit diagonal guaranteed)
    Note: PCA identifies rho_kl only.  kappa_k is not identifiable
    from PCA; use init_kappa_logspaced() for a log-spaced starting grid.

Initialisation summary (for G5++ calibration)
    rho_kl  <- from PCA of R (this module)
    beta    <- moment-matched to adjacent rho_kl via e^{-beta} = mean(rho_{k,k+1})
    kappa_k <- log-spaced grid [kappa_min, kappa_max] (init_kappa_logspaced)
    sigma_k <- one inner NNLS pass at fixed kappa^{(0)}, rho^{(0)} (in g5pp_calibration.py)

Usage
-----
Quick start with synthetic data:
    python sofr_fwd_pca.py

With real data:
    import sofr_fwd_pca as sfp
    result = sfp.run_pipeline(
        dates         = my_dates,        # (T,) array of date strings or ordinals
        zero_rates    = my_zero_rates,   # (T, n_pillars) in decimal (e.g. 0.05)
        pillar_tenors = my_tenors,       # (n_pillars,) in years
    )
    # Key outputs:
    print(result['rho'].round(3))               # (5x5) G5++ factor correlation
    print(result['kappa0'])                      # (5,) log-spaced kappa initialisation
    print(result['beta0'])                       # scalar beta initialisation
    print(result['jump_result']['n_stage1_days']) # Stage-1 cells adjusted
    print(result['jump_result']['n_stage2_days']) # Stage-2 days winsorised

Author: generated alongside G5++ Gaussian HJM Swaption Paper
"""

from __future__ import annotations

import warnings
import os
import re
import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2 as _chi2


# =============================================================================
# Section 0 — CSV I/O
# =============================================================================

def _parse_tenor(s: str) -> float:
    """
    Parse a tenor string into a float number of years.

    Handles the following formats (case-insensitive):
      "0.25"  "0.25Y"  ".25y"   -> 0.25
      "1"     "1Y"     "1y"     -> 1.0
      "10"    "10Y"    "10y"    -> 10.0
      "6M"    "6m"               -> 0.5
      "3M"    "3m"               -> 0.25
      "18M"   "18m"              -> 1.5

    Raises ValueError if the string cannot be parsed.
    """
    s = s.strip()
    # Try plain float first
    try:
        return float(s)
    except ValueError:
        pass
    # Year suffix: e.g. "1Y", "0.25Y", "10y"
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)[Yy]", s)
    if m:
        return float(m.group(1))
    # Month suffix: e.g. "6M", "3m", "18M"
    m = re.fullmatch(r"([0-9]+)[Mm]", s)
    if m:
        return float(m.group(1)) / 12.0
    raise ValueError(
        f"Cannot parse tenor '{s}'. "
        "Expected formats: '1Y', '6M', '0.25', '0.25Y', etc."
    )


def load_zero_rates_csv(path: str) -> tuple:
    """
    Load a zero rate time series from a CSV file.

    Expected CSV format
    -------------------
    - First column   : dates (any string label, e.g. 'date', 'Date', 'DATE')
    - Remaining columns : one per tenor, with the tenor as the column header.

    Tenor headers are parsed flexibly:
        "0.25Y", "0.5Y", "1Y", "2Y", "5Y", "10Y", "30Y"
        "3M", "6M", "18M"
        "0.25", "1", "10"  (plain floats interpreted as years)

    Zero rates must be in **decimal** form (e.g. 0.05 for 5%).
    If all values in the rate block exceed 1.0 the function raises a
    ValueError with a clear message so the caller can rescale.

    Empty rows and rows where ALL rate values are NaN are dropped silently.
    Rows with any but not all NaN values raise a warning and are dropped.

    Parameters
    ----------
    path : str   path to the CSV file

    Returns
    -------
    zero_rates    : (T, n) ndarray  zero rates in decimal
    pillar_tenors : (n,)   ndarray  tenor grid in years (ascending)
    dates         : (T,)   ndarray  date strings from the first column

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if the header cannot produce at least 2 tenor columns,
                       or if rates appear to be in percent rather than decimal
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    with open(path, "r", newline="") as fh:
        raw = fh.read()

    # Detect delimiter (comma or semicolon or tab)
    first_line = raw.split("\n")[0]
    if "\t" in first_line:
        delim = "\t"
    elif ";" in first_line:
        delim = ";"
    else:
        delim = ","

    lines = [ln.rstrip("\r") for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln.strip()]   # drop blank lines

    if len(lines) < 2:
        raise ValueError(f"CSV has fewer than 2 non-empty lines: {path}")

    # --- Parse header ---
    header = [h.strip().strip('"').strip("'") for h in lines[0].split(delim)]
    if len(header) < 3:
        raise ValueError(
            f"CSV header must have at least 3 columns "
            f"(date + ≥2 tenor columns); found {len(header)}: {header}"
        )

    # First column is the date column regardless of its name
    tenor_strs = header[1:]
    pillar_tenors = np.array([_parse_tenor(t) for t in tenor_strs], dtype=float)

    # Enforce ascending tenors
    if not np.all(np.diff(pillar_tenors) > 0):
        order = np.argsort(pillar_tenors)
        pillar_tenors = pillar_tenors[order]
        # Will reorder columns after reading data
    else:
        order = None

    n_tenors = len(pillar_tenors)

    # --- Parse data rows ---
    date_list  = []
    rate_rows  = []
    skipped    = 0

    for ln in lines[1:]:
        parts = [p.strip().strip('"').strip("'") for p in ln.split(delim)]
        if len(parts) < n_tenors + 1:
            skipped += 1
            continue

        date_str = parts[0]
        try:
            vals = [float(p) if p not in ("", "NA", "NaN", "nan", "N/A") else np.nan
                    for p in parts[1: n_tenors + 1]]
        except ValueError:
            skipped += 1
            continue

        row = np.array(vals, dtype=float)
        n_nan = int(np.isnan(row).sum())

        if n_nan == n_tenors:
            skipped += 1
            continue                      # all-NaN row: skip silently
        if n_nan > 0:
            warnings.warn(
                f"Row '{date_str}' has {n_nan}/{n_tenors} NaN values — row skipped.",
                RuntimeWarning,
            )
            skipped += 1
            continue

        date_list.append(date_str)
        rate_rows.append(row)

    if skipped > 0:
        warnings.warn(f"{skipped} row(s) skipped due to missing/malformed values.",
                      RuntimeWarning)

    if len(rate_rows) < 2:
        raise ValueError(
            f"CSV produced fewer than 2 valid data rows after parsing: {path}"
        )

    zero_rates = np.array(rate_rows, dtype=float)

    # Re-order columns if tenors were not already ascending
    if order is not None:
        zero_rates    = zero_rates[:, order]
        pillar_tenors = pillar_tenors  # already sorted above

    # Sanity check: rates should be in decimal, not percent
    rate_max = np.nanmax(zero_rates)
    if rate_max > 1.0:
        raise ValueError(
            f"Zero rates appear to be in PERCENT (max value = {rate_max:.4f}). "
            "Please divide by 100 to convert to decimal before saving the CSV, "
            "or rescale: zero_rates / 100."
        )

    dates = np.array(date_list)
    return zero_rates, pillar_tenors, dates


def write_zero_rates_csv(
    path: str,
    zero_rates: np.ndarray,
    pillar_tenors: np.ndarray,
    dates=None,
) -> None:
    """
    Write a zero rate time series to a CSV file in the format expected by
    load_zero_rates_csv().

    Parameters
    ----------
    path          : str         output file path
    zero_rates    : (T, n)      zero rates in decimal
    pillar_tenors : (n,)        tenor grid in years
    dates         : (T,) or None  date labels; defaults to t0, t1, ...
    """
    T, n = zero_rates.shape
    if dates is None:
        dates = np.array([f"t{i}" for i in range(T)])

    tenor_headers = ",".join(f"{tau:.4g}Y" for tau in pillar_tenors)
    header = f"date,{tenor_headers}"

    lines = [header]
    for i, d in enumerate(dates):
        vals = ",".join(f"{v:.8g}" for v in zero_rates[i])
        lines.append(f"{d},{vals}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# =============================================================================
# Section 1 — Zero Rates to Instantaneous Forward Rates
# =============================================================================

# ── Shared anchor logic ───────────────────────────────────────────────────────
#
# For ALL interpolation schemes the origin anchor  Y(0) = 0  must be used.
#
# Why: P(t,t) = 1  =>  Z(0) = 0  =>  Y(0) = tau*Z|_{tau=0} = 0  (exact).
# Without the anchor the area integral from 0 to tau_0 is undefined.
# With it:  integral_0^{tau_0} f du = Y_0  (exact or O(h^2) depending on scheme).
#
# Each helper below prepends  (tau=0, Y=0)  to the knot sequence before
# fitting, so that every scheme naturally passes through the origin.
# ─────────────────────────────────────────────────────────────────────────────


def _zero_to_fwd_fd(tau: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Finite-difference scheme — vectorised over T dates.

    tau : (n,)    pillar tenors, tau[0] > 0
    Y   : (T, n)  cumulative yield  Y_i = tau_i * Z_i

    Stencils (all use Y(-1) = 0 at tau(-1) = 0 as anchor):

        f_0     = (Y_0 - 0) / (tau_0 - 0)  = Z_0          exact on [0, tau_0]
        f_i     = (Y_{i+1} - Y_{i-1}) / (tau_{i+1} - tau_{i-1})   centred
        f_{n-1} = (Y_{n-1} - Y_{n-2}) / (tau_{n-1} - tau_{n-2})   one-sided right

    Area errors: [0, tau_0] is EXACT; interior intervals O(h^2).
    """
    n = len(tau)
    f = np.empty_like(Y)
    f[:, 0] = Y[:, 0] / tau[0]                                           # = Z_0
    for i in range(1, n - 1):
        f[:, i] = (Y[:, i + 1] - Y[:, i - 1]) / (tau[i + 1] - tau[i - 1])
    f[:, -1] = (Y[:, -1] - Y[:, -2]) / (tau[-1] - tau[-2])
    return f


def _zero_to_fwd_cubic(tau: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    C2 cubic spline (not-a-knot) anchored at Y(0) = 0 — vectorised.

    tau : (n,)    pillar tenors, tau[0] > 0
    Y   : (T, n)  cumulative yield

    The origin anchor (0, 0) is prepended to the knot sequence before
    fitting, forcing the spline to pass through Y(0)=0.  The derivative
    dY/dtau evaluated at each pillar gives the instantaneous forward rate.

    Area errors: O(h^4) on all intervals including [0, tau_0].
    """
    from scipy.interpolate import CubicSpline
    T, n   = Y.shape
    tau_a  = np.concatenate(([0.0], tau))          # prepend origin anchor
    f      = np.empty((T, n))
    for t in range(T):
        Y_a    = np.concatenate(([0.0], Y[t]))     # prepend Y(0) = 0
        spl    = CubicSpline(tau_a, Y_a, bc_type='not-a-knot')
        f[t]   = spl(tau, 1)                       # derivative at pillar nodes
    return f


def _hyman_slopes_anchored(tau_a: np.ndarray, Y_a: np.ndarray) -> np.ndarray:
    """Bessel slopes with Hyman (1983) monotonicity filter on anchored grid."""
    n     = len(tau_a)
    delta = np.diff(Y_a) / np.diff(tau_a)
    d     = np.zeros(n)
    for i in range(1, n - 1):
        d[i] = (Y_a[i + 1] - Y_a[i - 1]) / (tau_a[i + 1] - tau_a[i - 1])
    d[0]  = delta[0]
    d[-1] = delta[-1]
    for i in range(n):
        if i == 0:
            lim = 3.0 * abs(delta[0])
        elif i == n - 1:
            lim = 3.0 * abs(delta[-1])
        else:
            lim = 3.0 * min(abs(delta[i - 1]), abs(delta[i]))
        if abs(d[i]) > lim:
            d[i] = np.sign(d[i]) * lim
    return d


def _zero_to_fwd_monotone(tau: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Monotone Hyman-filtered Hermite spline anchored at Y(0) = 0 — vectorised.

    tau : (n,)    pillar tenors, tau[0] > 0
    Y   : (T, n)  cumulative yield

    The origin anchor (0, 0) is prepended before computing Hyman slopes,
    ensuring the spline passes through Y(0)=0.  Monotonicity of Y
    (i.e. f >= 0) is guaranteed by the Hyman filter.

    Area errors: O(h^2) on all intervals including [0, tau_0].
    """
    from scipy.interpolate import CubicHermiteSpline
    T, n   = Y.shape
    tau_a  = np.concatenate(([0.0], tau))
    f      = np.empty((T, n))
    for t in range(T):
        Y_a  = np.concatenate(([0.0], Y[t]))
        dY_a = _hyman_slopes_anchored(tau_a, Y_a)
        spl  = CubicHermiteSpline(tau_a, Y_a, dY_a)
        f[t] = np.array([float(spl(ti, 1)) for ti in tau])
    return f


def _zero_to_fwd_flat(tau: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Flat-forward (piecewise-constant) scheme anchored at Y(0) = 0 — vectorised.

    tau : (n,)    pillar tenors, tau[0] > 0
    Y   : (T, n)  cumulative yield

    The instantaneous forward rate is constant on each interval and equals
    the finite-difference slope of Y:

        f_0 = (Y_0 - 0) / (tau_0 - 0)  = Z_0    covers [0, tau_0]   EXACT
        f_i = (Y_i - Y_{i-1}) / (tau_i - tau_{i-1})   covers [tau_{i-1}, tau_i]

    Each f_i is assigned to the RIGHT endpoint of its interval (so the
    output at pillar tau_i is the slope of the interval ending there).
    The first interval [0, tau_0] gives f_0 = Z_0 exactly.

    Area errors: EXACT on all intervals (rectangle rule, step-constant f).
    """
    T, n = Y.shape
    f    = np.empty((T, n))
    f[:, 0] = Y[:, 0] / tau[0]                                    # Z_0, exact
    for i in range(1, n):
        f[:, i] = (Y[:, i] - Y[:, i - 1]) / (tau[i] - tau[i - 1])
    return f


def zero_to_fwd(
    pillar_tenors: np.ndarray,
    zero_rates: np.ndarray,
    method: str = "fd",
) -> np.ndarray:
    """
    Convert zero rates to instantaneous forward rates at the same pillar tenors.

    All four schemes share a critical common anchor: Y(0) = 0
    ──────────────────────────────────────────────────────────
    P(t,t) = 1  =>  Z(0) = 0  =>  Y(0) = tau*Z|_{tau=0} = 0  (exact).

    Without anchoring at the origin, the area integral over [0, tau_0]
    — i.e. the discount factor P(t, t+tau_0) — is undefined.  All four
    schemes below prepend the ghost point (tau=0, Y=0) before computation.

    Schemes
    -------
    "fd"       Finite-difference derivatives of Y = tau*Z at pillar nodes.
               f_0 = Z_0 (EXACT on [0,tau_0]); interior centred O(h²);
               last node one-sided backward.  Default; fastest; no scipy.

    "cubic"    C2 not-a-knot cubic spline fit to Y with origin anchor prepended.
               f = dY/dτ evaluated at pillars.  Smooth forward curve; O(h^4)
               area error on all intervals.  Best for smooth curves.

    "monotone" Hyman (1983) monotone Hermite spline with origin anchor prepended.
               Guarantees Y non-decreasing (f >= 0) on stressed/inverted curves.
               O(h^2) area error.  Use when cubic spline produces negative f.

    "flat"     Piecewise-constant (log-linear discount factor) with origin anchor.
               f_0 = Z_0 (EXACT on [0,tau_0]); f_i = ΔY/Δτ on each interval.
               EXACT area preservation on every interval.
               Assigns each f_i to the right endpoint of its interval.

    Area preservation — [0, tau_0] interval
    ----------------------------------------
    Scheme       [0, tau_0] error     All other intervals
    "fd"         EXACT (0.00e+00)     O(h²) trapezoidal
    "cubic"      O(h^4)               O(h^4) trapezoidal
    "monotone"   O(h^2)               O(h^2) trapezoidal
    "flat"       EXACT (0.00e+00)     EXACT (rectangle rule)

    Parameters
    ----------
    pillar_tenors : (n,) array   tenor grid in years, strictly increasing,
                                 tau[0] > 0  (origin is the implicit anchor)
    zero_rates    : (n,) or (T, n) array   zero rates in decimal.
                   2-D: rows = dates, columns = tenors.
    method        : str   one of "fd" (default), "cubic", "monotone", "flat"

    Returns
    -------
    fwd_rates : same shape as zero_rates   instantaneous forward rates in decimal
    """
    tau = np.asarray(pillar_tenors, dtype=float)
    Z   = np.asarray(zero_rates,    dtype=float)

    scalar = (Z.ndim == 1)
    if scalar:
        Z = Z[np.newaxis, :]

    T, n = Z.shape
    if n != len(tau):
        raise ValueError(
            f"zero_rates has {n} columns but pillar_tenors has {len(tau)} entries."
        )
    if n < 2:
        raise ValueError("Need at least 2 pillar tenors.")
    if tau[0] <= 0:
        raise ValueError(
            f"First pillar tenor must be > 0 (got {tau[0]}). "
            "tau=0 is the implicit origin anchor Y(0)=0."
        )

    Y = tau[np.newaxis, :] * Z    # cumulative yield  Y_i = tau_i * Z_i  (T, n)

    if method == "fd":
        f = _zero_to_fwd_fd(tau, Y)
    elif method == "cubic":
        f = _zero_to_fwd_cubic(tau, Y)
    elif method == "monotone":
        f = _zero_to_fwd_monotone(tau, Y)
    elif method == "flat":
        f = _zero_to_fwd_flat(tau, Y)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose: 'fd', 'cubic', 'monotone', 'flat'."
        )

    return f[0] if scalar else f


def check_area_preservation(
    pillar_tenors: np.ndarray,
    zero_rates: np.ndarray,
    fwd_rates: np.ndarray,
    tol: float = 1e-4,
    method: str = "fd",
) -> dict:
    """
    Verify area preservation on ALL intervals including [0, tau_0].

    The fundamental identity is:

        integral_0^{tau_i} f(u) du  =  Y_i  =  tau_i * Z_i   for all i

    which is equivalent to requiring that the discount factor
    P(t, t+tau) = exp(-Y(tau)) is internally consistent from tau=0 onward.

    Integration rules used per scheme
    ----------------------------------
    "flat"     : rectangle rule on each interval (f is constant per interval;
                 trapezoid would mix two different constants and give wrong answer)
                 [0, tau_0]       : f_0 * tau_0          EXACT
                 [tau_i, tau_{i+1}] : f_{i+1} * (tau_{i+1} - tau_i)  EXACT

    "fd", "cubic", "monotone" : trapezoidal rule (f is smooth)
                 [0, tau_0]       : f_0 * tau_0  (rectangle, since f(0) not given)
                 [tau_i, tau_{i+1}] : (f_i + f_{i+1})/2 * Δtau

    Parameters
    ----------
    pillar_tenors : (n,) array
    zero_rates    : (n,) array   single date, decimal
    fwd_rates     : (n,) array   output of zero_to_fwd for the same date
    tol           : float        pass/fail threshold (default 1e-4)
    method        : str          scheme used to produce fwd_rates; controls
                                 which integration rule is applied (default "fd")

    Returns
    -------
    dict with keys:
        errors     : (n,) array   absolute errors; index 0 = [0, tau_0],
                                  index i = [tau_{i-1}, tau_i] for i >= 1
        max_error  : float
        passed     : bool
        intervals  : list of (tau_lo, tau_hi) pairs
        rules      : list of integration rule names used per interval
    """
    tau = np.asarray(pillar_tenors, dtype=float)
    Z   = np.asarray(zero_rates,    dtype=float)
    f   = np.asarray(fwd_rates,     dtype=float)
    Y   = tau * Z

    errors    = []
    intervals = []
    rules     = []

    # ── [0, tau_0] ────────────────────────────────────────────────────────────
    # For all schemes: f_0 = Z_0, so the rectangle f_0 * tau_0 = Y_0 exactly.
    # (For cubic/monotone f_0 ≠ Z_0 strictly, so use rectangle as best estimate.)
    area_0 = f[0] * tau[0]
    errors.append(abs(area_0 - Y[0]))
    intervals.append((0.0, float(tau[0])))
    rules.append("rectangle")

    # ── [tau_i, tau_{i+1}] ────────────────────────────────────────────────────
    for i in range(len(tau) - 1):
        exact = Y[i + 1] - Y[i]
        if method == "flat":
            # f_{i+1} is the constant rate on [tau_i, tau_{i+1}]
            area = f[i + 1] * (tau[i + 1] - tau[i])
            rule = "rectangle"
        else:
            # Smooth scheme: trapezoidal approximation
            area = 0.5 * (f[i] + f[i + 1]) * (tau[i + 1] - tau[i])
            rule = "trapezoid"
        errors.append(abs(area - exact))
        intervals.append((float(tau[i]), float(tau[i + 1])))
        rules.append(rule)

    errors = np.array(errors)
    return {
        "errors":    errors,
        "max_error": float(np.max(errors)),
        "passed":    bool(np.max(errors) < tol),
        "intervals": intervals,
        "rules":     rules,
    }


# =============================================================================
# Section 2 — Build Daily Forward Rate Changes
# =============================================================================

def build_delta_f(fwd_rates: np.ndarray) -> np.ndarray:
    """
    Compute daily first differences of the forward rate matrix.

    Parameters
    ----------
    fwd_rates : (T, m) array  instantaneous forward rates, rows = dates

    Returns
    -------
    delta_f : (T-1, m) array  daily changes delta_f[t] = fwd_rates[t+1] - fwd_rates[t]
    """
    return np.diff(fwd_rates, axis=0)


# =============================================================================
# =============================================================================
# Section 3 — Jump Detection
# =============================================================================

# =============================================================================
# Order-statistic helpers for Stage 1 jump detection
# (ported from reference implementation)
#
# The j-th order statistic of n i.i.d. Normal(0,1) samples has a known
# CDF.  These helpers compute that CDF, its PDF, and solve for the tau-
# and (1-tau)-quantiles of each order statistic — used as per-rank bounds
# in the Stage 1 bipower test.
# =============================================================================

def _ord_comb(n: int, r: int, fac: float) -> float:
    """Generalised binomial coefficient C(n,r) * fac^r, computed recursively."""
    if r > n:  return 0.0
    if r == 0: return 1.0
    if r > n / 2: return _ord_comb(n, n - r, fac)
    return fac * n * _ord_comb(n - 1, r - 1, fac) / r


def _order_stat_cdf(n: int, r: int, cdf, x: float) -> float:
    """
    CDF of the r-th order statistic (1-indexed) of n i.i.d. samples from
    distribution with CDF `cdf`, evaluated at x.

    P(X_(r) <= x) = sum_{j=r}^{n} C(n,j) * F(x)^j * (1-F(x))^(n-j)

    The two branches of _ord_comb avoid numerical overflow for large n.
    """
    Fx = cdf(x)
    val = 0.0
    for j in range(r, n + 1):
        if j > n / 2:
            val += _ord_comb(n, n - j, Fx * (1 - Fx)) * Fx ** (2 * j - n)
        else:
            val += _ord_comb(n, j, Fx / (1 - Fx)) * (1 - Fx) ** n
    return val


def _order_stat_pdf(n: int, r: int, cdf, pdf, x: float) -> float:
    """PDF of the r-th order statistic of n i.i.d. samples."""
    Fx = cdf(x)
    fx = pdf(x)
    if r > n / 2:
        val = _ord_comb(n, n - r, 1 - Fx) * r * fx * Fx ** (r - 1)
    else:
        val = _ord_comb(n, r, Fx / (1 - Fx)) * r * fx * (1 - Fx) ** n / Fx
    return val


def _order_stat_plot(n: int, r: int, cdf, tau: float):
    """
    Coarse grid search over [-4, 4] to bracket the tau, 0.5, and 1-tau
    quantiles of the r-th order statistic CDF.  Returns
    [x_lo_tau, x_hi_tau, x_lo_med, x_hi_med, x_lo_1mtau, x_hi_1mtau].
    """
    x_min, x_max   = -10.0, 10.0
    y_min, y_max   = -10.0, 10.0
    c_min, c_max   = -10.0, 10.0
    vx_min = vx_max = vy_min = vy_max = vc_min = vc_max = None

    for ii in range(-80, 80, 2):
        x   = ii / 20.0
        val = _order_stat_cdf(n, r, cdf, x)
        if x > x_min and val < tau:        x_min, vx_min = x, val
        if x < x_max and val > tau:        x_max, vx_max = x, val
        if x > c_min and val < 0.5:        c_min, vc_min = x, val
        if x < c_max and val > 0.5:        c_max, vc_max = x, val
        if x > y_min and val < 1 - tau:    y_min, vy_min = x, val
        if x < y_max and val > 1 - tau:    y_max, vy_max = x, val

    return [x_min, x_max, c_min, c_max, y_min, y_max]


def _solve_order_stat_quantile(n: int, r: int, cdf, pdf,
                               tau: float, bracket) -> float:
    """
    Solve P(X_(r) <= x) = tau for x, starting from the bracket returned
    by _order_stat_plot.  Uses scipy least_squares with analytic Jacobian.
    """
    import scipy.optimize as _sopt

    func = lambda x: _order_stat_cdf(n, r, cdf, x[0]) - tau
    jac  = lambda x: [[_order_stat_pdf(n, r, cdf, pdf, x[0])]]
    x0   = [(bracket[0] + bracket[1]) / 2.0]
    res  = _sopt.least_squares(
        func, x0, jac=jac, method='trf', bounds=bracket,
        ftol=1e-15, xtol=1e-15, gtol=1e-15,
    )
    return float(res.x[0])


# Pre-computed order-statistic bounds for WIN=21, tau=0.0025.
# Each entry [lo, hi, med] gives the tau, 1-tau, and 0.5 quantiles of the
# distribution of vtest = (X_(r+1) - mu_hat) / bvar at rank r, where
# bvar is the _bipower_scale_ref formula and mu_hat is the sample mean,
# both estimated from the same WIN=21 i.i.d. N(0,1) window.
#
# These were calibrated by Monte Carlo (N=500000) to give exactly 0.5% false
# positive rate per (rank, window) cell for clean Gaussian data.
# The analytical Normal order-statistic quantiles (as in the reference's
# run_order_stat_solver) are NOT used here because bvar >> sigma for WIN=21,
# making vtest much narrower than N(0,1) and the analytical bounds too wide.
# Exact output of run_order_stat_solver(21) — tau=0.0025.
# [lo, hi, med] = [tau-quantile, (1-tau)-quantile, median] of the
# r-th order statistic of 21 i.i.d. N(0,1) samples (0-indexed, r=1..21).
# vtest = (val - mu) / bvar is compared directly against these bounds.
_OS_BOUNDS_WIN21 = [
    [-3.67443197, -0.68011063, -1.84569542],
    [-2.69417301, -0.43889305, -1.41425299],
    [-2.22354567, -0.25816669, -1.14882949],
    [-1.91154057, -0.10476397, -0.94594022],
    [-1.67366065,  0.03321115, -0.77589546],
    [-1.47772155,  0.16175148, -0.62574920],
    [-1.30820664,  0.28449651, -0.48853889],
    [-1.15643839,  0.40397864, -0.35997138],
    [-1.01702011,  0.52217885, -0.23710441],
    [-0.88629759,  0.64082693, -0.11772318],
    [-0.76160100,  0.76160100,  0.00000000],
    [-0.64082693,  0.88629759,  0.11772318],
    [-0.52217885,  1.01702011,  0.23710441],
    [-0.40397864,  1.15643839,  0.35997138],
    [-0.28449651,  1.30820664,  0.48853889],
    [-0.16175148,  1.47772155,  0.62574920],
    [-0.03321115,  1.67366065,  0.77589546],
    [ 0.10476397,  1.91154057,  0.94594022],
    [ 0.25816669,  2.22354567,  1.14882949],
    [ 0.43889305,  2.69417301,  1.41425299],
    [ 0.68011063,  3.67443197,  1.84569542],
]


def _build_order_stat_bounds(win: int, tau: float = 0.0025) -> list:
    """
    Return order-statistic bounds for Stage 1 jump detection.

    Calls run_order_stat_solver(win) — exactly as the reference implementation
    does — to get the tau and 1-tau quantiles plus the median of each order
    statistic of win i.i.d. N(0,1) samples.  For win=21, tau=0.0025 the
    pre-computed table _OS_BOUNDS_WIN21 is returned immediately.

    Returns list of [lo, hi, med] for ranks 0..win-1 (0-indexed).
    """
    if win == 21 and abs(tau - 0.0025) < 1e-9:
        return [list(row) for row in _OS_BOUNDS_WIN21]

    # General case: call run_order_stat_solver exactly as the reference does.
    return _run_order_stat_solver(win, tau)


def _run_order_stat_solver(win: int, tau: float = 0.0025) -> list:
    """
    Port of run_order_stat_solver(n) from the reference implementation.
    Solves for [lo, hi, med] = [tau, 1-tau, 0.5] quantiles of the r-th
    order statistic of win i.i.d. N(0,1) samples, for r = 1..win.
    Returns a 0-indexed list of length win.
    """
    import math as _math
    import scipy.stats  as _scist
    import scipy.optimize as _sopt

    gauss = _scist.norm.cdf
    gpdf  = lambda x: _math.exp(-x**2 / 2) / (2 * _math.pi) ** 0.5

    res = []
    for r in range(1, win + 1):
        bounds = _order_stat_plot(win, r, gauss, tau)
        lo  = _solve_order_stat_quantile(win, r, gauss, gpdf, tau,     bounds[0:2])
        hi  = _solve_order_stat_quantile(win, r, gauss, gpdf, 1 - tau, bounds[4:6])
        med = _solve_order_stat_quantile(win, r, gauss, gpdf, 0.5,     bounds[2:4])
        res.append([lo, hi, med])
    return res



def _bipower_scale(window_col: np.ndarray, mu: float) -> float:
    """
    Barndorff-Nielsen & Shephard (2004) bipower variation scale estimate.

    Given a window of K returns r_j = window_col[j] - mu, estimates the
    diffusive volatility as:
        sigma* = [ sqrt(pi/2) * sum_{j=1}^{K-1} |r_j| |r_{j-1}| ]^{1/2}

    Unlike the sample standard deviation, bipower variation is robust to
    isolated large jumps in the window: a single outlier appears in at most
    two consecutive products, whereas sample variance squares the outlier.
    """
    r   = window_col - mu
    bpv = np.sqrt(np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    return float(np.sqrt(max(bpv, 1e-20)))


def _bipower_scale_ref(window_and_next: np.ndarray, mu: float) -> float:
    """
    Bipower variation scale exactly matching the reference implementation.

    Reference formula (find_jumps):
        bvar = sqrt(pi/2 * S * WIN) / (WIN-1)
             = 1.253314137 * sqrt(S * WIN) / (WIN-1)

    where S = sum_{y=i}^{i+WIN-1} |(r_y)(r_{y+1})|  — WIN products.

    The WIN is INSIDE the sqrt; (WIN-1) divides the whole expression.
    The last pair uses ds[i+WIN] — one element beyond the WIN-day window
    (look-ahead).  Therefore window_and_next must have WIN+1 elements:
    the WIN window days followed by the next day outside the window.

    mu is computed from the WIN-day window only (first WIN elements).
    Under i.i.d. N(0,sigma²), E[bvar] ≈ sigma, so vtest=(val-mu)/bvar
    is approximately standard Normal — enabling direct comparison with
    the analytical Normal order-statistic bounds.
    """
    K   = len(window_and_next) - 1          # K = WIN (window size)
    r   = window_and_next - mu              # WIN+1 residuals
    S   = float(np.sum(np.abs(r[:K] * r[1:K+1])))   # WIN products
    return float(1.253314137 * np.sqrt(max(S * K, 1e-40)) / (K - 1))


def detect_jumps(
    delta_f: np.ndarray,
    window: int = 21,
    alpha_bpv: float = 0.05,
    alpha_mah: float = 0.001,
    n_iter: int = 3,
    reg: float = 1e-6,
    use_stage2: bool = True,
) -> dict:
    """
    Jump decomposition for multivariate daily forward rate changes.

    Rather than discarding jump days, this function decomposes each daily
    change into a diffusive component and a jump component:

        delta_f(t, tau) = epsilon(t, tau)  +  J(t, tau)

    The jump J is estimated and subtracted, leaving the diffusive residual
    epsilon in place.  The output delta_f_diffusive is a FULL (T, n) matrix
    --- no rows are removed --- suitable for covariance estimation and PCA.

    Jump Estimation
    ---------------
    Stage 1 — Per-tenor bipower sliding window (Tawfik 2025 /
    Barndorff-Nielsen & Shephard 2004):
        Each tenor is treated as an INDEPENDENT time series. For every
        (t, j) pair, test whether delta_f(t, j) is a jump by comparing
        it against local bounds derived from the preceding K days:

            mu_j(t)     = window arithmetic mean
                          (centre for bipower formula, per BNS/Tawfik eq 4.3)
            med_j(t)    = window median
                          (robust diffusive level; breakdown point = 1/2)
            sigma*_j(t) = bipower variation scale centred on mu_j

        Flag (t, j) as a jump cell if delta_f(t,j) falls outside
        Phi^{-1}([alpha_bpv, 1-alpha_bpv]; mu_j, sigma*_j).

        Jump estimate:    J_hat(t,j)   = delta_f(t,j) - med_j(t)
        Diffusive residual: eps_hat(t,j) = med_j(t)

        There is no cross-tenor vote. A partial jump — e.g. an FOMC
        surprise that moves only the front end — is correctly identified
        as a jump in the short-tenor time series and clean diffusion in
        the long-tenor time series. No artificial threshold on "how many
        tenors must move" is imposed.

        Why median for the jump residual, not mean?
        The bipower formula must be centred on the mean (BNS/Tawfik).
        For the diffusive level estimate the median is preferable: with
        K=21 and 1-2 contaminated days in the window, the mean is pulled
        by up to ~10% of the jump size; the median is unaffected.

    Stage 2 — Iterative Mahalanobis winsorisation (Huber M-estimator):
        After Stage 1, delta_f_diffusive may still contain days where
        per-tenor moves were individually within their bipower bounds
        but the JOINT n-dimensional vector is anomalous.  Stage 2
        estimates the residual jump component for these days using
        Mahalanobis winsorisation and subtracts it, producing a fully
        cleaned matrix delta_f_clean2 used for BOTH PCA and Sigma_diff.

        For each day t compute:
            D_t^2 = (Y_t - mu)^T Sigma^{-1} (Y_t - mu),  Y_t = delta_f_diff[t]

        If D_t^2 > chi2_{1-alpha_mah}(n), the observation is outside the
        chi^2 confidence ellipsoid.  The jump estimate is the EXCESS
        beyond the ellipsoid boundary — the component of (Y_t - mu) that
        must be removed to bring D_t^2 exactly to the threshold:

            s_t       = sqrt(chi2_thresh / D_t^2)          (shrinkage factor, 0 < s_t < 1)
            J2_hat(t) = (Y_t - mu) * (1 - s_t)            (jump estimate)
            eps_hat(t)= mu + (Y_t - mu) * s_t = Y_t - J2_hat(t)

        This is Mahalanobis winsorisation: the DIRECTION of (Y_t - mu)
        is preserved (the cross-tenor correlation pattern of the move is
        kept); only the MAGNITUDE is shrunk back to the ellipsoid surface.
        For clean days D_t^2 <= thresh so s_t = 1, J2_hat = 0, and
        eps_hat = Y_t unchanged.

        The iteration:
          1. Estimate mu, Sigma from current delta_f_clean2 (all T rows).
          2. Compute D_t^2 for every t.
          3. Winsorise flagged rows: delta_f_clean2[t] = mu + (Y_t-mu)*s_t.
          4. Accumulate jump: jump_comp_s2[t] += J2_hat(t).
          5. Repeat until the set {t: D_t^2 > thresh} converges.

        After convergence Sigma_diff = Cov(delta_f_clean2) uses ALL T rows.
        PCA also runs on delta_f_clean2.  No rows are excluded or zeroed.

    Parameters
    ----------
    delta_f    : (T, n) ndarray  daily forward rate changes
    window     : int    sliding window length in business days (default 21)
    alpha_bpv  : float  per-tenor tail probability for bipower test (default 0.05)
    alpha_mah  : float  chi-squared tail probability for Mahalanobis (default 0.001)
    n_iter     : int    max Mahalanobis refinement iterations (default 3)
    use_stage2 : bool   run Stage-2 Mahalanobis winsorisation (default True)
    reg        : float  Tikhonov regularisation added to Sigma before inversion (default 1e-6)

    Returns
    -------
    dict with keys:
        delta_f_diffusive  : (T, n)    Stage-1 jump-cleaned changes (all T rows, PCA input)
        jump_component     : (T, n)    estimated jump J_hat; zero on clean (day,tenor) pairs
        sigma_diff         : (n, n)    Mahalanobis-refined diffusive covariance (calibration input)
        sigma_diff_idx     : (T,) bool True for rows included in sigma_diff estimation
        delta_f_diffusive  : (T, n)  Stage-1 cleaned (jump cells subtracted per tenor)
        delta_f_clean2     : (T, n)  Stage-1 + Stage-2 cleaned (PCA and Sigma_diff input)
        jump_component     : (T, n)  total jump estimate J_hat_s1 + J_hat_s2
        sigma_diff         : (n, n)  diffusive covariance from delta_f_clean2 (all T rows)
        stage1_day_mask    : (T,) bool any tenor jump-adjusted by Stage 1
        stage2_day_mask    : (T,) bool winsorised by Stage 2 (diagnostic)
        tenor_jump_mask    : (T, n) bool per-cell Stage-1 adjustment flag
        bpv_score          : (T,) float fraction of tenors with Stage-1 jump per day
        mahal_d2           : (T,) float final D_t^2 on delta_f_clean2
        chi2_threshold     : float  chi^2 critical value
        n_stage1_days      : int    days with at least one Stage-1 adjustment
        n_stage2_days      : int    days winsorised by Stage 2 (diagnostic)
    """
    T, n = delta_f.shape

    # Outputs — initialised to "no jumps"
    delta_f_diff = delta_f.copy()
    jump_comp    = np.zeros_like(delta_f)
    tenor_jmask  = np.zeros((T, n), dtype=bool)
    stage1_mask  = np.zeros(T, dtype=bool)
    bpv_score    = np.zeros(T)

    # ------------------------------------------------------------------
    # Stage 1: Per-tenor bipower sliding window with order-statistic bounds
    #
    # Reference: find_jumps() in the Tawfik implementation.
    #
    # The key insight vs a simple Normal test: each day in a sliding window
    # of size WIN occupies a particular RANK position among the WIN values.
    # The correct null distribution for the j-th smallest value in a sample
    # of WIN i.i.d. normals is the j-th ORDER STATISTIC, not Normal(0,1).
    # Using rank-specific bounds instead of symmetric Normal bounds makes
    # the test uniformly most powerful for each rank.
    #
    # Algorithm (faithfully ported from reference):
    #
    #   Pre-compute: for each rank j in {0..WIN-1}, solve for the tau and
    #   (1-tau) quantiles and the median of the j-th order statistic of
    #   WIN standard-normal samples.  Store as ords[j] = [lo, hi, med].
    #
    #   Outer loop: slide window of size WIN one step at a time.
    #   For window starting at day i (newest day = i+WIN-1):
    #
    #     For each tenor j:
    #       1. Compute mu = mean of window, bvar = bipower scale.
    #       2. Sort window days by their delta_f value → s_ds (rank-ordered).
    #       3. Inner loop over ranks k = 0..WIN-1:
    #            s   = original day index of rank-k element
    #            If cell (s, j) already flagged: skip.
    #            vtest = (value - mu) / bvar                (standardised)
    #            If vtest < ords[k][0] or vtest > ords[k][1]:  JUMP
    #              mid   = ords[k][2] * bvar + mu           (rank-k median)
    #              J_hat = value - mid
    #              record J_hat, clean delta_f_diff[s,j], flag tenor_jmask[s,j]
    #
    #   After the sliding window, any day t < WIN that was never inside a
    #   full window is handled by the warm-up MAD fallback.
    # ------------------------------------------------------------------
    # Order-statistic bounds always use tau=0.0025 (as in Tawfik reference).
    # alpha_bpv is reserved for future use / backward compatibility.
    _os_tau = 0.0025
    print(f"  [Stage 1] Pre-computing order-statistic bounds "
          f"(WIN={window}, tau={_os_tau}) ...", flush=True)
    ords = _build_order_stat_bounds(window, tau=_os_tau)
    print(f"  [Stage 1] Bounds ready. Sliding window over {T} days x {n} tenors.")

    # Outer loop: i = first day of window.  Window = delta_f[i:i+WIN].
    # The bipower formula needs one look-ahead element delta_f[i+WIN],
    # so we require i+WIN < T, i.e. i in range(0, T-WIN).
    for i in range(0, T - window):

        for j in range(n):
            # mu from WIN-day window only (matches reference)
            wj_raw = delta_f[i : i + window, j]
            mu     = wj_raw.mean()

            # bvar uses WIN+1 elements: window + one look-ahead day
            # (reference: range(i,i+WIN) uses ds[y] and ds[y+1],
            #  last pair = ds[i+WIN-1], ds[i+WIN])
            wj_plus = delta_f[i : i + window + 1, j]   # WIN+1 elements
            bvar    = _bipower_scale_ref(wj_plus, mu)

            # Sort window days by raw value, preserving (day_idx, value, flag).
            # flag is read NOW — if True in a prior window, skip this cell.
            s_ds = sorted(
                [(i + k, delta_f[i + k, j], tenor_jmask[i + k, j])
                 for k in range(window)],
                key=lambda x: x[1]
            )

            # Inner loop over ranks 0..WIN-1
            acc = 5   # decimal places for rounding (matches reference acc=5)
            for k in range(window):
                s, val, already = s_ds[k]
                if already:
                    continue               # already flagged in an earlier window
                if bvar < 1e-14:
                    continue               # degenerate window
                vtest  = (val - mu) / bvar
                lo, hi, med_os = ords[k]
                # Round before comparing — matches reference round(vtest,acc)
                if round(vtest, acc) < round(lo, acc) or round(vtest, acc) > round(hi, acc):
                    mid   = med_os * bvar + mu
                    J_hat = val - mid
                    jump_comp[s, j]    += J_hat
                    delta_f_diff[s, j]  = mid
                    tenor_jmask[s, j]   = True

    # Recompute day-level masks from the final per-cell tenor_jmask.
    # Every day 0..T-1 is covered by the main sliding window loop above
    # (each day appears as an interior element of at least one window),
    # so no warm-up fallback is needed.
    for t in range(T):
        count          = int(tenor_jmask[t].sum())
        bpv_score[t]   = count / n
        stage1_mask[t] = count > 0

    # ------------------------------------------------------------------
    # Stage 2: Iterative Mahalanobis winsorisation (Huber M-estimator)
    # Skipped entirely when use_stage2=False; outputs fall back to Stage-1 values.
    #
    # For each day t compute D_t^2 = (Y_t-mu)' Sigma^{-1} (Y_t-mu).
    # D_t^2 is ONE SCALAR PER DAY, computed independently for each t
    # using the SAME mu and Sigma from the current iteration.
    #
    # If D_t^2 > chi2_thresh, the jump estimate is the EXCESS beyond
    # the chi^2 ellipsoid:
    #   s_t        = sqrt(chi2_thresh / D_t^2)    shrinkage factor
    #   J2_hat(t)  = (Y_t - mu) * (1 - s_t)      jump component
    #   eps_hat(t) = Y_t - J2_hat(t)             winsorised residual
    #
    # This preserves the cross-tenor direction of each move (the
    # correlation pattern is unchanged); only the magnitude is shrunk
    # back to the chi^2 ellipsoid surface.
    #
    # WHY ITERATE?
    # Iteration 0 estimates Sigma from Stage-1 output including any
    # residual multivariate jumps, so D_t^2 is initially underestimated
    # for anomalous days. Winsorising them and re-estimating Sigma
    # tightens the ellipsoid, potentially exposing further borderline
    # days. Convergence (flagged set unchanged) is typically 2-3 passes.
    #
    # RESULT: delta_f_clean2 (all T rows, both Stage-1 and Stage-2
    # winsorised) is used for BOTH PCA and Sigma_diff.
    # No rows are excluded or zeroed.
    # ------------------------------------------------------------------
    chi2_thresh  = _chi2.ppf(1.0 - alpha_mah, df=n)
    stage2_mask  = np.zeros(T, dtype=bool)
    mahal_d2     = np.zeros(T)
    Sigma_diff   = np.eye(n)    # fallback; overwritten below

    # Working copy: will accumulate both Stage-1 and Stage-2 cleaning
    delta_f_clean2  = delta_f_diff.copy()
    jump_comp_s2    = np.zeros_like(delta_f)   # Stage-2 jump component only

    if use_stage2:
        prev_stage2 = np.zeros(T, dtype=bool)
        for iteration in range(n_iter):
            # Step (a): estimate mu and Sigma from ALL T rows of current clean2
            mu_c       = delta_f_clean2.mean(axis=0)   # ≈ 0 for forward rate changes
            Sigma_hat  = (delta_f_clean2 - mu_c).T @ (delta_f_clean2 - mu_c) / (T - 1)
            Sigma_inv  = np.linalg.inv(Sigma_hat + reg * np.eye(n))
            Sigma_diff = Sigma_hat

            # Step (b): compute D_t^2 for EVERY day t independently
            diff     = delta_f_clean2 - mu_c          # (T, n)
            mahal_d2 = np.einsum('ti,ij,tj->t', diff, Sigma_inv, diff)   # (T,)

            # Step (c): winsorise flagged days — shrink to chi^2 ellipsoid surface
            flagged = mahal_d2 > chi2_thresh
            if np.array_equal(flagged, prev_stage2):
                break                                  # flagged set has converged

            for t in np.where(flagged)[0]:
                s_t    = np.sqrt(chi2_thresh / mahal_d2[t])   # shrinkage factor
                y_t    = delta_f_clean2[t]
                y_new  = mu_c + (y_t - mu_c) * s_t            # winsorised residual
                j2_inc = y_t - y_new                           # incremental jump
                jump_comp_s2[t]    += j2_inc
                delta_f_clean2[t]   = y_new

            prev_stage2 = flagged.copy()

        stage2_mask = prev_stage2
    else:
        # Compute Sigma_diff from Stage-1 output only
        mu_c       = delta_f_clean2.mean(axis=0)
        Sigma_diff = (delta_f_clean2 - mu_c).T @ (delta_f_clean2 - mu_c) / (T - 1)

    # Total jump component: Stage-1 (per-tenor) + Stage-2 (multivariate winsorisation)
    jump_comp_total = jump_comp + jump_comp_s2

    return {
        "delta_f_diffusive": delta_f_diff,        # Stage-1 only (per-tenor jumps removed)
        "delta_f_clean2":    delta_f_clean2,       # Stage-1 + Stage-2 (PCA and Sigma_diff input)
        "jump_component":    jump_comp_total,      # total J_hat = J1 + J2
        "jump_component_s1": jump_comp,            # Stage-1 only
        "jump_component_s2": jump_comp_s2,         # Stage-2 winsorisation only
        "sigma_diff":        Sigma_diff,           # Cov(delta_f_clean2) using all T rows
        "stage1_day_mask":   stage1_mask,          # any tenor adjusted by Stage 1
        "stage2_day_mask":   stage2_mask,          # winsorised by Stage 2 (diagnostic)
        "tenor_jump_mask":   tenor_jmask,          # per-cell Stage-1 flag
        "bpv_score":         bpv_score,
        "mahal_d2":          mahal_d2,
        "chi2_threshold":    chi2_thresh,
        "n_stage1_days":     int(stage1_mask.sum()),
        "n_stage2_days":     int(stage2_mask.sum()),
    }

def print_jump_summary(result: dict, dates=None) -> None:
    """
    Print a readable summary of detect_jumps() output.

    Parameters
    ----------
    result : dict   output of detect_jumps()
    dates  : array-like or None   date labels aligned with delta_f rows
    """
    n1   = result["n_stage1_days"]
    n2   = result["n_stage2_days"]
    ntot = n1 + n2          # for display only — they serve different roles
    frac = ntot / len(result["delta_f_clean2"])

    print("=" * 65)
    print("JUMP DETECTION SUMMARY")
    print("=" * 65)
    T_total = len(result["delta_f_clean2"])
    print(f"  Stage 1 — per-tenor bipower     : {n1:5d} days (jump cells subtracted)")
    if n2 > 0:
        print(f"  Stage 2 — Mahalanobis winsors.  : {n2:5d} days (shrunk to chi^2 ellipsoid)")
    else:
        print(f"  Stage 2 — Mahalanobis winsors.  : skipped")
    print(f"  PCA + Sigma_diff input          : {T_total:5d} days (ALL rows, winsorised)")
    print(f"  Sigma_diff rows                  : {T_total:5d} days (ALL rows, winsorised)")
    print(f"  Chi^2 threshold (df=n)           : {result['chi2_threshold']:.2f}")
    print("-" * 65)

    # Show Stage-1 days (jumps subtracted) and Stage-2 days (excluded from Sigma) separately
    jump_days = np.where(result["stage1_day_mask"] | result["stage2_day_mask"])[0]
    if len(jump_days) > 0:
        print(f"  {'Day index':>10}  {'Date':>12}  {'BPV score':>10}  {'Mahal D^2':>11}  {'Tenors adj':>10}  Stage")
        print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*5}")
        darray = np.asarray(dates) if dates is not None else None
        for idx in jump_days:
            s1 = result["stage1_day_mask"][idx]
            s2 = result["stage2_day_mask"][idx]
            stage_str  = "1+2" if (s1 and s2) else ("1" if s1 else "2")
            date_str   = str(darray[idx]) if darray is not None else "—"
            n_adj      = int(result["tenor_jump_mask"][idx].sum())
            print(f"  {idx:>10}  {date_str:>12}  "
                  f"{result['bpv_score'][idx]:>10.3f}  "
                  f"{result['mahal_d2'][idx]:>11.1f}  "
                  f"{n_adj:>10}  "
                  f"{stage_str:>5}")
    print("=" * 65)


# =============================================================================
# Section 4 — PCA and Loading-Matrix Correlation
# =============================================================================

def run_pca(
    delta_f_clean: np.ndarray,
    n_factors: int = 5,
) -> dict:
    """
    PCA on the CORRELATION matrix of jump-cleaned daily forward rate changes.

    Rationale
    ---------
    PCA on the covariance matrix Sigma_diff conflates volatility levels with
    correlation structure: tenors with higher daily volatility dominate the
    eigenvectors, so the level PC tilts toward high-vol tenors rather than
    reflecting a pure parallel shift.  PCA on the correlation matrix R
    separates the two:

      - Eigenvectors of R  --> pure correlation shapes (level / slope /
        curvature), free of per-tenor volatility distortion.
      - Per-tenor volatilities sigma_j = sqrt(Sigma_diff[j,j])
        are re-introduced when forming the G5++ loading matrix L
        (see build_loading_correlation).

    Note: the decay rates kappa_k are NOT identifiable from PCA.
    PCA of R gives rho_kl (via the loading matrix) only.
    kappa_k are initialised separately via a log-spaced grid
    (see init_kappa_logspaced).  sigma_k are obtained from one
    inner NNLS pass at fixed kappa^{(0)}, rho^{(0)}.

    Procedure
    ---------
    1. Estimate Sigma = Cov(delta_f_clean).
    2. Form D = diag(sigma_1,...,sigma_n),  sigma_j = sqrt(Sigma[j,j]).
    3. Compute correlation matrix  R = D^{-1} Sigma D^{-1}.
    4. Eigendecompose R = V_R Lambda_R V_R^T  (all n components).
    5. Return top-K eigenvectors V_{R,K} and eigenvalues Lambda_{R,K}.

    The G5++ loading matrix is then  L = D V_{R,K} Lambda_{R,K}^{1/2}
    so that  L L^T ~= D R D = Sigma  (K-factor approximation).

    Parameters
    ----------
    delta_f_clean : (T, n) ndarray   Stage-1+2 cleaned diffusive changes
    n_factors     : int               number of components to retain (default 5)

    Returns
    -------
    dict with keys:
        eigenvalues      : (n,) eigenvalue spectrum of R (descending)
        eigenvectors     : (n, n) eigenvector matrix of R (columns)
        explained_var    : (n,) fraction of variance per component (from R)
        cumulative_var   : (n,) cumulative explained variance
        top_eigenvalues  : (n_factors,) top-K eigenvalues of R
        top_eigenvectors : (n, n_factors) top-K eigenvectors of R
        scores           : (T, n_factors) factor scores X_c @ V_{R,K}
        sigma_hat        : (n, n) sample covariance matrix Sigma_diff
        corr_matrix      : (n, n) sample correlation matrix R
        tenor_vols       : (n,) per-tenor daily volatility sqrt(Sigma[j,j])
        n_factors        : int
        n_clean_days     : int
    """
    T, n = delta_f_clean.shape

    # Step 1: sample covariance
    mu    = delta_f_clean.mean(axis=0)
    X_c   = delta_f_clean - mu
    Sigma = X_c.T @ X_c / (T - 1)

    # Step 2: per-tenor volatilities and scaling matrix
    tenor_vols = np.sqrt(np.diag(Sigma))                  # (n,)
    D_inv      = np.diag(1.0 / np.where(tenor_vols > 0, tenor_vols, 1.0))

    # Step 3: correlation matrix  R = D^{-1} Sigma D^{-1}
    R = D_inv @ Sigma @ D_inv
    np.fill_diagonal(R, 1.0)                              # enforce exact diagonal

    # Step 4: eigendecompose R (eigh: real symmetric, ascending order)
    eigenvalues, eigenvectors = eigh(R)
    eigenvalues  = eigenvalues[::-1]                      # descending
    eigenvectors = eigenvectors[:, ::-1]

    explained  = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    # Factor scores projected onto correlation eigenvectors
    scores = X_c @ eigenvectors[:, :n_factors]            # (T, n_factors)

    return {
        "eigenvalues":      eigenvalues,
        "eigenvectors":     eigenvectors,
        "explained_var":    explained,
        "cumulative_var":   cumulative,
        "top_eigenvalues":  eigenvalues[:n_factors],
        "top_eigenvectors": eigenvectors[:, :n_factors],
        "scores":           scores,
        "sigma_hat":        Sigma,
        "corr_matrix":      R,
        "tenor_vols":       tenor_vols,
        "n_factors":        n_factors,
        "n_clean_days":     T,
    }


def build_loading_correlation(
    pca_result: dict,
) -> dict:
    """
    Build the G5++ loading matrix and factor correlation matrix.

    Construction
    ------------
    PCA is performed on the correlation matrix R (see run_pca).
    The top-K eigenvectors V_{R,K} capture pure correlation shapes
    (level / slope / curvature) free of per-tenor volatility distortion.
    The G5++ loading matrix reintroduces per-tenor volatility:

        L = D * V_{R,K} * Lambda_{R,K}^{1/2}        (n x K)

    where D = diag(sigma_1,...,sigma_n) are the per-tenor daily vols
    from sqrt(Sigma_diff[j,j]).  This gives:

        L L^T  ~=  D R D  =  Sigma_diff               (K-factor approximation)

    so L[j,k] is the loading of G5++ factor k on tenor j, and the
    model forward rate covariance L L^T approximates Sigma_diff with
    correct per-tenor volatility scaling.

    For the G5++ factor correlation matrix (used for Cholesky
    decomposition of the model), extract the K x K factor block:

        L_fac    = L[:K, :]                           (K x K sub-block)
        L_fac_rn = L_fac / row_norms                  (row-normalised)
        rho      = L_fac_rn @ L_fac_rn.T              (K x K, unit diagonal)

    Row normalisation ensures rho[k,k] = 1 exactly, compensating for
    the variance not captured by the K-component truncation.

    Parameters
    ----------
    pca_result : dict   output of run_pca()

    Returns
    -------
    dict with keys:
        L          : (n, n_factors)   full loading matrix (tenor x factor)
        L_fac      : (n_factors, n_factors)  factor sub-block of L
        L_fac_rn   : (n_factors, n_factors)  row-normalised factor block
        rho        : (n_factors, n_factors)  G5++ factor correlation matrix
        row_norms  : (n_factors,) row norms of L_fac (pre-normalisation)
        var_explained : float  fraction of correlation variance in top-K
    """
    n_factors    = pca_result["n_factors"]
    eigvals_K    = pca_result["top_eigenvalues"]          # (K,)
    eigvecs_K    = pca_result["top_eigenvectors"]         # (n, K)
    tenor_vols   = pca_result["tenor_vols"]               # (n,)
    D            = np.diag(tenor_vols)                    # (n, n)

    # G5++ loading matrix: L = D V_{R,K} Lambda_{R,K}^{1/2}  shape (n, K)
    L = D @ eigvecs_K * np.sqrt(eigvals_K[np.newaxis, :])

    # Factor sub-block for correlation (first K rows of L)
    L_fac     = L[:n_factors, :]                          # (K, K)
    row_norms = np.sqrt(np.sum(L_fac ** 2, axis=1))       # (K,)

    # Row-normalise -> unit diagonal correlation matrix
    denom    = np.where(row_norms > 1e-12, row_norms, 1.0)
    L_fac_rn = L_fac / denom[:, np.newaxis]

    rho = L_fac_rn @ L_fac_rn.T
    np.fill_diagonal(rho, 1.0)   # enforce exact unit diagonal

    var_explained = eigvals_K.sum() / pca_result["eigenvalues"].sum()

    return {
        "L":              L,
        "L_fac":          L_fac,
        "L_fac_rn":       L_fac_rn,
        "rho":            rho,
        "row_norms":      row_norms,
        "var_explained":  var_explained,
    }


def init_kappa_logspaced(
    n_factors: int = 5,
    kappa_min: float = 0.05,
    kappa_max: float = 3.0,
) -> np.ndarray:
    """
    Initialise G5++ decay rates kappa_k on a log-spaced grid.

    Rationale
    ---------
    The decay rates kappa_k are NOT identifiable from PCA of the
    correlation matrix.  PCA only identifies rho_kl (via the
    row-normalised loading matrix).  sigma_k requires knowing kappa_k
    first (the loading sigma_k*exp(-kappa_k*tau) is not separable
    without kappa_k), so sigma_k is obtained from one inner NNLS pass
    at fixed kappa^{(0)} and rho^{(0)} — see g5pp_calibration.py.

    The per-tenor variance in G5++ is:
        Sigma_diff[j,j] = sum_k sigma_k^2 * exp(-2 * kappa_k * tau_j) * dt

    which is a mixture of decaying exponentials.  Recovering individual
    kappa_k from this mixture is ill-conditioned; instead we use a
    log-spaced grid spanning the full tenor range, placing one factor
    in each decay regime.  The nonlinear outer optimisation loop then
    refines all kappa_k from this starting point.

    Parameters
    ----------
    n_factors : int    number of factors K (default 5)
    kappa_min : float  slowest decay rate in yr^{-1} (default 0.05)
    kappa_max : float  fastest decay rate in yr^{-1} (default 3.0)

    Returns
    -------
    kappa0 : (n_factors,) array  initial decay rates, ascending
    """
    return np.exp(np.linspace(np.log(kappa_min), np.log(kappa_max), n_factors))


def print_pca_summary(pca_result: dict, loading_result: dict) -> None:
    """Print PCA variance decomposition and correlation matrix."""
    n_f  = pca_result["n_factors"]
    evs  = pca_result["top_eigenvalues"]
    expv = pca_result["explained_var"][:n_f]
    cumv = pca_result["cumulative_var"][:n_f]

    print("\n" + "=" * 58)
    print("PCA SUMMARY")
    print("=" * 58)
    print(f"  Clean days used : {pca_result['n_clean_days']}")
    print(f"  Tenors (n)      : {pca_result['sigma_hat'].shape[0]}")
    print(f"  Retained factors: {n_f}")
    print()
    print(f"  {'Factor':>7}  {'Eigenvalue':>12}  {'Expl. Var':>10}  {'Cumul.':>8}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*10}  {'-'*8}")
    for k in range(n_f):
        print(f"  {k+1:>7}  {evs[k]:>12.4e}  {100*expv[k]:>9.2f}%  {100*cumv[k]:>7.2f}%")
    print()
    print(f"  Variance in top-{n_f}: "
          f"{100*loading_result['var_explained']:.2f}%")
    print()
    print(f"  Row norms of L_fac (pre-normalisation):")
    for k, rn in enumerate(loading_result["row_norms"]):
        print(f"    Factor {k+1}: {rn:.4f}")
    print()
    print("  G5++ Correlation matrix R = L @ L.T:")
    rho = loading_result["rho"]
    header = "         " + "".join(f" F{k+1:>5}" for k in range(n_f))
    print(header)
    for k in range(n_f):
        row = f"  F{k+1:>5}  " + "  ".join(f"{rho[k,l]:>6.3f}" for l in range(n_f))
        print(row)
    print("=" * 58)


# =============================================================================
# Section 5 — Diagnostic Plots
# =============================================================================

def plot_jump_histograms(
    delta_f: np.ndarray,
    jump_result: dict,
    fwd_tenors: np.ndarray,
    representative_tenor: float = 2.0,
    n_bins: int = 80,
    figsize: tuple = (16, 11),
    save_path: str = None,
) -> None:
    """
    Four-panel diagnostic figure comparing daily forward rate change distributions
    before and after jump removal.

    Panels
    ------
    Top-left  : Overlaid histograms (raw vs clean) for a representative tenor,
                with fitted normal densities and excess-kurtosis annotations.
    Top-right : Normal Q-Q plot for the same tenor — raw (grey) vs clean (blue).
                A straight 45° line indicates Gaussianity; fat tails curve away.
    Bottom-left  : Per-day maximum MAD z-score distribution (Stage-1 statistic)
                   with the detection threshold marked.  Jump days shown in red.
    Bottom-right : Per-day Mahalanobis D² distribution (Stage-2 statistic) with
                   the chi² critical value and the theoretical chi² density overlaid.

    Parameters
    ----------
    delta_f              : (T, m) array   all daily forward rate changes (raw)
    jump_result          : dict           output of detect_jumps()
    fwd_tenors           : (m,) array     tenor labels for each column of delta_f
    representative_tenor : float          target tenor (years) for panels 1 & 2;
                                          nearest available tenor is used (default 2.0)
    n_bins               : int            histogram bin count (default 80)
    figsize              : tuple          figure size in inches
    save_path            : str or None    if given, save to this path (png/pdf);
                                          otherwise display interactively
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import norm as _norm, kurtosis as _kurt, chi2 as _chi2

    fwd_tenors = np.asarray(fwd_tenors)

    # Nearest available tenor to the requested representative
    col_idx   = int(np.argmin(np.abs(fwd_tenors - representative_tenor)))
    tau_label = f"{fwd_tenors[col_idx]:.2f}Y"

    jump_mask = jump_result["stage1_day_mask"]  # Stage-1 only: cells were modified
    df_raw    = delta_f[:, col_idx] * 1e4                           # raw (bps)
    df_clean  = jump_result["delta_f_clean2"][:, col_idx] * 1e4     # S1+S2 cleaned (bps)

    # ── palette ───────────────────────────────────────────────────────────────
    C_RAW   = "#9B9B9B"   # grey
    C_CLEAN = "#2166AC"   # blue
    C_JUMP  = "#D6604D"   # red
    C_THRSH = "#E08A00"   # amber — threshold lines
    C_CHI2  = "#4DAC26"   # green — chi² reference
    ALPHA_H = 0.55

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.suptitle(
        "Daily Forward Rate Changes — Before vs After Jump Subtraction\n"
        f"All {len(delta_f)} days retained  |  "
        f"{jump_result['n_stage1_days']} days had per-tenor jumps subtracted  "
        f"(+{jump_result['n_stage2_days']} excluded from Sigma_diff only)",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── Panel 1: Overlaid histograms ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    mu_c, sig_c = df_clean.mean(), df_clean.std()
    bin_lo = mu_c - 5.5 * sig_c
    bin_hi = mu_c + 5.5 * sig_c
    bins   = np.linspace(bin_lo, bin_hi, n_bins + 1)

    ax1.hist(df_raw,   bins=bins, density=True, color=C_RAW,
             alpha=ALPHA_H, label="Raw",   zorder=2, linewidth=0.3, edgecolor="white")
    ax1.hist(df_clean, bins=bins, density=True, color=C_CLEAN,
             alpha=ALPHA_H, label="Clean", zorder=3, linewidth=0.3, edgecolor="white")

    # Fitted normal densities
    x_fit = np.linspace(bin_lo, bin_hi, 300)
    ax1.plot(x_fit, _norm.pdf(x_fit, df_raw.mean(),   df_raw.std()),
             color=C_RAW,   lw=1.8, ls="--", zorder=4, label="_fitted raw")
    ax1.plot(x_fit, _norm.pdf(x_fit, mu_c, sig_c),
             color=C_CLEAN, lw=1.8, ls="--", zorder=5, label="_fitted clean")

    kurt_raw   = _kurt(df_raw,   fisher=True)
    kurt_clean = _kurt(df_clean, fisher=True)
    ax1.text(
        0.97, 0.97,
        f"Excess kurtosis\n"
        f"  Raw  : {kurt_raw:+.2f}\n"
        f"  Clean: {kurt_clean:+.2f}\n"
        f"  Normal: 0.00",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.9),
    )
    ax1.set_title(f"Histogram — Δf at {tau_label} tenor (bps)", fontsize=10)
    ax1.set_xlabel("Δf  (bps)", fontsize=9)
    ax1.set_ylabel("Density", fontsize=9)
    ax1.legend(fontsize=8.5, framealpha=0.85, loc="upper left")
    ax1.tick_params(labelsize=8)

    # ── Panel 2: Q-Q plots ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    def _qq(x):
        xs = np.sort((x - x.mean()) / x.std())
        n  = len(xs)
        return _norm.ppf((np.arange(1, n + 1) - 0.5) / n), xs

    th_r, em_r = _qq(df_raw)
    th_c, em_c = _qq(df_clean)

    ax2.scatter(th_r, em_r, color=C_RAW,   s=5,  alpha=0.45,
                label="Raw",   zorder=2, rasterized=True)
    ax2.scatter(th_c, em_c, color=C_CLEAN, s=6,  alpha=0.55,
                label="Clean", zorder=3, rasterized=True)

    lim = max(abs(th_r).max(), abs(em_r).max()) * 1.08
    ax2.plot([-lim, lim], [-lim, lim], color="black", lw=1.2,
             ls="--", zorder=4, label="N(0,1) line")

    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_title(f"Normal Q–Q — Δf at {tau_label} tenor", fontsize=10)
    ax2.set_xlabel("Theoretical N(0,1) quantile", fontsize=9)
    ax2.set_ylabel("Empirical quantile (standardised)", fontsize=9)
    ax2.legend(fontsize=8.5, framealpha=0.85)
    ax2.tick_params(labelsize=8)

    # ── Panel 3: Stage-1 bipower score ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bpv_s = jump_result["bpv_score"]
    bpv_bins = np.linspace(0, 1, n_bins + 1)

    ax3.hist(bpv_s[~jump_mask], bins=bpv_bins, color=C_CLEAN, alpha=ALPHA_H,
             label="Clean days", zorder=2, linewidth=0.3, edgecolor="white")
    ax3.hist(bpv_s[ jump_mask], bins=bpv_bins, color=C_JUMP,  alpha=0.75,
             label="Jump days",  zorder=3, linewidth=0.3, edgecolor="white")

    # bpv_score = fraction of tenors with a jump cell on this day (diagnostic)
    # No threshold line: each tenor is tested independently

    ax3.set_title("Stage 1 — Bipower sliding-window score per day", fontsize=10)
    ax3.set_xlabel(r"Fraction of tenors breaching bipower $\Phi^{-1}(\alpha_{bpv})$ bounds", fontsize=9)
    ax3.set_ylabel("Count", fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.legend(fontsize=8.5, framealpha=0.85)
    ax3.tick_params(labelsize=8)

    # ── Panel 4: Stage-2 Mahalanobis D² ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    d2      = jump_result["mahal_d2"]
    chi2_th = jump_result["chi2_threshold"]
    n_tens  = delta_f.shape[1]

    d2_cap      = max(chi2_th * 4.0,
                      np.percentile(d2[~jump_mask], 99) * 1.5 if (~jump_mask).any() else chi2_th * 4)
    d2_disp     = np.clip(d2, 0, d2_cap)
    d2_bins     = np.linspace(0, d2_cap, n_bins + 1)
    n_offscale2 = int((d2 > d2_cap).sum())

    ax4.hist(d2_disp[~jump_mask], bins=d2_bins, color=C_CLEAN, alpha=ALPHA_H,
             label="Clean days", zorder=2, linewidth=0.3, edgecolor="white")
    ax4.hist(d2_disp[ jump_mask], bins=d2_bins, color=C_JUMP,  alpha=0.75,
             label="Jump days",  zorder=3, linewidth=0.3, edgecolor="white")
    ax4.axvline(chi2_th, color=C_THRSH, lw=2.0, ls="--", zorder=5,
                label=f"$\\chi^2_{{0.999}}$({n_tens}) = {chi2_th:.0f}")

    # Theoretical chi² reference density (scaled to count units)
    n_clean_d2 = int(jump_result["chi2_threshold"] > 0)  # always all T days
    bw_d2      = d2_bins[1] - d2_bins[0]
    x_chi2     = np.linspace(max(0.1, n_tens * 0.3), d2_cap, 400)
    ax4.plot(x_chi2, _chi2.pdf(x_chi2, df=n_tens) * n_clean_d2 * bw_d2,
             color=C_CHI2, lw=1.8, ls="-", zorder=4,
             label=f"$\\chi^2$({n_tens}) reference")

    if n_offscale2 > 0:
        ax4.text(0.97, 0.91,
                 f"+{n_offscale2} obs off-scale →",
                 transform=ax4.transAxes, ha="right", va="top",
                 fontsize=8, color=C_JUMP)

    ax4.set_title("Stage 2 — Mahalanobis $D^2$ per day", fontsize=10)
    ax4.set_xlabel(
        r"$D_t^2 = \Delta f(t)^\top\,\hat\Sigma^{-1}\,\Delta f(t)$", fontsize=9)
    ax4.set_ylabel("Count", fontsize=9)
    ax4.legend(fontsize=8.5, framealpha=0.85)
    ax4.tick_params(labelsize=8)

    # ── polish ────────────────────────────────────────────────────────────────
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.5, color="#DDDDDD", zorder=0)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


# =============================================================================
# Section 6 — CSV Export and Jump Table Utilities
# =============================================================================

def save_pipeline_csvs(
    result: dict,
    output_dir: str = ".",
    prefix: str = "sofr",
    dates=None,
) -> dict:
    """
    Save forward rate levels and daily changes to CSV files.

    Files written
    -------------
    {prefix}_fwd_rates_raw.csv
        Raw instantaneous forward rates f(t, tau) in decimal.
        Rows = dates (T rows), columns = tenor grid.

    {prefix}_fwd_rates_clean.csv
        Cleaned forward rate levels: reconstructed by cumulatively summing
        the Stage-1+2 cleaned daily changes from the first raw level.
        Rows = dates (T rows), columns = tenor grid.

    {prefix}_delta_f_raw.csv
        Raw daily forward rate changes in basis points.
        Rows = change dates (T-1 rows), columns = tenor grid.

    {prefix}_delta_f_clean.csv
        Stage-1+2 cleaned daily changes in basis points.

    Parameters
    ----------
    result     : dict  output of run_pipeline()
    output_dir : str   directory to write CSV files (default: current dir)
    prefix     : str   filename prefix (default 'sofr')
    dates      : (T,) array or None  date labels for the T level rows

    Returns
    -------
    dict  {csv_filename: full_path}  for all four files written
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    fwd_tenors  = result["fwd_tenors"]                     # (m,)
    fwd_rates   = result["fwd_rates"]                      # (T, m)
    delta_f_raw = result["delta_f"]                        # (T-1, m)
    delta_f_cln = result["jump_result"]["delta_f_clean2"]  # (T-1, m)

    T   = fwd_rates.shape[0]
    T_d = delta_f_raw.shape[0]

    # Build date arrays
    if dates is not None:
        level_dates  = np.asarray(dates)[:T]
        change_dates = np.asarray(dates)[1:T]
    elif result.get("delta_dates") is not None:
        change_dates = np.asarray(result["delta_dates"])
        level_dates  = np.array([f"t{i}" for i in range(T)])
    else:
        level_dates  = np.array([f"t{i}" for i in range(T)])
        change_dates = np.array([f"t{i+1}" for i in range(T_d)])

    col_headers = [f"{tau:.4g}Y" for tau in fwd_tenors]

    def _write_csv(path, row_labels, data, col_names):
        header = "date," + ",".join(col_names)
        lines  = [header]
        for i, lbl in enumerate(row_labels):
            vals = ",".join(f"{v:.8g}" for v in data[i])
            lines.append(f"{lbl},{vals}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # Reconstruct cleaned levels: first raw level + cumsum of clean changes
    fwd_clean = np.empty_like(fwd_rates)
    fwd_clean[0] = fwd_rates[0]
    for t in range(1, T):
        fwd_clean[t] = fwd_clean[t - 1] + delta_f_cln[t - 1]

    paths = {}
    files = {
        f"{prefix}_fwd_rates_raw.csv":   (level_dates,  fwd_rates,         col_headers),
        f"{prefix}_fwd_rates_clean.csv": (level_dates,  fwd_clean,         col_headers),
        f"{prefix}_delta_f_raw.csv":     (change_dates, delta_f_raw * 1e4, col_headers),
        f"{prefix}_delta_f_clean.csv":   (change_dates, delta_f_cln * 1e4, col_headers),
    }
    for fname, (row_lbl, data, cols) in files.items():
        full = os.path.join(output_dir, fname)
        _write_csv(full, row_lbl, data, cols)
        paths[fname] = full

    return paths


def build_jump_table(
    jump_result: dict,
    fwd_tenors: np.ndarray,
    dates=None,
) -> list:
    """
    Build a structured table of all detected jump events.

    Each row represents one (date, tenor) cell where a Stage-1 bipower jump
    was subtracted or a Stage-2 Mahalanobis winsorisation occurred.
    Jump sizes are reported in basis points.

    Columns in each row dict
    ------------------------
    date            : date label of the change row
    day_index       : integer index in delta_f
    tenor_Y         : tenor in years
    tenor_index     : column index in delta_f
    stage           : '1' | '2' | '1+2'
    jump_s1_bps     : Stage-1 bipower jump component (bps)
    jump_s2_bps     : Stage-2 winsorisation component (bps)
    jump_total_bps  : total J1+J2 (bps)
    mahal_d2        : Mahalanobis D² for this day
    chi2_thresh     : chi² threshold used in Stage 2

    Parameters
    ----------
    jump_result : dict        output of detect_jumps()
    fwd_tenors  : (m,) array  tenor grid in years
    dates       : (T-1,) array or None  date labels aligned with delta_f rows

    Returns
    -------
    list of dicts (one per affected (day, tenor) cell)
    """
    fwd_tenors  = np.asarray(fwd_tenors)
    T, n        = jump_result["jump_component"].shape
    chi2_thresh = float(jump_result["chi2_threshold"])
    darray      = np.asarray(dates) if dates is not None else None

    rows = []
    for t in range(T):
        s1_day = bool(jump_result["stage1_day_mask"][t])
        s2_day = bool(jump_result["stage2_day_mask"][t])
        if not s1_day and not s2_day:
            continue

        date_lbl = str(darray[t]) if darray is not None else f"t{t}"
        d2       = float(jump_result["mahal_d2"][t])

        for j in range(n):
            j1   = float(jump_result["jump_component_s1"][t, j]) * 1e4   # bps
            j2   = float(jump_result["jump_component_s2"][t, j]) * 1e4
            jtot = j1 + j2

            cell_s1 = bool(jump_result["tenor_jump_mask"][t, j])
            cell_s2 = abs(j2) > 1e-8

            if not cell_s1 and not cell_s2:
                continue

            if cell_s1 and cell_s2:
                stage = "1+2"
            elif cell_s1:
                stage = "1"
            else:
                stage = "2"

            rows.append({
                "date":           date_lbl,
                "day_index":      int(t),
                "tenor_Y":        float(fwd_tenors[j]) if j < len(fwd_tenors) else float(j),
                "tenor_index":    int(j),
                "stage":          stage,
                "jump_s1_bps":    round(j1,   6),
                "jump_s2_bps":    round(j2,   6),
                "jump_total_bps": round(jtot, 6),
                "mahal_d2":       round(d2,   4),
                "chi2_thresh":    round(chi2_thresh, 4),
            })

    return rows


def save_jump_table_csv(jump_rows: list, path: str) -> None:
    """Write the output of build_jump_table() to a CSV file."""
    cols = ["date", "day_index", "tenor_Y", "tenor_index",
            "stage", "jump_s1_bps", "jump_s2_bps",
            "jump_total_bps", "mahal_d2", "chi2_thresh"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for row in jump_rows:
            fh.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


# =============================================================================
# Section 7 — Per-Tenor Histogram Grid
# =============================================================================

def plot_tenor_histograms(
    delta_f: np.ndarray,
    jump_result: dict,
    fwd_tenors: np.ndarray,
    n_bins: int = 50,
    max_cols: int = 4,
    figsize_per_panel: tuple = (3.8, 3.2),
    save_path: str = None,
) -> None:
    """
    Grid of per-tenor histograms: raw vs Stage-1+2 cleaned daily forward rate changes.

    One subplot per tenor.  Each panel shows:
      - Grey bars  : raw changes before jump cleaning (bps)
      - Blue bars  : Stage-1+2 cleaned changes (bps)
      - Dashed lines : fitted Gaussian density for each series
      - Annotation : excess kurtosis and std dev (raw and clean)

    The fat-tail reduction from FOMC/macro jump days is visible as a
    decrease in both excess kurtosis and standard deviation after cleaning.

    Parameters
    ----------
    delta_f           : (T, m) array   raw daily changes in decimal
    jump_result       : dict           output of detect_jumps()
    fwd_tenors        : (m,) array     tenor labels in years for each column
    n_bins            : int            histogram bins per panel (default 50)
    max_cols          : int            max subplot columns (default 4)
    figsize_per_panel : tuple          (width_in, height_in) per panel (default (3.8, 3.2))
    save_path         : str or None    save path (png/pdf) or display if None
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm as _norm, kurtosis as _kurt

    fwd_tenors = np.asarray(fwd_tenors)
    m          = delta_f.shape[1]
    df_clean   = jump_result["delta_f_clean2"]   # (T, m)

    n_cols = min(m, max_cols)
    n_rows = int(np.ceil(m / n_cols))
    fig_w  = figsize_per_panel[0] * n_cols
    fig_h  = figsize_per_panel[1] * n_rows

    C_RAW   = "#9B9B9B"
    C_CLEAN = "#2166AC"
    ALPHA_H = 0.52

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        facecolor="white",
        squeeze=False,
    )
    fig.suptitle(
        "Per-Tenor Daily Forward Rate Changes — Raw vs Cleaned (bps)\n"
        f"Grey = raw  |  Blue = cleaned  "
        f"|  Stage-1: {jump_result['n_stage1_days']} days  "
        f"|  Stage-2 winsorised: {jump_result['n_stage2_days']} days",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for idx in range(n_cols * n_rows):
        row_i = idx // n_cols
        col_i = idx %  n_cols
        ax    = axes[row_i][col_i]

        if idx >= m:
            ax.set_visible(False)
            continue

        raw_bps   = delta_f[:, idx]  * 1e4
        clean_bps = df_clean[:, idx] * 1e4
        tenor_lbl = f"{fwd_tenors[idx]:.4g}Y"

        # Bin range: wide enough to cover both distributions
        mu_c  = clean_bps.mean()
        sig_c = max(clean_bps.std(), 1e-8)
        sig_r = max(raw_bps.std(),   1e-8)
        lo    = mu_c - max(5.5 * sig_c, 4.0 * sig_r)
        hi    = mu_c + max(5.5 * sig_c, 4.0 * sig_r)
        bins  = np.linspace(lo, hi, n_bins + 1)

        ax.hist(raw_bps,   bins=bins, density=True, alpha=ALPHA_H,
                color=C_RAW,   label="Raw",   zorder=2)
        ax.hist(clean_bps, bins=bins, density=True, alpha=ALPHA_H,
                color=C_CLEAN, label="Clean", zorder=3)

        x = np.linspace(lo, hi, 300)
        ax.plot(x, _norm.pdf(x, raw_bps.mean(), sig_r),
                color=C_RAW,   lw=1.3, ls="--", alpha=0.75, zorder=4)
        ax.plot(x, _norm.pdf(x, mu_c, sig_c),
                color=C_CLEAN, lw=1.5, ls="--", zorder=5)

        kurt_raw = float(_kurt(raw_bps,   fisher=True))
        kurt_cln = float(_kurt(clean_bps, fisher=True))
        ax.text(
            0.97, 0.96,
            (f"σ raw ={sig_r:.2f} | clean={sig_c:.2f} bps\n"
             f"Kurt raw ={kurt_raw:+.1f} | clean={kurt_cln:+.1f}"),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.80),
        )

        ax.set_title(f"Tenor {tenor_lbl}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Δf (bps)", fontsize=8)
        ax.set_ylabel("Density",  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7.5, framealpha=0.85, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Section 8 — Full Pipeline
# =============================================================================

# =============================================================================
# Section 9 — Full Pipeline
# =============================================================================

def run_pipeline(
    zero_rates: np.ndarray,
    pillar_tenors: np.ndarray,
    dates=None,
    n_factors: int = 5,
    jump_kwargs: dict = None,
    verbose: bool = True,
    output_dir: str = None,
    csv_prefix: str = "sofr",
    method: str = "fd",
) -> dict:
    """
    End-to-end pipeline: zero rates -> forward rates -> jump detection -> PCA.

    Instantaneous forward rates are computed at exactly the same tenor grid
    as the input zero rates.  All four schemes share the critical anchor
    Y(0) = 0 (from P(t,t)=1), ensuring [0, tau_0] is correctly handled.
    Input grid = output forward rate grid.

    When output_dir is given the pipeline automatically writes:
      - {prefix}_zero_rates.csv          : input zero rates (copy)
      - {prefix}_fwd_rates_raw.csv       : raw instantaneous forward rates
      - {prefix}_fwd_rates_clean.csv     : cleaned forward rate levels
      - {prefix}_delta_f_raw.csv         : raw daily changes (bps)
      - {prefix}_delta_f_clean.csv       : cleaned daily changes (bps)
      - {prefix}_jump_table.csv          : per-(day,tenor) jump events
      - {prefix}_jump_diagnostic.png     : 4-panel jump diagnostic
      - {prefix}_tenor_histograms.png    : per-tenor histogram grid

    Typical usage
    -------------
        import sofr_fwd_pca as sfp

        zero_rates, pillar_tenors, dates = sfp.load_zero_rates_csv("sofr_zeros.csv")
        result = sfp.run_pipeline(
            zero_rates    = zero_rates,
            pillar_tenors = pillar_tenors,
            dates         = dates,
            output_dir    = "./outputs",
        )

    Parameters
    ----------
    zero_rates    : (T, n) array  zero rates in decimal, rows = dates,
                                  columns = pillar tenors (ascending)
    pillar_tenors : (n,) array    tenor grid in years, strictly ascending;
                                  must have same length as zero_rates columns
    dates         : (T,) array or None  date labels (strings)
    n_factors     : int           PCA components to retain (default 5)
    jump_kwargs   : dict or None  override detect_jumps() defaults:
                                  window, alpha_bpv, alpha_mah, n_iter, reg
    verbose       : bool          print step-by-step summaries (default True)
    output_dir    : str or None   if set, write all CSV and PNG outputs here
    csv_prefix    : str           filename prefix for all outputs (default "sofr")
    method        : str           zero->forward conversion scheme (default "fd"):
                                  "fd"       finite-difference, exact on [0,tau_0]
                                  "cubic"    C2 not-a-knot spline with Y(0)=0 anchor
                                  "monotone" Hyman Hermite, guarantees f>=0
                                  "flat"     piecewise-constant, exact on all intervals

    Returns
    -------
    dict with keys:
        pillar_tenors   : (n,) pillar tenor grid (= forward tenor grid)
        fwd_tenors      : (n,) same as pillar_tenors
        fwd_rates       : (T, n) raw instantaneous forward rates (decimal)
        fwd_rates_clean : (T, n) cleaned forward rate levels
        zero_rates      : (T, n) input zero rates (passed through)
        delta_f         : (T-1, n) raw daily forward rate changes (decimal)
        delta_dates     : (T-1,) dates aligned with delta_f
        jump_result     : dict  output of detect_jumps()
        jump_table      : list of dicts  one per (day, tenor) jump event
        pca_result      : dict  output of run_pca()
        loading_result  : dict  output of build_loading_correlation()
        rho             : (n_factors, n_factors) G5++ factor correlation
        L               : (n, n_factors) loading matrix D V_{R,K} Lambda_{R,K}^{1/2}
        kappa0          : (n_factors,) log-spaced kappa initialisation
        beta0           : float  beta init (moment-matched from rho_kl)
        csv_paths       : dict  {filename: full_path} for each file written
    """
    zero_rates    = np.asarray(zero_rates,    dtype=float)
    pillar_tenors = np.asarray(pillar_tenors, dtype=float)

    if zero_rates.ndim != 2:
        raise ValueError(
            f"zero_rates must be 2-D (T x n), got shape {zero_rates.shape}"
        )
    if zero_rates.shape[1] != len(pillar_tenors):
        raise ValueError(
            f"zero_rates has {zero_rates.shape[1]} columns but pillar_tenors "
            f"has {len(pillar_tenors)} entries — they must match."
        )
    min_pillars = 3 if method in ("cubic", "monotone") else 2
    if len(pillar_tenors) < min_pillars:
        raise ValueError(
            f"Method '{method}' requires at least {min_pillars} pillar tenors; "
            f"got {len(pillar_tenors)}."
        )

    # Forward tenor grid = pillar tenors exactly
    fwd_tenors = pillar_tenors.copy()

    if verbose:
        print(f"[1] Zero -> instantaneous forward rates  (method='{method}')")
        print(f"    Pillars : {len(pillar_tenors)} tenors  "
              f"[{pillar_tenors[0]:.4g}Y … {pillar_tenors[-1]:.4g}Y]")
        print(f"    Anchor  : Y(0)=0  =>  f_0 = Z_0 = {zero_rates[0,0]*100:.4f}%  "
              f"(exact on [0, {pillar_tenors[0]:.4g}Y])")
        print(f"    Output  : f(tau) at same {len(fwd_tenors)} pillar tenors")

    # ------------------------------------------------------------------
    # 1. Zero -> forward rates  (anchored at Y(0)=0 for all schemes)
    # ------------------------------------------------------------------
    fwd_rates = zero_to_fwd(pillar_tenors, zero_rates, method=method)

    # ------------------------------------------------------------------
    # 2. Daily changes
    # ------------------------------------------------------------------
    delta_f = build_delta_f(fwd_rates)
    T_level = fwd_rates.shape[0]
    T_delta = delta_f.shape[0]

    if dates is not None:
        dates_arr   = np.asarray(dates)
        delta_dates = dates_arr[1:]   # change[t] = date[t+1] - date[t]
    else:
        dates_arr   = None
        delta_dates = None

    if verbose:
        print(f"\n[2] Daily forward rate changes : {T_delta} observations  "
              f"({T_level} dates)")

    # ------------------------------------------------------------------
    # 3. Jump detection
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[3] Jump detection ...")

    kw          = jump_kwargs or {}
    jump_result = detect_jumps(delta_f, **kw)

    if verbose:
        print_jump_summary(jump_result, dates=delta_dates)

    # ------------------------------------------------------------------
    # 4. Cleaned forward rate levels  (cumulative sum of clean changes)
    # ------------------------------------------------------------------
    delta_f_cln  = jump_result["delta_f_clean2"]
    fwd_clean    = np.empty_like(fwd_rates)
    fwd_clean[0] = fwd_rates[0]
    for t in range(1, T_level):
        fwd_clean[t] = fwd_clean[t - 1] + delta_f_cln[t - 1]

    # ------------------------------------------------------------------
    # 5. Jump event table
    # ------------------------------------------------------------------
    jump_table = build_jump_table(jump_result, fwd_tenors, dates=delta_dates)

    if verbose:
        n_events   = len(jump_table)
        n_jmp_days = jump_result["n_stage1_days"] + jump_result["n_stage2_days"]
        print(f"\n[4] Jump table : {n_events} (day, tenor) cells  "
              f"across {n_jmp_days} event days")

    # ------------------------------------------------------------------
    # 6. PCA on cleaned changes
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[5] PCA on cleaned sample ...")

    pca_result     = run_pca(delta_f_cln, n_factors=n_factors)
    loading_result = build_loading_correlation(pca_result)

    if verbose:
        print_pca_summary(pca_result, loading_result)

    # ------------------------------------------------------------------
    # 7. G5++ initialisation
    # ------------------------------------------------------------------
    kappa0   = init_kappa_logspaced(n_factors=n_factors)
    rho_mat  = loading_result["rho"]
    adj_corr = np.array([rho_mat[k, k + 1] for k in range(n_factors - 1)])
    mean_adj = np.clip(adj_corr.mean(), 1e-6, 1.0 - 1e-6)
    beta0    = -np.log(mean_adj)

    if verbose:
        print(f"\n[6] G5++ initialisation:")
        print(f"    kappa0 = {np.round(kappa0, 4)}")
        print(f"    beta0  = {beta0:.4f}  "
              f"(moment-matched, mean adjacent rho = {mean_adj:.3f})")

    # ------------------------------------------------------------------
    # 8. CSV and figure outputs
    # ------------------------------------------------------------------
    csv_paths = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"\n[7] Writing outputs to: {output_dir}")

        # Minimal result dict for save_pipeline_csvs
        _tmp = {
            "fwd_tenors":  fwd_tenors,
            "fwd_rates":   fwd_rates,
            "delta_f":     delta_f,
            "delta_dates": delta_dates,
            "jump_result": jump_result,
        }
        csv_paths = save_pipeline_csvs(
            _tmp, output_dir=output_dir, prefix=csv_prefix, dates=dates_arr,
        )

        # Input zero rates (passed through unchanged)
        zr_csv = os.path.join(output_dir, f"{csv_prefix}_zero_rates.csv")
        write_zero_rates_csv(zr_csv, zero_rates, pillar_tenors, dates_arr)
        csv_paths[f"{csv_prefix}_zero_rates.csv"] = zr_csv

        # Jump table
        jump_csv = os.path.join(output_dir, f"{csv_prefix}_jump_table.csv")
        save_jump_table_csv(jump_table, jump_csv)
        csv_paths[f"{csv_prefix}_jump_table.csv"] = jump_csv

        if verbose:
            for fname in sorted(csv_paths):
                print(f"    Saved: {fname}")

        # Figure 1: four-panel jump diagnostic
        diag_png = os.path.join(output_dir, f"{csv_prefix}_jump_diagnostic.png")
        plot_jump_histograms(
            delta_f              = delta_f,
            jump_result          = jump_result,
            fwd_tenors           = fwd_tenors,
            representative_tenor = float(np.median(fwd_tenors)),
            n_bins               = 60,
            save_path            = diag_png,
        )
        csv_paths[f"{csv_prefix}_jump_diagnostic.png"] = diag_png

        # Figure 2: per-tenor histogram grid
        tenor_png = os.path.join(output_dir, f"{csv_prefix}_tenor_histograms.png")
        plot_tenor_histograms(
            delta_f     = delta_f,
            jump_result = jump_result,
            fwd_tenors  = fwd_tenors,
            n_bins      = 50,
            max_cols    = 4,
            save_path   = tenor_png,
        )
        csv_paths[f"{csv_prefix}_tenor_histograms.png"] = tenor_png

    return {
        "pillar_tenors":   pillar_tenors,
        "fwd_tenors":      fwd_tenors,
        "fwd_rates":       fwd_rates,
        "fwd_rates_clean": fwd_clean,
        "zero_rates":      zero_rates,
        "delta_f":         delta_f,
        "delta_dates":     delta_dates,
        "jump_result":     jump_result,
        "jump_table":      jump_table,
        "pca_result":      pca_result,
        "loading_result":  loading_result,
        "rho":             loading_result["rho"],
        "L":               loading_result["L"],
        "kappa0":          kappa0,
        "beta0":           beta0,
        "csv_paths":       csv_paths,
    }


# =============================================================================
# Section 6 — Self-test with synthetic data
# =============================================================================

def _make_synthetic_sofr(
    T: int = 600,
    n_jump_days: int = 12,
    seed: int = 42,
) -> tuple:
    """
    Generate a synthetic SOFR zero rate time series for testing.

    Simulates:
      - 5-factor Gaussian diffusion (level, slope, curvature, hump, ultra-short)
      - n_jump_days discrete jumps (simulating FOMC surprises) at random dates
    """
    rng = np.random.default_rng(seed)

    pillar_tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    n_pillars     = len(pillar_tenors)

    # --- Base zero curve (flat at 5%, slight inversion at short end) ---
    base_Z = (5.0 - 0.5 * np.exp(-pillar_tenors / 3)) / 100

    # --- 5-factor OU process parameters ---
    # Daily vol in rate units (bps/day ~ 5 bps/day for level factor)
    # Mean-reverting OU: dX_k = -kappa_k * X_k * dt + sigma_k * dW_k
    # where X_k are the state variables, not the rates directly.
    kappas      = np.array([0.001, 0.01, 0.05, 0.15, 0.4])   # daily mean-reversion
    sigmas_bps  = np.array([5.0,   3.0,  2.0,  1.5,  1.0])   # bps/sqrt(day)
    sigmas      = sigmas_bps * 1e-4                            # decimal

    def vol_shape(kappa, tau):
        """B_k(tau) = (1 - exp(-kappa*tau)) / kappa — factor loading shape."""
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(kappa < 1e-8, tau, (1 - np.exp(-kappa * tau)) / kappa)

    # Factor loadings on zero rates: dZ(tau) = sum_k sigma_k * B_k(tau) * dW_k
    B = np.stack([vol_shape(k, pillar_tenors) for k in kappas], axis=0)  # (5, n_pillars)
    # Normalise each row so that max|B_k| = 1 (pure shape, amplitude in sigma)
    B_norms = np.abs(B).max(axis=1, keepdims=True)
    B = B / np.where(B_norms < 1e-12, 1.0, B_norms)

    # Cholesky of simple exponential correlation  rho_kl = exp(-0.5*|k-l|)
    idx  = np.arange(5)
    Rho  = np.exp(-0.5 * np.abs(idx[:, None] - idx[None, :]))
    chol = np.linalg.cholesky(Rho)

    # --- Simulate zero rates (OU mean-reversion keeps levels realistic) ---
    Z_series = np.zeros((T, n_pillars))
    Z_series[0] = base_Z.copy()
    x_state = np.zeros(5)   # OU state variables
    for t in range(1, T):
        z      = rng.standard_normal(5)
        corr_z = chol @ z
        # OU step: dx = -kappa*x*dt + sigma*dW  (dt=1 day)
        x_state = x_state * (1 - kappas) + sigmas * corr_z
        dZ = x_state @ B          # (n_pillars,) zero rate daily change
        Z_series[t] = Z_series[t - 1] + dZ

    # Clip to positive rates
    Z_series = np.clip(Z_series, 1e-4, None)

    # --- Inject jump days ---
    # Jumps are ~8–15x the typical daily diffusive move (clearly detectable)
    jump_idx = rng.choice(T - 1, size=n_jump_days, replace=False) + 1
    for jt in jump_idx:
        level_jump = rng.choice([-1, 1]) * rng.uniform(0.0025, 0.006)   # 25–60 bps
        slope_jump = rng.uniform(-0.0015, 0.0015)
        Z_series[jt] += level_jump + slope_jump * np.exp(-pillar_tenors / 5)

    dates = np.array([f"2022-01-03 +{t}d" for t in range(T)])

    return Z_series, pillar_tenors, dates, jump_idx


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("SOFR Forward Rate PCA Correlation Calibration")
    print("=" * 70)

    OUT_DIR = "/mnt/user-data/outputs"

    # ------------------------------------------------------------------
    # Accept an optional CSV path as the first command-line argument.
    # If none is given, generate synthetic data, write it to a temp CSV,
    # then load it back through load_zero_rates_csv — so the same code
    # path is exercised as with a real input file.
    # ------------------------------------------------------------------
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        PREFIX   = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\nInput CSV  : {csv_path}")

    else:
        # --- Generate synthetic SOFR zero rate series ---
        print("\nNo CSV supplied — generating synthetic data (T=600, 12 jump days)...")
        Z_series, pillar_tenors_synth, dates_synth, true_jump_idx = \
            _make_synthetic_sofr(T=600, n_jump_days=12, seed=42)

        csv_path = os.path.join(OUT_DIR, "sofr_synthetic_zeros.csv")
        os.makedirs(OUT_DIR, exist_ok=True)
        write_zero_rates_csv(csv_path, Z_series, pillar_tenors_synth, dates_synth)
        print(f"Synthetic CSV written : {csv_path}")
        print(f"  Shape : {Z_series.shape[0]} dates x {Z_series.shape[1]} tenors")
        print(f"  Tenors: {pillar_tenors_synth.tolist()}")
        PREFIX = "sofr_synthetic"
        true_jump_idx_main = true_jump_idx  # keep for accuracy check below

    # ------------------------------------------------------------------
    # Load the CSV
    # ------------------------------------------------------------------
    print(f"\nLoading zero rates from: {csv_path}")
    zero_rates, pillar_tenors, dates = load_zero_rates_csv(csv_path)
    print(f"  Loaded : {zero_rates.shape[0]} dates x {zero_rates.shape[1]} tenors")
    print(f"  Tenors : {pillar_tenors.tolist()}")
    print(f"  Dates  : {dates[0]} … {dates[-1]}")
    print(f"  Rates range : [{zero_rates.min()*100:.3f}%, {zero_rates.max()*100:.3f}%]")



    # ------------------------------------------------------------------
    # Area-preservation check — all four schemes, all intervals including [0, tau_0]
    # This verifies that every scheme correctly handles P(t,t)->P(t,t+tau_0)
    # via the Y(0)=0 anchor, and that inter-pillar errors are within tolerance.
    # ------------------------------------------------------------------
    print("\n--- Area preservation check (first date, all four schemes) ---")
    print(f"  {'Scheme':<10}  {'[0,tau_0]':>11}  {'Max inter-pillar':>17}  {'Passed':>6}")
    print(f"  {'-'*10}  {'-'*11}  {'-'*17}  {'-'*6}")
    _tols = {"fd": 1e-2, "cubic": 1e-3, "monotone": 1e-2, "flat": 1e-12}
    for _m in ("fd", "cubic", "monotone", "flat"):
        _fc = zero_to_fwd(pillar_tenors, zero_rates[0], method=_m)
        _ap = check_area_preservation(pillar_tenors, zero_rates[0], _fc, tol=_tols[_m], method=_m)
        _e0  = _ap['errors'][0]          # [0, tau_0] interval
        _emax = max(_ap['errors'][1:])    # worst inter-pillar interval
        _ok   = _ap['passed']
        print(f"  {_m:<10}  {_e0:>11.2e}  {_emax:>17.2e}  {'OK' if _ok else 'FAIL':>6}")
    print()
    # Run the full pipeline
    # ------------------------------------------------------------------
    print()
    result = run_pipeline(
        zero_rates    = zero_rates,
        pillar_tenors = pillar_tenors,
        dates         = dates,
        n_factors     = 5,
        jump_kwargs   = {"window": 21, "alpha_bpv": 0.05, "alpha_mah": 0.001},
        verbose       = True,
        output_dir    = OUT_DIR,
        csv_prefix    = PREFIX,
    )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    jd = result["jump_result"]
    max_resid = np.abs(
        result["delta_f"] - jd["delta_f_clean2"] - jd["jump_component"]
    ).max()
    print(f"\nDecomposition check:")
    print(f"  delta_f = clean + jump   |  max residual : {max_resid:.2e}  (should be ~0)")

    level_diff = np.abs(result["fwd_rates"] - result["fwd_rates_clean"]).max()
    print(f"  Max level diff raw vs clean : {level_diff * 1e4:.2f} bps")

    # If synthetic data, report detection accuracy
    if len(sys.argv) < 2:
        detected  = np.where(jd["stage1_day_mask"])[0]
        true_in_d = true_jump_idx_main - 1
        true_in_d = true_in_d[
            (true_in_d >= 0) & (true_in_d < len(result["delta_f"]))
        ]
        recovered = np.intersect1d(detected, true_in_d)
        false_pos = np.setdiff1d(detected, true_in_d)
        print(f"\nJump detection accuracy (synthetic ground truth):")
        print(f"  True jumps injected : {len(true_in_d)}")
        print(f"  Detected correctly  : {len(recovered)} / {len(true_in_d)}")
        print(f"  False positives     : {len(false_pos)}")

    # ------------------------------------------------------------------
    # Jump table sample
    # ------------------------------------------------------------------
    jt = result["jump_table"]
    print(f"\nJump table : {len(jt)} (day, tenor) events  "
          f"[first 12 event days shown]")
    print(f"  {'date':>22}  {'tenor':>6}  {'stage':>5}  "
          f"{'J1(bps)':>10}  {'J2(bps)':>10}  {'Total(bps)':>11}  {'D²':>8}")
    print("  " + "-" * 83)
    shown = set()
    for row in jt:
        d = row["day_index"]
        if d not in shown:
            shown.add(d)
            print(f"  {row['date']:>22}  {row['tenor_Y']:>6.4g}Y  {row['stage']:>5}  "
                  f"{row['jump_s1_bps']:>+10.3f}  {row['jump_s2_bps']:>+10.3f}  "
                  f"{row['jump_total_bps']:>+11.3f}  {row['mahal_d2']:>8.1f}")
        if len(shown) >= 12:
            remaining = len({r['day_index'] for r in jt}) - 12
            if remaining > 0:
                print(f"  … {remaining} more event days — see {PREFIX}_jump_table.csv")
            break

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------
    print(f"\nG5++ PCA-implied correlation matrix ({result['rho'].shape[0]}x{result['rho'].shape[0]}):")
    rho    = result["rho"]
    K      = rho.shape[0]
    labels = [f"F{k+1}" for k in range(K)]
    print("        " + "".join(f"  {lb:>7}" for lb in labels))
    for k in range(K):
        print(f"  {labels[k]:>5}  " + "  ".join(f"{rho[k, l]:>7.3f}" for l in range(K)))

    # ------------------------------------------------------------------
    # Files written
    # ------------------------------------------------------------------
    print(f"\nFiles written to {OUT_DIR}/:")
    for fname in sorted(result["csv_paths"]):
        sz = os.path.getsize(result["csv_paths"][fname])
        unit = "KB" if sz >= 1024 else "B"
        val  = sz // 1024 if sz >= 1024 else sz
        print(f"  {fname:<50}  ({val:,} {unit})")
