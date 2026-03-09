"""
G5++ Gaussian HJM Swaption Calibration
========================================
Implements the full calibration of the G5++ model to a USD SOFR ATM
normal (Bachelier) swaption implied volatility matrix.

Model:
    df(t,T) = alpha(t,T) dt + sum_k sigma_k exp(-kappa_k(T-t)) dW_k(t)
    r(t)    = sum_k x_k(t) + phi(t)
    dx_k    = -kappa_k x_k dt + sigma_k dW_k,  x_k(0) = 0
    dW_k dW_l = rho_kl dt

Correlation modes:
    'exponential'  rho_kl = exp(-beta*|k-l|), beta calibrated.  11 parameters.
    'pca'          rho_kl from PCA of historical delta_f (data-driven, no
                   exponential-decay assumption on tenor correlations).  10 parameters.
    'identity'     rho_kl = delta_kl.  10 parameters.

Initialisation:
    rho_kl  <- PCA of correlation matrix of cleaned historical forward rate changes
    beta    <- moment-matched from adjacent PCA-implied rho_kl
    kappa_k <- log-spaced grid [0.05, 3.0] yr^{-1}  (NOT from PCA)
    sigma_k <- one inner NNLS pass at fixed kappa^{(0)}, rho^{(0)}

Bond price:
    P(t,T) = P(0,T)/P(0,t) * exp(-sum_k B_k(t,T)*x_k(t)
                                  + 0.5*sum_{kl} rho_kl*sigma_k*sigma_l*Gamma_kl(t,T))

Required inputs
---------------
P0              : callable  P(0, T) -> float  (or array-valued)
market_vols_bps : dict  {(T_e, T_s): sigma_n_bps}  ATM normal vols in bps
delta_f         : (T, n) ndarray  historical daily forward rate changes (for PCA rho)

Usage example
-------------
    from g5pp_calibration import G5ppCalibrator
    import numpy as np

    def P(T):
        return np.exp(-0.05 * np.asarray(T))

    market_vols = {
        (1, 1): 85.0, (1, 2): 90.0, (1, 5): 95.0, (1, 10): 88.0,
        (2, 1): 82.0, (2, 2): 86.0, (2, 5): 91.0, (2, 10): 85.0,
        (5, 1): 70.0, (5, 2): 74.0, (5, 5): 79.0, (5, 10): 75.0,
       (10, 1): 58.0,(10, 2): 61.0,(10, 5): 65.0,(10, 10): 62.0,
    }

    calib = G5ppCalibrator(P, market_vols, delta=0.5)
    result = calib.calibrate()
    print(result)
"""

import numpy as np
from scipy.optimize import nnls, minimize
from scipy.linalg import eigh
from scipy.stats import chi2 as _chi2
import warnings


# =============================================================================
# Section 1 – Model building blocks
# =============================================================================

def B_k(kappa: float, tau: np.ndarray) -> np.ndarray:
    """
    Mean-reversion integral B_k(t, T) = (1 - exp(-kappa*(T-t))) / kappa.

    Parameters
    ----------
    kappa : float  > 0, mean-reversion speed for factor k
    tau   : array  T - t  (time to maturity), >= 0

    Returns
    -------
    array  same shape as tau
    """
    tau = np.asarray(tau, dtype=float)
    # Stable for small kappa
    return np.where(
        np.abs(kappa) < 1e-10,
        tau,
        (1.0 - np.exp(-kappa * tau)) / kappa
    )


def Gamma_kl(kappa_k: float, kappa_l: float, tau: np.ndarray) -> np.ndarray:
    """
    Variance integral Gamma_kl(t, T) = int_t^T B_k(t,u) * B_l(t,u) du
    where tau = T - t.

    Closed form:
        Gamma_kl = (1 - exp(-(kk+kl)*tau)) / (kk*kl*(kk+kl))
                 - B_k / (kl*(kk+kl))
                 - B_l / (kk*(kk+kl))
                 + tau / (kk*kl)
    """
    tau = np.asarray(tau, dtype=float)
    kk, kl = kappa_k, kappa_l
    Bk = B_k(kk, tau)
    Bl = B_k(kl, tau)
    ksum = kk + kl
    term1 = (1.0 - np.exp(-ksum * tau)) / (kk * kl * ksum)
    term2 = Bk / (kl * ksum)
    term3 = Bl / (kk * ksum)
    term4 = tau / (kk * kl)
    return term1 - term2 - term3 + term4


# =============================================================================
# Jump Detection — Robust Pre-processing for PCA
# =============================================================================

def detect_jumps(
    delta_f: np.ndarray,
    window: int = 21,
    alpha_bpv: float = 0.05,
    alpha_mah: float = 0.001,
    n_iter: int = 3,
    reg: float = 1e-6,
) -> dict:
    """
    Two-stage jump decomposition for multivariate daily forward rate changes.

    Decomposes each daily change into diffusive and jump components:
        delta_f(t, tau) = epsilon(t, tau) + J1(t, tau) + J2(t, tau)

    All T rows are retained; jump components are subtracted / winsorised.

    Stage 1 — Per-tenor bipower sliding window (Barndorff-Nielsen & Shephard 2004)
        Each tenor is treated independently.  For window of K days preceding t:
          mu_j   = bipower mean (robust to jumps in window)
          m_j    = window median (robust centre for jump residual)
          sigma*_j = [sqrt(pi/2) * sum |r_k||r_{k-1}|]^{1/2}  (bipower scale)
        Flag cell (t,j) if |delta_f(t,j) - mu_j| > z_{1-alpha/2} * sigma*_j.
        Jump estimate: J1(t,j) = delta_f(t,j) - m_j.
        Diffusive residual: epsilon_hat(t,j) = m_j.

    Stage 2 — Iterative Mahalanobis winsorisation (Huber M-estimator)
        On the Stage-1 cleaned matrix (all T rows):
          mu_hat  ~ 0,  Sigma_hat = Cov(delta_f_clean2)
          D_t^2 = (Y_t - mu_hat)^T (Sigma_hat + reg*I)^{-1} (Y_t - mu_hat)
        Flag day t if D_t^2 > chi2.ppf(1 - alpha_mah, df=n).
        Winsorise: shrink radially to chi^2 ellipsoid surface:
          s_t = sqrt(chi2_thresh / D_t^2)  in (0,1)
          J2(t) += (Y_t - mu_hat)(1 - s_t)   (accumulated, +=)
          Y_t  <- mu_hat + (Y_t - mu_hat)*s_t
        Re-estimate Sigma_hat from ALL T rows (including winsorised);
        iterate until flagged set converges (typically 2-3 passes).

    Parameters
    ----------
    delta_f   : (T, n) ndarray  daily forward rate changes
    window    : int    bipower sliding window length (default 21)
    alpha_bpv : float  bipower tail probability for Stage-1 flag (default 0.05)
    alpha_mah : float  chi-squared tail probability for Stage-2 (default 0.001)
    n_iter    : int    max Mahalanobis iterations (default 3)
    reg       : float  Tikhonov regularisation for Sigma inversion (default 1e-6)

    Returns
    -------
    dict with keys:
        delta_f_diffusive : (T, n)  Stage-1 cleaned (J1 subtracted per cell)
        delta_f_clean2    : (T, n)  Stage-1 + Stage-2 cleaned (PCA/Sigma input)
        jump_component    : (T, n)  total J1 + J2
        jump_component_s1 : (T, n)  Stage-1 jump only
        jump_component_s2 : (T, n)  Stage-2 winsorisation component only
        sigma_diff        : (n, n)  Cov(delta_f_clean2) using all T rows
        stage1_day_mask   : (T,) bool  any cell adjusted by Stage 1
        stage2_day_mask   : (T,) bool  winsorised by Stage 2
        tenor_jump_mask   : (T, n) bool  per-cell Stage-1 flag
        mahal_d2          : (T,)  final Mahalanobis D^2 per day
        mahal_d2_raw      : (T,)  D^2 computed on raw delta_f (diagnostic)
        chi2_threshold    : float  chi^2 critical value used in Stage 2
        n_stage1_days     : int   days with at least one Stage-1 cell adjusted
        n_stage2_days     : int   days winsorised by Stage 2
    """
    from scipy.stats import chi2 as _chi2_dist
    from scipy.stats import norm as _norm

    T, n = delta_f.shape
    z_bpv = _norm.ppf(1.0 - alpha_bpv / 2.0)

    delta_f_diff = delta_f.copy()
    jump_comp    = np.zeros_like(delta_f)
    stage1_mask  = np.zeros(T, dtype=bool)
    tenor_jmask  = np.zeros((T, n), dtype=bool)
    bpv_score    = np.zeros(T)

    # Stage 1: per-tenor bipower sliding window
    for j in range(n):
        col = delta_f[:, j]
        for t in range(T):
            lo = max(0, t - window)
            win = col[lo:t] if t > 0 else col[0:1]
            if len(win) < 2:
                mu_j = col[t]
                m_j  = col[t]
                s_j  = abs(col[t]) if abs(col[t]) > 1e-20 else 1e-20
            else:
                r     = win - win.mean()
                bpv   = np.sqrt(np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
                s_j   = float(np.sqrt(max(bpv, 1e-20)))
                mu_j  = win.mean()
                m_j   = np.median(win)
            if abs(delta_f[t, j] - mu_j) > z_bpv * s_j:
                J_hat = delta_f[t, j] - m_j
                jump_comp[t, j]    = J_hat
                delta_f_diff[t, j] = m_j
                tenor_jmask[t, j]  = True
        stage1_mask |= tenor_jmask[:, j]

    for t in range(T):
        bpv_score[t] = tenor_jmask[t].sum() / n

    # Stage 2: iterative Mahalanobis winsorisation
    chi2_thresh  = _chi2_dist.ppf(1.0 - alpha_mah, df=n)
    stage2_mask  = np.zeros(T, dtype=bool)
    mahal_d2_raw = np.zeros(T)
    mahal_d2     = np.zeros(T)
    Sigma_diff   = np.eye(n)

    try:
        mu_raw  = delta_f.mean(0)
        Sig_raw = (delta_f - mu_raw).T @ (delta_f - mu_raw) / (T - 1)
        Sinv_r  = np.linalg.inv(Sig_raw + reg * np.eye(n))
        diff_r  = delta_f - mu_raw
        mahal_d2_raw = np.einsum('ti,ij,tj->t', diff_r, Sinv_r, diff_r)
    except np.linalg.LinAlgError:
        pass

    delta_f_clean2 = delta_f_diff.copy()
    jump_comp_s2   = np.zeros_like(delta_f)
    prev_stage2    = np.zeros(T, dtype=bool)

    for iteration in range(n_iter):
        mu_c      = delta_f_clean2.mean(axis=0)
        Sigma_hat = (delta_f_clean2 - mu_c).T @ (delta_f_clean2 - mu_c) / (T - 1)
        Sigma_inv = np.linalg.inv(Sigma_hat + reg * np.eye(n))
        Sigma_diff = Sigma_hat

        diff     = delta_f_clean2 - mu_c
        mahal_d2 = np.einsum('ti,ij,tj->t', diff, Sigma_inv, diff)

        flagged = mahal_d2 > chi2_thresh
        if np.array_equal(flagged, prev_stage2):
            break

        for t in np.where(flagged)[0]:
            s_t   = np.sqrt(chi2_thresh / mahal_d2[t])
            y_t   = delta_f_clean2[t]
            y_new = mu_c + (y_t - mu_c) * s_t
            jump_comp_s2[t]   += y_t - y_new
            delta_f_clean2[t]  = y_new

        prev_stage2 = flagged.copy()

    stage2_mask = prev_stage2

    return {
        "delta_f_diffusive": delta_f_diff,
        "delta_f_clean2":    delta_f_clean2,
        "jump_component":    jump_comp + jump_comp_s2,
        "jump_component_s1": jump_comp,
        "jump_component_s2": jump_comp_s2,
        "sigma_diff":        Sigma_diff,
        "stage1_day_mask":   stage1_mask,
        "stage2_day_mask":   stage2_mask,
        "tenor_jump_mask":   tenor_jmask,
        "bpv_score":         bpv_score,
        "mahal_d2":          mahal_d2,
        "mahal_d2_raw":      mahal_d2_raw,
        "chi2_threshold":    chi2_thresh,
        "n_stage1_days":     int(stage1_mask.sum()),
        "n_stage2_days":     int(stage2_mask.sum()),
        # legacy keys for backward compatibility
        "delta_f_clean":     delta_f_clean2,
        "clean_idx":         np.where(~stage2_mask)[0],
        "jump_idx":          np.where(stage2_mask)[0],
        "n_jumps":           int(stage2_mask.sum()),
        "jump_fraction":     float(stage2_mask.mean()),
    }


def jump_detection_summary(result: dict, dates=None) -> None:
    """Print a human-readable summary of detect_jumps() output."""
    n1 = result["n_stage1_days"]
    n2 = result["n_stage2_days"]
    T  = len(result["delta_f_clean2"])
    print("=" * 60)
    print("JUMP DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Stage 1 — per-tenor bipower     : {n1:5d} days (cells adjusted)")
    print(f"  Stage 2 — Mahalanobis winsors.  : {n2:5d} days (shrunk to chi^2 ellipsoid)")
    print(f"  PCA + Sigma_diff input          : {T:5d} days (ALL rows, winsorised)")
    print(f"  Chi^2 threshold (df=n)          : {result['chi2_threshold']:.2f}")
    print("=" * 60)


def pca_loading_correlation(
    delta_f: np.ndarray,
    n_factors: int = 5,
    remove_jumps: bool = True,
    jump_kwargs: dict = None,
    verbose: bool = False,
) -> tuple:
    """
    PCA-implied G5++ factor correlation matrix from historical forward rate changes.

    Procedure
    ---------
    1. Optional jump decomposition (remove_jumps=True):
       Two-stage bipower + Mahalanobis winsorisation; all T rows retained
       with jump components subtracted / winsorised in place.
    2. PCA on the CORRELATION matrix R = D^{-1} Sigma_diff D^{-1},
       where D = diag(sqrt(Sigma_diff[j,j])) are per-tenor daily vols.
       PCA on R gives pure correlation shapes free of vol distortion.
       PCA on Sigma_diff directly would tilt eigenvectors toward high-vol
       tenors (typically short end), distorting the loading matrix.
    3. Loading matrix L = D V_{R,K} Lambda_{R,K}^{1/2}  (n x K)
       so that L L^T ~= Sigma_diff (K-factor approximation).
    4. Factor correlation rho = L_fac_rn @ L_fac_rn.T  (unit diagonal).

    Note
    ----
    PCA identifies rho_kl only.  kappa_k and sigma_k are NOT identifiable
    from PCA — use init_kappa_logspaced() and one inner NNLS pass.

    Parameters
    ----------
    delta_f     : (T, n) array  historical daily forward rate changes
    n_factors   : int           PCA components to retain (default 5)
    remove_jumps: bool          run jump decomposition before PCA (default True)
    jump_kwargs : dict or None  override defaults in detect_jumps()
    verbose     : bool          print jump summary (default False)

    Returns
    -------
    rho         : (n_factors, n_factors) ndarray  PSD correlation matrix, unit diagonal
    L           : (n, n_factors) ndarray  loading matrix D V_{R,K} Lambda_{R,K}^{1/2}
    jump_result : dict or None  output of detect_jumps(), or None if skipped
    """
    jump_result = None
    if remove_jumps:
        kw = jump_kwargs or {}
        jump_result = detect_jumps(delta_f, **kw)
        if verbose:
            jump_detection_summary(jump_result)
        delta_f_clean = jump_result["delta_f_clean2"]
    else:
        delta_f_clean = delta_f

    T, n = delta_f_clean.shape

    # Sample covariance of cleaned changes
    mu = delta_f_clean.mean(axis=0)
    X_c = delta_f_clean - mu
    Sigma = X_c.T @ X_c / (T - 1)

    # Per-tenor vols and correlation matrix R = D^{-1} Sigma D^{-1}
    tenor_vols = np.sqrt(np.diag(Sigma))
    D_inv = np.diag(1.0 / np.where(tenor_vols > 0, tenor_vols, 1.0))
    R = D_inv @ Sigma @ D_inv
    np.fill_diagonal(R, 1.0)

    # Eigendecompose R (eigh returns ascending order; reverse)
    eigenvalues, eigenvectors = eigh(R)
    eigenvalues  = eigenvalues[::-1][:n_factors]
    eigenvectors = eigenvectors[:, ::-1][:, :n_factors]

    # Loading matrix: L = D V_{R,K} Lambda_{R,K}^{1/2}  shape (n, K)
    D = np.diag(tenor_vols)
    L = D @ eigenvectors * np.sqrt(eigenvalues[np.newaxis, :])

    # Factor sub-block and row-normalisation
    L_fac     = L[:n_factors, :]                         # (K, K)
    row_norms = np.sqrt(np.sum(L_fac ** 2, axis=1, keepdims=True))
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    L_fac_rn  = L_fac / row_norms

    rho = L_fac_rn @ L_fac_rn.T
    np.fill_diagonal(rho, 1.0)

    return rho, L, jump_result


def correlation_matrix(
    beta: float = None,
    n_factors: int = 5,
    rho_mode: str = "exponential",
    delta_f: np.ndarray = None,
) -> np.ndarray:
    """
    Correlation matrix for the G5++ Brownian motions.

    Three modes:

    'exponential'  (default)
        rho_kl = exp(-beta * |k - l|),  beta > 0.  Guaranteed PSD.
        Parameter vector: Theta in R^11.

    'pca'
        rho_kl = (L @ L.T)_kl  where L is the row-normalised PCA loading matrix.
        dW_k = sum_j l_kj * d~W_j  (d~W_j independent, from PCA score directions).
        Row-normalised so sum_j l_kj^2 = 1 => unit diagonal.
        Requires delta_f. Parameter vector: Theta in R^10 (no beta).

    'identity'
        rho_kl = delta_kl  (R = I_5).  Limiting / default fallback.
        Parameter vector: Theta in R^10 (no beta).

    Parameters
    ----------
    beta      : float or None   required for 'exponential'
    n_factors : int             number of factors (default 5)
    rho_mode  : str             'exponential', 'pca', or 'identity'
    delta_f   : (T,n) array or None  required for 'pca'

    Returns
    -------
    rho : (n_factors, n_factors) ndarray
    """
    if rho_mode == "identity":
        return np.eye(n_factors)
    elif rho_mode == "exponential":
        if beta is None:
            raise ValueError("beta must be provided for rho_mode='exponential'")
        idx = np.arange(n_factors)
        return np.exp(-beta * np.abs(idx[:, None] - idx[None, :]))
    elif rho_mode == "pca":
        if delta_f is None:
            raise ValueError("delta_f must be provided for rho_mode='pca'")
        rho, _, _ = pca_loading_correlation(delta_f, n_factors)
        return rho
    else:
        raise ValueError(
            f"Unknown rho_mode '{rho_mode}'. "
            "Use 'exponential', 'pca', or 'identity'."
        )


# =============================================================================
# Section 2 – Bond pricing
# =============================================================================

def bond_price(
    t: float,
    T: float,
    x: np.ndarray,
    sigmas: np.ndarray,
    kappas: np.ndarray,
    rho: np.ndarray,
    P0: callable,
) -> float:
    """
    Affine bond price P(t, T) under G5++.

    P(t,T) = P(0,T)/P(0,t) * exp(-sum_k B_k(t,T)*x_k
                                   + 0.5*sum_{kl} rho_kl*sigma_k*sigma_l*Gamma_kl(t,T))

    Parameters
    ----------
    t, T     : floats  current time and maturity
    x        : (5,) array  state variables x_k(t)
    sigmas   : (5,) array  volatility parameters
    kappas   : (5,) array  mean-reversion speeds
    rho      : (5,5) array  correlation matrix
    P0       : callable  P0(T) -> float, initial discount curve

    Returns
    -------
    float  bond price P(t, T)
    """
    tau = T - t
    Bvec = np.array([B_k(kappas[k], tau) for k in range(5)])

    # Variance correction: +0.5 * sum_{kl} rho_kl * sigma_k * sigma_l * Gamma_kl
    variance_correction = 0.0
    for k in range(5):
        for l in range(5):
            variance_correction += (
                rho[k, l] * sigmas[k] * sigmas[l] * Gamma_kl(kappas[k], kappas[l], tau)
            )
    variance_correction *= 0.5

    exponent = -np.dot(Bvec, x) + variance_correction
    return (P0(T) / P0(t)) * np.exp(exponent)


# =============================================================================
# Section 3 – Forward swap rate and hedge ratios
# =============================================================================

def annuity(
    Te: float,
    Ts: float,
    delta: float,
    P0: callable,
) -> float:
    """
    Initial annuity A(0; Te, Ts) = sum_{i=1}^n delta_i * P(0, Te + i*delta).

    Parameters
    ----------
    Te    : float  swaption expiry in years
    Ts    : float  swap tenor in years
    delta : float  accrual fraction (0.5 semi-annual, 0.25 quarterly)
    P0    : callable  P0(T) -> float

    Returns
    -------
    float  A0
    """
    n = round(Ts / delta)
    payment_times = Te + np.arange(1, n + 1) * delta
    return delta * np.sum(P0(payment_times))


def forward_swap_rate(
    Te: float,
    Ts: float,
    delta: float,
    P0: callable,
) -> float:
    """
    ATM forward swap rate F(0; Te, Ts) = (P(0,Te) - P(0,Te+Ts)) / A(0;Te,Ts).
    """
    A0 = annuity(Te, Ts, delta, P0)
    return (P0(Te) - P0(Te + Ts)) / A0


def hedge_ratios(
    Te: float,
    Ts: float,
    delta: float,
    kappas: np.ndarray,
    P0: callable,
) -> np.ndarray:
    """
    Hedge ratios h_k = dF/dx_k at x=0.

    h_k = (1/A0) * [ -B_k(0,Te)*P(0,Te)
                      + B_k(0,Te+Ts)*P(0,Te+Ts)
                      + F0 * sum_i delta_i * B_k(0,Te+Ti) * P(0,Te+Ti) ]

    Parameters
    ----------
    Te, Ts  : floats  expiry and tenor
    delta   : float   accrual fraction
    kappas  : (5,) array
    P0      : callable

    Returns
    -------
    h : (5,) array
    """
    n = round(Ts / delta)
    payment_times = Te + np.arange(1, n + 1) * delta
    A0 = delta * np.sum(P0(payment_times))
    F0 = (P0(Te) - P0(Te + Ts)) / A0

    h = np.zeros(5)
    for k in range(5):
        # dN/dx_k = -B_k(0,Te)*P(0,Te) + B_k(0,Te+Ts)*P(0,Te+Ts)
        dN = -B_k(kappas[k], Te) * P0(Te) + B_k(kappas[k], Te + Ts) * P0(Te + Ts)
        # -F0 * dA/dx_k = F0 * sum_i delta_i * B_k(0,Te+Ti) * P(0,Te+Ti)
        dA_correction = F0 * delta * np.sum(
            B_k(kappas[k], payment_times) * P0(payment_times)
        )
        h[k] = (dN + dA_correction) / A0
    return h


# =============================================================================
# Section 4 – Normal (Bachelier) swaption volatility
# =============================================================================

def model_normal_vol(
    Te: float,
    Ts: float,
    delta: float,
    sigmas: np.ndarray,
    kappas: np.ndarray,
    beta: float,
    P0: callable,
    rho_mode: str = "exponential",
    delta_f: np.ndarray = None,
) -> float:
    """
    Closed-form G5++ normal (Bachelier) implied volatility.

    sigma_n = sqrt(V_F / Te)

    where V_F = sum_{kl} rho_kl * sigma_k * sigma_l * h_k * h_l
                         * (1 - exp(-(kappa_k+kappa_l)*Te)) / (kappa_k+kappa_l)

    Parameters
    ----------
    Te, Ts   : floats  expiry and tenor in years
    delta    : float   accrual fraction
    sigmas   : (5,) array  vol parameters
    kappas   : (5,) array  mean-reversion speeds
    beta     : float or None  correlation decay (required for 'exponential')
    P0       : callable
    rho_mode : str  'exponential', 'pca', or 'identity'
    delta_f  : (T,n) array or None  required when rho_mode='pca'

    Returns
    -------
    float  normal vol in ANNUAL units (multiply by 10000 for bps)
    """
    rho = correlation_matrix(beta, rho_mode=rho_mode, delta_f=delta_f)
    h = hedge_ratios(Te, Ts, delta, kappas, P0)

    VF = 0.0
    for k in range(5):
        for l in range(5):
            ksum = kappas[k] + kappas[l]
            VF += (
                rho[k, l]
                * sigmas[k] * sigmas[l]
                * h[k] * h[l]
                * (1.0 - np.exp(-ksum * Te)) / ksum
            )

    if VF < 0:
        return 0.0
    return np.sqrt(VF / Te)


def bachelier_price(
    F0: float,
    K: float,
    Te: float,
    Ts: float,
    delta: float,
    sigma_n: float,
    P0: callable,
    option_type: str = "payer",
) -> float:
    """
    Bachelier (normal) swaption price.

    V = A0 * [ (F0 - K) * Phi(d) + sqrt(V_F) * phi(d) ]
    where d = (F0 - K) / sqrt(V_F),  V_F = sigma_n^2 * Te

    Parameters
    ----------
    F0         : float  forward swap rate
    K          : float  strike (use F0 for ATM)
    Te         : float  expiry
    Ts         : float  tenor
    delta      : float  accrual fraction
    sigma_n    : float  normal vol (annual units)
    P0         : callable
    option_type: 'payer' or 'receiver'

    Returns
    -------
    float  swaption price
    """
    from scipy.stats import norm
    A0 = annuity(Te, Ts, delta, P0)
    VF = sigma_n ** 2 * Te
    std = np.sqrt(VF)
    if std < 1e-12:
        # Deep ITM/OTM
        payoff = max(F0 - K, 0.0) if option_type == "payer" else max(K - F0, 0.0)
        return A0 * payoff

    d = (F0 - K) / std
    if option_type == "payer":
        price = A0 * ((F0 - K) * norm.cdf(d) + std * norm.pdf(d))
    else:
        price = A0 * ((K - F0) * norm.cdf(-d) + std * norm.pdf(d))
    return price


# =============================================================================
# Section 5 – Parameter initialisation
# =============================================================================

def init_kappa_logspaced(
    n_factors: int = 5,
    kappa_min: float = 0.05,
    kappa_max: float = 3.0,
) -> np.ndarray:
    """
    Initialise G5++ decay rates kappa_k on a log-spaced grid.

    Rationale
    ---------
    kappa_k is NOT identifiable from PCA of the correlation matrix.
    PCA identifies only rho_kl (via the row-normalised loading matrix).
    sigma_k in turn requires knowing kappa_k first — the loading
    sigma_k * exp(-kappa_k * tau) is not separable without kappa_k.
    sigma_k is therefore obtained from one inner NNLS pass at fixed
    kappa^{(0)} and rho^{(0)} (see inner_nnls).

    A log-spaced grid places one factor in each decay regime (level,
    medium, curvature, ...) and gives the outer L-BFGS-B optimiser a
    good basin of attraction without any model-specific assumption.

    Parameters
    ----------
    n_factors : int    number of factors K (default 5)
    kappa_min : float  slowest decay rate yr^{-1} (default 0.05)
    kappa_max : float  fastest decay rate yr^{-1} (default 3.0)

    Returns
    -------
    kappa0 : (n_factors,) array  initial decay rates, ascending
    """
    return np.exp(np.linspace(np.log(kappa_min), np.log(kappa_max), n_factors))


def init_beta_from_rho(rho: np.ndarray) -> float:
    """
    Initialise beta by moment-matching to the PCA-implied correlation.

    The parametric form rho_kl = exp(-beta * |k-l|) implies that
    adjacent-factor correlations equal exp(-beta).  Inverting:

        beta^{(0)} = -log( mean_{k} rho_{k, k+1} )

    Parameters
    ----------
    rho : (K, K) array  PCA-implied factor correlation matrix

    Returns
    -------
    beta0 : float  initial beta value
    """
    K = rho.shape[0]
    adj = np.array([rho[k, k+1] for k in range(K - 1)])
    mean_adj = np.clip(adj.mean(), 1e-6, 1.0 - 1e-6)
    return float(-np.log(mean_adj))


# =============================================================================
# Section 6 – Calibration objective and nested optimisation
# =============================================================================

def vega_weights(
    swaption_grid: list,
    delta: float,
    P0: callable,
) -> dict:
    """
    Bachelier vega-proportional weights: omega(Te, Ts) ∝ A0 * sqrt(Te).

    Parameters
    ----------
    swaption_grid : list of (Te, Ts) tuples
    delta         : float  accrual fraction
    P0            : callable

    Returns
    -------
    dict {(Te, Ts): weight}  normalised so sum = len(grid)
    """
    raw = {}
    for (Te, Ts) in swaption_grid:
        A0 = annuity(Te, Ts, delta, P0)
        raw[(Te, Ts)] = A0 * np.sqrt(Te)
    total = sum(raw.values())
    n = len(swaption_grid)
    return {k: n * v / total for k, v in raw.items()}


def inner_nnls(
    kappas: np.ndarray,
    beta: float,
    swaption_grid: list,
    market_vols_bps: dict,
    weights: dict,
    delta: float,
    P0: callable,
    rho_mode: str = "exponential",
    delta_f: np.ndarray = None,
) -> tuple:
    """
    Inner NNLS loop: solve for non-negative sigmas given fixed kappas and beta.

    The normal vol is linear in sigmas^2 but not in sigmas.  We exploit
    partial convexity by noting that V_F = sum_{kl} rho_kl * sigma_k * sigma_l
    * h_k * h_l * C_kl where C_kl = (1-exp(-(kk+kl)*Te))/(kk+kl).

    Defining s_k = sigma_k^2 (always positive), V_F is linear in the outer
    products s_k * s_l, but this makes it a quadratic in s_k.  We instead
    solve for sigma directly via the formulation:

        sigma_n^model ≈ sqrt(sum_{kl} rho_kl * sigma_k*sigma_l * h_k*h_l * C_kl / Te)

    Iterating: fix sigma^(old), linearise sigma_k = eps_k * sigma_k^(old),
    solve for eps_k >= 0 via NNLS.  Here we use a simpler approach:
    build a (M x 5) design matrix A where A[m,k] captures the marginal
    contribution of sigma_k to sigma_n^model, and solve NNLS for sigmas.

    For each swaption m:
        sigma_n^model(m) ≈ sum_k a_mk * sigma_k
    where a_mk = sqrt(rho_kk) * |h_k| * sqrt(C_kk(m) / Te_m)
    (diagonal approximation used as linear surrogate for NNLS warm start).

    Parameters
    ----------
    kappas, beta  : current outer parameters
    swaption_grid : list of (Te, Ts)
    market_vols_bps : dict {(Te,Ts): vol in bps}
    weights       : dict of vega weights
    delta, P0     : accrual and discount curve

    Returns
    -------
    sigmas : (5,) array  non-negative solution
    residual : float  weighted RMSE in bps
    """
    rho = correlation_matrix(beta, rho_mode=rho_mode, delta_f=delta_f)
    M = len(swaption_grid)

    # Build linear design matrix using diagonal approximation
    A_mat = np.zeros((M, 5))
    b_vec = np.zeros(M)

    for m, (Te, Ts) in enumerate(swaption_grid):
        h = hedge_ratios(Te, Ts, delta, kappas, P0)
        w = np.sqrt(weights[(Te, Ts)])
        b_vec[m] = w * market_vols_bps[(Te, Ts)] * 1e-4  # convert bps -> annual

        for k in range(5):
            C_kk = (1.0 - np.exp(-2.0 * kappas[k] * Te)) / (2.0 * kappas[k])
            A_mat[m, k] = w * np.abs(h[k]) * np.sqrt(C_kk / Te)

    sigmas_nnls, _ = nnls(A_mat, b_vec)

    # Polish: one Newton-style refinement using full double-sum formula
    # Compute RMSE with initial NNLS solution
    def weighted_rmse(sigmas):
        total = 0.0
        for Te, Ts in swaption_grid:
            sv = model_normal_vol(
                Te, Ts, delta, sigmas, kappas, beta, P0,
                rho_mode=rho_mode, delta_f=delta_f,
            ) * 1e4
            diff = sv - market_vols_bps[(Te, Ts)]
            total += weights[(Te, Ts)] * diff ** 2
        return np.sqrt(total / M)

    return sigmas_nnls, weighted_rmse(sigmas_nnls)


def calibration_objective(
    theta: np.ndarray,
    swaption_grid: list,
    market_vols_bps: dict,
    weights: dict,
    delta: float,
    P0: callable,
    mu_reg: float = 1e-4,
    rho_mode: str = "exponential",
    delta_f: np.ndarray = None,
) -> float:
    """
    Outer objective L(kappas, beta) = min_{sigmas>=0} weighted RMSE^2
                                     + mu * regularisation.

    theta is reparameterised as log(kappas) and, when rho_mode='exponential',
    log(beta) to enforce positivity.

    Parameters
    ----------
    theta    : (6,) array  [log(kappa_1),...,log(kappa_5), log(beta)]
               or (5,) array  [log(kappa_1),...,log(kappa_5)]
               when rho_mode='identity'
    mu_reg   : float  regularisation weight on log(kappas)
    rho_mode : str    'exponential' or 'identity'

    Returns
    -------
    float  objective value
    """
    kappas = np.exp(theta[:5])
    kappas = np.clip(kappas, 1e-4, 10.0)

    if rho_mode == "exponential":
        beta = np.clip(np.exp(theta[5]), 1e-4, 5.0)
    else:
        beta = None  # not used for identity

    sigmas, rmse = inner_nnls(
        kappas, beta, swaption_grid, market_vols_bps, weights, delta, P0,
        rho_mode=rho_mode, delta_f=delta_f,
    )

    # Regularisation: penalise very small or very large kappas
    reg = mu_reg * np.sum((np.log(kappas)) ** 2)

    return rmse ** 2 + reg


# =============================================================================
# Section 7 – Main calibrator class
# =============================================================================

class G5ppCalibrator:
    """
    Full G5++ calibration to a USD SOFR ATM normal swaption vol matrix.

    Parameters
    ----------
    P0              : callable  P0(T) -> float (or ndarray), initial discount curve
    market_vols_bps : dict  {(Te, Ts): normal_vol_in_bps}
    delta           : float  accrual fraction (0.5 = semi-annual, 0.25 = quarterly)
    delta_f         : (T, n) ndarray or None  historical fwd rate changes for PCA init
    tenor_grid_pca  : (n,) ndarray or None    tenor values matching delta_f columns
    kappas0         : (5,) ndarray or None    manual initial kappas (overrides PCA)
    mu_reg          : float  regularisation weight (default 1e-4)
    """

    def __init__(
        self,
        P0: callable,
        market_vols_bps: dict,
        delta: float = 0.5,
        delta_f: np.ndarray = None,
        tenor_grid_pca: np.ndarray = None,
        kappas0: np.ndarray = None,
        mu_reg: float = 1e-4,
        rho_mode: str = "exponential",
    ):
        """
        rho_mode : str
            'exponential'  rho_kl = exp(-beta*|k-l|), beta calibrated. R^11.
            'pca'          rho_kl = (L@L.T)_kl, L = row-normalised loading
                           matrix from PCA of delta_f. Requires delta_f.
                           dW_k = sum_j l_kj * d~W_j (independent PCA BWs).
                           Row-normalised: sum_j l_kj^2=1. R^10 (no beta).
            'identity'     rho_kl = delta_kl. Fallback. R^10 (no beta).
        """
        self.P0 = P0
        self.market_vols_bps = market_vols_bps
        self.delta = delta
        self.delta_f = delta_f
        self.tenor_grid_pca = tenor_grid_pca
        self.kappas0_manual = kappas0
        self.mu_reg = mu_reg
        self.rho_mode = rho_mode
        self.swaption_grid = sorted(market_vols_bps.keys())

    # ------------------------------------------------------------------
    def _initialise_kappas(self) -> np.ndarray:
        """
        Initialise kappas via log-spaced grid.
        kappa_k is NOT identifiable from PCA — use a fixed grid.
        Manual override via kappas0 constructor argument.
        """
        if self.kappas0_manual is not None:
            print("Using manually supplied kappa initialisation.")
            return np.asarray(self.kappas0_manual, dtype=float)
        kappas0 = init_kappa_logspaced()
        print(f"kappa^{{(0)}} = {kappas0.round(4)}  (log-spaced grid)")
        return kappas0

    # ------------------------------------------------------------------
    def calibrate(
        self,
        n_restarts: int = 3,
        maxiter: int = 500,
        verbose: bool = True,
    ) -> dict:
        """
        Run the two-stage nested calibration:
            Stage 1 (inner): NNLS for sigmas given fixed kappas, beta.
            Stage 2 (outer): L-BFGS-B on log(kappas), log(beta).

        Multi-start: perturbs initial kappas to escape local minima.

        Parameters
        ----------
        n_restarts : int   number of random restarts (default 3)
        maxiter    : int   max iterations per L-BFGS-B run
        verbose    : bool  print progress

        Returns
        -------
        dict with keys:
            sigmas, kappas, beta, rho,
            fitted_vols_bps, market_vols_bps,
            rmse_bps, max_error_bps,
            success
        """
        # Compute vega weights once
        weights = vega_weights(self.swaption_grid, self.delta, self.P0)

        # Initialise kappas and beta
        kappas0 = self._initialise_kappas()

        # beta0: moment-match to PCA-implied correlation if available,
        # else use default 0.3
        if self.delta_f is not None and self.rho_mode == "exponential":
            rho_pca, _, _ = pca_loading_correlation(self.delta_f, remove_jumps=True)
            beta0 = init_beta_from_rho(rho_pca)
            print(f"beta^{{(0)}} = {beta0:.4f}  (moment-matched from PCA rho)")
        else:
            beta0 = 0.3

        # Bounds in log-space
        # kappa in [1e-4, 10]; beta in [1e-4, 5] only for exponential mode
        if self.rho_mode == "exponential":
            bounds_lbfgs = (
                [(np.log(1e-4), np.log(10.0))] * 5
                + [(np.log(1e-4), np.log(5.0))]
            )
            theta_dim = 6
        else:  # identity: no beta parameter
            bounds_lbfgs = [(np.log(1e-4), np.log(10.0))] * 5
            theta_dim = 5

        best_result = None
        best_obj    = np.inf

        rng = np.random.default_rng(42)

        for restart in range(n_restarts):
            if restart == 0:
                kappas_init = kappas0.copy()
            else:
                # Perturb in log-space
                noise = rng.uniform(-0.5, 0.5, size=5)
                kappas_init = kappas0 * np.exp(noise)
                kappas_init = np.clip(kappas_init, 1e-4, 10.0)

            if self.rho_mode == "exponential":
                theta0 = np.concatenate([np.log(kappas_init), [np.log(beta0)]])
            else:
                theta0 = np.log(kappas_init)

            if verbose:
                print(f"\nRestart {restart+1}/{n_restarts}: "
                      f"kappas0 = {kappas_init.round(3)}, beta0 = {beta0:.3f}")

            try:
                opt = minimize(
                    calibration_objective,
                    theta0,
                    args=(
                        self.swaption_grid,
                        self.market_vols_bps,
                        weights,
                        self.delta,
                        self.P0,
                        self.mu_reg,
                        self.rho_mode,
                        self.delta_f,
                    ),
                    method="L-BFGS-B",
                    bounds=bounds_lbfgs,
                    options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
                )
            except Exception as e:
                if verbose:
                    print(f"  Optimisation failed: {e}")
                continue

            if opt.fun < best_obj:
                best_obj    = opt.fun
                best_result = opt

        if best_result is None:
            raise RuntimeError("All calibration restarts failed.")

        # Extract final parameters
        theta_opt  = best_result.x
        kappas_opt = np.clip(np.exp(theta_opt[:5]), 1e-4, 10.0)
        if self.rho_mode == "exponential":
            beta_opt = np.clip(np.exp(theta_opt[5]), 1e-4, 5.0)
        else:
            beta_opt = None  # not used for identity mode

        # Final NNLS pass for sigmas
        sigmas_opt, _ = inner_nnls(
            kappas_opt, beta_opt,
            self.swaption_grid, self.market_vols_bps,
            weights, self.delta, self.P0,
            rho_mode=self.rho_mode, delta_f=self.delta_f,
        )

        # Compute fit quality
        fitted_vols  = {}
        errors       = {}
        for (Te, Ts) in self.swaption_grid:
            sv = model_normal_vol(
                Te, Ts, self.delta, sigmas_opt, kappas_opt, beta_opt, self.P0,
                rho_mode=self.rho_mode, delta_f=self.delta_f,
            ) * 1e4
            fitted_vols[(Te, Ts)] = sv
            errors[(Te, Ts)] = sv - self.market_vols_bps[(Te, Ts)]

        err_arr   = np.array(list(errors.values()))
        rmse_bps  = np.sqrt(np.mean(err_arr ** 2))
        max_error = np.max(np.abs(err_arr))

        rho = correlation_matrix(
            beta_opt, rho_mode=self.rho_mode, delta_f=self.delta_f
        )
        result = {
            "sigmas":           sigmas_opt,
            "kappas":           kappas_opt,
            "beta":             beta_opt,
            "rho_mode":         self.rho_mode,
            "rho":              rho,
            "fitted_vols_bps":  fitted_vols,
            "market_vols_bps":  self.market_vols_bps,
            "errors_bps":       errors,
            "rmse_bps":         rmse_bps,
            "max_error_bps":    max_error,
            "success":          best_result.success,
            "n_swaptions":      len(self.swaption_grid),
            "delta_f":          self.delta_f,
        }

        if verbose:
            self._print_summary(result)

        return result

    # ------------------------------------------------------------------
    def _print_summary(self, result: dict):
        print("\n" + "=" * 60)
        print("G5++ CALIBRATION RESULTS")
        print("=" * 60)
        print(f"  sigmas : {result['sigmas'].round(6)}")
        print(f"  kappas : {result['kappas'].round(4)}")
        if result['beta'] is not None:
            print(f"  beta     : {result['beta']:.4f}")
        print(f"  RMSE   : {result['rmse_bps']:.3f} bps")
        print(f"  MaxErr : {result['max_error_bps']:.3f} bps")
        print(f"  Status : {'CONVERGED' if result['success'] else 'NOT CONVERGED'}")
        print()
        print(f"  {'Expiry':>6} {'Tenor':>6} {'Market':>10} {'Model':>10} {'Error':>8}")
        print("  " + "-" * 44)
        for (Te, Ts) in sorted(result['market_vols_bps'].keys()):
            mkt = result['market_vols_bps'][(Te, Ts)]
            mdl = result['fitted_vols_bps'][(Te, Ts)]
            err = result['errors_bps'][(Te, Ts)]
            print(f"  {Te:>6.1f} {Ts:>6.1f} {mkt:>10.2f} {mdl:>10.2f} {err:>+8.2f}")
        print("=" * 60)


# =============================================================================
# Section 8 – Convenience functions for post-calibration use
# =============================================================================

def price_swaption(
    Te: float,
    Ts: float,
    K: float,
    result: dict,
    delta: float,
    P0: callable,
    option_type: str = "payer",
) -> dict:
    """
    Price an arbitrary swaption using calibrated G5++ parameters.

    Parameters
    ----------
    Te, Ts      : expiry and tenor
    K           : strike rate (use None for ATM)
    result      : calibration result dict from G5ppCalibrator.calibrate()
    delta       : accrual fraction
    P0          : discount curve
    option_type : 'payer' or 'receiver'

    Returns
    -------
    dict with keys: F0, K, sigma_n_bps, price, A0
    """
    sigmas   = result['sigmas']
    kappas   = result['kappas']
    beta     = result['beta']
    rho_mode = result.get('rho_mode', 'exponential')

    F0 = forward_swap_rate(Te, Ts, delta, P0)
    if K is None:
        K = F0

    delta_f  = result.get('delta_f', None)
    sigma_n = model_normal_vol(Te, Ts, delta, sigmas, kappas, beta, P0,
                               rho_mode=rho_mode, delta_f=delta_f)

    price   = bachelier_price(F0, K, Te, Ts, delta, sigma_n, P0, option_type)
    A0      = annuity(Te, Ts, delta, P0)

    return {
        "Te":          Te,
        "Ts":          Ts,
        "F0":          F0,
        "K":           K,
        "sigma_n_bps": sigma_n * 1e4,
        "price":       price,
        "A0":          A0,
        "option_type": option_type,
    }


def vol_matrix(
    expiries: list,
    tenors: list,
    result: dict,
    delta: float,
    P0: callable,
) -> np.ndarray:
    """
    Compute model normal vol matrix for a grid of expiries and tenors.

    Parameters
    ----------
    expiries, tenors : lists of floats
    result           : calibration result dict
    delta, P0        : accrual and discount curve

    Returns
    -------
    vols : (len(expiries), len(tenors)) array  in bps
    """
    sigmas   = result['sigmas']
    kappas   = result['kappas']
    beta     = result['beta']
    rho_mode = result.get('rho_mode', 'exponential')

    delta_f  = result.get('delta_f', None)
    vols = np.zeros((len(expiries), len(tenors)))
    for i, Te in enumerate(expiries):
        for j, Ts in enumerate(tenors):
            vols[i, j] = model_normal_vol(
                Te, Ts, delta, sigmas, kappas, beta, P0,
                rho_mode=rho_mode, delta_f=delta_f,
            ) * 1e4
    return vols


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    print("Running G5++ self-test with flat 5% discount curve...\n")

    def P0_flat(T):
        return np.exp(-0.05 * np.asarray(T, dtype=float))

    market_vols = {
        (1,  1): 85.0, (1,  2): 90.0, (1,  5): 95.0, (1, 10): 88.0,
        (2,  1): 82.0, (2,  2): 86.0, (2,  5): 91.0, (2, 10): 85.0,
        (5,  1): 70.0, (5,  2): 74.0, (5,  5): 79.0, (5, 10): 75.0,
        (10, 1): 58.0, (10, 2): 61.0, (10, 5): 65.0, (10,10): 62.0,
    }

    calib = G5ppCalibrator(
        P0=P0_flat,
        market_vols_bps=market_vols,
        delta=0.5,
    )

    result = calib.calibrate(n_restarts=2, verbose=True)

    print("\nPricing a 5Y x 10Y ATM payer swaption:")
    px = price_swaption(5, 10, None, result, 0.5, P0_flat, "payer")
    print(f"  F0        = {px['F0']*100:.4f}%")
    print(f"  sigma_n   = {px['sigma_n_bps']:.2f} bps")
    print(f"  Price     = {px['price']:.6f}")

    print("\nFull vol matrix (model, exponential rho):")
    expiries = [1, 2, 5, 10]
    tenors   = [1, 2, 5, 10]
    V = vol_matrix(expiries, tenors, result, 0.5, P0_flat)
    header = "       " + "".join(f"  {t:4.0f}Y" for t in tenors)
    print(header)
    for i, Te in enumerate(expiries):
        row = f"  {Te:4.0f}Y  " + "  ".join(f"{V[i,j]:6.2f}" for j in range(len(tenors)))
        print(row)

    print("\n--- Testing rho_mode='pca' (loading matrix correlation) ---")
    # Synthetic historical data: 500 days x 40 tenors
    rng2 = np.random.default_rng(0)
    synthetic_df = rng2.standard_normal((500, 40)) * 5e-4
    tenor_grid   = np.linspace(1/12, 30, 40)
    calib_pca = G5ppCalibrator(
        P0=P0_flat,
        market_vols_bps=market_vols,
        delta=0.5,
        delta_f=synthetic_df,
        tenor_grid_pca=tenor_grid,
        rho_mode="pca",
    )
    result_pca = calib_pca.calibrate(n_restarts=2, verbose=True)
    rho_pca, L_pca, jres = pca_loading_correlation(synthetic_df, verbose=True)
    print(f"PCA correlation matrix (top-left 3x3):\n{result_pca['rho'][:3,:3].round(3)}")
    if jres is not None:
        print(f"Jump days detected: {jres['n_jumps']} "
              f"({100*jres['jump_fraction']:.1f}% of sample)")
    print()

    print("--- Demonstrating detect_jumps standalone ---")
    # Inject 10 artificial jump days into synthetic data
    rng3 = np.random.default_rng(42)
    synthetic_with_jumps = synthetic_df.copy()
    jump_days_true = rng3.choice(500, size=10, replace=False)
    synthetic_with_jumps[jump_days_true] += rng3.standard_normal((10, 40)) * 5e-3
    jresult = detect_jumps(synthetic_with_jumps, c_mad=4.0, alpha=0.001, n_iter=3)
    jump_detection_summary(jresult)
    recovered = np.intersect1d(jresult["jump_idx"], jump_days_true)
    print(f"True jump days injected : {len(jump_days_true)}")
    print(f"Detected correctly      : {len(recovered)}")
    print()
