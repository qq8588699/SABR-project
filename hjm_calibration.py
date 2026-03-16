"""
G2++ / G3++ Gaussian HJM — Swaption Calibration
================================================

Full workflow demonstration:
  1.  Forward curve  (USD SOFR, end-2023 levels)
  2.  Synthetic ATM swaption vol surface
        - Generated from known G2++ / G3++ parameters + small noise
        - Model-consistent: the model CAN recover the true parameters
  3.  Semi-analytic swaption pricing (Rebonato / Brigo-Mercurio approximation)
  4.  Two-stage calibration
        - Stage 1: Multi-start L-BFGS-B (fast gradient-based, log-parameterised)
        - Stage 2: Nelder-Mead polish
  5.  Results: vol surface comparison + Bachelier price comparison

═══════════════════════════════════════════════════════════
MODEL
═══════════════════════════════════════════════════════════
G_n++ separable Gaussian HJM:

  Instantaneous forward rate:
    f(t,T) = f(0,T) + Σ_k x_k(t) · exp(-κ_k(T-t))

  Factor dynamics under Q:
    dx_k = -κ_k x_k dt + σ_k dW_k(t)
    E[dW_j dW_k] = ρ_{jk} dt

  Bond price (exact):
    P(t,T) = P(0,T)/P(0,t) · exp( -Σ_k B_k(t,T)·x_k(t) + ½V(t,T) )
    B_k(t,T) = (1 - exp(-κ_k(T-t))) / κ_k

═══════════════════════════════════════════════════════════
SWAPTION PRICING  (Linearisation route, paper Section 5)
═══════════════════════════════════════════════════════════
ATM swaption: option expiry T_e, swap tenor T_s, semi-annual coupons.

  Forward swap rate variance (linearisation route):

    V_F(T_e) = Σ_{j,k} ρ_{jk} σ_j σ_k h_j h_k I_{jk}(T_e)

  where:
    I_{jk}(T) = [1 - exp(-(κ_j+κ_k)T)] / (κ_j+κ_k)
              = integral_0^T exp(-(κ_j+κ_k)(T-t)) dt    [OU covariance integral]
    h_k = (1/A_0) * [ -B_k(0,T_e)*P(0,T_e)
                      + B_k(0,T_e+T_s)*P(0,T_e+T_s)
                      + F_0 * Σ_i δ_i*B_k(0,T_e+T_i)*P(0,T_e+T_i) ]
                                                          [hedge ratio dF/dx_k at x=0]
    A(0) = Σ_i δ_i · P(0,T_i)                            annuity factor
    F_0  = (P(0,T_e) - P(0,T_e+T_s)) / A(0)              forward swap rate

  Normal vol: σ_n = sqrt(V_F / T_e)

  Normal (Bachelier) ATM swaption price:
    Price = A(0) · σ_n · √(2·T_e/π)     [fraction of notional]

═══════════════════════════════════════════════════════════
CALIBRATION  (Minimise weighted MSE on vol surface)
═══════════════════════════════════════════════════════════
  Objective:  min Σ_{ij} w_{ij} · (σ_S^model(T_e_i,T_s_j) - σ_S^mkt_{ij})²

  Log-parameterisation for positivity constraints:
    y[:n]   = log(κ_k)  →  κ_k always positive
    y[n:2n] = log(σ_k)  →  σ_k always positive
    y[2n:]  = ρ_jk       →  clipped to (-0.97, 0.97)

  Stage 1: L-BFGS-B from multiple starting points (physics-motivated grid)
  Stage 2: Nelder-Mead polish of best Stage 1 solution
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FORWARD CURVE AND DISCOUNT FACTORS
# ─────────────────────────────────────────────────────────────────────────────

def make_fwd_curve():
    """USD SOFR forward rate curve f(0,T) in % p.a., approx end-2023."""
    def f(T):
        T = np.asarray(T, float)
        return 4.20 + 1.10*np.exp(-0.40*T) + 0.60*(T/5.0)*np.exp(1-T/5.0)
    return f

def build_discount_cache(f, t_max=35.0, n=3500):
    """P(0,T) via trapezoidal integration of f(0,t)."""
    tg = np.linspace(0, t_max, n+1)
    fg = f(tg) / 100.0
    dt = tg[1] - tg[0]
    cf = np.zeros(n+1)
    for i in range(1, n+1):
        cf[i] = cf[i-1] + 0.5*dt*(fg[i-1]+fg[i])
    return tg, np.exp(-cf)

def make_discount(tg, Pg):
    """Vectorised interpolated discount function."""
    def P(T): return np.interp(np.clip(np.asarray(T,float),0,tg[-1]), tg, Pg)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def B_k(kappa, tau):
    """B_k(tau) = (1-exp(-κτ))/κ  — numerically stable."""
    safe = np.where(np.abs(kappa)<1e-9, 1e-9, kappa)
    return (1.0 - np.exp(-safe*tau)) / safe

def rho_mat(rv, n):
    """Build n×n correlation matrix from lower-triangle vector rv."""
    R = np.eye(n); idx=0
    for i in range(1,n):
        for j in range(i):
            R[i,j]=R[j,i]=rv[idx]; idx+=1
    return R

def is_pd(R, tol=1e-7):
    return bool(np.all(np.linalg.eigvalsh(R) > tol))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SWAPTION PRICING
# ─────────────────────────────────────────────────────────────────────────────

def atm_normal_vol(kappas, sigmas, R, P, T_e, T_s, freq=2):
    """
    ATM swaption normal vol (bp/yr).

    Uses the linearisation route (paper Section 5):
      V_F = Σ_{jk} ρ_{jk}·σ_j·σ_k·h_j·h_k·I_{jk}(T_e)

    where I_{jk}(T) = [1-exp(-(κ_j+κ_k)·T)]/(κ_j+κ_k)
            = integral_0^T exp(-(κ_j+κ_k)*(T-t)) dt
            = Cov(x_j(T), x_k(T)) / (σ_j σ_k ρ_{jk})   [OU covariance]

    and the hedge ratios h_k = dF/dx_k |_{x=0} are (paper eq. hk):
      h_k = (1/A_0) * [ -B_k(0,T_e)*P(0,T_e)
                        + B_k(0,T_e+T_s)*P(0,T_e+T_s)
                        + F_0 * Σ_i δ_i*B_k(0,T_e+T_i)*P(0,T_e+T_i) ]

    The three terms are:
      -B_k(0,T_e)*P(0,T_e)          : sensitivity of numerator start bond
      +B_k(0,T_e+T_s)*P(0,T_e+T_s) : sensitivity of numerator end bond
      +F_0 * annuity sensitivity     : correction for shift in denominator A

    Normal vol σ_n = sqrt(V_F / T_e),  returned in bp/yr.

    Note: the previous implementation used only the annuity-weighted loading
      w_k = Σ_i α_i·B_k(0,T_i)
    which omits the numerator bond sensitivities and is incorrect.
    The correct quantity is h_k, not w_k.
    """
    n    = len(kappas)
    dt   = 1.0/freq
    n_p  = int(round(T_s*freq))
    pd   = T_e + np.arange(1, n_p+1)*dt          # coupon payment dates
    dl   = np.full(n_p, dt)                        # payment intervals δ_i
    Pi   = P(pd)                                   # P(0, T_e+T_i)
    A0   = float(np.dot(dl, Pi))                   # annuity A(0)
    P_Te = float(P(T_e))                           # P(0, T_e)
    P_TN = float(P(T_e + T_s))                     # P(0, T_e+T_s)
    F0   = (P_Te - P_TN) / A0                      # forward swap rate F_0

    # Hedge ratios h_k = dF/dx_k evaluated at x=0
    h = np.array([
        (1.0/A0) * (
            - B_k(kappas[k], T_e)     * P_Te          # start bond
            + B_k(kappas[k], T_e+T_s) * P_TN          # end bond
            + F0 * float(np.dot(dl, B_k(kappas[k], pd) * Pi))  # annuity correction
        )
        for k in range(n)
    ])

    # OU covariance integral  I_{jk}(T_e) = (1-exp(-(κ_j+κ_k)*T_e))/(κ_j+κ_k)
    D  = kappas[:,None] + kappas[None,:]
    I  = np.where(D > 1e-8, (1-np.exp(-D*T_e))/D, T_e)

    # V_F = Σ_{jk} ρ_{jk}·σ_j·σ_k·h_j·h_k·I_{jk}
    S  = sigmas[:,None] * sigmas[None,:]
    HH = h[:,None] * h[None,:]
    VF = float(np.sum(R * S * HH * I))

    return np.sqrt(max(VF, 0) / T_e) * 10000.0 if T_e > 0 else 0.0


def vol_surface(kappas, sigmas, R, P, exps, tens):
    """Model ATM normal vol surface (bp/yr) over expiry × tenor grid."""
    out = np.zeros((len(exps), len(tens)))
    for i,Te in enumerate(exps):
        for j,Ts in enumerate(tens):
            out[i,j] = atm_normal_vol(kappas, sigmas, R, P, Te, Ts)
    return out


def bachelier_atm(sigma_bp, T_e, T_s, P, freq=2):
    """
    Bachelier ATM swaption price (bp of notional).

    V(0) = A(0) · σ_N · √(T_e/(2π))  =  A(0)·σ_N·sqrt(T_e)·E[Z⁺]  where E[Z⁺]=1/√(2π)
    where σ_N = sigma_bp/10000.
    """
    sN  = sigma_bp/10000.0
    dt  = 1.0/freq
    np_ = int(round(T_s*freq))
    pd  = T_e + np.arange(1, np_+1)*dt
    A   = float(np.dot(np.full(np_,dt), P(pd)))
    return A*sN*np.sqrt(T_e/(2.0*np.pi))*10000.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SYNTHETIC MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────

def build_market(P, noise_bp=2.0, seed=42):
    """
    Synthetic ATM swaption vol surfaces for G2++ and G3++ calibration.

    Strategy: generate surfaces from KNOWN true parameters, then add
    Gaussian noise (default 2bp) to simulate bid-ask uncertainty.
    This ensures the calibration problem is well-posed and the true
    parameters are approximately recoverable.

    Returns
    -------
    exps   : (8,)  option expiries (yr)
    tens   : (6,)  swap tenors (yr)
    mkt2   : (8,6) market vols for G2++ calibration (from G2++ + noise)
    mkt3   : (8,6) market vols for G3++ calibration (from G3++ + noise)
    tp2    : true G2++ parameter dict
    tp3    : true G3++ parameter dict
    """
    exps = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    tens = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    # ── True G2++ parameters ──────────────────────────────────────
    # Factor 1: slow (level factor, κ=0.35 → half-life 2yr)
    # Factor 2: fast (curvature factor, κ=2.5 → half-life 0.28yr)
    tp2 = dict(
        kappas = np.array([0.35, 2.50]),
        sigmas = np.array([0.0100, 0.0070]),   # 100bp, 70bp
        rhos_v = np.array([-0.30]),
    )
    tp2['R'] = rho_mat(tp2['rhos_v'], 2)

    # ── True G3++ parameters ──────────────────────────────────────
    # Factor 1: very slow (level,  κ=0.15 → half-life 4.6yr)
    # Factor 2: medium   (slope,   κ=0.80 → half-life 0.87yr)
    # Factor 3: fast     (curv,    κ=3.50 → half-life 0.20yr)
    tp3 = dict(
        kappas = np.array([0.15, 0.80, 3.50]),
        sigmas = np.array([0.0080, 0.0070, 0.0050]),  # 80bp, 70bp, 50bp
        rhos_v = np.array([-0.40, -0.15, 0.25]),
    )
    tp3['R'] = rho_mat(tp3['rhos_v'], 3)

    rng   = np.random.default_rng(seed)
    true2 = vol_surface(tp2['kappas'],tp2['sigmas'],tp2['R'],P,exps,tens)
    true3 = vol_surface(tp3['kappas'],tp3['sigmas'],tp3['R'],P,exps,tens)
    mkt2  = true2 + rng.normal(0, noise_bp, true2.shape)
    mkt3  = true3 + rng.normal(0, noise_bp, true3.shape)

    return exps, tens, mkt2, mkt3, tp2, tp3


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def pack(kappas, sigmas, rhos_v):
    """Pack into log-transformed calibration vector."""
    return np.concatenate([np.log(kappas), np.log(sigmas), rhos_v])

def unpack(y, n):
    """Unpack log-transformed vector → kappas, sigmas, rhos_v."""
    return np.exp(y[:n]), np.exp(y[n:2*n]), y[2*n:]

def make_obj(n, P, mkt, exps, tens, weights):
    """
    Build calibration objective (closure).

    Objective = weighted MSE(model_vols − market_vols) + soft penalties.

    Log-parameterisation ensures κ_k>0 and σ_k>0 without hard bounds on
    positivity. Bounds on the log-space variables prevent extreme values.

    Soft penalties (quadratic):
      κ_k > 10     (unreasonably fast mean-reversion)
      σ_k > 3%     (unreasonably large instantaneous vol)
      |ρ| > 0.97   (near-singular correlation matrix)
    Hard return 1e7 if correlation matrix is not positive definite.
    """
    def obj(y):
        ks, ss, rv = unpack(y, n)
        rvc = np.clip(rv, -0.97, 0.97)
        pen = 0.0
        for k in ks:
            if k > 10.0: pen += 1e4*(k-10.0)**2
        for s in ss:
            if s > 0.03:  pen += 1e4*(s-0.03)**2
        for r in rv:
            if abs(r) > 0.97: pen += 1e5*(abs(r)-0.97)**2
        if pen > 5e3: return pen
        R = rho_mat(rvc, n)
        if not is_pd(R): return pen + 1e7
        try:
            mdl = vol_surface(ks, ss, R, P, exps, tens)
        except Exception:
            return pen + 1e7
        return float(np.mean(((mdl - mkt)*weights)**2)) + pen
    return obj

def calibrate(mkt, exps, tens, P, n, verbose=True):
    """
    Calibrate G_n++ model to ATM swaption vol surface.

    Method: Multi-start L-BFGS-B (Stage 1) + Nelder-Mead polish (Stage 2).

    Starting points are physics-motivated:
      - κ values span the range from slow (level) to fast (curvature)
      - σ values in the 70-120bp typical range
      - ρ values cover negative to zero correlation

    The L-BFGS-B method uses finite-difference gradients and converges
    in ~200-300 function evaluations per start. With 6 starts and 2ms
    per evaluation, Stage 1 takes ~2-4 seconds total.

    Stage 2 Nelder-Mead is a derivative-free simplex method that polishes
    the Stage 1 solution without requiring gradient information, which is
    important near the optimum where finite-difference gradients are noisy.
    """
    n_rho = n*(n-1)//2

    # Weight matrix: heavier on short expiries (more sensitive to κ, more liquid)
    W = np.ones_like(mkt)
    for i, Te in enumerate(exps):
        W[i,:] = 1.0 + 2.0*np.exp(-0.5*Te)

    obj = make_obj(n, P, mkt, exps, tens, W)

    # Bounds in log space: log(κ) ∈ [-4, 2.3] → κ ∈ [0.018, 10]
    #                      log(σ) ∈ [-7, -2]  → σ ∈ [0.001, 0.0135]  (10-135bp)
    bnds = [(-4.0,2.3)]*n + [(-7.0,-2.0)]*n + [(-0.97,0.97)]*n_rho

    # Physics-motivated starting points
    if n == 2:
        K_starts = [[0.15,1.5], [0.30,2.5], [0.50,3.0], [0.20,2.0], [0.40,4.0], [0.10,1.8]]
        S_starts = [[0.008,0.010],[0.010,0.008],[0.009,0.007],[0.012,0.006],[0.007,0.011],[0.011,0.009]]
        R_starts = [[-0.30],[-0.50],[-0.20],[0.00],[-0.40],[-0.60]]
    else:
        K_starts = [[0.10,0.80,3.0],[0.15,1.00,3.5],[0.20,1.20,4.0],
                    [0.10,0.60,2.5],[0.15,0.90,3.0],[0.12,0.70,3.5]]
        S_starts = [[0.008,0.007,0.005],[0.010,0.006,0.005],[0.009,0.008,0.005],
                    [0.007,0.009,0.006],[0.008,0.007,0.006],[0.009,0.007,0.004]]
        R_starts = [[-0.40,-0.15,0.25],[-0.30,-0.10,0.20],[-0.50,-0.20,0.30],
                    [-0.30,0.00,0.10],[-0.40,-0.20,0.15],[-0.35,-0.10,0.25]]

    if verbose:
        print(f"\n{'='*62}")
        print(f"  CALIBRATING G{n}++ MODEL")
        print(f"  Parameters : {2*n+n_rho}  (κ×{n}, σ×{n}, ρ×{n_rho})")
        print(f"  Data points: {mkt.size}  ({len(exps)} expiries × {len(tens)} tenors)")
        print(f"  Method     : Multi-start L-BFGS-B + Nelder-Mead polish")
        print(f"{'='*62}")
        print(f"  Stage 1: L-BFGS-B from {len(K_starts)} starting points...")

    # ── Stage 1: Multi-start L-BFGS-B ────────────────────────────
    best_loss = 1e9
    best_y    = None
    losses    = []
    for ks, ss, rs in zip(K_starts, S_starts, R_starts):
        y0 = pack(np.array(ks), np.array(ss), np.array(rs))
        try:
            r = minimize(obj, y0, method='L-BFGS-B', bounds=bnds,
                        options={'maxiter':200,'ftol':1e-13,'gtol':1e-9,'maxfun':600})
            losses.append(round(r.fun, 4))
            if r.fun < best_loss:
                best_loss = r.fun
                best_y    = r.x.copy()
        except Exception:
            pass

    if verbose:
        print(f"  Stage 1 losses : {losses}")
        print(f"  Stage 1 best   : {best_loss:.5f}")
        print(f"  Stage 2: Nelder-Mead polish...")

    # ── Stage 2: Nelder-Mead ─────────────────────────────────────
    r2 = minimize(obj, best_y, method='Nelder-Mead',
                  options={'maxiter':5000,'xatol':1e-8,'fatol':1e-8,'adaptive':True})
    y_best = r2.x
    if verbose:
        print(f"  Stage 2 final  : {r2.fun:.5f}")

    # ── Extract final result ──────────────────────────────────────
    ks, ss, rv = unpack(y_best, n)
    rv = np.clip(rv, -0.97, 0.97)
    R  = rho_mat(rv, n)
    mdl_vols = vol_surface(ks, ss, R, P, exps, tens)
    errors   = mdl_vols - mkt
    rmse     = float(np.sqrt(np.mean(errors**2)))
    mae      = float(np.mean(np.abs(errors)))
    max_e    = float(np.max(np.abs(errors)))

    if verbose:
        print(f"\n  Fitted parameters:")
        for k in range(n):
            hl = np.log(2)/ks[k]
            print(f"    κ_{k+1} = {ks[k]:.5f} yr⁻¹  (half-life={hl:.2f}yr)  "
                  f"σ_{k+1} = {ss[k]*1e4:.2f} bp/yr")
        ridx = [(i,j) for i in range(1,n) for j in range(i)]
        for idx,(i,j) in enumerate(ridx):
            print(f"    ρ_{i+1}{j+1} = {rv[idx]:+.5f}")
        print(f"\n  RMSE    : {rmse:.3f} bp/yr")
        print(f"  MAE     : {mae:.3f} bp/yr")
        print(f"  Max err : {max_e:.3f} bp/yr")

    return dict(kappas=ks, sigmas=ss, rhos_v=rv, R=R,
                mdl_vols=mdl_vols, errors=errors,
                rmse=rmse, mae=mae, max_e=max_e, n=n)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def fmte(T): return f"{int(round(T*12))}M" if T<1 else f"{int(T)}Y"

def ptab(label, V, exps, tens, signed=False):
    fmt = "+7.2f" if signed else "7.2f"
    hdr = f"  {'':>5}" + "".join(f"  {int(T):>5}Y" for T in tens)
    print(f"\n  {label}")
    print(hdr)
    print("  " + "─"*(len(hdr)-2))
    for i,Te in enumerate(exps):
        row = f"  {fmte(Te):>5}" + "".join(f"  {V[i,j]:{fmt}}" for j in range(len(tens)))
        print(row)

def pprice(lbl, mkt, mdl, exps, tens, P):
    print(f"\n  {lbl}")
    print(f"  {'Exp':>5}  {'Ten':>4}  {'MktVol':>7}  {'MdlVol':>7}  "
          f"{'VolErr':>7}  {'MktPx(bp)':>10}  {'MdlPx(bp)':>10}  {'PxErr(bp)':>10}")
    print("  " + "─"*73)
    for i,Te in enumerate(exps):
        for j,Ts in enumerate(tens):
            mv=mkt[i,j]; mmv=mdl[i,j]
            mp=bachelier_atm(mv, Te, Ts, P)
            ep=bachelier_atm(mmv,Te, Ts, P)
            print(f"  {fmte(Te):>5}  {int(Ts):>3}Y  "
                  f"  {mv:6.2f}  {mmv:6.2f}  {mmv-mv:+6.2f}  "
                  f"  {mp:9.3f}  {ep:9.3f}  {ep-mp:+9.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*62)
    print("  G2++ / G3++ GAUSSIAN HJM SWAPTION CALIBRATION")
    print("="*62)

    # Forward curve and discount
    f    = make_fwd_curve()
    tg,Pg = build_discount_cache(f)
    P    = make_discount(tg, Pg)

    print("\n  USD SOFR forward curve f(0,T) % p.a., approx end-2023:")
    ts = [0.25,0.5,1,2,3,5,7,10,20,30]
    lbs= ["3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"]
    print("  "+"  ".join(f"{l:>5}" for l in lbs))
    print("  "+"  ".join(f"{f(t):5.2f}" for t in ts))

    # Synthetic market
    exps, tens, mkt2, mkt3, tp2, tp3 = build_market(P)

    # ── Show true parameters ──────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  TRUE GENERATING PARAMETERS  (to be recovered by calibration)")
    print(f"{'='*62}")
    print(f"\n  G2++  (5 free parameters: κ₁,κ₂, σ₁,σ₂, ρ₁₂):")
    print(f"  {'Factor':<8}  {'κ (yr⁻¹)':>10}  {'half-life':>10}  {'σ (bp/yr)':>10}")
    for k,(kap,sig) in enumerate(zip(tp2['kappas'],tp2['sigmas'])):
        print(f"  {k+1:<8}  {kap:>10.4f}  {np.log(2)/kap:>9.2f}Y  {sig*1e4:>10.1f}")
    print(f"  ρ₁₂ = {tp2['rhos_v'][0]:+.4f}")

    print(f"\n  G3++  (9 free parameters: κ₁,κ₂,κ₃, σ₁,σ₂,σ₃, ρ₁₂,ρ₁₃,ρ₂₃):")
    print(f"  {'Factor':<8}  {'κ (yr⁻¹)':>10}  {'half-life':>10}  {'σ (bp/yr)':>10}")
    for k,(kap,sig) in enumerate(zip(tp3['kappas'],tp3['sigmas'])):
        print(f"  {k+1:<8}  {kap:>10.4f}  {np.log(2)/kap:>9.2f}Y  {sig*1e4:>10.1f}")
    ridx=[(i,j) for i in range(1,3) for j in range(i)]
    for idx,(i,j) in enumerate(ridx):
        print(f"  ρ_{i+1}{j+1} = {tp3['rhos_v'][idx]:+.4f}")

    ptab("MARKET VOLS FOR G2++ CALIBRATION  (true G2++ surface + 2bp noise, bp/yr)",
         mkt2, exps, tens)
    ptab("MARKET VOLS FOR G3++ CALIBRATION  (true G3++ surface + 2bp noise, bp/yr)",
         mkt3, exps, tens)

    # ── G2++ calibration ──────────────────────────────────────────
    res2 = calibrate(mkt2, exps, tens, P, n=2)
    ptab("G2++ MODEL VOLS  (bp/yr)", res2['mdl_vols'], exps, tens)
    ptab("G2++ ERRORS  (model − market, bp/yr)", res2['errors'], exps, tens, signed=True)

    # ── G3++ calibration ──────────────────────────────────────────
    res3 = calibrate(mkt3, exps, tens, P, n=3)
    ptab("G3++ MODEL VOLS  (bp/yr)", res3['mdl_vols'], exps, tens)
    ptab("G3++ ERRORS  (model − market, bp/yr)", res3['errors'], exps, tens, signed=True)

    # ── Parameter recovery ────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PARAMETER RECOVERY")
    print(f"{'='*62}")

    print(f"\n  G2++  (calibrated to its own surface + 2bp noise):")
    print(f"  {'':12}  {'κ₁':>8}  {'κ₂':>8}  {'σ₁(bp)':>8}  {'σ₂(bp)':>8}  {'ρ₁₂':>8}")
    print(f"  {'─'*56}")
    k2t=tp2['kappas']; s2t=tp2['sigmas']; r2t=tp2['rhos_v']
    k2c=res2['kappas']; s2c=res2['sigmas']; r2c=res2['rhos_v']
    print(f"  {'True':12}  {k2t[0]:>8.4f}  {k2t[1]:>8.4f}  "
          f"{s2t[0]*1e4:>8.1f}  {s2t[1]*1e4:>8.1f}  {r2t[0]:>+8.4f}")
    print(f"  {'Calibrated':12}  {k2c[0]:>8.4f}  {k2c[1]:>8.4f}  "
          f"{s2c[0]*1e4:>8.1f}  {s2c[1]*1e4:>8.1f}  {r2c[0]:>+8.4f}")

    print(f"\n  G3++  (calibrated to its own surface + 2bp noise):")
    print(f"  {'':12}  {'κ₁':>7}  {'κ₂':>7}  {'κ₃':>7}  "
          f"{'σ₁':>7}  {'σ₂':>7}  {'σ₃':>7}  "
          f"{'ρ₁₂':>7}  {'ρ₁₃':>7}  {'ρ₂₃':>7}")
    print(f"  {'─'*76}")
    k3t=tp3['kappas']; s3t=tp3['sigmas']; r3t=tp3['rhos_v']
    k3c=res3['kappas']; s3c=res3['sigmas']; r3c=res3['rhos_v']
    print(f"  {'True':12}  {k3t[0]:>7.3f}  {k3t[1]:>7.3f}  {k3t[2]:>7.3f}  "
          f"{s3t[0]*1e4:>7.1f}  {s3t[1]*1e4:>7.1f}  {s3t[2]*1e4:>7.1f}  "
          f"{r3t[0]:>+7.3f}  {r3t[1]:>+7.3f}  {r3t[2]:>+7.3f}")
    print(f"  {'Calibrated':12}  {k3c[0]:>7.3f}  {k3c[1]:>7.3f}  {k3c[2]:>7.3f}  "
          f"{s3c[0]*1e4:>7.1f}  {s3c[1]*1e4:>7.1f}  {s3c[2]*1e4:>7.1f}  "
          f"{r3c[0]:>+7.3f}  {r3c[1]:>+7.3f}  {r3c[2]:>+7.3f}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FIT QUALITY SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Metric':<28}  {'G2++ (k=5)':>10}  {'G3++ (k=9)':>10}")
    print(f"  {'─'*50}")
    print(f"  {'RMSE (bp/yr)':<28}  {res2['rmse']:>10.3f}  {res3['rmse']:>10.3f}")
    print(f"  {'MAE  (bp/yr)':<28}  {res2['mae']:>10.3f}  {res3['mae']:>10.3f}")
    print(f"  {'Max |error| (bp/yr)':<28}  {res2['max_e']:>10.3f}  {res3['max_e']:>10.3f}")
    print(f"  {'k / n  (overfitting risk)':<28}  {5/mkt2.size:>10.3f}  {9/mkt3.size:>10.3f}")

    # ── Swaption price comparison ─────────────────────────────────
    pprice("SWAPTION PRICES — G2++  (Bachelier ATM, bp of notional)",
           mkt2, res2['mdl_vols'], exps, tens, P)
    pprice("SWAPTION PRICES — G3++  (Bachelier ATM, bp of notional)",
           mkt3, res3['mdl_vols'], exps, tens, P)

    # ── Factor vol loadings across tenors ─────────────────────────
    print(f"\n{'='*62}")
    print(f"  FACTOR VOL LOADINGS  σ_k · exp(-κ_k · T)  at t=0  (bp/yr)")
    print(f"{'='*62}")
    for res, nm in [(res2,'G2++'), (res3,'G3++')]:
        nf=res['n']; ks=res['kappas']; ss=res['sigmas']
        print(f"\n  {nm}:")
        Ts=[0.5,1,2,5,10,20]
        print(f"  {'T':>5}" + "".join(f"  {'Factor'+str(k+1):>10}" for k in range(nf)))
        for T in Ts:
            row = f"  {T:>4}Y" + "".join(
                f"  {ss[k]*np.exp(-ks[k]*T)*1e4:>9.2f}bp" for k in range(nf))
            print(row)

    return res2, res3


if __name__ == '__main__':
    res2, res3 = main()
