"""
zero_ifr_pipeline.py  (correct architecture)
=============================================
Key design principle
---------------------
Cleaning is performed entirely in ZERO RATE space.
The conversion to piecewise-constant IFR happens ONCE, after cleaning.

Why this matters for piecewise-constant IFRs
---------------------------------------------
The forward rate between pillars i-1 and i is:

    f_i = ( r(T_i)*T_i  -  r(T_{i-1})*T_{i-1} )  /  (T_i - T_{i-1})

Each f_i is a DIFFERENCE of two adjacent zero-rate quantities.
Cleaning in IFR space is therefore wrong for four reasons:

  1. A spike in zero rate r_k simultaneously corrupts f_k AND f_{k+1}.
     The spike cleaner would see two suspicious IFR series instead of one.

  2. The jump detector in the spike cleaner is already operating on
     differences of the input. Applied to IFRs it is taking differences
     of differences — i.e. the second difference of zeros — which has
     completely different statistical properties.

  3. Savitzky-Golay curvature smoothing freely modifies individual f_i
     values. But the consistency identity  sum(f_i * dt_i) = r(T_k)*T_k
     must hold at every pillar. Smoothing destroys this, making the
     round-trip zero != original zero even for unmodified dates.

  4. The second-derivative curvature check assumes a smooth underlying
     curve. Piecewise-constant IFRs are DESIGNED to be discontinuous at
     every pillar — the d2 detector fires at every boundary.

Cleaning in zero rate space avoids all four problems:
  - A spike in r_k is isolated to one time series (one tenor column).
  - The curvature check on zeros measures genuine economic smoothness.
  - After cleaning, a single exact conversion to IFR preserves the
    cumulative-sum identity perfectly.
  - Round-trip error on untouched dates is exactly 0.

Usage
-----
    from zero_ifr_pipeline import ZeroIFRPipeline

    # zero_df: pd.DataFrame, index=dates, columns=tenors (float years)
    # values : continuously compounded zero rates in decimal (e.g. 0.045)
    # may contain np.nan

    pipeline = ZeroIFRPipeline(zero_df)
    result   = pipeline.run()

    result.zero_clean   # cleaned zero rates
    result.ifr_clean    # piecewise-constant IFRs derived from clean zeros
    result.report       # full audit log (spike_flags, nan_flags, curve_report)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════════════════
# Robust statistics  (NaN-safe)
# ══════════════════════════════════════════════════════════════════════════════

def _mad_zscore(series: np.ndarray) -> np.ndarray:
    """
    MAD z-score using only finite values.
    NaN positions in input -> NaN in output (never silently flagged).
    """
    finite = series[np.isfinite(series)]
    if len(finite) < 2:
        return np.full_like(series, np.nan, dtype=float)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad == 0:
        return np.where(np.isfinite(series), 0.0, np.nan)
    z                         = 0.6745 * (series - med) / mad
    z[~np.isfinite(series)]   = np.nan
    return z


def _rolling_mad_zscore(series: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Rolling MAD z-score. Falls back to global z-score where the rolling
    window has fewer than 5 observed values.
    NaN positions -> NaN output.
    """
    s        = pd.Series(series, dtype=float)
    roll_med = s.rolling(window, center=True, min_periods=5).median()
    roll_mad = (s - roll_med).abs().rolling(window, center=True, min_periods=5).median()
    roll_mad = roll_mad.replace(0, np.nan)
    roll_z   = 0.6745 * (s - roll_med) / roll_mad

    global_z             = pd.Series(_mad_zscore(series))
    fallback             = roll_z.isna() & s.notna()
    roll_z[fallback]     = global_z[fallback]
    roll_z[s.isna()]     = np.nan
    return roll_z.values


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Spike cleaner  (operates on zero rate time series, one tenor)
# ══════════════════════════════════════════════════════════════════════════════

class SpikeCleaner:
    """
    Detect and replace spikes in a 1-D zero rate time series.

    Because we are cleaning in zero rate space, a spike in r(T_k, t) affects
    exactly ONE series. Three complementary detectors are run in parallel:

      A. Jump detector  — |Δr| is large on both sides of the point,
                          ruling out genuine level shifts (one large jump).
                          Computed only between consecutive *observed* pairs
                          so NaN gaps do not corrupt the differences.

      B. Global MAD     — level is far from the full-sample median.
                          Catches e.g. decimal/percent conversion errors.

      C. Rolling MAD    — level is far from the local (windowed) median.
                          Catches regime-local anomalies invisible to (B).

    NaN positions are NEVER flagged as spikes; they are tracked separately
    and filled from the same clean-neighbour interpolation pool.

    Parameters
    ----------
    diff_zscore_thresh : float, default 4.0
    level_mad_thresh   : float, default 6.0
    rolling_mad_thresh : float, default 5.0
    rolling_window     : int,   default 20
    min_rate           : float, default -0.02   hard floor (−2 %)
    max_nan_frac       : float, default 0.5     skip series if too sparse
    """

    def __init__(
        self,
        diff_zscore_thresh: float = 4.0,
        level_mad_thresh:   float = 6.0,
        rolling_mad_thresh: float = 5.0,
        rolling_window:     int   = 20,
        min_rate:           float = -0.02,
        max_nan_frac:       float = 0.5,
    ):
        self.diff_zscore_thresh = diff_zscore_thresh
        self.level_mad_thresh   = level_mad_thresh
        self.rolling_mad_thresh = rolling_mad_thresh
        self.rolling_window     = rolling_window
        self.min_rate           = min_rate
        self.max_nan_frac       = max_nan_frac

    def _diff_spike_mask(self, values: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
        """
        Jump detector on consecutive OBSERVED pairs only.
        A point is a spike if the absolute jump coming in AND going out
        both exceed the threshold — ruling out genuine level shifts.
        """
        n       = len(values)
        obs_idx = np.where(~nan_mask)[0]
        obs_val = values[obs_idx]
        if len(obs_val) < 3:
            return np.zeros(n, dtype=bool)

        abs_diff  = np.abs(np.diff(obs_val))
        dz        = _mad_zscore(abs_diff)
        big       = np.abs(dz) > self.diff_zscore_thresh

        spike_obs           = np.zeros(len(obs_idx), dtype=bool)
        spike_obs[1:-1]     = big[:-1] & big[1:]
        mask                = np.zeros(n, dtype=bool)
        mask[obs_idx[spike_obs]] = True
        return mask

    def _level_spike_mask(self, values: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
        obs = np.where(nan_mask, np.nan, values)
        z   = _mad_zscore(obs)
        return np.where(np.isnan(z), False, np.abs(z) > self.level_mad_thresh)

    def _rolling_spike_mask(self, values: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
        obs = np.where(nan_mask, np.nan, values)
        z   = _rolling_mad_zscore(obs, self.rolling_window)
        return np.where(np.isnan(z), False, np.abs(z) > self.rolling_mad_thresh)

    def detect(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (spike_mask, nan_mask).
        spike_mask: True = observed spike to be replaced (never NaN positions).
        nan_mask:   True = originally missing.
        """
        nan_mask   = ~np.isfinite(values)
        spike_mask = (
            self._diff_spike_mask(values, nan_mask)
            | self._level_spike_mask(values, nan_mask)
            | self._rolling_spike_mask(values, nan_mask)
            | ((~nan_mask) & (values < self.min_rate))
        )
        spike_mask &= ~nan_mask      # NaN positions are never spikes
        return spike_mask, nan_mask

    def fix(
        self,
        values:     np.ndarray,
        spike_mask: np.ndarray,
        nan_mask:   np.ndarray,
    ) -> np.ndarray:
        """
        Linearly interpolate spike and NaN positions from clean neighbours.
        'Clean' = not a spike and not NaN.
        """
        fill_mask = spike_mask | nan_mask
        clean_obs = ~fill_mask
        fixed     = values.copy()
        if clean_obs.sum() < 2:
            warnings.warn("Fewer than 2 clean observations — cannot interpolate.")
            fixed[fill_mask] = np.nan
            return fixed
        idx              = np.arange(len(values))
        fixed[fill_mask] = np.interp(idx[fill_mask], idx[clean_obs], values[clean_obs])
        return fixed

    def clean(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-pass spike cleaning.

        Returns
        -------
        fixed_values : ndarray   (spikes and NaNs interpolated)
        spike_mask   : bool ndarray  (True = was a spike, excludes NaNs)
        nan_mask     : bool ndarray  (True = was originally NaN)
        """
        nan_frac = np.mean(~np.isfinite(values))
        if nan_frac > self.max_nan_frac:
            warnings.warn(f"Series is {nan_frac:.0%} NaN — skipping spike cleaning.")
            _, nan_mask = self.detect(values)
            return values.copy(), np.zeros(len(values), dtype=bool), nan_mask

        spike_mask, nan_mask = self.detect(values)
        fixed                = self.fix(values, spike_mask, nan_mask)

        spike_mask2, _  = self.detect(fixed)
        spike_mask2    &= ~nan_mask
        fixed2          = self.fix(fixed, spike_mask2, np.zeros(len(fixed), dtype=bool))

        return fixed2, spike_mask | spike_mask2, nan_mask


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Curvature cleaner  (operates on zero rate curve, one date)
# ══════════════════════════════════════════════════════════════════════════════

class CurvatureCleaner:
    """
    Detect and smooth abnormal curvature in a single zero rate curve.

    In zero rate space curvature checking is economically meaningful:
    a well-behaved zero curve should be smooth across tenors (monotone or
    gently humped). Kinks and oscillations in the zero curve indicate data
    problems that would propagate into multiple IFR pillars after conversion.

    Why NOT in IFR space
    --------------------
    Piecewise-constant IFRs are step functions — they are deliberately
    discontinuous at every pillar. A second-derivative test applied to
    piecewise-constant IFRs fires at every step boundary and produces
    meaningless kink flags. Curvature smoothing of the IFR surface via
    Savitzky-Golay also destroys the cumulative-sum identity:
        sum(f_i * dt_i) = r(T_k) * T_k
    making the round-trip conversion inexact even for unmodified dates.

    NaN handling
    ------------
    All diagnostics operate on observed tenors only.
    NaN tenors are left as NaN in the output (not invented).
    Curves with > max_nan_frac missing tenors are flagged 'too_sparse'
    and skipped (insufficient shape information).

    Parameters
    ----------
    d2_zscore_thresh : float, default 4.0
    max_sign_changes : int,   default 8
    sg_window_frac   : float, default 0.25
    sg_polyorder     : int,   default 3
    min_rate         : float, default -0.02
    max_nan_frac     : float, default 0.4
    """

    def __init__(
        self,
        d2_zscore_thresh: float = 4.0,
        max_sign_changes: int   = 8,
        sg_window_frac:   float = 0.25,
        sg_polyorder:     int   = 3,
        min_rate:         float = -0.02,
        max_nan_frac:     float = 0.4,
    ):
        self.d2_zscore_thresh = d2_zscore_thresh
        self.max_sign_changes = max_sign_changes
        self.sg_window_frac   = sg_window_frac
        self.sg_polyorder     = sg_polyorder
        self.min_rate         = min_rate
        self.max_nan_frac     = max_nan_frac

    def _d2(self, tenors: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Central-difference d2r/dt2 on irregular observed grid."""
        n  = len(rates)
        d2 = np.full(n, np.nan)
        for i in range(1, n - 1):
            h1 = tenors[i] - tenors[i - 1]
            h2 = tenors[i + 1] - tenors[i]
            if h1 > 0 and h2 > 0:
                d2[i] = (
                    2 * (rates[i+1]/h2 - rates[i]*(1/h1 + 1/h2) + rates[i-1]/h1)
                    / (h1 + h2)
                )
        if n > 1:
            if np.isfinite(d2[1]):   d2[0]  = d2[1]
            if np.isfinite(d2[-2]):  d2[-1] = d2[-2]
        return d2

    def _savgol(self, rates_obs: np.ndarray) -> np.ndarray:
        n_obs = len(rates_obs)
        if n_obs < self.sg_polyorder + 2:
            return rates_obs.copy()
        win = max(self.sg_polyorder + 2, int(n_obs * self.sg_window_frac))
        if win % 2 == 0:
            win += 1
        win = min(win, n_obs if n_obs % 2 == 1 else n_obs - 1)
        return savgol_filter(rates_obs, window_length=win, polyorder=self.sg_polyorder)

    def diagnose(self, tenors: np.ndarray, rates: np.ndarray) -> dict:
        """
        Run curvature diagnostics on observed tenors only.
        Returns dict with keys:
            obs_mask, nan_frac, d2_zscores, kink_mask,
            n_sign_changes, oscillating, n_negative,
            too_sparse, needs_fix
        """
        nan_mask   = ~np.isfinite(rates)
        obs_mask   = ~nan_mask
        nan_frac   = float(nan_mask.mean())
        too_sparse = nan_frac > self.max_nan_frac

        t_obs = tenors[obs_mask]
        r_obs = rates[obs_mask]

        if len(r_obs) < 3:
            return dict(
                obs_mask=obs_mask, nan_frac=nan_frac,
                d2_zscores=np.full(len(rates), np.nan),
                kink_mask=np.zeros(len(rates), dtype=bool),
                n_sign_changes=0, oscillating=False,
                n_negative=0, too_sparse=True, needs_fix=False,
            )

        d2_obs   = self._d2(t_obs, r_obs)
        d2z_obs  = _mad_zscore(d2_obs)
        kink_obs = np.where(np.isfinite(d2z_obs),
                            np.abs(d2z_obs) > self.d2_zscore_thresh, False)

        d2z_full            = np.full(len(rates), np.nan)
        kink_full           = np.zeros(len(rates), dtype=bool)
        d2z_full[obs_mask]  = d2z_obs
        kink_full[obs_mask] = kink_obs

        dr          = np.diff(r_obs)
        sign_chg    = int(np.sum(np.diff(np.sign(dr)) != 0)) if len(dr) > 1 else 0
        oscillating = sign_chg > self.max_sign_changes
        n_negative  = int(np.sum(r_obs < self.min_rate))

        needs_fix = (
            (not too_sparse)
            and (bool(kink_full.any()) or oscillating or n_negative > 0)
        )
        return dict(
            obs_mask=obs_mask, nan_frac=nan_frac,
            d2_zscores=d2z_full, kink_mask=kink_full,
            n_sign_changes=sign_chg, oscillating=oscillating,
            n_negative=n_negative, too_sparse=too_sparse,
            needs_fix=needs_fix,
        )

    def fix(self, tenors: np.ndarray, rates: np.ndarray, diag: dict) -> np.ndarray:
        """
        Savitzky-Golay smooth observed tenors; preserve NaN positions and endpoints.
        """
        if not diag["needs_fix"]:
            return rates.copy()
        obs_mask         = diag["obs_mask"]
        r_obs            = rates[obs_mask]
        sm               = self._savgol(r_obs)
        sm[0]            = r_obs[0]
        sm[-1]           = r_obs[-1]
        sm               = np.maximum(sm, self.min_rate)
        result           = rates.copy()
        result[obs_mask] = sm
        return result

    def clean(self, tenors: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, dict]:
        """Returns (cleaned_rates, diagnostic_dict). NaN positions stay NaN."""
        diag = self.diagnose(tenors, rates)
        return self.fix(tenors, rates, diag), diag


# ══════════════════════════════════════════════════════════════════════════════
# Piecewise-constant  Zero <-> IFR  converter
# ══════════════════════════════════════════════════════════════════════════════

class PiecewiseConstantConverter:
    """
    Exact conversion between continuously compounded zero rates and
    piecewise-constant instantaneous forward rates.

        f_i = ( r(T_i)*T_i  -  r(T_{i-1})*T_{i-1} )  /  (T_i - T_{i-1})

        r(T_k) = (1/T_k) * sum_{i=1}^{k}  f_i * (T_i - T_{i-1})

    After cleaning in zero rate space, zero_to_ifr is called exactly once.
    The round-trip  zero -> ifr -> zero  is exact (machine epsilon) by
    construction — no smoothing has touched the IFRs.

    NaN propagation
    ---------------
    zero_to_ifr : NaN in pillar i makes f_i and f_{i+1} NaN (both neighbours
                  of the gap are affected, since f uses a difference).
    ifr_to_zero : NaN in f_i propagates to all r(T_k) with k >= i.
    Because cleaning is done before conversion, NaN positions in the
    clean zero surface are filled by SpikeCleaner. The only remaining NaNs
    after cleaning are positions flagged 'too_sparse' — those propagate.
    """

    def __init__(self, tenors: np.ndarray):
        self.tenors  = np.asarray(tenors, dtype=float)
        if np.any(self.tenors <= 0):
            raise ValueError("All tenors must be strictly positive.")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("Tenors must be strictly increasing.")
        self._T_left  = np.concatenate([[0.0], self.tenors[:-1]])
        self._T_right = self.tenors
        self._dt      = self._T_right - self._T_left

    def zero_to_ifr(self, zero_df: pd.DataFrame) -> pd.DataFrame:
        Z         = zero_df.values.astype(float)
        T         = self.tenors
        area      = Z * T[np.newaxis, :]
        area_left = np.zeros_like(area)
        area_left[:, 1:] = area[:, :-1]
        ifr       = (area - area_left) / self._dt[np.newaxis, :]
        return pd.DataFrame(ifr, index=zero_df.index, columns=zero_df.columns)

    def ifr_to_zero(self, ifr_df: pd.DataFrame) -> pd.DataFrame:
        F        = ifr_df.values.astype(float)
        dt       = self._dt[np.newaxis, :]
        cum      = np.nancumsum(F * dt, axis=1)
        nan_prop = np.cumsum(~np.isfinite(F), axis=1) > 0
        cum      = np.where(nan_prop, np.nan, cum)
        zero     = cum / self.tenors[np.newaxis, :]
        return pd.DataFrame(zero, index=ifr_df.index, columns=ifr_df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    zero_raw:   pd.DataFrame   # original input zero rates
    zero_clean: pd.DataFrame   # zero rates after spike + curvature cleaning
    ifr_clean:  pd.DataFrame   # piecewise-constant IFRs from clean zeros
    report:     dict           # audit log


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

class ZeroIFRPipeline:
    """
    Correct round-trip: clean zero rates -> convert once to piecewise IFRs.

    Pipeline
    --------
    Step 1 — Spike removal (per tenor, across time)
        SpikeCleaner runs on each zero rate time series independently.
        A spike in r(T_k, t) affects exactly one column — no cross-tenor
        contamination.

    Step 2 — Curvature correction (per date, across tenors)
        CurvatureCleaner checks the zero curve shape on each date.
        d2 on the zero curve has clear economic meaning (smoothness of
        the term structure). Fixing here prevents multiple IFR pillars
        from being distorted after conversion.

    Step 3 — Single conversion to piecewise-constant IFRs
        zero_to_ifr is called once on the fully cleaned zero surface.
        The cumulative-sum identity is preserved exactly.

    Parameters
    ----------
    zero_df : pd.DataFrame
        Index = dates, columns = tenors (float years), values = c.c. zero
        rates in decimal. May contain np.nan.

    spike_kwargs : dict, optional
        Overrides for SpikeCleaner.

    curve_kwargs : dict, optional
        Overrides for CurvatureCleaner.
    """

    def __init__(
        self,
        zero_df:      pd.DataFrame,
        spike_kwargs: dict | None = None,
        curve_kwargs: dict | None = None,
    ):
        self.zero_df   = zero_df.copy().astype(float)
        self.tenors    = np.array(zero_df.columns, dtype=float)
        self.converter = PiecewiseConstantConverter(self.tenors)
        self.spike_kw  = spike_kwargs or {}
        self.curve_kw  = curve_kwargs or {}

    # ── cleaning steps ────────────────────────────────────────────────────────

    def _step1_spikes(
        self, zero_df: pd.DataFrame, verbose: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Spike removal on each zero rate tenor series."""
        sc          = SpikeCleaner(**self.spike_kw)
        cleaned     = zero_df.copy()
        spike_flags = pd.DataFrame(False, index=zero_df.index, columns=zero_df.columns)
        nan_flags   = pd.DataFrame(False, index=zero_df.index, columns=zero_df.columns)

        for tenor in zero_df.columns:
            vals, s_mask, n_mask = sc.clean(zero_df[tenor].values)
            cleaned[tenor]       = vals
            spike_flags[tenor]   = s_mask
            nan_flags[tenor]     = n_mask

        if verbose:
            n_spikes = int(spike_flags.values.sum())
            n_nans   = int(nan_flags.values.sum())
            print(f"  NaNs found   : {n_nans} "
                  f"({nan_flags.any(axis=1).sum()} dates, "
                  f"{nan_flags.any(axis=0).sum()} tenors)")
            print(f"  Spikes fixed : {n_spikes} "
                  f"({spike_flags.any(axis=1).sum()} dates affected)")
        return cleaned, spike_flags, nan_flags

    def _step2_curvature(
        self, zero_df: pd.DataFrame, verbose: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Curvature correction on each date's zero rate curve."""
        cc         = CurvatureCleaner(**self.curve_kw)
        cleaned    = zero_df.copy()
        curve_rows = []

        for date in zero_df.index:
            rates        = zero_df.loc[date].values
            fixed, diag  = cc.clean(self.tenors, rates)
            cleaned.loc[date] = fixed
            curve_rows.append({
                "date":           date,
                "needs_fix":      diag["needs_fix"],
                "too_sparse":     diag["too_sparse"],
                "nan_frac":       round(diag["nan_frac"], 4),
                "n_sign_changes": diag["n_sign_changes"],
                "n_kinks":        int(diag["kink_mask"].sum()),
                "n_negative":     diag["n_negative"],
            })

        curve_report = pd.DataFrame(curve_rows).set_index("date")
        if verbose:
            n_fixed  = int(curve_report["needs_fix"].sum())
            n_sparse = int(curve_report["too_sparse"].sum())
            print(f"  Curves fixed : {n_fixed} / {len(curve_report)} dates")
            if n_sparse:
                print(f"  Curves skipped (too sparse) : {n_sparse}")
        return cleaned, curve_report

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> PipelineResult:
        """
        Execute the three-step pipeline.

        Returns PipelineResult with:
            .zero_raw    original input
            .zero_clean  after spike + curvature cleaning
            .ifr_clean   piecewise-constant IFRs (single conversion)
            .report      dict: spike_flags, nan_flags, curve_report,
                               n_spikes, n_nans, n_abnormal_curves
        """
        if verbose:
            print("Step 1 — spike removal on zero rates (per tenor)")
        zero_s1, spike_flags, nan_flags = self._step1_spikes(
            self.zero_df, verbose
        )

        if verbose:
            print("Step 2 — curvature correction on zero rates (per date)")
        zero_clean, curve_report = self._step2_curvature(zero_s1, verbose)

        if verbose:
            print("Step 3 — single conversion to piecewise-constant IFRs")
        ifr_clean = self.converter.zero_to_ifr(zero_clean)

        # Round-trip sanity: zero -> ifr -> zero on clean surface = exact
        rt_err = float(np.nanmax(np.abs(
            self.converter.ifr_to_zero(ifr_clean).values - zero_clean.values
        )))
        if verbose:
            print(f"  Round-trip error (zero -> IFR -> zero) : {rt_err:.2e}  <- should be ~0")

        # Additionally verify untouched dates match original exactly
        not_touched = (
            ~spike_flags.any(axis=1)
            & ~nan_flags.any(axis=1)
            & ~curve_report["needs_fix"]
        )
        if not_touched.any():
            orig_err = float(np.nanmax(np.abs(
                zero_clean.loc[not_touched].values
                - self.zero_df.loc[not_touched].values
            )))
            if verbose:
                print(f"  Max change on untouched dates        : {orig_err:.2e}  <- should be ~0")

        report = {
            "spike_flags":       spike_flags,
            "nan_flags":         nan_flags,
            "curve_report":      curve_report,
            "n_spikes":          int(spike_flags.values.sum()),
            "n_nans":            int(nan_flags.values.sum()),
            "n_abnormal_curves": int(curve_report["needs_fix"].sum()),
            "n_sparse_curves":   int(curve_report["too_sparse"].sum()),
        }
        return PipelineResult(
            zero_raw   = self.zero_df,
            zero_clean = zero_clean,
            ifr_clean  = ifr_clean,
            report     = report,
        )

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot_tenor_timeseries(self, tenor, result, figsize=(13, 4)):
        """Zero rate time series for one tenor: raw vs clean, spikes marked."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed."); return
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(result.zero_raw.index,   result.zero_raw[tenor],
                lw=1,   alpha=0.6, color="steelblue",  label="zero raw")
        ax.plot(result.zero_clean.index, result.zero_clean[tenor],
                lw=1.2, color="darkorange", label="zero clean")
        nan_pos = result.report["nan_flags"][tenor]
        ax.scatter(result.zero_raw.index[nan_pos],
                   result.zero_clean[tenor][nan_pos],
                   color="green", s=15, zorder=5, label="NaN (filled)")
        spk_pos = result.report["spike_flags"][tenor]
        ax.scatter(result.zero_raw.index[spk_pos],
                   result.zero_raw[tenor][spk_pos],
                   color="red", s=25, zorder=6, label="spike fixed")
        ax.set_title(f"Zero rate — tenor {tenor}y")
        ax.set_ylabel("Zero rate"); ax.legend()
        plt.tight_layout(); plt.show()

    def plot_curve_date(self, date, result, figsize=(13, 4)):
        """
        For one date: zero curve (raw vs clean) and the resulting
        piecewise-constant IFR derived from the clean zeros.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed."); return
        T   = self.tenors
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.plot(T, result.zero_raw.loc[date].values,   lw=1.5,
                color="steelblue",              label="zero raw")
        ax.plot(T, result.zero_clean.loc[date].values, lw=1.5,
                color="darkorange", ls="--",    label="zero clean")
        nan_mask = ~np.isfinite(result.zero_raw.loc[date].values)
        if nan_mask.any():
            ax.scatter(T[nan_mask], result.zero_clean.loc[date].values[nan_mask],
                       color="green", s=30, zorder=5, label="NaN (filled)")
        ax.set_title(f"Zero curve — {date}"); ax.set_xlabel("Tenor (y)")
        ax.set_ylabel("Zero rate"); ax.legend()

        ax = axes[1]
        ax.step(T, result.ifr_clean.loc[date].values, where="post",
                lw=1.5, color="purple", label="IFR (from clean zeros)")
        ax.set_title(f"Piecewise-constant IFR — {date}")
        ax.set_xlabel("Tenor (y)"); ax.set_ylabel("Forward rate"); ax.legend()

        plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng    = np.random.default_rng(0)
    dates  = pd.date_range("2018-01-01", periods=500, freq="B")
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

    base  = np.array([0.015, 0.018, 0.022, 0.028, 0.032, 0.038,
                      0.042, 0.046, 0.049, 0.050, 0.051])
    drift = np.cumsum(rng.normal(0, 0.0002, (len(dates), len(tenors))), axis=0)
    zeros = np.clip(base + drift, 0.001, 0.12)

    # Inject spikes in zero rates (each spike contaminates exactly 1 series)
    for _ in range(25):
        r, c = rng.integers(0, len(dates)), rng.integers(0, len(tenors))
        zeros[r, c] += rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)

    # Inject NaNs
    for r, c in zip(rng.choice(len(dates), 40, replace=False),
                    rng.choice(len(tenors), 40, replace=True)):
        zeros[r, c] = np.nan
    zeros[100, 7:] = np.nan     # illiquid long end on one date

    zero_df  = pd.DataFrame(zeros, index=dates, columns=tenors)
    pipeline = ZeroIFRPipeline(zero_df)
    result   = pipeline.run(verbose=True)

    print("\n── Round-trip: zero_clean -> IFR -> zero  (should be ~0 everywhere) ──")
    rt = np.abs(
        pipeline.converter.ifr_to_zero(result.ifr_clean).values
        - result.zero_clean.values
    )
    print(f"  Max : {np.nanmax(rt):.2e}   Mean : {np.nanmean(rt):.2e}")

    print("\n── Date 100 (long-end NaNs) ──")
    d    = dates[100]
    comp = pd.DataFrame({
        "zero_raw":   result.zero_raw.loc[d],
        "zero_clean": result.zero_clean.loc[d],
        "ifr_clean":  result.ifr_clean.loc[d],
    })
    print(comp.to_string())

    # pipeline.plot_tenor_timeseries(5, result)
    # pipeline.plot_curve_date(dates[10], result)
