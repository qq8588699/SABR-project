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

def _dates_to_numeric(dates: pd.Index) -> np.ndarray:
    """
    Convert any pandas DatetimeIndex (or other index) to a numeric array
    of days elapsed from the first date, preserving the true calendar gaps
    between observations.

    Examples
    --------
    Daily business-day index  → [0, 1, 2, 3, ...]         (no gaps)
    Missing Wednesday         → [0, 1, 3, 4, ...]         (gap of 2 on Thursday)
    Monthly dates             → [0, 31, 59, 90, ...]
    Integer index (fallback)  → [0, 1, 2, 3, ...]         (treated as days)

    This array is used as the x-axis for np.interp so that interpolated
    values are weighted by actual elapsed time, not by row position.
    """
    if isinstance(dates, pd.DatetimeIndex):
        days = (dates - dates[0]).days.to_numpy(dtype=float)
    else:
        # Fallback for integer or other index types: treat as equally spaced
        days = np.arange(len(dates), dtype=float)
    return days


# ══════════════════════════════════════════════════════════════════════════════
# Input normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

# Days-per-year convention used throughout (ACT/365)
_DAYS_PER_YEAR = 365.0


def _normalise_index(index: pd.Index) -> tuple[pd.DatetimeIndex, str]:
    """
    Accept any of these row-index formats and return a DatetimeIndex plus
    a format tag so the original format can be restored on output.

    Accepted formats
    ----------------
    1. pd.DatetimeIndex                 → used as-is          tag = "datetime"
    2. strings  "YYYYMMDD"              → parsed with %Y%m%d  tag = "%Y%m%d"
    3. strings  "YYYY-MM-DD"            → parsed with %Y-%m-%d tag = "%Y-%m-%d"
    4. anything else                    → pd.to_datetime()     tag = "datetime"

    The returned DatetimeIndex has no timezone so it is always compatible
    with (dates - dates[0]).days used in _dates_to_numeric().
    """
    if isinstance(index, pd.DatetimeIndex):
        return index.normalize(), "datetime"

    sample = str(index[0]).strip()

    if len(sample) == 8 and sample.isdigit():
        # "20180103" → %Y%m%d
        dt = pd.to_datetime(index, format="%Y%m%d")
        return dt, "%Y%m%d"

    if len(sample) == 10 and sample[4] == "-" and sample[7] == "-":
        # "2018-01-03" → %Y-%m-%d
        dt = pd.to_datetime(index, format="%Y-%m-%d")
        return dt, "%Y-%m-%d"

    # Fallback: let pandas infer
    dt = pd.to_datetime(index)
    return dt, "datetime"


def _restore_index(dt_index: pd.DatetimeIndex, fmt_tag: str) -> pd.Index:
    """
    Convert a DatetimeIndex back to the original format.

    fmt_tag == "datetime"  → return DatetimeIndex unchanged
    fmt_tag == "%Y%m%d"    → return Index of strings "YYYYMMDD"
    fmt_tag == "%Y-%m-%d"  → return Index of strings "YYYY-MM-DD"
    """
    if fmt_tag == "datetime":
        return dt_index
    return pd.Index(dt_index.strftime(fmt_tag), name=dt_index.name)


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

    Non-consecutive date handling
    -----------------------------
    Three places depend on the x-axis used for interpolation or windowing:

      fix()              — np.interp uses calendar days, not integer positions,
                           so a spike between dates 5 days apart is interpolated
                           differently from one between dates 1 day apart.

      _diff_spike_mask() — jump sizes are divided by the calendar gap between
                           consecutive observed dates, converting absolute
                           differences to per-day rates of change before the
                           MAD z-score is applied. This prevents a genuine
                           slow drift over a long holiday gap from being
                           misclassified as a spike.

      _rolling_mad_zscore (via rolling_window) — the rolling window parameter
                           is expressed in number of *observations* (rows), not
                           calendar days, so it already adapts naturally to gaps.
                           No change needed there.

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

    def _diff_spike_mask(
        self,
        values:   np.ndarray,
        nan_mask: np.ndarray,
        t:        np.ndarray,        # calendar-day positions, same length as values
    ) -> np.ndarray:
        """
        Jump detector on consecutive OBSERVED pairs only.

        The raw difference between two observed values is divided by the
        calendar gap between their dates to give a per-day rate of change:

            rate_of_change = |r_j - r_i| / (t_j - t_i)

        Without this normalisation a genuine slow drift spanning a long
        holiday gap (e.g. 10 days) would produce a large absolute difference
        and be wrongly flagged as a spike, even though the per-day move is
        perfectly normal.

        A point is a spike if the normalised jump coming IN and the
        normalised jump going OUT both exceed the threshold — ruling out
        genuine level shifts (which produce only one large jump).
        """
        n       = len(values)
        obs_idx = np.where(~nan_mask)[0]
        obs_val = values[obs_idx]
        obs_t   = t[obs_idx]                          # calendar positions of observed points

        if len(obs_val) < 3:
            return np.zeros(n, dtype=bool)

        # Calendar gaps between consecutive observed dates
        gaps     = np.diff(obs_t)                     # always > 0 (dates are sorted)
        gaps     = np.maximum(gaps, 1e-6)             # guard against duplicate timestamps

        # Per-day absolute rate of change between consecutive observed pairs
        abs_roc  = np.abs(np.diff(obs_val)) / gaps    # length = len(obs_idx) - 1

        dz       = _mad_zscore(abs_roc)
        big      = np.abs(dz) > self.diff_zscore_thresh

        spike_obs            = np.zeros(len(obs_idx), dtype=bool)
        spike_obs[1:-1]      = big[:-1] & big[1:]     # large jump in AND out
        mask                 = np.zeros(n, dtype=bool)
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

    def detect(
        self,
        values: np.ndarray,
        t:      np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (spike_mask, nan_mask).
        spike_mask: True = observed spike to be replaced (never NaN positions).
        nan_mask:   True = originally missing.

        Parameters
        ----------
        values : ndarray   — zero rate values
        t      : ndarray   — numeric date positions (days from origin),
                             same length as values, from _dates_to_numeric()
        """
        nan_mask   = ~np.isfinite(values)
        spike_mask = (
            self._diff_spike_mask(values, nan_mask, t)
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
        t:          np.ndarray,
    ) -> np.ndarray:
        """
        Linearly interpolate spike and NaN positions from clean neighbours,
        using actual calendar-day positions as the x-axis.

        Why calendar days matter
        ------------------------
        np.interp(x_fill, x_clean, y_clean) weights the interpolation by
        the distance along the x-axis. Using integer row positions (0,1,2...)
        treats every row gap as equal, so a spike surrounded by a 1-day gap
        on the left and a 10-day gap on the right would be filled as the
        simple midpoint. Using calendar days, the interpolated value is pulled
        toward the closer (in time) clean neighbour — which is the correct
        linear interpolation along the actual time axis.

        Parameters
        ----------
        values     : ndarray — original values
        spike_mask : ndarray — True where a spike was detected
        nan_mask   : ndarray — True where value is NaN
        t          : ndarray — calendar-day positions from _dates_to_numeric()
        """
        fill_mask = spike_mask | nan_mask
        clean_obs = ~fill_mask
        fixed     = values.copy()
        if clean_obs.sum() < 2:
            warnings.warn("Fewer than 2 clean observations — cannot interpolate.")
            fixed[fill_mask] = np.nan
            return fixed

        # Use calendar-day positions, not integer row indices
        fixed[fill_mask] = np.interp(
            t[fill_mask],          # x-positions to fill (calendar days)
            t[clean_obs],          # x-positions of clean anchors
            values[clean_obs],     # y-values of clean anchors
        )
        return fixed

    def clean(
        self,
        values: np.ndarray,
        t:      np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-pass spike cleaning using calendar-aware interpolation.

        Parameters
        ----------
        values : ndarray — zero rate time series for one tenor
        t      : ndarray — numeric date positions from _dates_to_numeric()

        Returns
        -------
        fixed_values : ndarray   (spikes and NaNs interpolated)
        spike_mask   : bool ndarray  (True = was a spike, excludes NaNs)
        nan_mask     : bool ndarray  (True = was originally NaN)
        """
        nan_frac = np.mean(~np.isfinite(values))
        if nan_frac > self.max_nan_frac:
            warnings.warn(f"Series is {nan_frac:.0%} NaN — skipping spike cleaning.")
            _, nan_mask = self.detect(values, t)
            return values.copy(), np.zeros(len(values), dtype=bool), nan_mask

        # Pass 1
        spike_mask, nan_mask = self.detect(values, t)
        fixed                = self.fix(values, spike_mask, nan_mask, t)

        # Pass 2 — catch artefacts introduced by first interpolation
        spike_mask2, _  = self.detect(fixed, t)
        spike_mask2    &= ~nan_mask
        fixed2          = self.fix(
            fixed, spike_mask2, np.zeros(len(fixed), dtype=bool), t
        )

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
    Step 2 — Curvature correction (per date, across tenors)
    Step 3 — Single conversion to piecewise-constant IFRs

    Input format
    ------------
    Row index (dates)
        Accepted: pd.DatetimeIndex, strings "YYYYMMDD", strings "YYYY-MM-DD".
        Detected automatically; original format is restored on all output
        DataFrames.

    Columns (tenors)
        The column values are taken directly as the tenor axis — no detection
        or heuristic is applied.  You declare the unit via `tenor_unit`:
          • tenor_unit="days"  (default) — columns are used as-is in days,
                                e.g. [91, 182, 365, 730, 1825, 3650, 10950]
          • tenor_unit="years"           — columns are multiplied by 365
                                to convert to days internally,
                                e.g. [0.25, 0.5, 1, 2, 5, 10, 30]
        Original column labels are always preserved on all output DataFrames.

    Output guarantee
    ----------------
    result.zero_raw   — identical index and columns to the input zero_df
    result.zero_clean — same index format and column labels as input zero_df
    result.ifr_clean  — same index format and column labels as input zero_df

    Parameters
    ----------
    zero_df : pd.DataFrame
        Index = dates (see above), columns = tenors (numeric, in the unit
        declared by tenor_unit). Values = continuously compounded zero rates
        in decimal. May contain NaN.
    tenor_unit : str, default "days"
        Unit of the tenor columns: "days" or "years".
    spike_kwargs : dict, optional
        Overrides for SpikeCleaner.
    curve_kwargs : dict, optional
        Overrides for CurvatureCleaner.
    """

    def __init__(
        self,
        zero_df:      pd.DataFrame,
        tenor_unit:   str = "days",
        spike_kwargs: dict | None = None,
        curve_kwargs: dict | None = None,
    ):
        if tenor_unit not in ("days", "years"):
            raise ValueError(f"tenor_unit must be 'days' or 'years', got {tenor_unit!r}")

        # ── 1. normalise row index → DatetimeIndex ────────────────────────────
        dt_index, self._index_fmt = _normalise_index(zero_df.index)

        # ── 2. read tenor columns directly; convert to days if needed ─────────
        try:
            col_vals = np.array(zero_df.columns, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column labels must be numeric tenors, got: {zero_df.columns[:5].tolist()!r}"
            ) from exc

        tenors_days = col_vals * _DAYS_PER_YEAR if tenor_unit == "years" else col_vals

        if np.any(tenors_days <= 0):
            raise ValueError("All tenors must be strictly positive.")
        if np.any(np.diff(tenors_days) <= 0):
            raise ValueError("Tenors must be strictly increasing.")

        # ── 3. build internal working copy: DatetimeIndex + float-day columns ─
        self._internal_df = pd.DataFrame(
            zero_df.values.astype(float),
            index   = dt_index,
            columns = tenors_days,
        )

        # Store original df and column labels for output restoration
        self._orig_df      = zero_df.copy()
        self._orig_columns = zero_df.columns
        self._tenor_unit   = tenor_unit

        self.tenors    = tenors_days          # float days, used internally
        self.converter = PiecewiseConstantConverter(self.tenors)
        self.spike_kw  = spike_kwargs or {}
        self.curve_kw  = curve_kwargs or {}

    # ── private: restore original index and column labels ────────────────────

    def _restore_output(self, df_internal: pd.DataFrame) -> pd.DataFrame:
        """
        Given an internal DataFrame (DatetimeIndex, float-day columns),
        return a copy with the original index format and original column labels.
        """
        restored_index = _restore_index(df_internal.index, self._index_fmt)
        return pd.DataFrame(
            df_internal.values,
            index   = restored_index,
            columns = self._orig_columns,
        )

    # ── cleaning steps ────────────────────────────────────────────────────────

    def _step1_spikes(
        self, zero_df: pd.DataFrame, verbose: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Spike removal on each zero rate tenor series."""
        sc          = SpikeCleaner(**self.spike_kw)
        cleaned     = zero_df.copy()
        spike_flags = pd.DataFrame(False, index=zero_df.index, columns=zero_df.columns)
        nan_flags   = pd.DataFrame(False, index=zero_df.index, columns=zero_df.columns)

        # Convert the date index to calendar-day numeric positions once.
        # This is shared across all tenors since they all share the same index.
        t = _dates_to_numeric(zero_df.index)

        for tenor in zero_df.columns:
            vals, s_mask, n_mask = sc.clean(zero_df[tenor].values, t)
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
            .zero_raw    original input (unchanged index and columns)
            .zero_clean  cleaned zeros  (same index format and columns as input)
            .ifr_clean   IFRs           (same index format and columns as input)
            .report      audit log
        """
        if verbose:
            print(f"  tenor_unit   : {self._tenor_unit!r}"
                  + (" (× 365 → days internally)" if self._tenor_unit == "years" else ""))
            print(f"  Index format : {self._index_fmt!r}")

        if verbose:
            print("Step 1 — spike removal on zero rates (per tenor)")
        zero_s1, spike_flags, nan_flags = self._step1_spikes(
            self._internal_df, verbose
        )

        if verbose:
            print("Step 2 — curvature correction on zero rates (per date)")
        zero_clean_internal, curve_report = self._step2_curvature(zero_s1, verbose)

        if verbose:
            print("Step 3 — single conversion to piecewise-constant IFRs")
        ifr_clean_internal = self.converter.zero_to_ifr(zero_clean_internal)

        # Round-trip sanity check (on internal float-year representation)
        rt_err = float(np.nanmax(np.abs(
            self.converter.ifr_to_zero(ifr_clean_internal).values
            - zero_clean_internal.values
        )))
        if verbose:
            print(f"  Round-trip error (zero -> IFR -> zero) : {rt_err:.2e}  <- should be ~0")

        not_touched = (
            ~spike_flags.any(axis=1)
            & ~nan_flags.any(axis=1)
            & ~curve_report["needs_fix"]
        )
        if not_touched.any():
            orig_err = float(np.nanmax(np.abs(
                zero_clean_internal.loc[not_touched].values
                - self._internal_df.loc[not_touched].values
            )))
            if verbose:
                print(f"  Max change on untouched dates        : {orig_err:.2e}  <- should be ~0")

        # ── restore original index and column labels on all outputs ───────────
        zero_clean = self._restore_output(zero_clean_internal)
        ifr_clean  = self._restore_output(ifr_clean_internal)

        # spike_flags and nan_flags also get original labels
        spike_flags = self._restore_output(
            spike_flags.astype(float)
        ).astype(bool)
        nan_flags = self._restore_output(
            nan_flags.astype(float)
        ).astype(bool)

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
            zero_raw   = self._orig_df,
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
    rng         = np.random.default_rng(0)
    dates       = pd.date_range("2018-01-01", periods=500, freq="B")
    tenors_days = [91, 182, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300, 10950]
    tenors_yrs  = [d / 365.0 for d in tenors_days]

    base  = np.array([0.015, 0.018, 0.022, 0.028, 0.032, 0.038,
                      0.042, 0.046, 0.049, 0.050, 0.051])
    drift = np.cumsum(rng.normal(0, 0.0002, (len(dates), len(tenors_days))), axis=0)
    zeros = np.clip(base + drift, 0.001, 0.12)
    for _ in range(25):
        r, c = rng.integers(0, len(dates)), rng.integers(0, len(tenors_days))
        zeros[r, c] += rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
    for r, c in zip(rng.choice(len(dates), 40, replace=False),
                    rng.choice(len(tenors_days), 40, replace=True)):
        zeros[r, c] = np.nan

    # ── Test A: DatetimeIndex + tenor_unit="days" ─────────────────────────────
    print("=" * 60)
    print("TEST A — DatetimeIndex  +  tenor_unit='days'")
    print("=" * 60)
    df_A     = pd.DataFrame(zeros, index=dates, columns=tenors_days)
    result_A = ZeroIFRPipeline(df_A, tenor_unit="days").run(verbose=True)
    print(f"  zero_clean columns : {list(result_A.zero_clean.columns[:3])} ...")
    print(f"  zero_clean index[0]: {result_A.zero_clean.index[0]!r}")

    # ── Test B: DatetimeIndex + tenor_unit="years" ────────────────────────────
    print("\n" + "=" * 60)
    print("TEST B — DatetimeIndex  +  tenor_unit='years'")
    print("=" * 60)
    df_B     = pd.DataFrame(zeros, index=dates, columns=tenors_yrs)
    result_B = ZeroIFRPipeline(df_B, tenor_unit="years").run(verbose=True)
    print(f"  zero_clean columns : {[round(c,4) for c in result_B.zero_clean.columns[:3]]} ...")
    print(f"  zero_clean index[0]: {result_B.zero_clean.index[0]!r}")

    # ── Test C: string index YYYYMMDD + tenor_unit="days" ─────────────────────
    print("\n" + "=" * 60)
    print("TEST C — string index YYYYMMDD  +  tenor_unit='days'")
    print("=" * 60)
    df_C     = pd.DataFrame(zeros,
                            index   = pd.Index(dates.strftime("%Y%m%d")),
                            columns = tenors_days)
    result_C = ZeroIFRPipeline(df_C, tenor_unit="days").run(verbose=True)
    print(f"  zero_clean columns : {list(result_C.zero_clean.columns[:3])} ...")
    print(f"  zero_clean index[0]: {result_C.zero_clean.index[0]!r}  (restored YYYYMMDD)")

    # ── Test D: string index YYYY-MM-DD + tenor_unit="years" ──────────────────
    print("\n" + "=" * 60)
    print("TEST D — string index YYYY-MM-DD  +  tenor_unit='years'")
    print("=" * 60)
    df_D     = pd.DataFrame(zeros,
                            index   = pd.Index(dates.strftime("%Y-%m-%d")),
                            columns = tenors_yrs)
    result_D = ZeroIFRPipeline(df_D, tenor_unit="years").run(verbose=True)
    print(f"  zero_clean columns : {[round(c,4) for c in result_D.zero_clean.columns[:3]]} ...")
    print(f"  zero_clean index[0]: {result_D.zero_clean.index[0]!r}  (restored YYYY-MM-DD)")

    # ── Format round-trip verification ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FORMAT ROUND-TRIP: output columns/index match input exactly")
    print("=" * 60)
    for name, df_in, result in [
        ("A", df_A, result_A), ("B", df_B, result_B),
        ("C", df_C, result_C), ("D", df_D, result_D),
    ]:
        cols_ok  = df_in.columns.equals(result.zero_clean.columns)
        index_ok = df_in.index.equals(result.zero_clean.index)
        print(f"  Test {name}: columns match={cols_ok}  index match={index_ok}")


