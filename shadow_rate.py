import numpy as np
from typing import Union, Tuple, Optional


class ShadowRateTransformation:
    """
    Piecewise shadow rate transformation for instantaneous forward rates.

    The transformation maps observed rates r into a shadow space s where
    the full real line is accessible, making s suitable for simulation
    under a normal stochastic process. The map is C1-continuous at the
    switch point r*.

    Parameters
    ----------
    switch_point : float
        The rate level r* at which the process switches from lognormal
        (below) to normal (above). Recommended at or below zero for
        currencies like EUR that exhibit negative rates.
    floor : float
        The asymptotic floor r_floor for the instantaneous forward rate.
        As shadow rate s -> -inf, the actual rate r -> floor from above.
        Should be set well below observed historical lows (e.g. -0.02 for EUR).

    Notes
    -----
    Delta is pinned by the C1 continuity condition at r*:

        d(r)/d(s)|_{s=r*} = 1  =>  delta = r* - r_floor

    This ensures the slope of the transformation is continuous at the
    switch point, preventing artefacts in the simulated rate distribution.

    Regimes
    -------
    Forward map T(r):
        r >= r*:  s = r
        r <  r*:  s = r* + delta * ln((r - r_floor) / (r* - r_floor))

    Inverse map T^{-1}(s):
        s >= r*:  r = s
        s <  r*:  r = r_floor + (r* - r_floor) * exp((s - r*) / delta)
    """

    def __init__(self, switch_point: float, floor: float):
        if floor >= switch_point:
            raise ValueError(
                f"floor ({floor}) must be strictly less than "
                f"switch_point ({switch_point})."
            )
        self.switch_point = switch_point
        self.floor = floor
        self.delta = switch_point - floor

    # ------------------------------------------------------------------
    # Forward transformation: r -> s
    # ------------------------------------------------------------------

    def to_shadow(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Forward transformation T(r): observed rate -> shadow rate.

        For r >= r*:  s = r
        For r <  r*:  s = r* + delta * ln((r - r_floor) / (r* - r_floor))

        Parameters
        ----------
        r : float or np.ndarray
            Instantaneous forward rate(s). Must satisfy r > r_floor.

        Returns
        -------
        float or np.ndarray
            Shadow rate(s) s = T(r).

        Raises
        ------
        ValueError
            If any rate is at or below the floor.
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 0
        r = np.atleast_1d(r)

        if np.any(r <= self.floor):
            raise ValueError(
                f"All rates must be strictly above the floor ({self.floor}). "
                f"Got min(r) = {r.min():.6f}."
            )

        s = np.where(
            r >= self.switch_point,
            r,
            self.switch_point
            + self.delta * np.log((r - self.floor) / (self.switch_point - self.floor))
        )

        return float(s[0]) if scalar_input else s

    # ------------------------------------------------------------------
    # Inverse transformation: s -> r
    # ------------------------------------------------------------------

    def from_shadow(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Inverse transformation T^{-1}(s): shadow rate -> observed rate.

        For s >= r*:  r = s
        For s <  r*:  r = r_floor + (r* - r_floor) * exp((s - r*) / delta)

        Parameters
        ----------
        s : float or np.ndarray
            Shadow rate(s).

        Returns
        -------
        float or np.ndarray
            Instantaneous forward rate(s). Guaranteed r > r_floor for all finite s.
        """
        s = np.asarray(s, dtype=float)
        scalar_input = s.ndim == 0
        s = np.atleast_1d(s)

        r = np.where(
            s >= self.switch_point,
            s,
            self.floor
            + (self.switch_point - self.floor) * np.exp((s - self.switch_point) / self.delta)
        )

        return float(r[0]) if scalar_input else r

    # ------------------------------------------------------------------
    # Derivative: dr/ds
    # ------------------------------------------------------------------

    def jacobian(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Derivative dr/ds of the inverse transformation.

        For s >= r*:  dr/ds = 1
        For s <  r*:  dr/ds = (r* - r_floor) / delta * exp((s - r*) / delta)
                             = (r - r_floor) / delta

        At s = r* both expressions equal 1, confirming C1 continuity.

        Parameters
        ----------
        s : float or np.ndarray
            Shadow rate(s).

        Returns
        -------
        float or np.ndarray
            Jacobian dr/ds evaluated at s.
        """
        s = np.asarray(s, dtype=float)
        scalar_input = s.ndim == 0
        s = np.atleast_1d(s)

        jac = np.where(
            s >= self.switch_point,
            1.0,
            ((self.switch_point - self.floor) / self.delta)
            * np.exp((s - self.switch_point) / self.delta)
        )

        return float(jac[0]) if scalar_input else jac

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        r0: float,
        mu: float,
        sigma: float,
        dt: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate instantaneous forward rate paths via Euler-Maruyama on
        the shadow rate, then invert back to rate space.

        The shadow rate follows:
            ds = mu * dt + sigma * dW,   dW ~ N(0, dt)

        Parameters
        ----------
        r0 : float
            Initial instantaneous forward rate.
        mu : float
            Drift of the shadow rate process.
        sigma : float
            Volatility of the shadow rate process.
        dt : float
            Time step size (in years).
        n_steps : int
            Number of simulation steps.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps + 1)
            Simulated instantaneous forward rate paths.
            All paths start at r0, i.e. paths[:, 0] == r0.
        """
        rng = np.random.RandomState(seed)

        s0 = self.to_shadow(r0)
        s = np.full((n_paths, n_steps + 1), s0)

        dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

        for t in range(n_steps):
            s[:, t + 1] = s[:, t] + mu * dt + sigma * dW[:, t]

        return self.from_shadow(s)

    # ------------------------------------------------------------------
    # Analytical expected value of r given s ~ N(m, sigma^2)
    # ------------------------------------------------------------------

    def expected_rate(self, m: float, sigma: float) -> float:
        """
        Closed-form expected value of the instantaneous forward rate r
        given that the shadow rate s ~ N(m, sigma^2).

        Derivation
        ----------
        Splitting E[r] at the switch point r*:

            E[r] = I1 + I2

        where

            I1 = integral_{r*}^{inf} s * phi(s) ds
               = m * [1 - Phi(alpha)] + sigma * phi_std(alpha)

            I2 = r_floor * Phi(alpha)
               + delta * exp((m - r*)/delta + sigma^2 / (2*delta^2))
                       * Phi(alpha - sigma/delta)

        and alpha = (r* - m) / sigma, Phi is the standard normal CDF,
        phi_std is the standard normal PDF.

        The exponential term in I2 arises from the moment-generating
        function of the normal distribution evaluated at 1/delta,
        integrated only over the lognormal regime.

        Parameters
        ----------
        m : float
            Mean of the shadow rate s.
        sigma : float
            Standard deviation of the shadow rate s. Must be positive.

        Returns
        -------
        float
            E[r] = E[T^{-1}(s)] under s ~ N(m, sigma^2).
        """
        from scipy.stats import norm

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")

        alpha     = (self.switch_point - m) / sigma
        phi_alpha = norm.pdf(alpha)
        Phi_alpha = norm.cdf(alpha)

        I1 = m * (1.0 - Phi_alpha) + sigma * phi_alpha

        # Lognormal term: delta * exp(mgf_exponent) * Phi(alpha - sigma/delta)
        #
        # Numerical stability issue: when m >> switch_point, mgf_exponent grows
        # as (m - r*)/delta which can be enormous for small delta (e.g. delta = 5bps
        # and m = 2.5% gives exponent ~ 50). Simultaneously Phi(alpha - sigma/delta)
        # collapses to zero exponentially fast, so the product -> 0.
        #
        # Solution: evaluate in log-domain and cap at a safe exponent threshold.
        # If log_term >= LOG_CAP the mantissa overflows; but in that regime
        # Phi2 is so small the product is negligibly close to zero anyway.
        LOG_CAP = 700.0   # exp(700) is near float64 overflow boundary
        arg2    = alpha - sigma / self.delta
        Phi2    = norm.cdf(arg2)

        if Phi2 > 0.0:
            mgf_exponent = (
                (m - self.switch_point) / self.delta
                + sigma**2 / (2.0 * self.delta**2)
            )
            log_term     = np.log(self.delta) + mgf_exponent + np.log(Phi2)
            I2_lognormal = np.exp(log_term) if log_term < LOG_CAP else 0.0
        else:
            I2_lognormal = 0.0

        I2 = self.floor * Phi_alpha + I2_lognormal

        return I1 + I2

    def fit_from_mean_and_quantile(
        self,
        r_bar: float,
        r_quantile: float,
        percentile: float,
    ) -> Tuple[float, float]:
        """
        Find the shadow rate mean m and standard deviation sigma such that
        the implied instantaneous forward rate distribution matches two targets:

            E[r | m, sigma]       = r_bar         (mean condition)
            Q_p[r | m, sigma]     = r_quantile    (quantile condition)

        The quantile condition is exact in closed form. Since T^{-1} is
        strictly increasing, quantiles transform as:

            Q_p[r] = T^{-1}(Q_p[s]) = T^{-1}(m + sigma * z_p)

        Applying T to both sides:

            T(r_quantile) = m + sigma * z_p   =>   m = T(r_quantile) - sigma * z_p

        Substituting into the mean condition leaves a single scalar equation
        in sigma, which is solved via Brent's method.

        Parameters
        ----------
        r_bar : float
            Target expected instantaneous forward rate E[r].
        r_quantile : float
            Target instantaneous forward rate at the given percentile.
        percentile : float
            Percentile level in (0, 1). Use 0.05 for 5th, 0.95 for 95th.

        Returns
        -------
        (m, sigma) : tuple of float
            Shadow rate mean and standard deviation.

        Raises
        ------
        ValueError
            If inputs are inconsistent (e.g. r_bar and r_quantile on
            the wrong side of each other given the percentile).
        """
        from scipy.stats import norm
        from scipy.optimize import brentq

        if not (0 < percentile < 1):
            raise ValueError(f"percentile must be in (0, 1), got {percentile}.")
        if r_bar <= self.floor:
            raise ValueError(f"r_bar ({r_bar}) must be above the floor ({self.floor}).")
        if r_quantile <= self.floor:
            raise ValueError(f"r_quantile ({r_quantile}) must be above the floor ({self.floor}).")

        z_p = norm.ppf(percentile)

        # Consistency check: direction of r_quantile vs r_bar must match z_p
        if z_p < 0 and r_quantile >= r_bar:
            raise ValueError(
                f"For percentile={percentile} (z_p={z_p:.3f} < 0), "
                f"r_quantile ({r_quantile}) must be strictly below r_bar ({r_bar})."
            )
        if z_p > 0 and r_quantile <= r_bar:
            raise ValueError(
                f"For percentile={percentile} (z_p={z_p:.3f} > 0), "
                f"r_quantile ({r_quantile}) must be strictly above r_bar ({r_bar})."
            )

        # Feasibility check for upper percentile (z_p > 0):
        # As sigma -> inf, m = T(r_p) - sigma*z_p -> -inf, so E[r] -> floor.
        # As sigma -> 0,  E[r] -> r_quantile.
        # Therefore E[r] is bounded in (floor, r_quantile) — r_bar must lie strictly within.
        if z_p > 0 and r_bar >= r_quantile:
            raise ValueError(
                f"No solution exists: for percentile={percentile}, E[r] is bounded "
                f"above by r_quantile={r_quantile:.6f}. "
                f"Target r_bar={r_bar:.6f} is not reachable. "
                f"Reduce r_bar or increase r_quantile."
            )

        # Feasibility check for lower percentile (z_p < 0):
        # As sigma -> 0,  E[r] -> r_quantile.
        # As sigma -> inf, m -> +inf so E[r] -> +inf (unbounded above).
        # Therefore E[r] is bounded below by r_quantile — r_bar must be strictly above it.
        # Edge case: r_quantile near floor forces sigma -> inf (practically infeasible).
        if z_p < 0 and r_quantile <= self.floor + 1e-8:
            raise ValueError(
                f"r_quantile={r_quantile:.6f} is at or below the floor={self.floor:.6f}. "
                f"No finite sigma can produce this quantile."
            )

        # T(r_quantile): shadow rate corresponding to r_quantile
        s_quantile = self.to_shadow(r_quantile)

        # From quantile condition: m = s_quantile - sigma * z_p
        # Substitute into mean condition: E[r | m(sigma), sigma] = r_bar
        def residual(sigma):
            m = s_quantile - sigma * z_p
            return self.expected_rate(m, sigma) - r_bar

        # Bracket sigma: must be positive; expand upper bound until sign change
        sigma_lo = 1e-8
        sigma_hi = abs(s_quantile - self.to_shadow(r_bar)) / abs(z_p) * 10
        sigma_hi = max(sigma_hi, 0.01)
        for _ in range(50):
            if residual(sigma_lo) * residual(sigma_hi) < 0:
                break
            sigma_hi *= 2.0

        sigma = brentq(residual, sigma_lo, sigma_hi, xtol=1e-12, rtol=1e-12)
        m = s_quantile - sigma * z_p

        return m, sigma

    def invert_expected_rate(self, r_bar: float, sigma: float) -> float:
        """
        Given a target expected instantaneous forward rate E[r] = r_bar
        and shadow rate volatility sigma, find the shadow rate mean m
        such that expected_rate(m, sigma) == r_bar.

        There is no closed-form inverse since m appears both linearly
        and nonlinearly (inside alpha, the exponential, and the CDF).
        We solve numerically via Brent's method, which is guaranteed to
        converge given a valid bracket.

        The bracket is constructed by noting that:
            - E[r] is strictly increasing in m
            - For m >> r*, E[r] ~ m  (normal regime dominates)
            - For m << r*, E[r] ~ r_floor  (floor dominates)
        So we expand the bracket until the residual changes sign.

        Parameters
        ----------
        r_bar : float
            Target expected instantaneous forward rate. Must satisfy
            r_bar > r_floor.
        sigma : float
            Standard deviation of the shadow rate. Must be positive.

        Returns
        -------
        float
            Shadow rate mean m such that E[r | m, sigma] = r_bar.

        Raises
        ------
        ValueError
            If r_bar <= r_floor (target is unreachable).
        """
        from scipy.optimize import brentq

        if r_bar <= self.floor:
            raise ValueError(
                f"r_bar ({r_bar}) must be strictly above the floor ({self.floor})."
            )

        f = lambda m: self.expected_rate(m, sigma) - r_bar

        # Initial bracket: start from r_bar and expand until sign change
        m_lo, m_hi = r_bar - 10 * sigma, r_bar + 10 * sigma
        for _ in range(50):
            if f(m_lo) < 0 and f(m_hi) > 0:
                break
            m_lo -= sigma
            m_hi += sigma

        return brentq(f, m_lo, m_hi, xtol=1e-12, rtol=1e-12)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ShadowRateTransformation("
            f"switch_point={self.switch_point}, "
            f"floor={self.floor}, "
            f"delta={self.delta})"
        )

    def summary(self) -> str:
        """Return a human-readable summary of the transformation parameters."""
        return (
            f"ShadowRateTransformation\n"
            f"  Switch point (r*) : {self.switch_point:.4f}\n"
            f"  Floor (r_floor)   : {self.floor:.4f}\n"
            f"  Delta             : {self.delta:.4f}\n"
            f"  C1 check at r*    : dr/ds = {self.jacobian(self.switch_point):.6f} (should be 1.0)\n"
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    srt = ShadowRateTransformation(switch_point=0.0, floor=-0.02)
    print(srt)
    print(srt.summary())

    # --- Roundtrip check ---
    r_test = np.array([-0.015, -0.010, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05])
    s_test = srt.to_shadow(r_test)
    r_recovered = srt.from_shadow(s_test)
    print("Roundtrip check (r -> s -> r):")
    for r, s, rr in zip(r_test, s_test, r_recovered):
        print(f"  r={r:+.4f}  ->  s={s:+.6f}  ->  r={rr:+.6f}  diff={abs(r-rr):.2e}")

    # --- Simulate ---
    paths = srt.simulate(
        r0=0.005, mu=0.0, sigma=0.005,
        dt=1/252, n_steps=252, n_paths=500, seed=42
    )
    print(f"\nSimulation: {paths.shape[0]} paths x {paths.shape[1]} steps")
    print(f"  Min rate observed : {paths.min():.4f}")
    print(f"  Max rate observed : {paths.max():.4f}")
    print(f"  Paths below floor : {(paths <= srt.floor).any(axis=1).sum()}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    r_grid = np.linspace(srt.floor + 1e-6, 0.06, 300)
    s_grid = srt.to_shadow(r_grid)
    axes[0].plot(r_grid * 100, s_grid * 100, color="steelblue", lw=1.8)
    axes[0].axvline(srt.switch_point * 100, color="coral", lw=1, ls="--", label=f"r* = {srt.switch_point*100:.0f}%")
    axes[0].axhline(srt.switch_point * 100, color="coral", lw=1, ls="--")
    axes[0].set_xlabel("Instantaneous forward rate r (%)")
    axes[0].set_ylabel("Shadow rate s (%)")
    axes[0].set_title("Forward transformation T(r)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    t = np.arange(paths.shape[1]) / 252
    for p in paths[:50]:
        axes[1].plot(t, p * 100, lw=0.4, alpha=0.4, color="steelblue")
    axes[1].axhline(srt.switch_point * 100, color="coral", lw=1, ls="--", label=f"r* = {srt.switch_point*100:.0f}%")
    axes[1].axhline(srt.floor * 100, color="firebrick", lw=1, ls=":", label=f"floor = {srt.floor*100:.0f}%")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("50 simulated forward rate paths")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/shadow_rate_demo.png", dpi=150)
    print("\nPlot saved to shadow_rate_demo.png")
