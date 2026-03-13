"""
tenor_utils.py
==============
Class-based utilities for parsing tenor strings and converting them to
year fractions under different day count conventions, with currency-aware
defaults.

Classes
-------
  DayCount      Day count convention engine (parse, convert, compare)
  TenorParser   Stateless tenor string parser (no dates needed)

Quick start
-----------
    from tenor_utils import DayCount

    dc = DayCount("USD")                          # USD -> ACT/360
    dc.tenor_to_years("6M", date(2024, 1, 15))    # -> 0.5056

    dc = DayCount(currency="GBP")                 # GBP -> ACT/365
    dc.tenor_to_years("6M", date(2024, 1, 15))    # -> 0.4986

    dc = DayCount(convention="30/360")             # explicit convention
    dc.tenor_to_years("6M", date(2024, 1, 15))    # -> 0.5000

    dc.dt                                          # -> 1/360  (for sigma_k extraction)
    dc.summary()                                   # print tenor grid table
"""

import re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


# =============================================================================
# Module-level constants  (used by both classes)
# =============================================================================

# ── Tenor regex ───────────────────────────────────────────────────────────────
_TENOR_RE = re.compile(
    r"^\s*(?P<value>[0-9]*\.?[0-9]+)\s*(?P<unit>[DdWwMmYy])?\s*$"
)

# Nominal year fractions per unit (convention-agnostic, for grid labels)
_UNIT_TO_YEARS = {
    "D": 1.0 / 365,
    "W": 7.0 / 365,
    "M": 1.0 / 12,
    "Y": 1.0,
}

# ── Day count convention names ────────────────────────────────────────────────
_SUPPORTED_CONVENTIONS = ("ACT/360", "ACT/365", "ACT/ACT", "30/360", "BUS/252")

# Aliases: upper-case normalised key -> canonical name
_CONV_ALIASES = {
    "ACT360":        "ACT/360",
    "ACTUAL360":     "ACT/360",
    "ACTUAL/360":    "ACT/360",
    "ACT365":        "ACT/365",
    "ACTUAL365":     "ACT/365",
    "ACTUAL/365":    "ACT/365",
    "ACTACT":        "ACT/ACT",
    "ACTUALACTUAL":  "ACT/ACT",
    "ACTUAL/ACTUAL": "ACT/ACT",
    "ACTACTISDA":    "ACT/ACT",
    "30360":         "30/360",
    "BONDBASIS":     "30/360",
    "BUS252":        "BUS/252",
    "BUSINESS252":   "BUS/252",
}

# ── Currency -> convention mapping ────────────────────────────────────────────
# Standard OIS / overnight rate day count convention per ISO 4217 code.
# Reference: ISDA definitions, central bank and ICMA documentation.
_CURRENCY_MAP = {
    # ── ACT/360 ───────────────────────────────────────────────────────────────
    "USD": ("ACT/360", "SOFR"),           # Secured Overnight Financing Rate
    "EUR": ("ACT/360", "€STR"),           # Euro Short-Term Rate
    "JPY": ("ACT/360", "TONAR"),          # Tokyo Overnight Average Rate
    "CHF": ("ACT/360", "SARON"),          # Swiss Average Rate Overnight
    "NOK": ("ACT/360", "NOWA"),           # Norwegian Overnight Weighted Average
    "SEK": ("ACT/360", "SWESTR"),         # Swedish krona Short-Term Rate
    "DKK": ("ACT/360", "DESTR"),          # Danmarks Nationalbank
    "MXN": ("ACT/360", "TIIE"),           # Tasa Interbancaria de Equilibrio
    "CZK": ("ACT/360", "CZEONIA"),        # Czech Overnight Index Average
    "HUF": ("ACT/360", "HUFONIA"),        # Hungarian Overnight Index Average
    "PLN": ("ACT/360", "WIBOR"),          # Warsaw Interbank Offered Rate
    "CNY": ("ACT/360", "SHIBOR"),         # Shanghai Interbank Offered Rate
    "SAR": ("ACT/360", "SAIBOR"),         # Saudi Interbank Offered Rate
    "COP": ("ACT/360", "IBR"),            # Indicador Bancario de Referencia
    "CLP": ("ACT/360", "TNA"),            # Tasa Nominal Anual (Chile)
    "TRY": ("ACT/360", "TLREF"),          # Turkish Lira Overnight Reference Rate
    "RUB": ("ACT/360", "RUONIA"),         # Ruble Overnight Index Average
    "RON": ("ACT/360", "ROBOR"),          # Romanian Interbank Offered Rate
    "TWD": ("ACT/360", "TAIBOR"),         # Taipei Interbank Offered Rate
    "KRW": ("ACT/360", "KOFR"),           # Korea Overnight Financing Repo Rate

    # ── ACT/365 ───────────────────────────────────────────────────────────────
    "GBP": ("ACT/365", "SONIA"),          # Sterling Overnight Index Average
    "AUD": ("ACT/365", "AONIA"),          # Australian Overnight Index Average
    "CAD": ("ACT/365", "CORRA"),          # Canadian Overnight Repo Rate Average
    "NZD": ("ACT/365", "OCR"),            # Official Cash Rate
    "HKD": ("ACT/365", "HONIA"),          # Hong Kong Overnight Index Average
    "SGD": ("ACT/365", "SORA"),           # Singapore Overnight Rate Average
    "ZAR": ("ACT/365", "ZARONIA"),        # South African Rand Overnight Index Average
    "THB": ("ACT/365", "THOR"),           # Thai Overnight Repurchase Rate
    "INR": ("ACT/365", "MIBOR"),          # Mumbai Interbank Offered Rate
    "IDR": ("ACT/365", "IndONIA"),        # Indonesia Overnight Index Average
    "MYR": ("ACT/365", "MYOR"),           # Malaysia Overnight Rate
    "ILS": ("ACT/365", "TELBOR"),         # Tel Aviv Interbank Offered Rate

    # ── BUS/252 ───────────────────────────────────────────────────────────────
    "BRL": ("BUS/252", "CDI"),            # Certificado de Depósito Interbancário
}


# =============================================================================
# TenorParser  —  stateless string parser, no dates required
# =============================================================================

class TenorParser:
    """
    Stateless parser for tenor strings.

    Converts tenor strings like "6M", "1Y", "2W", "30D" into nominal year
    fractions using fixed denominators (D=1/365, W=7/365, M=1/12, Y=1).
    No date or convention needed — suitable for building tenor grids and
    axis labels.

    Methods
    -------
    parse(tenor)            str  -> float  (nominal years)
    to_date(tenor, start)   str  -> date   (calendar-accurate end date)
    parse_many(tenors)      list -> list of float

    Examples
    --------
    >>> tp = TenorParser()
    >>> tp.parse("6M")
    0.5
    >>> tp.parse("1Y")
    1.0
    >>> tp.parse("3M")
    0.25
    >>> tp.parse_many(["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"])
    [0.0833, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    """

    # ------------------------------------------------------------------
    def parse(self, tenor: str) -> float:
        """
        Parse a tenor string into a nominal year fraction.

        Parameters
        ----------
        tenor : str   e.g. "6M", "1Y", "2W", "30D", "0.5", "1.5Y"

        Returns
        -------
        float  nominal year fraction

        Raises
        ------
        ValueError  if the string cannot be parsed
        """
        s = tenor.strip()
        try:
            return float(s)
        except ValueError:
            pass

        m = _TENOR_RE.match(s)
        if not m:
            raise ValueError(
                f"Cannot parse tenor '{tenor}'. "
                "Expected formats: '6M', '1Y', '2W', '30D', '0.5', '1.5Y'."
            )
        value = float(m.group("value"))
        unit  = (m.group("unit") or "Y").upper()
        if unit not in _UNIT_TO_YEARS:
            raise ValueError(
                f"Unknown unit '{unit}' in '{tenor}'. "
                f"Supported: {list(_UNIT_TO_YEARS)}"
            )
        return value * _UNIT_TO_YEARS[unit]

    # ------------------------------------------------------------------
    def to_date(self, tenor: str, start: date) -> date:
        """
        Advance ``start`` by the tenor using calendar-accurate arithmetic.

        Month/year steps use dateutil.relativedelta so that end-of-month
        dates are handled correctly (e.g. "1M" from 31 Jan -> 28 Feb).

        Parameters
        ----------
        tenor : str    tenor string
        start : date   start date

        Returns
        -------
        date  end date
        """
        s = tenor.strip()
        m = _TENOR_RE.match(s)
        if not m:
            raise ValueError(f"Cannot parse tenor '{tenor}'.")

        raw   = float(m.group("value"))
        unit  = (m.group("unit") or "Y").upper()

        if unit == "D":
            return start + timedelta(days=int(raw))
        elif unit == "W":
            return start + timedelta(weeks=int(raw))
        elif unit == "M":
            return start + relativedelta(months=int(raw))
        elif unit == "Y":
            return start + relativedelta(years=int(raw))
        else:
            raise ValueError(f"Unknown unit '{unit}'.")

    # ------------------------------------------------------------------
    def parse_many(self, tenors: list) -> list:
        """
        Parse a list of tenor strings into nominal year fractions.

        Parameters
        ----------
        tenors : list of str

        Returns
        -------
        list of float
        """
        return [self.parse(t) for t in tenors]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "TenorParser()"


# =============================================================================
# DayCount  —  main class, convention + currency aware
# =============================================================================

class DayCount:
    """
    Day count convention engine with optional currency-aware defaults.

    Instantiate with a currency (looks up market-standard convention)
    or an explicit convention string, then call instance methods to
    compute year fractions, build tenor grids, or retrieve ``dt``.

    Resolution priority
    -------------------
    1. currency   (market-standard convention for that currency's OIS rate)
    2. convention (explicit override)
    3. default    "ACT/360"

    Parameters
    ----------
    currency   : str or None   ISO 4217 currency code (case-insensitive).
    convention : str or None   Explicit day count convention.  Ignored when
                               currency is supplied.

    Supported conventions
    ---------------------
    "ACT/360"   Actual days / 360           USD SOFR, EUR €STR, JPY TONAR ...
    "ACT/365"   Actual days / 365           GBP SONIA, AUD AONIA, CAD CORRA ...
    "ACT/ACT"   Actual days / actual days   US Treasuries, EUR govt bonds
    "30/360"    30-day months / 360         Corporate bonds, legacy swaps
    "BUS/252"   Business days / 252         BRL CDI

    Supported currencies (34)
    -------------------------
    ACT/360 : USD EUR JPY CHF NOK SEK DKK MXN CZK HUF PLN CNY SAR COP CLP
              TRY RUB RON TWD KRW
    ACT/365 : GBP AUD CAD NZD HKD SGD ZAR THB INR IDR MYR ILS
    BUS/252 : BRL

    Attributes
    ----------
    convention : str    canonical convention name, e.g. "ACT/360"
    currency   : str or None
    dt         : float  year fraction per calendar day (use for sigma_k extraction)
    ois_name   : str or None  name of the overnight rate for this currency

    Examples
    --------
    >>> from datetime import date
    >>> from tenor_utils import DayCount

    >>> dc = DayCount("USD")
    >>> dc.convention
    'ACT/360'
    >>> dc.dt
    0.002777...       # 1/360

    >>> dc.tenor_to_years("6M", date(2024, 1, 15))
    0.5055...

    >>> dc.tenors_to_years(["1M","3M","6M","1Y"], date(2024, 1, 15))
    [0.0861, 0.2528, 0.5056, 1.0167]

    >>> dc.year_fraction(date(2024, 1, 15), date(2024, 7, 15))
    0.5055...

    >>> DayCount("GBP").dt
    0.002739...       # 1/365

    >>> DayCount("BRL").dt
    0.003968...       # 1/252

    >>> DayCount(convention="ACT/ACT").convention
    'ACT/ACT'
    """

    _parser = TenorParser()   # shared, stateless

    # ------------------------------------------------------------------
    def __init__(
        self,
        currency: str = None,
        convention: str = None,
    ):
        if currency is not None:
            key = currency.strip().upper()
            if key not in _CURRENCY_MAP:
                supported = sorted(_CURRENCY_MAP.keys())
                raise ValueError(
                    f"Unknown currency '{currency}'. "
                    f"Supported ({len(supported)}): {supported}"
                )
            self._convention = _CURRENCY_MAP[key][0]
            self._currency   = key
            self._ois_name   = _CURRENCY_MAP[key][1]
        elif convention is not None:
            self._convention = self._resolve_convention(convention)
            self._currency   = None
            self._ois_name   = None
        else:
            # default: ACT/360
            self._convention = "ACT/360"
            self._currency   = None
            self._ois_name   = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def convention(self) -> str:
        """Canonical convention name, e.g. 'ACT/360'."""
        return self._convention

    @property
    def currency(self) -> str | None:
        """ISO 4217 currency code, or None if instantiated from convention."""
        return self._currency

    @property
    def ois_name(self) -> str | None:
        """Name of the overnight rate for this currency (e.g. 'SOFR', 'SONIA')."""
        return self._ois_name

    @property
    def dt(self) -> float:
        """
        Year fraction per calendar day under this convention.

        This is the value to pass as ``dt`` to ``run_pca()`` and
        ``pca_loading_correlation()`` for sigma_k extraction:

            sigma_k = sqrt(M_kk / dt)

        Values:
            ACT/360  -> 1/360  = 0.002778
            ACT/365  -> 1/365  = 0.002740
            ACT/ACT  -> 1/366  = 0.002732  (2024 is a leap year)
            30/360   -> 1/360  = 0.002778
            BUS/252  -> 1/252  = 0.003968
        """
        ref   = date(2024, 1, 15)
        end   = ref + timedelta(days=1)
        return self._apply(ref, end)

    # ------------------------------------------------------------------
    # Core year fraction methods
    # ------------------------------------------------------------------

    def year_fraction(self, start: date, end: date) -> float:
        """
        Compute the year fraction between two explicit dates.

        Parameters
        ----------
        start : date
        end   : date

        Returns
        -------
        float

        Examples
        --------
        >>> DayCount("USD").year_fraction(date(2024, 1, 15), date(2024, 7, 15))
        0.5055555555555555
        >>> DayCount("GBP").year_fraction(date(2024, 1, 15), date(2024, 7, 15))
        0.4986301369863014
        """
        return self._apply(start, end)

    def tenor_to_years(self, tenor: str, start: date) -> float:
        """
        Convert a tenor string to a year fraction given a start date.

        Parameters
        ----------
        tenor : str    e.g. "6M", "1Y", "3M", "10Y", "2W", "30D"
        start : date   start date (trade date or curve date)

        Returns
        -------
        float

        Examples
        --------
        >>> DayCount("USD").tenor_to_years("6M", date(2024, 1, 15))
        0.5055555555555555
        >>> DayCount("GBP").tenor_to_years("6M", date(2024, 1, 15))
        0.4986301369863014
        >>> DayCount("BRL").tenor_to_years("6M", date(2024, 1, 15))
        0.5158730158730159
        """
        end = self._parser.to_date(tenor, start)
        return self._apply(start, end)

    def tenors_to_years(self, tenors: list, start: date) -> list:
        """
        Convert a list of tenor strings to year fractions.

        Parameters
        ----------
        tenors : list of str
        start  : date

        Returns
        -------
        list of float

        Examples
        --------
        >>> DayCount("USD").tenors_to_years(["1M","3M","6M","1Y"], date(2024, 1, 15))
        [0.0861, 0.2528, 0.5056, 1.0167]
        """
        return [self.tenor_to_years(t, start) for t in tenors]

    def tenor_grid(
        self,
        tenors: list,
        start: date,
    ) -> dict:
        """
        Build a tenor grid mapping tenor strings to (end_date, year_fraction).

        Parameters
        ----------
        tenors : list of str
        start  : date

        Returns
        -------
        dict  {tenor_str: {"end_date": date, "year_fraction": float}}

        Examples
        --------
        >>> grid = DayCount("USD").tenor_grid(["3M","6M","1Y"], date(2024,1,15))
        >>> grid["6M"]
        {'end_date': datetime.date(2024, 7, 15), 'year_fraction': 0.5056}
        """
        result = {}
        for t in tenors:
            end = self._parser.to_date(t, start)
            result[t] = {
                "end_date":      end,
                "year_fraction": self._apply(start, end),
            }
        return result

    # ------------------------------------------------------------------
    # Comparison and display
    # ------------------------------------------------------------------

    def summary(
        self,
        tenors: list = None,
        start: date = None,
    ) -> None:
        """
        Print a tenor grid table for this convention / currency.

        Parameters
        ----------
        tenors : list of str   defaults to standard pillar tenors
        start  : date          defaults to 2024-01-15
        """
        if tenors is None:
            tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        if start is None:
            start = date(2024, 1, 15)

        ccy_str = f" [{self._currency} / {self._ois_name}]" if self._currency else ""
        print(f"\n{self._convention}{ccy_str}  —  start date: {start}")
        print(f"  dt = {self.dt:.8f}  (= 1/{round(1/self.dt):.0f})")
        print("  " + "─" * 44)
        print(f"  {'Tenor':<8}  {'End date':<12}  {'Year fraction':>14}")
        print("  " + "─" * 44)
        for t in tenors:
            end = self._parser.to_date(t, start)
            yf  = self._apply(start, end)
            print(f"  {t:<8}  {str(end):<12}  {yf:>14.8f}")
        print("  " + "─" * 44)

    @staticmethod
    def compare(
        tenors: list = None,
        start: date = None,
        currencies: list = None,
    ) -> None:
        """
        Print a side-by-side comparison of year fractions across currencies.

        Parameters
        ----------
        tenors     : list of str   defaults to standard pillar tenors
        start      : date          defaults to 2024-01-15
        currencies : list of str   defaults to major liquid currencies
        """
        if tenors is None:
            tenors = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        if start is None:
            start = date(2024, 1, 15)
        if currencies is None:
            currencies = ["USD", "EUR", "GBP", "JPY", "CHF",
                          "AUD", "CAD", "SGD", "BRL"]

        instances = [DayCount(c) for c in currencies]
        col_w     = 9

        # Header row: CCY(CONV)
        print(f"\nCurrency comparison — start date: {start}")
        sep = "─" * (8 + 12 + col_w * len(currencies))
        print(sep)
        hdr = f"{'Tenor':<8}{'End date':<12}"
        for dc in instances:
            short = dc.convention.replace("ACT/", "").replace("BUS/", "")
            hdr  += f"{dc.currency+'('+short+')':>{col_w}}"
        print(hdr)
        print(sep)

        parser = TenorParser()
        for t in tenors:
            end = parser.to_date(t, start)
            row = f"{t:<8}{str(end):<12}"
            for dc in instances:
                row += f"{dc._apply(start, end):>{col_w}.4f}"
            print(row)
        print(sep)

        # dt summary grouped by convention
        print(f"\ndt per calendar day:")
        seen = set()
        for dc in instances:
            if dc.convention not in seen:
                ccys = sorted(
                    c for c, (conv, _) in _CURRENCY_MAP.items()
                    if conv == dc.convention
                )
                print(f"  {dc.convention:<12}  "
                      f"dt = {dc.dt:.8f}  (= 1/{round(1/dc.dt):.0f})"
                      f"  —  {', '.join(ccys)}")
                seen.add(dc.convention)

    @staticmethod
    def list_currencies() -> None:
        """Print all supported currencies grouped by convention."""
        groups: dict = {}
        for ccy, (conv, ois) in sorted(_CURRENCY_MAP.items()):
            groups.setdefault(conv, []).append((ccy, ois))

        print("\nSupported currencies by convention:")
        for conv, entries in sorted(groups.items()):
            print(f"\n  {conv}  ({len(entries)} currencies)")
            print(f"  {'Code':<6}  {'OIS Rate':<12}")
            print("  " + "─" * 22)
            for ccy, ois in sorted(entries):
                print(f"  {ccy:<6}  {ois:<12}")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._currency:
            return (f"DayCount(currency='{self._currency}', "
                    f"convention='{self._convention}', "
                    f"ois='{self._ois_name}')")
        return f"DayCount(convention='{self._convention}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, DayCount):
            return self._convention == other._convention
        return NotImplemented

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, start: date, end: date) -> float:
        """Apply this instance's convention to (start, end)."""
        conv = self._convention
        if conv == "ACT/360":
            return (end - start).days / 360.0
        elif conv == "ACT/365":
            return (end - start).days / 365.0
        elif conv == "ACT/ACT":
            return self._act_act(start, end)
        elif conv == "30/360":
            return self._thirty_360(start, end)
        elif conv == "BUS/252":
            return self._bus_252(start, end)
        else:
            raise RuntimeError(f"Unhandled convention '{conv}'.")  # should never happen

    @staticmethod
    def _act_act(start: date, end: date) -> float:
        """ACT/ACT ISDA: split across calendar years, each weighted by year length."""
        if start.year == end.year:
            denom = 366.0 if DayCount._is_leap(start.year) else 365.0
            return (end - start).days / denom
        # Remaining days in start year
        y1_end  = date(start.year, 12, 31)
        d1      = (y1_end - start).days + 1
        denom1  = 366.0 if DayCount._is_leap(start.year) else 365.0
        # Days elapsed in end year
        y2_start = date(end.year, 1, 1)
        d2       = (end - y2_start).days
        denom2   = 366.0 if DayCount._is_leap(end.year) else 365.0
        # Full years in between
        full = end.year - start.year - 1
        return d1 / denom1 + full + d2 / denom2

    @staticmethod
    def _thirty_360(start: date, end: date) -> float:
        """30/360 Bond Basis (US)."""
        Y1, M1, D1 = start.year, start.month, start.day
        Y2, M2, D2 = end.year,   end.month,   end.day
        if D1 == 31:
            D1 = 30
        if D2 == 31 and D1 >= 30:
            D2 = 30
        return (360 * (Y2 - Y1) + 30 * (M2 - M1) + (D2 - D1)) / 360.0

    @staticmethod
    def _bus_252(start: date, end: date) -> float:
        """BUS/252: count Mon–Fri days (no holiday calendar) / 252."""
        count   = 0
        current = start
        while current < end:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return count / 252.0

    @staticmethod
    def _is_leap(year: int) -> bool:
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    @staticmethod
    def _resolve_convention(s: str) -> str:
        """Normalise a convention string to its canonical form."""
        # Exact match (case-insensitive)
        key = s.strip().upper()
        for canon in _SUPPORTED_CONVENTIONS:
            if key == canon:
                return canon
        # Alias match (strip / and spaces for fuzzy matching)
        norm = key.replace("/", "").replace(" ", "")
        for canon in _SUPPORTED_CONVENTIONS:
            if norm == canon.replace("/", "").replace(" ", ""):
                return canon
        # Registered aliases
        if norm in _CONV_ALIASES:
            return _CONV_ALIASES[norm]
        for alias, canon in _CONV_ALIASES.items():
            if norm == alias.replace("/", "").replace(" ", ""):
                return canon
        raise ValueError(
            f"Unknown convention '{s}'. "
            f"Supported: {list(_SUPPORTED_CONVENTIONS)}"
        )


# =============================================================================
# Module-level convenience functions (thin wrappers around DayCount)
# =============================================================================

def parse_tenor(tenor: str) -> float:
    """Parse a tenor string to a nominal year fraction (no dates needed)."""
    return TenorParser().parse(tenor)


def tenor_to_years(
    tenor: str,
    start: date,
    currency: str = None,
    convention: str = None,
) -> float:
    """
    Convert a tenor string to a year fraction.

    Parameters
    ----------
    tenor      : str
    start      : date
    currency   : str or None   ISO 4217 code; takes priority over convention
    convention : str or None   explicit day count convention

    Returns
    -------
    float
    """
    return DayCount(currency=currency, convention=convention).tenor_to_years(tenor, start)


def tenors_to_years(
    tenors: list,
    start: date,
    currency: str = None,
    convention: str = None,
) -> list:
    """Convert a list of tenor strings to year fractions."""
    dc = DayCount(currency=currency, convention=convention)
    return dc.tenors_to_years(tenors, start)


def year_fraction(
    start: date,
    end: date,
    currency: str = None,
    convention: str = None,
) -> float:
    """Compute year fraction between two dates."""
    return DayCount(currency=currency, convention=convention).year_fraction(start, end)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    from datetime import date

    all_ok = True

    def check(label, got, expected, tol=1e-10):
        global all_ok
        ok = abs(got - expected) < tol
        mark = "✓" if ok else f"✗  expected {expected:.8f}"
        print(f"  {label:<55}  {got:.8f}  {mark}")
        if not ok:
            all_ok = False

    def check_str(label, got, expected):
        global all_ok
        ok = got == expected
        mark = "✓" if ok else f"✗  expected '{expected}'"
        print(f"  {label:<55}  {got!r}  {mark}")
        if not ok:
            all_ok = False

    # ── TenorParser ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("TenorParser.parse  (nominal year fractions)")
    print("=" * 70)
    tp = TenorParser()
    cases = [
        ("1D",   1/365), ("1W",   7/365), ("1M",  1/12),
        ("3M",   3/12),  ("6M",   0.5),   ("18M", 1.5),
        ("1Y",   1.0),   ("10Y",  10.0),  ("0.5", 0.5),
        ("0.5Y", 0.5),   ("1.5Y", 1.5),
    ]
    for s, exp in cases:
        check(f"parse('{s}')", tp.parse(s), exp)
    print()

    # ── DayCount: currency lookup ─────────────────────────────────────────────
    print("=" * 70)
    print("DayCount: currency -> convention")
    print("=" * 70)
    ccy_cases = [
        ("USD", "ACT/360", "SOFR"),
        ("EUR", "ACT/360", "€STR"),
        ("GBP", "ACT/365", "SONIA"),
        ("JPY", "ACT/360", "TONAR"),
        ("CHF", "ACT/360", "SARON"),
        ("AUD", "ACT/365", "AONIA"),
        ("CAD", "ACT/365", "CORRA"),
        ("BRL", "BUS/252", "CDI"),
        ("SGD", "ACT/365", "SORA"),
        ("KRW", "ACT/360", "KOFR"),
        ("MYR", "ACT/365", "MYOR"),
        ("CLP", "ACT/360", "TNA"),
    ]
    for ccy, exp_conv, exp_ois in ccy_cases:
        dc = DayCount(ccy)
        check_str(f"DayCount('{ccy}').convention", dc.convention, exp_conv)
        check_str(f"DayCount('{ccy}').ois_name  ", dc.ois_name,   exp_ois)
    print()

    # ── DayCount: dt per convention ───────────────────────────────────────────
    print("=" * 70)
    print("DayCount.dt  (year fraction per calendar day)")
    print("=" * 70)
    dt_cases = [
        ("USD", 1/360),
        ("GBP", 1/365),
        ("BRL", 1/252),
    ]
    for ccy, exp_dt in dt_cases:
        check(f"DayCount('{ccy}').dt", DayCount(ccy).dt, exp_dt, tol=1e-8)
    print()

    # ── DayCount: tenor_to_years ──────────────────────────────────────────────
    print("=" * 70)
    print("DayCount.tenor_to_years  (start = 2024-01-15)")
    print("=" * 70)
    start = date(2024, 1, 15)
    yf_cases = [
        ("USD", "6M",  182/360),
        ("GBP", "6M",  182/365),
        ("USD", "1Y",  366/360),   # 2024 is a leap year, Jan15->Jan15 = 366 days
        ("GBP", "1Y",  366/365),
    ]
    for ccy, tenor, exp in yf_cases:
        got = DayCount(ccy).tenor_to_years(tenor, start)
        check(f"DayCount('{ccy}').tenor_to_years('{tenor}')", got, exp)
    print()

    # ── DayCount: currency overrides convention ───────────────────────────────
    print("=" * 70)
    print("DayCount: repr and equality")
    print("=" * 70)
    dc1 = DayCount("USD")
    dc2 = DayCount(convention="ACT/360")
    print(f"  DayCount('USD')               = {dc1!r}")
    print(f"  DayCount(convention='ACT/360')= {dc2!r}")
    ok = (dc1 == dc2)
    print(f"  DayCount('USD') == DayCount(convention='ACT/360'):  "
          f"{'✓' if ok else '✗'}")
    if not ok:
        all_ok = False
    print()

    # ── module-level convenience wrappers ─────────────────────────────────────
    print("=" * 70)
    print("Module-level convenience functions")
    print("=" * 70)
    check("tenor_to_years('6M', start, currency='USD')",
          tenor_to_years("6M", start, currency="USD"), 182/360)
    check("tenor_to_years('6M', start, currency='GBP')",
          tenor_to_years("6M", start, currency="GBP"), 182/365)
    check("tenor_to_years('6M', start, convention='30/360')",
          tenor_to_years("6M", start, convention="30/360"), 0.5)
    check("year_fraction(start, date(2024,7,15), currency='USD')",
          year_fraction(start, date(2024, 7, 15), currency="USD"), 182/360)
    print()

    # ── Summary tables ────────────────────────────────────────────────────────
    DayCount("USD").summary(
        tenors=["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"],
        start=start,
    )
    DayCount("GBP").summary(
        tenors=["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"],
        start=start,
    )
    DayCount.compare(start=start)
    DayCount.list_currencies()

    print()
    print("=" * 70)
    print("All tests passed ✓" if all_ok else "SOME TESTS FAILED ✗")
    print("=" * 70)
