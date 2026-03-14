"""
tenor_utils.py
==============
Class-based utilities for parsing tenor strings and converting them to
year fractions under different day count conventions, with currency-aware
defaults and holiday calendar support.

Classes
-------
  HolidayCalendar   Holiday set for a currency; generates holidays for any year
  TenorParser       Stateless tenor string parser (no dates needed)
  DayCount          Day count convention engine integrating calendar + convention

Dependencies
------------
  holidays >= 0.46   pip install holidays
  python-dateutil    pip install python-dateutil

Quick start
-----------
    from tenor_utils import DayCount, HolidayCalendar
    from datetime import date

    # Currency-aware with holidays
    dc = DayCount("USD")
    dc.tenor_to_years("6M", date(2024, 1, 15))      # ACT/360 with NYC holidays
    dc.business_days(date(2024, 1, 15), date(2024, 7, 15))   # holiday-adjusted

    # Holiday calendar standalone
    cal = HolidayCalendar("GBP")
    cal.is_holiday(date(2024, 12, 25))               # True
    cal.is_business_day(date(2024, 12, 25))          # False
    cal.holidays(2024)                               # set of all 2024 holidays
    cal.next_business_day(date(2024, 12, 24))        # date(2024, 12, 27)

Notes
-----
  Holiday rules are approximations suitable for financial modelling.
  For production use, a market data provider's official holiday feed
  (e.g. ICE, Bloomberg, Refinitiv) should be preferred.

  BUS/252 year fractions use the holiday calendar for the currency to
  count only true business days (excluding both weekends and holidays).
"""

import re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import holidays as _holidays_lib


# =============================================================================
# Section 0 — Module-level constants
# =============================================================================

_TENOR_RE = re.compile(
    r"^\s*(?P<value>[0-9]*\.?[0-9]+)\s*(?P<unit>[DdWwMmYy])?\s*$"
)

_UNIT_TO_YEARS = {
    "D": 1.0 / 365,
    "W": 7.0 / 365,
    "M": 1.0 / 12,
    "Y": 1.0,
}

_SUPPORTED_CONVENTIONS = ("ACT/360", "ACT/365", "ACT/ACT", "30/360", "BUS/252")

_CONV_ALIASES = {
    "ACT360": "ACT/360", "ACTUAL360": "ACT/360", "ACTUAL/360": "ACT/360",
    "ACT365": "ACT/365", "ACTUAL365": "ACT/365", "ACTUAL/365": "ACT/365",
    "ACTACT": "ACT/ACT", "ACTUALACTUAL": "ACT/ACT", "ACTUAL/ACTUAL": "ACT/ACT",
    "ACTACTISDA": "ACT/ACT",
    "30360": "30/360", "BONDBASIS": "30/360",
    "BUS252": "BUS/252", "BUSINESS252": "BUS/252",
}

# currency -> (convention, OIS name, holiday centre label)
_CURRENCY_MAP = {
    # ACT/360
    "USD": ("ACT/360", "SOFR",    "USD"),
    "EUR": ("ACT/360", "€STR",    "EUR"),
    "JPY": ("ACT/360", "TONAR",   "JPY"),
    "CHF": ("ACT/360", "SARON",   "CHF"),
    "NOK": ("ACT/360", "NOWA",    "NOK"),
    "SEK": ("ACT/360", "SWESTR",  "SEK"),
    "DKK": ("ACT/360", "DESTR",   "DKK"),
    "MXN": ("ACT/360", "TIIE",    "MXN"),
    "CZK": ("ACT/360", "CZEONIA", "CZK"),
    "HUF": ("ACT/360", "HUFONIA", "HUF"),
    "PLN": ("ACT/360", "WIBOR",   "PLN"),
    "CNY": ("ACT/360", "SHIBOR",  "CNY"),
    "SAR": ("ACT/360", "SAIBOR",  "SAR"),
    "COP": ("ACT/360", "IBR",     "COP"),
    "CLP": ("ACT/360", "TNA",     "CLP"),
    "TRY": ("ACT/360", "TLREF",   "TRY"),
    "RUB": ("ACT/360", "RUONIA",  "RUB"),
    "RON": ("ACT/360", "ROBOR",   "RON"),
    "TWD": ("ACT/360", "TAIBOR",  "TWD"),
    "KRW": ("ACT/360", "KOFR",    "KRW"),
    # ACT/365
    "GBP": ("ACT/365", "SONIA",   "GBP"),
    "AUD": ("ACT/365", "AONIA",   "AUD"),
    "CAD": ("ACT/365", "CORRA",   "CAD"),
    "NZD": ("ACT/365", "OCR",     "NZD"),
    "HKD": ("ACT/365", "HONIA",   "HKD"),
    "SGD": ("ACT/365", "SORA",    "SGD"),
    "ZAR": ("ACT/365", "ZARONIA", "ZAR"),
    "THB": ("ACT/365", "THOR",    "THB"),
    "INR": ("ACT/365", "MIBOR",   "INR"),
    "IDR": ("ACT/365", "IndONIA", "IDR"),
    "MYR": ("ACT/365", "MYOR",    "MYR"),
    "ILS": ("ACT/365", "TELBOR",  "ILS"),
    # BUS/252
    "BRL": ("BUS/252", "CDI",     "BRL"),
}


# =============================================================================
# Section 1 — Easter and moveable feast helpers
# =============================================================================

def _easter(year: int) -> date:
    """Compute Easter Sunday for a given year (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _next_monday(d: date) -> date:
    """If d is Saturday/Sunday, return following Monday; else return d."""
    if d.weekday() == 5:
        return d + timedelta(days=2)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _substitute(d: date) -> date:
    """
    Standard UK/US substitute-day rule: if a fixed holiday falls on
    Saturday -> Friday; Sunday -> Monday.
    """
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, n: int, weekday: int) -> date:
    """
    Return the n-th occurrence (1-based) of ``weekday`` (Mon=0..Sun=6)
    in the given year/month.  n=-1 means the last occurrence.
    """
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + (n - 1) * 7)
    else:  # last occurrence
        # Start from the last day of the month and walk back
        if month == 12:
            last = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)
        offset = (last.weekday() - weekday) % 7
        return last - timedelta(days=offset)


# =============================================================================
# Section 2 — HolidayCalendar
# =============================================================================

class HolidayCalendar:
    """
    Holiday calendar for a given currency / financial centre.

    Generates the set of public holidays that cause banks to be closed in
    the relevant financial centre(s).  Weekends (Saturday, Sunday) are
    treated as non-business days separately and are NOT included in the
    holiday set returned by ``holidays()``.

    Parameters
    ----------
    currency : str   ISO 4217 currency code (case-insensitive)

    Attributes
    ----------
    currency : str   normalised ISO code, e.g. "USD"
    centre   : str   primary financial centre, e.g. "New York"

    Methods
    -------
    holidays(year)              -> frozenset of date
    is_holiday(d)               -> bool   (public holiday, not weekend)
    is_business_day(d)          -> bool   (not weekend AND not holiday)
    business_days(start, end)   -> int    (count of business days in [start, end))
    next_business_day(d)        -> date
    prev_business_day(d)        -> date
    add_business_days(d, n)     -> date

    Examples
    --------
    >>> from datetime import date
    >>> cal = HolidayCalendar("USD")
    >>> cal.is_holiday(date(2024, 12, 25))
    True
    >>> cal.is_business_day(date(2024, 12, 25))
    False
    >>> cal.is_business_day(date(2024, 12, 23))
    True
    >>> cal.next_business_day(date(2024, 12, 24))
    datetime.date(2024, 12, 26)
    >>> cal.business_days(date(2024, 1, 1), date(2024, 1, 10))
    6
    """

    # Human-readable centre names
    _CENTRES = {
        "USD": "New York",       "EUR": "Frankfurt/TARGET",
        "GBP": "London",         "JPY": "Tokyo",
        "CHF": "Zurich",         "AUD": "Sydney",
        "CAD": "Toronto",        "NZD": "Wellington",
        "HKD": "Hong Kong",      "SGD": "Singapore",
        "NOK": "Oslo",           "SEK": "Stockholm",
        "DKK": "Copenhagen",     "PLN": "Warsaw",
        "HUF": "Budapest",       "CZK": "Prague",
        "RON": "Bucharest",      "ILS": "Tel Aviv",
        "ZAR": "Johannesburg",   "BRL": "São Paulo",
        "MXN": "Mexico City",    "COP": "Bogotá",
        "CLP": "Santiago",       "CNY": "Shanghai",
        "INR": "Mumbai",         "IDR": "Jakarta",
        "MYR": "Kuala Lumpur",   "THB": "Bangkok",
        "SAR": "Riyadh",         "KRW": "Seoul",
        "TWD": "Taipei",         "RUB": "Moscow",
        "TRY": "Istanbul",
    }

    def __init__(self, currency: str):
        key = currency.strip().upper()
        if key not in _CURRENCY_MAP:
            raise ValueError(
                f"Unknown currency '{currency}'. "
                f"Supported: {sorted(_CURRENCY_MAP)}"
            )
        self._currency = key
        self._centre   = self._CENTRES.get(key, key)

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def centre(self) -> str:
        return self._centre

    # ------------------------------------------------------------------
    # Currencies that require post-generation Furikae/Kokumin processing
    _NEEDS_FURIKAE = frozenset({"JPY"})


    @lru_cache(maxsize=64)
    def holidays(self, year: int) -> frozenset:
        """
        Return the set of public holidays (weekdays only) for the given year.

        For JPY, Furikae Kyujitsu and Kokumin no Kyujitsu rules are applied
        via a simple single-pass date walk (see _apply_furikae_simple).
        When the ``holidays`` library is installed its JPY implementation
        is used directly (already handles all rules) and the walk is skipped.

        Parameters
        ----------
        year : int

        Returns
        -------
        frozenset of date  (weekday holidays only; weekends excluded)
        """
        fn  = self._GENERATORS.get(self._currency, self._western_generic)
        raw = fn(self, year)

        if self._currency in self._NEEDS_FURIKAE:
            raw = self._apply_furikae_simple(raw, year)
            # Add BoJ bank holidays AFTER Furikae so they don't block
            # substitute searches (e.g. Jan 1 Sun → sub is Jan 2, not Jan 4)
            boj = [date(year, m, d) for m, d in self._JPY_BOJ_EXTRA]
            raw = raw + [d for d in boj if d not in set(raw)]

        return frozenset(d for d in raw if d.weekday() < 5)
    @staticmethod
    def _apply_furikae_simple(raw: list, year: int) -> list:
        """
        Apply Furikae Kyujitsu and Kokumin no Kyujitsu in a single O(365)
        forward pass — no convergence loop, no risk of hanging.

        Approach (adapted from the date-walk pattern):
          Pass 1 — Furikae: walk Jan 1 to Dec 31; for each Sunday holiday,
                   assign the next non-holiday Monday as a substitute.
                   Because we walk forward, each substitute is placed before
                   we encounter it, so a single pass suffices for all but
                   the most pathological clusters.
          Pass 2 — Kokumin: walk Jan 1 to Dec 31; add any weekday that is
                   sandwiched between two holidays (including Sat/Sun ones).
          Both passes terminate in O(365) iterations with no inner loops
          that can run away.
        """
        hol_set = set(raw)

        # ── Pass 1: Furikae ───────────────────────────────────────────
        # Walk every day of the year; when we find a Saturday or Sunday
        # holiday, search forward (at most 7 days) for the next free
        # weekday that is not a Sunday and not already a holiday.
        d = date(year, 1, 1)
        while d.year == year:
            if d in hol_set and d.weekday() in (5, 6):   # Saturday or Sunday holiday
                sub = d + timedelta(days=1)
                for _ in range(7):                        # safety cap: max 7 steps
                    if sub.weekday() != 6 and sub not in hol_set:
                        if sub.year == year:              # keep within same year
                            hol_set.add(sub)
                        break
                    sub += timedelta(days=1)
            d += timedelta(days=1)

        # ── Pass 2: Kokumin ───────────────────────────────────────────
        # Walk every day; add weekdays sandwiched between two holidays.
        # Saturday/Sunday entries in hol_set count as holidays for this
        # check even though they will be stripped at the end.
        d = date(year, 1, 1)
        while d.year == year:
            if (d.weekday() < 5
                    and d not in hol_set
                    and (d - timedelta(days=1)) in hol_set
                    and (d + timedelta(days=1)) in hol_set):
                hol_set.add(d)
            d += timedelta(days=1)

        return list(hol_set)



    # ------------------------------------------------------------------
    def is_holiday(self, d: date) -> bool:
        """Return True if d is a public holiday (not including weekends)."""
        return d in self.holidays(d.year)

    def is_business_day(self, d: date) -> bool:
        """Return True if d is a weekday and not a public holiday."""
        return d.weekday() < 5 and not self.is_holiday(d)

    def business_days(self, start: date, end: date) -> int:
        """
        Count business days in the half-open interval [start, end).

        Parameters
        ----------
        start : date  inclusive
        end   : date  exclusive

        Returns
        -------
        int
        """
        if end <= start:
            return 0
        count   = 0
        current = start
        # Pre-fetch holiday sets for all years in range to avoid repeated calls
        years   = set(range(start.year, end.year + 1))
        hols    = set()
        for y in years:
            hols |= self.holidays(y)
        while current < end:
            if current.weekday() < 5 and current not in hols:
                count += 1
            current += timedelta(days=1)
        return count

    def next_business_day(self, d: date) -> date:
        """Return d if it is a business day, else the next business day."""
        while not self.is_business_day(d):
            d += timedelta(days=1)
        return d

    def prev_business_day(self, d: date) -> date:
        """Return d if it is a business day, else the previous business day."""
        while not self.is_business_day(d):
            d -= timedelta(days=1)
        return d

    def add_business_days(self, d: date, n: int) -> date:
        """
        Add n business days to d.  n may be negative (subtract).
        """
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        current   = d
        while remaining > 0:
            current += timedelta(days=step)
            if self.is_business_day(current):
                remaining -= 1
        return current

    def holidays_between(self, start: date, end: date, inclusive: bool = True) -> list:
        """
        Return a sorted list of public holidays between two dates.

        Parameters
        ----------
        start     : date  start of the range (inclusive)
        end       : date  end of the range (inclusive by default)
        inclusive : bool  if True (default), include both start and end dates;
                          if False, the interval is half-open [start, end)

        Returns
        -------
        list of date, sorted ascending

        Examples
        --------
        >>> from datetime import date
        >>> cal = HolidayCalendar("USD")
        >>> cal.holidays_between(date(2024, 12, 1), date(2024, 12, 31))
        [datetime.date(2024, 12, 25)]

        >>> cal.holidays_between(date(2024, 1, 1), date(2024, 12, 31))
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15), ...]
        """
        if end < start:
            return []
        # Pre-fetch all holiday sets for years spanned by the range
        years = set(range(start.year, end.year + 1))
        all_hols: set = set()
        for y in years:
            all_hols |= self.holidays(y)
        # Filter to the requested date range
        result = [
            d for d in all_hols
            if d >= start and (d <= end if inclusive else d < end)
        ]
        return sorted(result)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"HolidayCalendar('{self._currency}', centre='{self._centre}')"

    # ==================================================================
    # Holiday generation methods — one per currency / region
    # ==================================================================

    def _western_christian(self, year: int) -> list:
        """
        Core set shared by most Western/Christian markets:
        New Year, Good Friday, Easter Monday, Christmas, Boxing Day.
        """
        e = _easter(year)
        return [
            _substitute(date(year, 1, 1)),    # New Year's Day
            e - timedelta(days=2),             # Good Friday
            e + timedelta(days=1),             # Easter Monday
            _substitute(date(year, 12, 25)),   # Christmas Day
            _substitute(date(year, 12, 26)),   # Boxing Day
        ]

    def _western_generic(self, year: int) -> list:
        """Fallback: just the core Western set."""
        return self._western_christian(year)

    # ── USD — New York ────────────────────────────────────────────────
    def _usd(self, year: int) -> list:
        hols = [
            _substitute(date(year, 1, 1)),             # New Year's Day
            _nth_weekday(year, 1, 3, 0),               # MLK Day (3rd Mon Jan)
            _nth_weekday(year, 2, 3, 0),               # Presidents Day (3rd Mon Feb)
            _nth_weekday(year, 5, -1, 0),              # Memorial Day (last Mon May)
            _substitute(date(year, 7, 4)),              # Independence Day
            _nth_weekday(year, 9, 1, 0),               # Labor Day (1st Mon Sep)
            _nth_weekday(year, 10, 2, 0),              # Columbus Day (2nd Mon Oct)
            _substitute(date(year, 11, 11)),            # Veterans Day
            _nth_weekday(year, 11, 4, 3),              # Thanksgiving (4th Thu Nov)
            _substitute(date(year, 12, 25)),            # Christmas Day
        ]
        # Juneteenth: federally recognised from 2021 onwards only
        if year >= 2021:
            hols.append(_substitute(date(year, 6, 19)))
        # Cross-year observed New Year's: when Jan 1 of the FOLLOWING year falls
        # on Saturday, Dec 31 of this year is the observed holiday.
        if date(year + 1, 1, 1).weekday() == 5:       # next Jan 1 is Saturday
            hols.append(date(year, 12, 31))
        return hols

    # ── EUR — TARGET2 calendar ────────────────────────────────────────
    def _eur(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # St Stephen's Day
        ]

    # ── GBP — London ──────────────────────────────────────────────────
    @staticmethod
    def _uk_new_year(year: int) -> list:
        """
        UK New Year observed rule: Jan 1 Sat -> Jan 3 Mon; Jan 1 Sun -> Jan 2 Mon.
        Always forward, never back (unlike the US Sat->Fri rule).
        Also handles cross-year: if Jan 1 of the NEXT year falls on Sat/Sun,
        add the observed date (which falls in this year) now.
        """
        dates = []
        d = date(year, 1, 1)
        if d.weekday() == 5:        # Saturday -> Monday Jan 3
            dates.append(date(year, 1, 3))
        elif d.weekday() == 6:      # Sunday -> Monday Jan 2
            dates.append(date(year, 1, 2))
        else:
            dates.append(d)
        # Cross-year: next Jan 1 on Saturday -> Dec 31 this year is NOT used in UK;
        # instead Jan 3 of next year is the observed date (handled when year+1 is queried).
        return dates

    @staticmethod
    def _uk_xmas(year: int) -> list:
        """
        UK Christmas + Boxing Day observed rules (always forward):
          Dec 25 Fri, Dec 26 Sat -> Dec 25 (Fri), Dec 28 (Mon observed Boxing)
          Dec 25 Sat, Dec 26 Sun -> Dec 27 (Mon observed Christmas), Dec 28 (Tue observed Boxing)
          Dec 25 Sun, Dec 26 Mon -> Dec 27 (Tue observed Christmas), Dec 26 (Mon Boxing, fine)
          All other combos: Dec 25 and Dec 26 as-is (both weekdays)
        """
        xmas  = date(year, 12, 25)
        boxing = date(year, 12, 26)
        wd25 = xmas.weekday()
        wd26 = boxing.weekday()
        if wd25 == 5 and wd26 == 6:    # Sat + Sun -> Mon + Tue
            return [date(year, 12, 27), date(year, 12, 28)]
        if wd25 == 6 and wd26 == 0:    # Sun + Mon -> Tue observed Christmas, Mon Boxing
            return [date(year, 12, 27), boxing]
        if wd25 == 4 and wd26 == 5:    # Fri + Sat -> Fri Christmas, Mon observed Boxing
            return [xmas, date(year, 12, 28)]
        # All other cases: both are weekdays (or only one adjustment needed)
        result = []
        result.append(xmas  if wd25 < 5 else xmas  + timedelta(days=(7 - wd25)))
        result.append(boxing if wd26 < 5 else boxing + timedelta(days=(7 - wd26)))
        return result

    def _gbp(self, year: int) -> list:
        e    = _easter(year)

        # Early May Bank Holiday — moved to May 8 in 2020 for VE Day 75th anniversary
        if year == 2020:
            may_bh = date(2020, 5, 8)
        else:
            may_bh = _nth_weekday(year, 5, 1, 0)

        # Spring Bank Holiday — moved in Jubilee years
        if year == 2012:
            spring_bh = date(2012, 6, 4)           # Diamond Jubilee: moved to Jun 4
        elif year == 2022:
            spring_bh = date(2022, 6, 2)           # Platinum Jubilee: moved to Jun 2
        else:
            spring_bh = _nth_weekday(year, 5, -1, 0)

        hols = (
            self._uk_new_year(year)                 # New Year's Day (UK forward rule)
            + [
                e - timedelta(days=2),              # Good Friday
                e + timedelta(days=1),              # Easter Monday
                may_bh,                             # Early May Bank Holiday
                spring_bh,                          # Spring Bank Holiday
                _nth_weekday(year, 8, -1, 0),       # Late Summer Bank Holiday
            ]
            + self._uk_xmas(year)                   # Christmas + Boxing (UK forward rule)
        )

        # ── One-off special holidays ──────────────────────────────────
        if year == 2011:
            hols.append(date(2011, 4, 29))          # Wedding of William & Catherine
        if year == 2012:
            hols.append(date(2012, 6, 5))           # Diamond Jubilee of Elizabeth II
        if year == 2022:
            hols += [date(2022, 6, 3),              # Platinum Jubilee of Elizabeth II
                     date(2022, 9, 19)]             # State Funeral of Queen Elizabeth II
        if year == 2023:
            hols.append(date(2023, 5, 8))           # Coronation of Charles III

        return hols

    # ── JPY equinox lookup (Cabinet Office Japan, fallback when holidays
    #    library is not available)
    _JPY_VERNAL = {            # Vernal Equinox day in March
        2018:20,2019:21,2020:20,2021:20,2022:21,2023:21,
        2024:20,2025:20,2026:20,2027:21,2028:20,2029:20,2030:20,
    }
    _JPY_AUTUMNAL = {          # Autumnal Equinox day in September
        2018:23,2019:23,2020:22,2021:23,2022:23,2023:23,
        2024:22,2025:23,2026:23,2027:23,2028:22,2029:23,2030:23,
    }

    # BoJ bank holidays not in the national holiday list
    _JPY_BOJ_EXTRA = (
        (1, 2),   # Bank Holiday (New Year period)
        (1, 3),   # Bank Holiday (New Year period)
        (12, 31), # Bank Holiday (Year-end)
    )

    def _jpy(self, year: int) -> list:
        """
        Generate JPY holidays, preferring the ``holidays`` library when
        available (handles Sunday Furikae + Kokumin internally and is updated
        annually).  Falls back to a lookup-table implementation otherwise.

        The ``holidays`` library does not generate substitute days for
        Saturday holidays, so a Saturday Furikae pass is applied here
        regardless of whether the library is used.

        The three BoJ bank holidays (Jan 2, Jan 3, Dec 31) are appended
        AFTER Furikae processing so they do not interfere with substitute
        day searches (e.g. Jan 1 Sunday -> substitute should be Jan 2, not
        Jan 4 just because Jan 2/3 are BoJ extras).
        """
        jp  = _holidays_lib.Japan(years=year)
        raw = list(jp.keys())
        # The holidays library handles Sunday Furikae and Kokumin but
        # does NOT generate substitutes for Saturday holidays.
        # Apply a Saturday-only Furikae pass to fill that gap.
        raw = self._apply_saturday_furikae(raw, year)
        # Add BoJ extras AFTER Furikae so they don't block substitute
        # searches.
        boj = [date(year, m, d) for m, d in self._JPY_BOJ_EXTRA]
        return raw + [d for d in boj if d not in set(raw)]
    @staticmethod
    def _apply_saturday_furikae(raw: list, year: int) -> list:
        """
        Apply Furikae for Saturday holidays only.

        Under Japan's holiday law (Act No. 178 of 1948, as amended 2007),
        when a national holiday falls on Saturday the next non-holiday
        weekday is designated as the substitute (振替休日).  The
        ``holidays`` library handles Sunday substitutes internally but
        omits the Saturday case, so this pass fills that gap.

        Walk Jan 1 → Dec 31; for each Saturday holiday, search forward
        (capped at 7 steps) for the next free weekday that is not already
        a holiday and not a Sunday.
        """
        hol_set = set(raw)
        d = date(year, 1, 1)
        while d.year == year:
            if d in hol_set and d.weekday() == 5:   # Saturday holiday
                sub = d + timedelta(days=1)
                for _ in range(7):                   # safety cap
                    if sub.weekday() != 6 and sub not in hol_set:
                        if sub.year == year:
                            hol_set.add(sub)
                        break
                    sub += timedelta(days=1)
            d += timedelta(days=1)
        return list(hol_set)

    def _jpy_national_only(self, year: int) -> list:
        """National holidays without BoJ bank holiday extras (used by holidays())."""
        jp = _holidays_lib.Japan(years=year)
        return list(jp.keys())
    def _jpy_fallback(self, year: int) -> list:
        """
        Pure-Python fallback JPY national holiday generator used when the
        ``holidays`` library is not installed.

        Furikae and Kokumin rules are applied by ``holidays()`` via
        ``_apply_furikae()`` after this raw list is returned.
        """
        vd = self._JPY_VERNAL.get(year, 20)
        ad = self._JPY_AUTUMNAL.get(year, 23)
        return [
            date(year, 1, 1),                           # New Year's Day (Ganjitsu)
            _nth_weekday(year, 1, 2, 0),               # Coming of Age Day (2nd Mon Jan)
            date(year, 2, 11),                          # National Foundation Day
            date(year, 2, 23),                          # Emperor's Birthday
            date(year, 3, vd),                          # Vernal Equinox
            date(year, 4, 29),                          # Showa Day
            date(year, 5, 3),                           # Constitution Day
            date(year, 5, 4),                           # Greenery Day
            date(year, 5, 5),                           # Children's Day
            _nth_weekday(year, 7, 3, 0),               # Marine Day (3rd Mon Jul)
            date(year, 8, 11),                          # Mountain Day
            _nth_weekday(year, 9, 3, 0),               # Respect for the Aged Day
            date(year, 9, ad),                          # Autumnal Equinox
            _nth_weekday(year, 10, 2, 0),              # Sports Day (2nd Mon Oct)
            date(year, 11, 3),                          # Culture Day
            date(year, 11, 23),                         # Labour Thanksgiving Day
        ]


    # ── CHF — Zurich ──────────────────────────────────────────────────
    def _chf(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 2),                          # Berchtoldstag
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            e + timedelta(days=39),                    # Ascension Day
            e + timedelta(days=50),                    # Whit Monday
            date(year, 8, 1),                          # Swiss National Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # St Stephen's Day
        ]

    # ── AUD — Sydney ──────────────────────────────────────────────────
    def _aud(self, year: int) -> list:
        e = _easter(year)

        # New Year's Day: forward substitute (Sat -> Mon Jan 3; Sun -> Mon Jan 2)
        ny = date(year, 1, 1)
        if ny.weekday() == 5:
            ny_obs = date(year, 1, 3)
        elif ny.weekday() == 6:
            ny_obs = date(year, 1, 2)
        else:
            ny_obs = ny

        # Australia Day Jan 26:
        #   From 2010+: Sat -> Mon Jan 28; Sun -> Mon Jan 27
        #   Pre-2010:   Sat -> no substitute (just the Saturday)
        au_day_raw = date(year, 1, 26)
        wd = au_day_raw.weekday()
        if wd == 6:                          # Sunday -> Monday Jan 27 (all years)
            au_day = date(year, 1, 27)
        elif wd == 5 and year >= 2010:       # Saturday from 2010+ -> Monday Jan 28
            au_day = date(year, 1, 28)
        else:
            au_day = au_day_raw              # Weekday, or pre-2010 Saturday (stripped later)

        # ANZAC Day Apr 25:
        #   Sunday -> observed Mon Apr 26 ONLY before 2014 (NSW Anzac Day Act amendment)
        #             then re-added from 2021+ (subsequent amendment reversal)
        #   Saturday -> observed Mon Apr 27 from 2021+ (skip Sunday Apr 26)
        #               no substitute before 2021
        anzac_raw = date(year, 4, 25)
        if anzac_raw.weekday() == 6:
            if year < 2014 or year >= 2021:
                anzac_obs = date(year, 4, 26)    # Sunday -> Monday Apr 26
            else:
                anzac_obs = None                 # 2014-2020: no substitute for Sunday
        elif anzac_raw.weekday() == 5 and year >= 2021:
            anzac_obs = date(year, 4, 27)        # Saturday from 2021+ -> Monday (skip Sunday)
        elif anzac_raw.weekday() < 5:
            anzac_obs = anzac_raw                # Weekday: as-is
        else:
            anzac_obs = None                     # Saturday pre-2021: no substitute

        # Christmas + Boxing Day:
        #   From 2010+: Saturday holiday -> observed Monday (forward rule)
        #   Pre-2010:   Saturday -> no substitute
        xmas  = date(year, 12, 25)
        boxing = date(year, 12, 26)
        wd25, wd26 = xmas.weekday(), boxing.weekday()
        if wd25 == 5 and wd26 == 6:          # Sat+Sun double-weekend
            # Pre-2011: only the Sunday (Boxing) gets a sub -> Dec 27 Mon
            # From 2011+: both get subs -> Dec 27 Mon (Christmas obs) + Dec 28 Tue (Boxing obs)
            if year >= 2011:
                xmas_boxing = [date(year, 12, 27), date(year, 12, 28)]
            else:
                xmas_boxing = [date(year, 12, 27)]
        elif wd25 == 6 and wd26 == 0:        # Sun Christmas, Mon Boxing -> Tue observed Xmas
            xmas_boxing = [date(year, 12, 27), boxing]
        elif wd26 == 5 and year >= 2010:     # Boxing Sat from 2010+: Fri Xmas + Mon observed Boxing
            xmas_boxing = [xmas, date(year, 12, 28)]
        elif wd26 == 5:                      # Boxing Sat pre-2010: no sub
            xmas_boxing = [xmas]
        else:
            xmas_boxing = [d for d in (xmas, boxing) if d.weekday() < 5]

        hols = (
            [ny_obs, au_day,
             e - timedelta(days=2),          # Good Friday
             e + timedelta(days=1),          # Easter Monday
             _nth_weekday(year, 6, 2, 0),    # Queen's/King's Birthday (2nd Mon Jun)
             _nth_weekday(year, 10, 1, 0),   # Labour Day NSW (1st Mon Oct)
            ]
            + xmas_boxing
        )

        # Bank Holiday NSW (1st Mon Aug): existed 2008–2010 only
        if year <= 2010:
            hols.append(_nth_weekday(year, 8, 1, 0))

        # ANZAC Day
        if anzac_obs is not None:
            hols.append(anzac_obs)

        # One-off special holidays
        if year == 2022:
            hols.append(date(2022, 9, 22))   # National Day of Mourning

        return hols

    # ── CAD — Toronto ─────────────────────────────────────────────────
    def _cad(self, year: int) -> list:
        e = _easter(year)

        # New Year's Day: forward substitute (Sat -> Mon Jan 3; Sun -> Mon Jan 2)
        ny = date(year, 1, 1)
        if ny.weekday() == 5:
            ny_obs = date(year, 1, 3)
        elif ny.weekday() == 6:
            ny_obs = date(year, 1, 2)
        else:
            ny_obs = ny

        # Canada Day Jul 1: forward substitute (Sat -> Mon Jul 3; Sun -> Mon Jul 2)
        jul1 = date(year, 7, 1)
        if jul1.weekday() == 5:
            canada_day = date(year, 7, 3)
        elif jul1.weekday() == 6:
            canada_day = date(year, 7, 2)
        else:
            canada_day = jul1

        # Victoria Day: last Monday on or before May 24
        victoria = date(year, 5, 24)
        while victoria.weekday() != 0:
            victoria -= timedelta(days=1)

        # Christmas + Boxing Day: forward substitute (Sat/Sun -> Mon/Tue)
        xmas, boxing = date(year, 12, 25), date(year, 12, 26)
        wd25, wd26 = xmas.weekday(), boxing.weekday()
        if wd25 == 5 and wd26 == 6:     # Sat+Sun -> Mon+Tue
            xmas_boxing = [date(year, 12, 27), date(year, 12, 28)]
        elif wd25 == 6:                 # Sun Xmas -> Dec 26 Mon observed (merges with Boxing)
            xmas_boxing = [boxing]
        elif wd26 == 5:                 # Boxing Sat -> Mon Dec 28 observed; Xmas Fri stays
            xmas_boxing = [xmas, date(year, 12, 28)]
        else:
            xmas_boxing = [d for d in (xmas, boxing) if d.weekday() < 5]

        return [
            ny_obs,                                    # New Year's Day
            _nth_weekday(year, 2, 3, 0),               # Family Day (3rd Mon Feb)
            e - timedelta(days=2),                     # Good Friday
            victoria,                                  # Victoria Day (last Mon on/before May 24)
            canada_day,                                # Canada Day
            _nth_weekday(year, 9, 1, 0),               # Labour Day (1st Mon Sep)
            _nth_weekday(year, 10, 2, 0),              # Thanksgiving (2nd Mon Oct)
        ] + xmas_boxing

    # ── NZD — Wellington ──────────────────────────────────────────────

    # Matariki (Maori New Year): floating date, gazetted annually from 2022
    _NZD_MATARIKI = {
        2022: (6, 24), 2023: (7, 14), 2024: (6, 28), 2025: (6, 20),
        2026: (7, 10), 2027: (6, 25), 2028: (7, 14), 2029: (7,  6),
        2030: (6, 21), 2031: (7, 11), 2032: (7,  2), 2033: (6, 24),
        2034: (7,  7), 2035: (6, 29), 2036: (7, 18), 2037: (7, 10),
        2038: (6, 25), 2039: (7, 15), 2040: (7,  6), 2041: (7, 19),
        2042: (7, 11), 2043: (7,  3), 2044: (6, 24), 2045: (7,  7),
    }

    def _nzd(self, year: int) -> list:
        e = _easter(year)

        # New Year's Day (Jan 1) + Day after New Year (Jan 2):
        # NZ uses fully-forward substitution for both.
        # Jan 1 Sat + Jan 2 Sun -> Jan 3 Mon + Jan 4 Tue
        # Jan 1 Sun            -> Jan 3 Mon observed; Jan 2 Mon stays
        # Jan 2 Sat            -> Jan 4 Mon observed; Jan 1 Fri stays
        jan1, jan2 = date(year, 1, 1), date(year, 1, 2)
        wd1, wd2 = jan1.weekday(), jan2.weekday()
        if wd1 == 5 and wd2 == 6:       # Both weekend -> Jan 3 Mon + Jan 4 Tue
            ny_dates = [date(year, 1, 3), date(year, 1, 4)]
        elif wd1 == 6:                   # Jan 1 Sun -> Jan 3 Mon observed; Jan 2 Mon as-is
            ny_dates = [date(year, 1, 3), jan2]
        elif wd2 == 5:                   # Jan 2 Sat -> Jan 1 as-is; Jan 4 Mon observed
            ny_dates = [jan1, date(year, 1, 4)]
        elif wd2 == 6:                   # Jan 2 Sun -> Jan 1 as-is; Jan 4 Tue observed
            ny_dates = [jan1, date(year, 1, 4)]
        else:
            ny_dates = [d for d in (jan1, jan2) if d.weekday() < 5]

        # Waitangi Day (Feb 6):
        #   Pre-2016: no substitute for Sat/Sun
        #   From 2016: Sat -> Mon Feb 8; Sun -> Mon Feb 7
        feb6 = date(year, 2, 6)
        if feb6.weekday() < 5:
            waitangi = [feb6]
        elif year >= 2016:
            waitangi = [feb6 + timedelta(days=(7 - feb6.weekday()))]  # next Monday
        else:
            waitangi = []                # pre-2016 Sat/Sun: no weekday holiday

        # ANZAC Day (Apr 25):
        #   Pre-2015: no substitute for Sat/Sun
        #   From 2015: Sat -> Mon Apr 27 (skip Sun); Sun -> Mon Apr 26
        #   If the substitute would clash with Easter Monday, Easter Monday already
        #   serves as the combined holiday — no additional substitute is added.
        apr25 = date(year, 4, 25)
        easter_monday = e + timedelta(days=1)
        if apr25.weekday() < 5:
            anzac = [apr25]
        elif year >= 2015:
            if apr25.weekday() == 5:     # Saturday -> Monday Apr 27
                sub = date(year, 4, 27)
            else:                        # Sunday -> Monday Apr 26
                sub = date(year, 4, 26)
            # If substitute clashes with Easter Monday, Easter Monday already
            # serves as the combined holiday — don't add a duplicate.
            anzac = [] if sub == easter_monday else [sub]
        else:
            anzac = []                   # pre-2015 Sat/Sun: no weekday holiday

        # Christmas + Boxing Day: NZ always substitutes both Sat and Sun forward
        xmas, boxing = date(year, 12, 25), date(year, 12, 26)
        wd25, wd26 = xmas.weekday(), boxing.weekday()
        if wd25 == 5 and wd26 == 6:     # Sat+Sun -> Mon+Tue
            xmas_boxing = [date(year, 12, 27), date(year, 12, 28)]
        elif wd25 == 6 and wd26 == 0:   # Sun Xmas + Mon Boxing -> Tue observed Xmas
            xmas_boxing = [date(year, 12, 27), boxing]
        elif wd26 == 5:                 # Boxing Sat -> Mon observed Boxing; Xmas Fri stays
            xmas_boxing = [xmas, date(year, 12, 28)]
        else:
            xmas_boxing = [d for d in (xmas, boxing) if d.weekday() < 5]

        hols = (
            ny_dates
            + waitangi
            + anzac
            + [
                e - timedelta(days=2),              # Good Friday
                e + timedelta(days=1),              # Easter Monday
                _nth_weekday(year, 6, 1, 0),        # Queen's/King's Birthday (1st Mon Jun)
                _nth_weekday(year, 10, 4, 0),       # Labour Day (4th Mon Oct)
            ]
            + xmas_boxing
        )

        # Matariki: from 2022 onwards
        if year in self._NZD_MATARIKI:
            m, d = self._NZD_MATARIKI[year]
            hols.append(date(year, m, d))

        # One-off special holidays
        if year == 2022:
            hols.append(date(2022, 9, 26))          # Queen Elizabeth II Memorial Day

        return hols

    # ── HKD — Hong Kong ───────────────────────────────────────────────
    def _hkd(self, year: int) -> list:
        # HKD holidays include many lunar-calendar dates (CNY, Qingming, Dragon Boat,
        # Mid-Autumn, Chung Yeung) plus observed/substitute days that shift year to year.
        # Delegate to the holidays library when available for accurate dates.
        lib = _holidays_lib.HongKong(years=year)
        return [d for d in lib.keys() if d.year == year]
    # ── SGD — Singapore ───────────────────────────────────────────────
    def _sgd(self, year: int) -> list:
        # SGD holidays include Islamic (Hari Raya Puasa/Haji), Hindu (Deepavali),
        # Buddhist (Vesak Day) and Chinese (CNY) lunar calendar dates that shift
        # significantly year to year. Delegate to the holidays library for accuracy.
        lib = _holidays_lib.Singapore(years=year)
        return [d for d in lib.keys() if d.year == year]
    # ── NOK — Oslo ────────────────────────────────────────────────────
    def _nok(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=3),                     # Maundy Thursday
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            e + timedelta(days=39),                    # Ascension Day
            e + timedelta(days=49),                    # Whit Sunday
            e + timedelta(days=50),                    # Whit Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 17),                         # Constitution Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]

    # ── SEK — Stockholm ───────────────────────────────────────────────
    def _sek(self, year: int) -> list:
        e = _easter(year)
        # Midsummer Eve: Friday immediately before Midsummer Day (Sat between Jun 20-26)
        midsummer_sat = date(year, 6, 19)
        while midsummer_sat.weekday() != 5:
            midsummer_sat += timedelta(days=1)
        midsummer_eve = midsummer_sat - timedelta(days=1)  # Friday before
        return [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 6),                          # Epiphany
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            e + timedelta(days=39),                    # Ascension Day
            date(year, 6, 6),                          # National Day
            midsummer_eve,                             # Midsummer Eve (Fri before Midsummer Sat)
            _nth_weekday(year, 11, 1, 5),             # All Saints' Day (Sat on/after Oct 31)
            date(year, 12, 24),                        # Christmas Eve
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
            date(year, 12, 31),                        # New Year's Eve
        ]

    # ── DKK — Copenhagen ──────────────────────────────────────────────
    def _dkk(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=3),                     # Maundy Thursday
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            e + timedelta(days=39),                    # Ascension Day
            e + timedelta(days=50),                    # Whit Monday
            date(year, 6, 5),                          # Constitution Day (bank half-day)
            date(year, 12, 24),                        # Christmas Eve (bank half-day)
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
            date(year, 12, 31),                        # New Year's Eve (bank half-day)
        ]
        # Store Bededag (4th Friday after Easter) — abolished from 2024
        if year <= 2023:
            hols.append(e + timedelta(days=26))
        return hols

    # ── PLN — Warsaw ──────────────────────────────────────────────────
    def _pln(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 3),                          # Constitution Day
            e + timedelta(days=49),                    # Whit Sunday (stripped as weekend)
            e + timedelta(days=60),                    # Corpus Christi
            date(year, 8, 15),                         # Assumption Day
            date(year, 11, 1),                         # All Saints' Day
            date(year, 11, 11),                        # Independence Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]
        # Epiphany restored as public holiday from 2011
        if year >= 2011:
            hols.append(date(year, 1, 6))
        # Independence Day centenary — extra day in 2018
        if year == 2018:
            hols.append(date(2018, 11, 12))
        # Christmas Eve added as public holiday from 2025
        if year >= 2025:
            hols.append(date(year, 12, 24))
        return hols

    # ── HUF — Budapest ────────────────────────────────────────────────
    def _huf(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            date(year, 3, 15),                         # Nemzeti ünnep (National Day)
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            e + timedelta(days=50),                    # Whit Monday
            date(year, 8, 20),                         # St Stephen's Day
            date(year, 10, 23),                        # Republic Day
            date(year, 11, 1),                         # All Saints' Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]
        # Good Friday added as public holiday from 2017
        if year >= 2017:
            hols.append(e - timedelta(days=2))
        # Pihenőnap: government-decreed bridge/rest days (vary each year)
        # Use the holidays library when available to get accurate dates
        lib = _holidays_lib.Hungary(years=year)
        for d, name in lib.items():
            if 'Pihenőnap' in name and d.weekday() < 5:
                hols.append(d)
        return hols

    # ── CZK — Prague ──────────────────────────────────────────────────
    def _czk(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's / Restoration Day
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 8),                          # Liberation Day
            date(year, 7, 5),                          # SS Cyril & Methodius
            date(year, 7, 6),                          # Jan Hus Day
            date(year, 9, 28),                         # Czech Statehood Day
            date(year, 10, 28),                        # Independence Day
            date(year, 11, 17),                        # Freedom & Democracy Day
            date(year, 12, 24),                        # Christmas Eve
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]
        # Good Friday added as a public holiday from 2016
        if year >= 2016:
            hols.append(e - timedelta(days=2))
        return hols

    # ── RON — Bucharest ───────────────────────────────────────────────
    def _ron(self, year: int) -> list:
        # Romania uses Orthodox Easter — always fetch from library when available
        lib = _holidays_lib.Romania(years=year)
        return [d for d in lib.keys() if d.year == year]
        # Fallback: approximate with Western Easter (will be wrong some years)
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 2),                          # New Year Holiday
            e - timedelta(days=2),                     # Good Friday (approx)
            e + timedelta(days=1),                     # Easter Monday (approx)
            date(year, 5, 1),                          # Labour Day
            date(year, 6, 1),                          # Children's Day
            e + timedelta(days=50),                    # Whit Monday (approx)
            date(year, 8, 15),                         # Assumption Day
            date(year, 11, 30),                        # St Andrew's Day
            date(year, 12, 1),                         # National Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]
        if year >= 2016:
            hols.append(date(year, 1, 24))             # Unification Day (from 2016)
        if year >= 2024:
            hols.extend([date(year, 1, 6), date(year, 1, 7)])  # Bobotează, Sfântul Ion
        return hols

    # ── ILS — Tel Aviv ────────────────────────────────────────────────
    def _ils(self, year: int) -> list:
        # Jewish holidays use the Hebrew lunar calendar; exact dates vary each year.
        # Delegate to the holidays library when available for accurate dates.
        lib = _holidays_lib.Israel(years=year)
        return [d for d in lib.keys() if d.year == year]
    # ── ZAR — Johannesburg ────────────────────────────────────────────
    def _zar(self, year: int) -> list:
        # ZAR uses forward substitution (Saturday → Monday, not Friday) and includes
        # unpredictable election days and presidential decree holidays.
        # Delegate to the library for accurate dates.
        lib = _holidays_lib.SouthAfrica(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── BRL — São Paulo ───────────────────────────────────────────────
    def _brl(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=2),                     # Good Friday
            date(year, 4, 21),                         # Tiradentes
            date(year, 5, 1),                          # Labour Day
            date(year, 9, 7),                          # Independence Day
            date(year, 10, 12),                        # Our Lady of Aparecida
            date(year, 11, 2),                         # All Souls' Day
            date(year, 11, 15),                        # Proclamation of the Republic
            date(year, 12, 25),                        # Christmas Day
        ]
        # Black Consciousness Day became a national federal holiday from 2024
        if year >= 2024:
            hols.append(date(year, 11, 20))
        return hols

    # ── MXN — Mexico City ─────────────────────────────────────────────
    def _mxn(self, year: int) -> list:
        hols = [
            date(year, 1, 1),                          # New Year's Day
            _nth_weekday(year, 2, 1, 0),               # Constitution Day (1st Mon Feb)
            _nth_weekday(year, 3, 3, 0),               # Benito Juárez Birthday (3rd Mon Mar)
            date(year, 5, 1),                          # Labour Day
            date(year, 9, 16),                         # Independence Day
            _nth_weekday(year, 11, 3, 0),              # Revolution Day (3rd Mon Nov)
            date(year, 12, 25),                        # Christmas Day
        ]
        # Transfer of Executive Power (Oct 1) — every 6 years: 2024, 2030, 2036...
        if year in (2024, 2030, 2036, 2042, 2048):
            hols.append(date(year, 10, 1))
        return hols

    # ── COP — Bogotá ──────────────────────────────────────────────────
    @staticmethod
    def _cop_next_monday(d: date) -> date:
        """Colombia: if d is Monday, keep it; otherwise advance to next Monday."""
        days = (7 - d.weekday()) % 7
        return d if days == 0 else d + timedelta(days=days)

    def _cop(self, year: int) -> list:
        e = _easter(year)
        nm = self._cop_next_monday
        return [
            date(year, 1, 1),
            nm(date(year, 1, 6)),                      # Epiphany
            nm(date(year, 3, 19)),                     # St Joseph
            e - timedelta(days=3),                     # Maundy Thursday
            e - timedelta(days=2),                     # Good Friday
            date(year, 5, 1),                          # Labour Day
            nm(e + timedelta(days=43)),                # Ascension
            nm(e + timedelta(days=64)),                # Corpus Christi
            nm(e + timedelta(days=71)),                # Sacred Heart
            nm(date(year, 6, 29)),                     # SS Peter & Paul
            date(year, 7, 20),                         # Independence Day
            date(year, 8, 7),                          # Battle of Boyacá
            nm(date(year, 8, 15)),                     # Assumption
            nm(date(year, 10, 12)),                    # Columbus Day
            nm(date(year, 11, 1)),                     # All Saints
            nm(date(year, 11, 11)),                    # Independence of Cartagena
            date(year, 12, 8),                         # Immaculate Conception
            date(year, 12, 25),
        ]

    # ── CLP — Santiago ────────────────────────────────────────────────
    def _clp(self, year: int) -> list:
        # CLP holidays include many shifted/renamed dates (Columbus Day → Día del Encuentro,
        # San Pedro y San Pablo moved to nearest Monday, Fiestas Patrias bridge days,
        # census days, and other one-offs). Delegate to the library for accuracy.
        lib = _holidays_lib.Chile(years=year)
        return [d for d in lib.keys() if d.year == year]
    # ── CNY — Shanghai ────────────────────────────────────────────────
    def _cny(self, year: int) -> list:
        # CNY holidays include Qingming (Solar Term), Dragon Boat, Mid-Autumn
        # and Golden Week — all with precise dates and government-decreed bridge days
        # (休息日) that shift annually. Delegate to the library for accuracy.
        lib = _holidays_lib.China(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── INR — Mumbai ──────────────────────────────────────────────────
    def _inr(self, year: int) -> list:
        # INR holidays include Diwali, Holi, Eid al-Fitr, Eid al-Adha, Buddha Purnima,
        # Guru Nanak Jayanti and others on Hindu/Islamic/Buddhist lunar calendars
        # that shift significantly each year. Delegate to the library for accuracy.
        lib = _holidays_lib.India(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── IDR — Jakarta ─────────────────────────────────────────────────
    def _idr(self, year: int) -> list:
        # IDR holidays include Eid al-Fitr, Eid al-Adha, Isra Mi'raj, Islamic New Year,
        # Maulid, Nyepi (Hindu Balinese), Waisak (Buddhist) — all on lunar calendars —
        # plus Chinese New Year, Ascension and election days. Delegate to library.
        lib = _holidays_lib.Indonesia(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── MYR — Kuala Lumpur ────────────────────────────────────────────
    def _myr(self, year: int) -> list:
        # MYR federal holidays include Hari Raya Puasa, Hari Raya Qurban, Awal Muharram,
        # Maulidur Rasul (all Islamic lunar calendar), Chinese New Year, Wesak Day
        # (Buddhist lunar), and a King's Birthday that changes with each new monarch.
        # Observed/substitute days and election days also vary. Delegate to library.
        lib = _holidays_lib.Malaysia(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── THB — Bangkok ─────────────────────────────────────────────────
    def _thb(self, year: int) -> list:
        # THB holidays include Makha Bucha, Visakha Bucha, Asanha Bucha and Khao Phansa
        # (Buddhist lunar calendar), Songkran (Thai New Year, sometimes moved), royal
        # birthday holidays that changed with King Vajiralongkorn's accession, substitute
        # days (ชดเชย) and government-decreed special holidays. Delegate to library.
        lib = _holidays_lib.Thailand(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── SAR — Riyadh ──────────────────────────────────────────────────
    def _sar(self, year: int) -> list:
        # SAR holidays are primarily Islamic (Eid al-Fitr, Eid al-Adha, Arafat Day),
        # which shift ~11 days earlier each Gregorian year. Founding Day (Feb 22)
        # was added from 2022. Delegate to the library for accurate Islamic dates.
        # Note: Saudi Arabia observes Fri/Sat weekends (not Sat/Sun).
        lib = _holidays_lib.SaudiArabia(years=year)
        # Saudi weekend is Fri+Sat; treat Mon-Thu as business days
        return [d for d in lib.keys() if d.weekday() < 4 and d.year == year]
    # ── KRW — Seoul ───────────────────────────────────────────────────
    def _krw(self, year: int) -> list:
        # KRW holidays include Lunar New Year (Seollal), Chuseok (Harvest Festival),
        # and Buddha's Birthday — all on lunar calendar dates that shift each year —
        # plus substitute/bridge days and occasional election days.
        # Delegate to the holidays library for accurate dates.
        lib = _holidays_lib.SouthKorea(years=year)
        return [d for d in lib.keys() if d.year == year]
    # ── TWD — Taipei ──────────────────────────────────────────────────
    def _twd(self, year: int) -> list:
        # TWD holidays include Lunar New Year (Spring Festival), Mid-Autumn Festival,
        # Dragon Boat Festival, and Qingming — all lunar/solar dates that shift each year —
        # plus government-decreed bridge days (補假/放假日) that vary annually.
        # Delegate to the holidays library for accurate dates.
        lib = _holidays_lib.Taiwan(years=year)
        return [d for d in lib.keys() if d.year == year]
    # ── RUB — Moscow ──────────────────────────────────────────────────
    def _rub(self, year: int) -> list:
        # Russian holidays include government-decreed bridge/transfer days
        # (Выходной перенесено) that are announced each year and vary unpredictably.
        # Delegate to the library for accurate dates including these transfers.
        lib = _holidays_lib.Russia(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── TRY — Istanbul ────────────────────────────────────────────────
    def _try(self, year: int) -> list:
        # TRY holidays include Ramazan Bayramı (Eid al-Fitr) and Kurban Bayramı
        # (Eid al-Adha), which shift ~11 days earlier each Gregorian year.
        # Labour Day was added in 2009; Democracy Day in 2017.
        # Republic Day Eve (Oct 28 half-day) is NOT an official public holiday.
        # Delegate to the library for accurate Islamic holiday dates.
        lib = _holidays_lib.Turkey(years=year)
        return [d for d in lib.keys() if d.weekday() < 5 and d.year == year]
    # ── Dispatch table ────────────────────────────────────────────────
    # Populated after class definition (references instance methods)
    _GENERATORS: dict = {}


# Populate dispatch table after class is defined
HolidayCalendar._GENERATORS = {
    "USD": HolidayCalendar._usd,
    "EUR": HolidayCalendar._eur,
    "GBP": HolidayCalendar._gbp,
    "JPY": HolidayCalendar._jpy,
    "CHF": HolidayCalendar._chf,
    "AUD": HolidayCalendar._aud,
    "CAD": HolidayCalendar._cad,
    "NZD": HolidayCalendar._nzd,
    "HKD": HolidayCalendar._hkd,
    "SGD": HolidayCalendar._sgd,
    "NOK": HolidayCalendar._nok,
    "SEK": HolidayCalendar._sek,
    "DKK": HolidayCalendar._dkk,
    "PLN": HolidayCalendar._pln,
    "HUF": HolidayCalendar._huf,
    "CZK": HolidayCalendar._czk,
    "RON": HolidayCalendar._ron,
    "ILS": HolidayCalendar._ils,
    "ZAR": HolidayCalendar._zar,
    "BRL": HolidayCalendar._brl,
    "MXN": HolidayCalendar._mxn,
    "COP": HolidayCalendar._cop,
    "CLP": HolidayCalendar._clp,
    "CNY": HolidayCalendar._cny,
    "INR": HolidayCalendar._inr,
    "IDR": HolidayCalendar._idr,
    "MYR": HolidayCalendar._myr,
    "THB": HolidayCalendar._thb,
    "SAR": HolidayCalendar._sar,
    "KRW": HolidayCalendar._krw,
    "TWD": HolidayCalendar._twd,
    "RUB": HolidayCalendar._rub,
    "TRY": HolidayCalendar._try,
}


# =============================================================================
# Section 3 — TenorParser
# =============================================================================

class TenorParser:
    """Stateless tenor string parser (no date or convention required)."""

    def parse(self, tenor: str) -> float:
        s = tenor.strip()
        try:
            return float(s)
        except ValueError:
            pass
        m = _TENOR_RE.match(s)
        if not m:
            raise ValueError(
                f"Cannot parse tenor '{tenor}'. "
                "Expected: '6M', '1Y', '2W', '30D', '0.5', '1.5Y'."
            )
        value = float(m.group("value"))
        unit  = (m.group("unit") or "Y").upper()
        if unit not in _UNIT_TO_YEARS:
            raise ValueError(f"Unknown unit '{unit}' in '{tenor}'.")
        return value * _UNIT_TO_YEARS[unit]

    def to_date(self, tenor: str, start: date) -> date:
        s = tenor.strip()
        m = _TENOR_RE.match(s)
        if not m:
            raise ValueError(f"Cannot parse tenor '{tenor}'.")
        raw  = float(m.group("value"))
        unit = (m.group("unit") or "Y").upper()
        if unit == "D":
            return start + timedelta(days=int(raw))
        elif unit == "W":
            return start + timedelta(weeks=int(raw))
        elif unit == "M":
            return start + relativedelta(months=int(raw))
        elif unit == "Y":
            return start + relativedelta(years=int(raw))
        raise ValueError(f"Unknown unit '{unit}'.")

    def parse_many(self, tenors: list) -> list:
        return [self.parse(t) for t in tenors]

    def __repr__(self) -> str:
        return "TenorParser()"


# =============================================================================
# Section 4 — DayCount
# =============================================================================

class DayCount:
    """
    Day count convention engine with holiday-calendar-aware business day counts.

    Instantiate with a currency (looks up market-standard convention and
    holiday calendar) or an explicit convention string.

    Resolution priority
    -------------------
    1. currency   -> convention + holiday calendar
    2. convention -> explicit convention; no holiday calendar (weekends only)
    3. default    -> ACT/360; no holiday calendar

    Parameters
    ----------
    currency   : str or None   ISO 4217 code (case-insensitive)
    convention : str or None   Explicit convention; ignored when currency given

    Attributes
    ----------
    convention  : str
    currency    : str or None
    ois_name    : str or None
    calendar    : HolidayCalendar or None
    dt          : float  year fraction per calendar day

    Examples
    --------
    >>> from datetime import date
    >>> dc = DayCount("USD")
    >>> dc.convention
    'ACT/360'
    >>> dc.dt
    0.002778
    >>> dc.tenor_to_years("6M", date(2024, 1, 15))
    0.5056
    >>> dc.business_days(date(2024, 1, 1), date(2024, 1, 10))
    6                    # Jan 1 (New Year) excluded
    >>> dc.is_business_day(date(2024, 7, 4))
    False                # Independence Day
    """

    _parser = TenorParser()

    def __init__(self, currency: str = None, convention: str = None):
        if currency is not None:
            key = currency.strip().upper()
            if key not in _CURRENCY_MAP:
                raise ValueError(
                    f"Unknown currency '{currency}'. "
                    f"Supported: {sorted(_CURRENCY_MAP)}"
                )
            self._convention = _CURRENCY_MAP[key][0]
            self._currency   = key
            self._ois_name   = _CURRENCY_MAP[key][1]
            self._calendar   = HolidayCalendar(key)
        elif convention is not None:
            self._convention = self._resolve_convention(convention)
            self._currency   = None
            self._ois_name   = None
            self._calendar   = None
        else:
            self._convention = "ACT/360"
            self._currency   = None
            self._ois_name   = None
            self._calendar   = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def convention(self) -> str:
        return self._convention

    @property
    def currency(self) -> str | None:
        return self._currency

    @property
    def ois_name(self) -> str | None:
        return self._ois_name

    @property
    def calendar(self) -> HolidayCalendar | None:
        return self._calendar

    @property
    def dt(self) -> float:
        """
        Year fraction per calendar day — derived analytically from the
        convention definition, with no dependence on any reference date.

        Values:
            ACT/360  ->  1/360    = 0.002778  (exact)
            ACT/365  ->  1/365    = 0.002740  (exact)
            ACT/ACT  ->  1/365.25 = 0.002738  (long-run average, accounts
                                               for 1 leap year in 4)
            30/360   ->  1/360    = 0.002778  (exact)
            BUS/252  ->  1/252    = 0.003968  (market convention; use
                                               dt_for_year(y) for the exact
                                               business day count in year y)
        """
        if self._convention in ("ACT/360", "30/360"):
            return 1.0 / 360.0
        elif self._convention == "ACT/365":
            return 1.0 / 365.0
        elif self._convention == "ACT/ACT":
            # 365.25 is the long-run average (97 non-leap + 3 leap per 400 yrs
            # gives (97*365 + 3*366) / 100 = 365.25 exactly over a 4-yr cycle).
            return 1.0 / 365.25
        elif self._convention == "BUS/252":
            return 1.0 / 252.0
        raise RuntimeError(f"Unhandled convention '{self._convention}'.")

    def dt_for_year(self, year: int) -> float:
        """
        Year fraction per calendar day for a specific calendar year.

        More precise than ``dt`` for ACT/ACT and BUS/252, where the
        correct denominator depends on whether the year is a leap year
        (ACT/ACT) or how many business days fall in the year (BUS/252).

        For all other conventions the result is identical to ``dt``.

        Parameters
        ----------
        year : int   the calendar year of the data being analysed

        Returns
        -------
        float

        Examples
        --------
        >>> DayCount("USD").dt_for_year(2024)
        0.002778             # ACT/360, unchanged
        >>> DayCount(convention="ACT/ACT").dt_for_year(2024)
        0.002732             # 2024 is a leap year -> 1/366
        >>> DayCount(convention="ACT/ACT").dt_for_year(2023)
        0.002740             # 2023 is not a leap year -> 1/365
        >>> DayCount("BRL").dt_for_year(2024)
        ~0.003984            # actual CDI business days in 2024 / 252
        """
        if self._convention in ("ACT/360", "30/360"):
            return 1.0 / 360.0
        elif self._convention == "ACT/365":
            return 1.0 / 365.0
        elif self._convention == "ACT/ACT":
            days_in_year = 366 if self._is_leap(year) else 365
            return 1.0 / days_in_year
        elif self._convention == "BUS/252":
            # Count actual business days in this calendar year using the
            # holiday calendar so the result matches the index (e.g. CDI)
            start = date(year, 1, 1)
            end   = date(year + 1, 1, 1)
            bd    = self.business_days(start, end)
            return 1.0 / bd
        raise RuntimeError(f"Unhandled convention '{self._convention}'.")

    # ------------------------------------------------------------------
    # Year fraction methods
    # ------------------------------------------------------------------

    def year_fraction(self, start: date, end: date) -> float:
        """Year fraction between two explicit dates under this convention."""
        return self._apply(start, end)

    def tenor_to_years(self, tenor: str, start: date) -> float:
        """Convert tenor string to year fraction from start date."""
        end = self._parser.to_date(tenor, start)
        return self._apply(start, end)

    def tenors_to_years(self, tenors: list, start: date) -> list:
        """Convert list of tenor strings to year fractions."""
        return [self.tenor_to_years(t, start) for t in tenors]

    def tenor_grid(self, tenors: list, start: date) -> dict:
        """Build {tenor: {end_date, year_fraction}} grid."""
        return {
            t: {"end_date": self._parser.to_date(t, start),
                "year_fraction": self.tenor_to_years(t, start)}
            for t in tenors
        }

    # ------------------------------------------------------------------
    # Business day methods (use holiday calendar when available)
    # ------------------------------------------------------------------

    def is_business_day(self, d: date) -> bool:
        """
        Return True if d is a business day (weekday + not a public holiday).

        Uses the holiday calendar if one is attached (currency-aware).
        Falls back to weekday-only check when no calendar is available.
        """
        if d.weekday() >= 5:
            return False
        if self._calendar is not None:
            return not self._calendar.is_holiday(d)
        return True

    def is_holiday(self, d: date) -> bool:
        """Return True if d is a public holiday (not a weekend day)."""
        if self._calendar is not None:
            return self._calendar.is_holiday(d)
        return False

    def business_days(self, start: date, end: date) -> int:
        """
        Count business days in [start, end) using the holiday calendar.

        Parameters
        ----------
        start : date  inclusive
        end   : date  exclusive

        Returns
        -------
        int
        """
        if self._calendar is not None:
            return self._calendar.business_days(start, end)
        # No calendar — count weekdays only
        count   = 0
        current = start
        while current < end:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return count

    def next_business_day(self, d: date) -> date:
        """Return d or the next business day if d is not a business day."""
        if self._calendar is not None:
            return self._calendar.next_business_day(d)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        return d

    def add_business_days(self, d: date, n: int) -> date:
        """Add n business days to d (n may be negative)."""
        if self._calendar is not None:
            return self._calendar.add_business_days(d, n)
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        while remaining > 0:
            d += timedelta(days=step)
            if d.weekday() < 5:
                remaining -= 1
        return d

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self, tenors: list = None, start: date = None) -> None:
        """Print a tenor grid table for this convention/currency."""
        if tenors is None:
            tenors = ["1M","3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"]
        if start is None:
            start = date(2024, 1, 15)

        ccy_str = f" [{self._currency} / {self._ois_name}]" if self._currency else ""
        cal_str = f"  calendar: {self._calendar.centre}" if self._calendar else "  no holiday calendar"
        print(f"\n{self._convention}{ccy_str}  —  start: {start}")
        print(f"  dt = {self.dt:.8f}  (= 1/{round(1/self.dt):.0f}){cal_str}")
        print("  " + "─" * 52)
        print(f"  {'Tenor':<8}  {'End date':<12}  {'Year frac':>12}  {'Biz days':>9}")
        print("  " + "─" * 52)
        for t in tenors:
            end = self._parser.to_date(t, start)
            yf  = self._apply(start, end)
            bd  = self.business_days(start, end)
            print(f"  {t:<8}  {str(end):<12}  {yf:>12.6f}  {bd:>9}")
        print("  " + "─" * 52)

    @staticmethod
    def compare(
        tenors: list = None,
        start: date = None,
        currencies: list = None,
    ) -> None:
        """Print side-by-side year fraction comparison across currencies."""
        if tenors is None:
            tenors = ["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"]
        if start is None:
            start = date(2024, 1, 15)
        if currencies is None:
            currencies = ["USD","EUR","GBP","JPY","CHF","AUD","CAD","SGD","BRL"]

        instances = [DayCount(c) for c in currencies]
        col_w     = 9
        sep = "─" * (8 + 12 + col_w * len(currencies))
        print(f"\nCurrency comparison — start: {start}")
        print(sep)
        hdr = f"{'Tenor':<8}{'End date':<12}"
        for dc in instances:
            short = dc.convention.replace("ACT/","").replace("BUS/","")
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
        seen = set()
        print(f"\ndt per calendar day:")
        for dc in instances:
            if dc.convention not in seen:
                ccys = sorted(c for c,(conv,*_) in _CURRENCY_MAP.items()
                              if conv == dc.convention)
                print(f"  {dc.convention:<12}  dt={dc.dt:.8f}"
                      f"  (=1/{round(1/dc.dt):.0f})  — {', '.join(ccys)}")
                seen.add(dc.convention)

    @staticmethod
    def list_currencies() -> None:
        """Print all supported currencies grouped by convention."""
        groups: dict = {}
        for ccy, (conv, ois, *_) in sorted(_CURRENCY_MAP.items()):
            groups.setdefault(conv, []).append((ccy, ois))
        print("\nSupported currencies by convention:")
        for conv, entries in sorted(groups.items()):
            print(f"\n  {conv}  ({len(entries)} currencies)")
            print(f"  {'Code':<6}  {'OIS Rate':<14}  {'Centre'}")
            print("  " + "─" * 38)
            for ccy, ois in sorted(entries):
                centre = HolidayCalendar._CENTRES.get(ccy, "—")
                print(f"  {ccy:<6}  {ois:<14}  {centre}")

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
            return self.business_days(start, end) / 252.0
        raise RuntimeError(f"Unhandled convention '{conv}'.")

    @staticmethod
    def _act_act(start: date, end: date) -> float:
        if start.year == end.year:
            denom = 366.0 if DayCount._is_leap(start.year) else 365.0
            return (end - start).days / denom
        y1_end   = date(start.year, 12, 31)
        d1       = (y1_end - start).days + 1
        denom1   = 366.0 if DayCount._is_leap(start.year) else 365.0
        y2_start = date(end.year, 1, 1)
        d2       = (end - y2_start).days
        denom2   = 366.0 if DayCount._is_leap(end.year) else 365.0
        full     = end.year - start.year - 1
        return d1 / denom1 + full + d2 / denom2

    @staticmethod
    def _thirty_360(start: date, end: date) -> float:
        Y1, M1, D1 = start.year, start.month, start.day
        Y2, M2, D2 = end.year,   end.month,   end.day
        if D1 == 31:
            D1 = 30
        if D2 == 31 and D1 >= 30:
            D2 = 30
        return (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360.0

    @staticmethod
    def _is_leap(year: int) -> bool:
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    @staticmethod
    def _resolve_convention(s: str) -> str:
        key = s.strip().upper()
        for canon in _SUPPORTED_CONVENTIONS:
            if key == canon:
                return canon
        norm = key.replace("/","").replace(" ","")
        for canon in _SUPPORTED_CONVENTIONS:
            if norm == canon.replace("/","").replace(" ",""):
                return canon
        if norm in _CONV_ALIASES:
            return _CONV_ALIASES[norm]
        for alias, canon in _CONV_ALIASES.items():
            if norm == alias.replace("/","").replace(" ",""):
                return canon
        raise ValueError(
            f"Unknown convention '{s}'. Supported: {list(_SUPPORTED_CONVENTIONS)}"
        )


# =============================================================================
# Section 5 — Module-level convenience wrappers
# =============================================================================

def parse_tenor(tenor: str) -> float:
    """Parse a tenor string to a nominal year fraction (no dates needed)."""
    return TenorParser().parse(tenor)


def tenor_to_years(
    tenor: str, start: date,
    currency: str = None, convention: str = None,
) -> float:
    """Convert tenor string to year fraction; currency takes priority."""
    return DayCount(currency=currency, convention=convention).tenor_to_years(tenor, start)


def tenors_to_years(
    tenors: list, start: date,
    currency: str = None, convention: str = None,
) -> list:
    """Convert list of tenor strings to year fractions."""
    return DayCount(currency=currency, convention=convention).tenors_to_years(tenors, start)


def year_fraction(
    start: date, end: date,
    currency: str = None, convention: str = None,
) -> float:
    """Compute year fraction between two dates."""
    return DayCount(currency=currency, convention=convention).year_fraction(start, end)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    from datetime import date

    all_ok = True
    def chk(label, got, exp, tol=1e-9):
        global all_ok
        ok = abs(got - exp) < tol
        mark = "✓" if ok else f"✗  expected {exp}"
        print(f"  {label:<58} {got:.6f}  {mark}")
        all_ok = all_ok and ok

    def chk_bool(label, got, exp):
        global all_ok
        ok = got == exp
        mark = "✓" if ok else f"✗  expected {exp}"
        print(f"  {label:<58} {got}  {mark}")
        all_ok = all_ok and ok

    def chk_int(label, got, exp):
        global all_ok
        ok = got == exp
        mark = "✓" if ok else f"✗  expected {exp}"
        print(f"  {label:<58} {got}  {mark}")
        all_ok = all_ok and ok

    # ── TenorParser ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("TenorParser")
    print("=" * 70)
    tp = TenorParser()
    for s, exp in [("6M",0.5),("1Y",1.0),("3M",0.25),("2W",14/365),("1D",1/365)]:
        chk(f"parse('{s}')", tp.parse(s), exp)
    print()

    # ── HolidayCalendar ──────────────────────────────────────────────────────
    print("=" * 70)
    print("HolidayCalendar")
    print("=" * 70)
    start = date(2024, 1, 15)

    # USD: New Year, MLK Day, Independence Day, Thanksgiving, Christmas
    cal_usd = HolidayCalendar("USD")
    chk_bool("USD: Jan 1 2024 is holiday",   cal_usd.is_holiday(date(2024,1,1)),   True)
    chk_bool("USD: Jan 15 2024 is holiday",  cal_usd.is_holiday(date(2024,1,15)),  True)  # MLK
    chk_bool("USD: Jul 4 2024 is holiday",   cal_usd.is_holiday(date(2024,7,4)),   True)
    chk_bool("USD: Dec 25 2024 is holiday",  cal_usd.is_holiday(date(2024,12,25)), True)
    chk_bool("USD: Jan 16 2024 is biz day",  cal_usd.is_business_day(date(2024,1,16)), True)
    chk_bool("USD: Dec 25 is NOT biz day",   cal_usd.is_business_day(date(2024,12,25)), False)

    # GBP: Good Friday 2024 = Mar 29
    cal_gbp = HolidayCalendar("GBP")
    chk_bool("GBP: Good Friday 2024",        cal_gbp.is_holiday(date(2024,3,29)),  True)
    chk_bool("GBP: Easter Monday 2024",      cal_gbp.is_holiday(date(2024,4,1)),   True)
    chk_bool("GBP: Dec 26 Boxing Day 2024",  cal_gbp.is_holiday(date(2024,12,26)), True)

    # BRL: Carnival Tuesday 2024
    cal_brl = HolidayCalendar("BRL")
    chk_bool("BRL: Carnival Tue 2024 (Feb 13)", cal_brl.is_holiday(date(2024,2,13)), True)

    # EUR: May 1 Labour Day
    cal_eur = HolidayCalendar("EUR")
    chk_bool("EUR: May 1 2024 Labour Day",   cal_eur.is_holiday(date(2024,5,1)),   True)

    # Business day count: USD Jan 1–10 2024 (Jan 1 holiday, Jan 6-7 weekend)
    bd_usd = cal_usd.business_days(date(2024,1,1), date(2024,1,10))
    chk_int("USD: biz days Jan 1–9 2024 (excl Jan1+weekends)", bd_usd, 6)

    # next_business_day
    nbd = cal_usd.next_business_day(date(2024,12,25))
    print(f"  USD: next biz day after Dec 25 2024 = {nbd}  "
          f"{'✓' if nbd == date(2024,12,26) else '✗'}")

    # add_business_days: Dec 23 (Mon) + 1 biz = Dec 24, + 1 biz = Dec 26 (skip Dec 25 holiday)
    abd = cal_usd.add_business_days(date(2024,12,23), 2)
    print(f"  USD: Dec 23 + 2 biz days = {abd}  "
          f"{'✓' if abd == date(2024,12,26) else '✗'}")
    print()

    # ── DayCount with calendar ───────────────────────────────────────────────
    print("=" * 70)
    print("DayCount — business day methods")
    print("=" * 70)
    dc_usd = DayCount("USD")
    chk_bool("DayCount USD: Jul 4 2024 not biz day",
             dc_usd.is_business_day(date(2024,7,4)), False)
    chk_bool("DayCount USD: Jul 5 2024 is biz day",
             dc_usd.is_business_day(date(2024,7,5)), True)

    # BUS/252 year fraction should use holiday-aware business days
    dc_brl = DayCount("BRL")
    # Jan 1 (holiday) + Jan 13-14 (weekend) -> fewer than 10*5/7 days
    bd_brl_yf = dc_brl.year_fraction(date(2024,1,1), date(2024,1,15))
    print(f"  BRL BUS/252 Jan 1–14 2024 year frac = {bd_brl_yf:.6f}  (holiday-adjusted)")
    print()

    # ── DayCount.summary ─────────────────────────────────────────────────────
    DayCount("USD").summary(
        tenors=["1M","3M","6M","1Y","2Y","5Y","10Y"],
        start=date(2024, 1, 15),
    )
    DayCount("BRL").summary(
        tenors=["1M","3M","6M","1Y"],
        start=date(2024, 1, 15),
    )

    DayCount.compare(start=date(2024,1,15))
    DayCount.list_currencies()

    print()
    print("=" * 70)
    print("All tests passed ✓" if all_ok else "SOME TESTS FAILED ✗")
    print("=" * 70)
