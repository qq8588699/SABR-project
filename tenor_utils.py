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
    in the given year/month.  n=-1 means last occurrence.
    """
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + (n - 1) * 7)
    else:  # last
        last = date(year, month, 28) + timedelta(days=4)   # always in month
        last = last - timedelta(days=last.day - 1)          # first of month
        last = last + relativedelta(months=1) - timedelta(days=1)
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
    @lru_cache(maxsize=64)
    def holidays(self, year: int) -> frozenset:
        """
        Return the set of public holidays (weekdays only) for the given year.

        Weekends are excluded — use ``is_business_day()`` to check both.

        Parameters
        ----------
        year : int

        Returns
        -------
        frozenset of date
        """
        fn = self._GENERATORS.get(self._currency, self._western_generic)
        raw = fn(self, year)
        # Remove any dates that fall on Saturday or Sunday
        return frozenset(d for d in raw if d.weekday() < 5)

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
        return [
            _substitute(date(year, 1, 1)),             # New Year's Day
            _nth_weekday(year, 1, 3, 0),               # MLK Day (3rd Mon Jan)
            _nth_weekday(year, 2, 3, 0),               # Presidents Day (3rd Mon Feb)
            _nth_weekday(year, 5, -1, 0),              # Memorial Day (last Mon May)
            _substitute(date(year, 6, 19)),             # Juneteenth
            _substitute(date(year, 7, 4)),              # Independence Day
            _nth_weekday(year, 9, 1, 0),               # Labor Day (1st Mon Sep)
            _nth_weekday(year, 10, 2, 0),              # Columbus Day (2nd Mon Oct)
            _substitute(date(year, 11, 11)),            # Veterans Day
            _nth_weekday(year, 11, 4, 3),              # Thanksgiving (4th Thu Nov)
            _substitute(date(year, 12, 25)),            # Christmas Day
        ]

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
    def _gbp(self, year: int) -> list:
        e    = _easter(year)
        hols = [
            _substitute(date(year, 1, 1)),             # New Year's Day
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            _nth_weekday(year, 5, 1, 0),              # Early May Bank Holiday
            _nth_weekday(year, 5, -1, 0),             # Spring Bank Holiday
            _nth_weekday(year, 8, -1, 0),             # Summer Bank Holiday
            _substitute(date(year, 12, 25)),           # Christmas Day
            _substitute(date(year, 12, 26)),           # Boxing Day
        ]
        # Queen's/King's Jubilee and other one-offs handled via fixed year checks
        if year == 2022:
            hols += [date(2022, 6, 2), date(2022, 6, 3),   # Platinum Jubilee
                     date(2022, 9, 19)]                     # State Funeral
        if year == 2023:
            hols.append(date(2023, 5, 8))              # Coronation Bank Holiday
        return hols

    # ── JPY — Tokyo ───────────────────────────────────────────────────
    def _jpy(self, year: int) -> list:
        hols = [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 2),                          # Bank Holiday
            date(year, 1, 3),                          # Bank Holiday
            _nth_weekday(year, 1, 2, 0),              # Coming of Age (2nd Mon)
            date(year, 2, 11),                         # National Foundation Day
            date(year, 2, 23),                         # Emperor's Birthday
            date(year, 3, 20),                         # Vernal Equinox (approx)
            date(year, 4, 29),                         # Showa Day
            date(year, 5, 3),                          # Constitution Day
            date(year, 5, 4),                          # Greenery Day
            date(year, 5, 5),                          # Children's Day
            _nth_weekday(year, 7, 3, 0),              # Marine Day (3rd Mon)
            date(year, 8, 11),                         # Mountain Day
            _nth_weekday(year, 9, 3, 0),              # Respect for Aged (3rd Mon)
            date(year, 9, 23),                         # Autumnal Equinox (approx)
            _nth_weekday(year, 10, 2, 0),             # Sports Day (2nd Mon)
            date(year, 11, 3),                         # Culture Day
            date(year, 11, 23),                        # Labour Thanksgiving Day
            date(year, 12, 31),                        # Bank Holiday
        ]
        return hols

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
        return [
            _substitute(date(year, 1, 1)),             # New Year's Day
            date(year, 1, 26) if date(year, 1, 26).weekday() < 5
                else date(year, 1, 27),                # Australia Day
            e - timedelta(days=2),                     # Good Friday
            e - timedelta(days=1),                     # Easter Saturday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 4, 25) if date(year, 4, 25).weekday() < 5
                else date(year, 4, 26),                # Anzac Day
            _nth_weekday(year, 6, 2, 0),              # Queen's/King's Birthday (2nd Mon Jun)
            _nth_weekday(year, 8, 1, 0),              # Bank Holiday NSW (1st Mon Aug)
            _nth_weekday(year, 10, 1, 0),             # Labour Day NSW (1st Mon Oct)
            _substitute(date(year, 12, 25)),           # Christmas Day
            _substitute(date(year, 12, 26)),           # Boxing Day
        ]

    # ── CAD — Toronto ─────────────────────────────────────────────────
    def _cad(self, year: int) -> list:
        e = _easter(year)
        return [
            _substitute(date(year, 1, 1)),             # New Year's Day
            _nth_weekday(year, 2, 3, 0),              # Family Day (3rd Mon Feb)
            e - timedelta(days=2),                     # Good Friday
            _nth_weekday(year, 5, -1, 0),             # Victoria Day (Mon before May 25)
            _substitute(date(year, 7, 1)),             # Canada Day
            _nth_weekday(year, 8, 1, 0),              # Civic Holiday (1st Mon Aug)
            _nth_weekday(year, 9, 1, 0),              # Labour Day (1st Mon Sep)
            _nth_weekday(year, 10, 2, 0),             # Thanksgiving (2nd Mon Oct)
            _substitute(date(year, 11, 11)),           # Remembrance Day
            _substitute(date(year, 12, 25)),           # Christmas Day
            _substitute(date(year, 12, 26)),           # Boxing Day
        ]

    # ── NZD — Wellington ──────────────────────────────────────────────
    def _nzd(self, year: int) -> list:
        e = _easter(year)
        return [
            _substitute(date(year, 1, 1)),             # New Year's Day
            _substitute(date(year, 1, 2)),             # Day after New Year
            date(year, 2, 6) if date(year, 2, 6).weekday() < 5
                else date(year, 2, 7),                 # Waitangi Day
            date(year, 4, 25) if date(year, 4, 25).weekday() < 5
                else date(year, 4, 26),                # Anzac Day
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            _nth_weekday(year, 6, 1, 0),              # Queen's/King's Birthday (1st Mon Jun)
            _nth_weekday(year, 10, 4, 0),             # Labour Day (4th Mon Oct)
            _substitute(date(year, 12, 25)),           # Christmas Day
            _substitute(date(year, 12, 26)),           # Boxing Day
        ]

    # ── HKD — Hong Kong ───────────────────────────────────────────────
    def _hkd(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=2),                     # Good Friday
            e - timedelta(days=1),                     # Holy Saturday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 7, 1),                          # HKSAR Establishment Day
            date(year, 10, 1),                         # National Day
            date(year, 10, 2),                         # National Day (Golden Week)
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # Boxing Day
        ]
        # Chinese New Year: approximate (lunar, varies Jan/Feb)
        # Using a lookup for 2020-2030; default to Jan 25 otherwise
        cny = {
            2020: date(2020, 1, 25), 2021: date(2021, 2, 12),
            2022: date(2022, 2, 1),  2023: date(2023, 1, 22),
            2024: date(2024, 2, 10), 2025: date(2025, 1, 29),
            2026: date(2026, 2, 17), 2027: date(2027, 2, 6),
            2028: date(2028, 1, 26), 2029: date(2029, 2, 13),
            2030: date(2030, 2, 3),
        }
        cny_date = cny.get(year, date(year, 1, 25))
        for offset in range(-1, 4):   # CNY Eve + 3 days
            hols.append(cny_date + timedelta(days=offset))
        return hols

    # ── SGD — Singapore ───────────────────────────────────────────────
    def _sgd(self, year: int) -> list:
        e = _easter(year)
        hols = [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=2),                     # Good Friday
            date(year, 5, 1),                          # Labour Day
            date(year, 8, 9),                          # National Day
            date(year, 12, 25),                        # Christmas Day
        ]
        # Chinese New Year (same lookup as HKD)
        cny = {
            2020: date(2020, 1, 25), 2021: date(2021, 2, 12),
            2022: date(2022, 2, 1),  2023: date(2023, 1, 22),
            2024: date(2024, 2, 10), 2025: date(2025, 1, 29),
            2026: date(2026, 2, 17), 2027: date(2027, 2, 6),
            2028: date(2028, 1, 26), 2029: date(2029, 2, 13),
            2030: date(2030, 2, 3),
        }
        cny_date = cny.get(year, date(year, 1, 25))
        hols.append(cny_date)
        hols.append(cny_date + timedelta(days=1))
        # Hari Raya Puasa and Hari Raya Haji: Islamic calendar, approximate
        # Deepavali: Hindu calendar, approximate (Oct/Nov)
        hols.append(_nth_weekday(year, 11, 1, 0))     # approximate Deepavali
        return hols

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
        return [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 6),                          # Epiphany
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            e + timedelta(days=39),                    # Ascension Day
            date(year, 6, 6),                          # National Day
            _nth_weekday(year, 6, -1, 4),             # Midsummer Day (Fri)
            _nth_weekday(year, 11, 1, 5),             # All Saints' Day (Sat on/after Oct 31)
            date(year, 12, 24),                        # Christmas Eve
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
            date(year, 12, 31),                        # New Year's Eve
        ]

    # ── DKK — Copenhagen ──────────────────────────────────────────────
    def _dkk(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=3),                     # Maundy Thursday
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            e + timedelta(days=26),                    # Store Bededag (4th Fri after Easter) — removed 2024+
            e + timedelta(days=39),                    # Ascension Day
            e + timedelta(days=50),                    # Whit Monday
            date(year, 6, 5),                          # Constitution Day
            date(year, 12, 24),                        # Christmas Eve
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
            date(year, 12, 31),                        # New Year's Eve
        ]

    # ── PLN — Warsaw ──────────────────────────────────────────────────
    def _pln(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 6),                          # Epiphany
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 3),                          # Constitution Day
            e + timedelta(days=50),                    # Whit Sunday
            e + timedelta(days=60),                    # Corpus Christi
            date(year, 8, 15),                         # Assumption Day
            date(year, 11, 1),                         # All Saints' Day
            date(year, 11, 11),                        # Independence Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]

    # ── HUF — Budapest ────────────────────────────────────────────────
    def _huf(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            e + timedelta(days=50),                    # Whit Monday
            date(year, 8, 20),                         # St Stephen's Day
            date(year, 10, 23),                        # Republic Day
            date(year, 11, 1),                         # All Saints' Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]

    # ── CZK — Prague ──────────────────────────────────────────────────
    def _czk(self, year: int) -> list:
        e = _easter(year)
        return [
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

    # ── RON — Bucharest ───────────────────────────────────────────────
    def _ron(self, year: int) -> list:
        e = _easter(year)   # Romanian Orthodox Easter (approximate Western date)
        return [
            date(year, 1, 1),                          # New Year's Day
            date(year, 1, 2),                          # New Year Holiday
            date(year, 1, 24),                         # Unification Day
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Easter Monday
            date(year, 5, 1),                          # Labour Day
            date(year, 6, 1),                          # Children's Day
            e + timedelta(days=50),                    # Whit Monday
            date(year, 8, 15),                         # Assumption Day
            date(year, 11, 30),                        # St Andrew's Day
            date(year, 12, 1),                         # National Day
            date(year, 12, 25),                        # Christmas Day
            date(year, 12, 26),                        # 2nd Day of Christmas
        ]

    # ── ILS — Tel Aviv ────────────────────────────────────────────────
    def _ils(self, year: int) -> list:
        # Jewish holidays use lunar calendar; using fixed approximations
        # Friday is a half day; Saturday is full weekend
        # Major approximate holidays:
        hols = [date(year, 9, 16)]    # Rosh Hashanah approx (varies)
        return hols  # simplified — full Hebrew calendar would require lunardate

    # ── ZAR — Johannesburg ────────────────────────────────────────────
    def _zar(self, year: int) -> list:
        e = _easter(year)
        return [
            _substitute(date(year, 1, 1)),             # New Year's Day
            _substitute(date(year, 3, 21)),            # Human Rights Day
            e - timedelta(days=2),                     # Good Friday
            e + timedelta(days=1),                     # Family Day
            _substitute(date(year, 4, 27)),            # Freedom Day
            _substitute(date(year, 5, 1)),             # Workers' Day
            _substitute(date(year, 6, 16)),            # Youth Day
            _substitute(date(year, 8, 9)),             # National Women's Day
            _substitute(date(year, 9, 24)),            # Heritage Day
            _substitute(date(year, 12, 16)),           # Day of Reconciliation
            _substitute(date(year, 12, 25)),           # Christmas Day
            _substitute(date(year, 12, 26)),           # Day of Goodwill
        ]

    # ── BRL — São Paulo ───────────────────────────────────────────────
    def _brl(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            e - timedelta(days=48),                    # Carnival Monday
            e - timedelta(days=47),                    # Carnival Tuesday
            e - timedelta(days=2),                     # Good Friday
            date(year, 4, 21),                         # Tiradentes
            date(year, 5, 1),                          # Labour Day
            e + timedelta(days=60),                    # Corpus Christi
            date(year, 9, 7),                          # Independence Day
            date(year, 10, 12),                        # Our Lady of Aparecida
            date(year, 11, 2),                         # All Souls' Day
            date(year, 11, 15),                        # Proclamation of the Republic
            date(year, 11, 20),                        # Black Consciousness Day
            date(year, 12, 25),                        # Christmas Day
        ]

    # ── MXN — Mexico City ─────────────────────────────────────────────
    def _mxn(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),                          # New Year's Day
            _nth_weekday(year, 2, 1, 0),              # Constitution Day (1st Mon Feb)
            _nth_weekday(year, 3, 3, 0),              # Benito Juarez Birthday (3rd Mon Mar)
            e - timedelta(days=3),                     # Maundy Thursday
            e - timedelta(days=2),                     # Good Friday
            date(year, 5, 1),                          # Labour Day
            date(year, 9, 16),                         # Independence Day
            _nth_weekday(year, 10, 3, 0),             # Columbus Day (3rd Mon Oct)
            date(year, 11, 2),                         # Day of the Dead
            date(year, 11, 20),                        # Revolution Day
            date(year, 12, 12),                        # Our Lady of Guadalupe
            date(year, 12, 25),                        # Christmas Day
        ]

    # ── COP — Bogotá ──────────────────────────────────────────────────
    def _cop(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),
            _next_monday(date(year, 1, 6)),            # Epiphany
            _next_monday(date(year, 3, 19)),           # St Joseph
            e - timedelta(days=3),                     # Maundy Thursday
            e - timedelta(days=2),                     # Good Friday
            date(year, 5, 1),                          # Labour Day
            _next_monday(e + timedelta(days=43)),      # Ascension
            _next_monday(e + timedelta(days=64)),      # Corpus Christi
            _next_monday(e + timedelta(days=71)),      # Sacred Heart
            _next_monday(date(year, 6, 29)),           # SS Peter & Paul
            date(year, 7, 20),                         # Independence Day
            date(year, 8, 7),                          # Battle of Boyacá
            _next_monday(date(year, 8, 15)),           # Assumption
            _next_monday(date(year, 10, 12)),          # Columbus Day
            _next_monday(date(year, 11, 1)),           # All Saints
            _next_monday(date(year, 11, 11)),          # Independence of Cartagena
            date(year, 12, 8),                         # Immaculate Conception
            date(year, 12, 25),
        ]

    # ── CLP — Santiago ────────────────────────────────────────────────
    def _clp(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),
            e - timedelta(days=2),                     # Good Friday
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 21),                         # Navy Day
            date(year, 6, 20),                         # Indigenous Peoples Day (approx)
            date(year, 7, 16),                         # Our Lady of Mount Carmel
            date(year, 8, 15),                         # Assumption Day
            date(year, 9, 18),                         # Independence Day
            date(year, 9, 19),                         # Army Day
            date(year, 10, 12),                        # Columbus Day
            date(year, 10, 27),                        # Evangelical Church Day
            date(year, 11, 1),                         # All Saints' Day
            date(year, 12, 8),                         # Immaculate Conception
            date(year, 12, 25),
        ]

    # ── CNY — Shanghai ────────────────────────────────────────────────
    def _cny(self, year: int) -> list:
        cny_map = {
            2020: date(2020, 1, 25), 2021: date(2021, 2, 12),
            2022: date(2022, 2, 1),  2023: date(2023, 1, 22),
            2024: date(2024, 2, 10), 2025: date(2025, 1, 29),
            2026: date(2026, 2, 17), 2027: date(2027, 2, 6),
        }
        cny_date = cny_map.get(year, date(year, 2, 5))
        hols = []
        for i in range(7):
            hols.append(cny_date + timedelta(days=i))
        hols += [
            date(year, 1, 1),                          # New Year's Day
            date(year, 4, 4),                          # Qingming (approx)
            date(year, 5, 1),                          # Labour Day
            date(year, 10, 1),                         # National Day Golden Week
            date(year, 10, 2),
            date(year, 10, 3),
            date(year, 10, 4),
            date(year, 10, 5),
            date(year, 10, 6),
            date(year, 10, 7),
        ]
        return hols

    # ── INR — Mumbai ──────────────────────────────────────────────────
    def _inr(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 26),                         # Republic Day
            e - timedelta(days=2),                     # Good Friday
            date(year, 4, 14),                         # Dr Ambedkar Jayanti (approx)
            date(year, 5, 1),                          # Maharashtra Day
            date(year, 8, 15),                         # Independence Day
            date(year, 10, 2),                         # Gandhi Jayanti
            date(year, 11, 1),                         # Diwali (approx)
            date(year, 12, 25),
        ]

    # ── IDR — Jakarta ─────────────────────────────────────────────────
    def _idr(self, year: int) -> list:
        e = _easter(year)
        return [
            date(year, 1, 1),
            e - timedelta(days=2),                     # Good Friday
            date(year, 5, 1),                          # Labour Day
            date(year, 8, 17),                         # Independence Day
            date(year, 12, 25),
        ]

    # ── MYR — Kuala Lumpur ────────────────────────────────────────────
    def _myr(self, year: int) -> list:
        return [
            date(year, 1, 1),
            date(year, 2, 1),                          # Federal Territory Day
            date(year, 5, 1),                          # Labour Day
            date(year, 8, 31),                         # National Day
            date(year, 9, 16),                         # Malaysia Day
            date(year, 12, 25),
        ]

    # ── THB — Bangkok ─────────────────────────────────────────────────
    def _thb(self, year: int) -> list:
        return [
            date(year, 1, 1),
            date(year, 4, 6),                          # Chakri Memorial Day
            date(year, 4, 13),                         # Songkran
            date(year, 4, 14),
            date(year, 4, 15),
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 4),                          # Coronation Day
            date(year, 7, 28),                         # King's Birthday
            date(year, 8, 12),                         # Queen Mother's Birthday
            date(year, 10, 13),                        # King Bhumibol Memorial Day
            date(year, 10, 23),                        # Chulalongkorn Day
            date(year, 12, 5),                         # King Bhumibol's Birthday
            date(year, 12, 10),                        # Constitution Day
            date(year, 12, 31),
        ]

    # ── SAR — Riyadh ──────────────────────────────────────────────────
    def _sar(self, year: int) -> list:
        # Saudi Arabia observes Fri/Sat weekends; Islamic calendar holidays
        # approximate dates only
        return [
            date(year, 2, 22),                         # Founding Day
            date(year, 9, 23),                         # National Day
        ]

    # ── KRW — Seoul ───────────────────────────────────────────────────
    def _krw(self, year: int) -> list:
        return [
            date(year, 1, 1),
            date(year, 3, 1),                          # Independence Movement Day
            date(year, 5, 5),                          # Children's Day
            date(year, 6, 6),                          # Memorial Day
            date(year, 8, 15),                         # Liberation Day
            date(year, 10, 3),                         # National Foundation Day
            date(year, 10, 9),                         # Hangeul Proclamation Day
            date(year, 12, 25),
        ]

    # ── TWD — Taipei ──────────────────────────────────────────────────
    def _twd(self, year: int) -> list:
        cny_map = {
            2023: date(2023, 1, 22), 2024: date(2024, 2, 10),
            2025: date(2025, 1, 29), 2026: date(2026, 2, 17),
        }
        cny = cny_map.get(year, date(year, 2, 5))
        hols = [cny + timedelta(days=i) for i in range(5)]
        hols += [
            date(year, 1, 1),
            date(year, 2, 28),                         # Peace Memorial Day
            date(year, 4, 4),                          # Children's Day / Qingming
            date(year, 5, 1),                          # Labour Day
            date(year, 10, 10),                        # National Day
        ]
        return hols

    # ── RUB — Moscow ──────────────────────────────────────────────────
    def _rub(self, year: int) -> list:
        return [
            date(year, 1, 1),
            date(year, 1, 2),
            date(year, 1, 3),
            date(year, 1, 4),
            date(year, 1, 5),
            date(year, 1, 6),
            date(year, 1, 7),                          # Orthodox Christmas
            date(year, 1, 8),
            date(year, 2, 23),                         # Defender of Fatherland Day
            date(year, 3, 8),                          # International Women's Day
            date(year, 5, 1),                          # Spring/Labour Day
            date(year, 5, 9),                          # Victory Day
            date(year, 6, 12),                         # Russia Day
            date(year, 11, 4),                         # National Unity Day
        ]

    # ── TRY — Istanbul ────────────────────────────────────────────────
    def _try(self, year: int) -> list:
        return [
            date(year, 1, 1),
            date(year, 4, 23),                         # National Sovereignty Day
            date(year, 5, 1),                          # Labour Day
            date(year, 5, 19),                         # Atatürk Day
            date(year, 7, 15),                         # Democracy Day
            date(year, 8, 30),                         # Victory Day
            date(year, 10, 28),                        # Republic Day Eve (half day)
            date(year, 10, 29),                        # Republic Day
        ]

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
        """Year fraction per calendar day (pass as dt to run_pca)."""
        ref = date(2024, 1, 15)
        return self._apply(ref, ref + timedelta(days=1))

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
