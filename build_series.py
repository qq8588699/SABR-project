"""
spread_ts.py
------------
Build spread time series between financial rate curves (LIBOR/OIS/SOFR/FEDFUND etc.).

Usage examples
--------------
# Single tenor:
  python spread_ts.py -c SOFR -t 012M -o sofr_12m -i ./data -d ./out

# Batch from file (rows: tenor,output_name):
  python spread_ts.py -c OIS  -f tenors.txt     -i ./data -d ./out

# All tenors:
  python spread_ts.py -c SOFR -F               -i ./data -d ./out
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENORS: list[str] = [
    "001M", "002M", "003M", "006M", "012M", "018M", "024M",
    "036M", "048M", "060M", "084M", "120M", "180M", "240M",
    "300M", "360M", "480M", "600M", "720M",
]

# Column index 0 is always the date; tenor columns start at index 1.
TENOR_COL: dict[str, int] = {t: i + 1 for i, t in enumerate(TENORS)}

# Registry: curve label -> CSV filename stem (without .csv)
# Add new curves here without touching any other code.
CURVE_FILES: dict[str, str] = {
    "LIBOR":   "IRSP.USD.IBR.US.STD",
    "OIS":     "IRSP.USD.OIS.US.STD",
    "SOFR":    "IRSP.USD.SOF.US.STD",
    "FEDFUND": "IRSP.USD.FED.US.STD",   # example – add the real stem
}

# Curves always loaded as the base pair (index 0 and 1 in slist).
BASE_CURVES: list[str] = ["LIBOR", "OIS"]

# Adhoc correction file applied to OIS data.
CORRECTION_FILE = "AdhocOIS20250414.csv"

# Number of header lines to skip when reading curve CSVs.
HEADER_LINES = 2

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Row = list[str]          # One CSV row as a list of strings (col 0 = date)
DtRow = list             # One merged row: [date_int, val1, val2, ...]


@dataclass
class CurveSeries:
    """Holds all loaded data for a single rate curve."""
    label: str
    filepath: Path
    ts: list[Row]              = field(default_factory=list)
    dtmap: dict[str, int]      = field(default_factory=dict)   # date_str -> ts index
    start_idx: Optional[int]   = None   # index of first fully-populated row
    end_idx: Optional[int]     = None   # index of last  fully-populated row
    start_date: Optional[str]  = None
    end_date: Optional[str]    = None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _open_and_skip_headers(self):
        """Open the file, skip header lines, and return the file object."""
        f = self.filepath.open()
        for _ in range(HEADER_LINES):
            f.readline()
        return f

    def scan(self) -> None:
        """Find the first and last fully-populated rows (all tenor columns non-empty)."""
        with self._open_and_skip_headers() as f:
            first_idx = last_idx = None
            first_date = last_date = None
            for idx, raw in enumerate(f):
                row = raw.strip().split(",")
                if all(row):
                    if first_idx is None:
                        first_idx, first_date = idx, row[0]
                    last_idx, last_date = idx, row[0]

        self.start_idx, self.start_date = first_idx, first_date
        self.end_idx,   self.end_date   = last_idx,  last_date

    def load(self) -> None:
        """Load all rows between scan boundaries into self.ts."""
        if self.start_idx is None or self.end_idx is None:
            raise RuntimeError(f"Call scan() before load() for curve '{self.label}'.")

        lo = min(self.start_idx, self.end_idx)
        hi = max(self.start_idx, self.end_idx)

        with self._open_and_skip_headers() as f:
            for idx, raw in enumerate(f):
                if lo <= idx <= hi:
                    row = raw.strip().split(",")
                    self.ts.append(row)
                    self.dtmap[row[0]] = len(self.ts) - 1

    def apply_correction(self, correction_path: Path) -> None:
        """
        Patch self.ts in-place from an adhoc correction CSV.

        Expected columns: DATE, IDENTIFIER, PX_LAST
        DATE format: YYYY-MM-DD  (converted to YYYYMMDD for dtmap lookup).
        IDENTIFIER: Bloomberg-style OIS tenor identifier (e.g. USSOA, USSO1, USSO10).
        """
        if not correction_path.exists():
            print(f"[warn] Correction file not found: {correction_path}", file=sys.stderr)
            return

        with correction_path.open() as f:
            headers = f.readline().strip().split(",")
            col = {name: i for i, name in enumerate(headers)}

            for raw in f:
                row = raw.strip().split(",")
                if not row or not row[0]:
                    continue
                date_str = re.sub(r"-", "", row[col["DATE"]])
                tenor_str = _map_bloomberg_tenor(row[col["IDENTIFIER"]])
                price = row[col["PX_LAST"]]

                if tenor_str and tenor_str in TENOR_COL and date_str in self.dtmap:
                    ts_idx = self.dtmap[date_str]
                    col_idx = TENOR_COL[tenor_str]
                    self.ts[ts_idx][col_idx] = price

    def fill_all_tenors(self, linear: bool = True) -> None:
        """Fill missing values for every tenor column."""
        for col_idx in TENOR_COL.values():
            _fill_missing(self.ts, col_idx, linear)

    def __repr__(self) -> str:
        return (
            f"CurveSeries(label={self.label!r}, rows={len(self.ts)}, "
            f"start={self.start_date}, end={self.end_date})"
        )


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interpolate(data: list[Row], lo: int, hi: int, col_idx: int, linear: bool) -> None:
    """
    Fill data[lo+1 .. hi-1][col_idx] by interpolating between
    data[lo][col_idx] and data[hi][col_idx].

    Parameters
    ----------
    lo, hi : int
        Indices of known values (lo < hi).
    linear : bool
        True  -> linear interpolation.
        False -> alternating +100 / -100 sentinel (diagnostic / testing mode).
    """
    gap = hi - lo
    if gap <= 1:
        return  # Nothing to fill.

    try:
        v_lo = float(data[lo][col_idx])
        v_hi = float(data[hi][col_idx])
    except (ValueError, IndexError):
        return  # Cannot interpolate from non-numeric endpoints.

    increment = (v_hi - v_lo) / gap

    for step in range(1, gap):
        i = lo + step
        if linear:
            data[i][col_idx] = str(v_lo + step * increment)
        else:
            data[i][col_idx] = str(100 if step % 2 else -100)


def _fill_missing(data: list[Row], col_idx: int, linear: bool = True) -> None:
    """
    Scan *data* and fill any interior gaps in column *col_idx* by interpolation.
    Leading and trailing missing values are left as-is.
    """
    prev_known: Optional[int] = None

    for i, row in enumerate(data):
        val = row[col_idx] if col_idx < len(row) else ""
        if val:
            if prev_known is not None and prev_known < i - 1:
                _interpolate(data, prev_known, i, col_idx, linear)
            prev_known = i


# ---------------------------------------------------------------------------
# Bloomberg tenor identifier mapping
# ---------------------------------------------------------------------------

_BLOOMBERG_LETTER_MAP: dict[str, str] = {
    "A": "1", "B": "2", "C": "3", "F": "6", "I": "9",
}


def _map_bloomberg_tenor(identifier: str) -> Optional[str]:
    """
    Convert a Bloomberg OIS identifier to a standard tenor string.

    Examples
    --------
    'USSOA'   -> '001M'
    'USSO3'   -> '036M'
    'USSO10'  -> '120M'
    """
    # Letter suffix: A=1M, B=2M, C=3M, F=6M, I=9M
    m = re.search(r"USSO([A-I])", identifier)
    if m:
        months = _BLOOMBERG_LETTER_MAP[m.group(1)]
        return months.zfill(3) + "M"

    # Numeric suffix: number of years -> months
    m = re.search(r"USSO(\d{1,2})$", identifier)
    if m:
        months = str(12 * int(m.group(1)))
        return months.zfill(3) + "M"

    return None


# ---------------------------------------------------------------------------
# Building the merged date-aligned time series
# ---------------------------------------------------------------------------

def build_dt(s1: CurveSeries, s2: CurveSeries, tenor_str: str) -> list[DtRow]:
    """
    Merge two curve series on the date column for a single tenor.

    Returns a list of [date_int, val_s1, val_s2] rows covering all dates
    present in either series. Missing values are represented as None.
    """
    col = TENOR_COL[tenor_str]
    rows: list[DtRow] = []
    both_started = False
    i1 = i2 = 0

    while i1 < len(s1.ts) and i2 < len(s2.ts):
        d1 = int(s1.ts[i1][0])
        d2 = int(s2.ts[i2][0])

        if d1 < d2:
            if both_started:
                rows.append([d1, s1.ts[i1][col], None])
            i1 += 1
        elif d1 > d2:
            if both_started:
                rows.append([d2, None, s2.ts[i2][col]])
            i2 += 1
        else:
            v1 = s1.ts[i1][col]
            v2 = s2.ts[i2][col]
            if v1 and v2:
                both_started = True
            if both_started:
                rows.append([d1, v1, v2])
            i1 += 1
            i2 += 1

    return rows


def merge_dt(dts: list[DtRow], si: CurveSeries, tenor_str: str) -> list[DtRow]:
    """
    Append a new value column from *si* to each row of *dts*, date-aligned.

    Rows present in *dts* but not in *si* get None appended.
    Rows present in *si* but not in *dts* are dropped (only interior dates
    present in both are retained once both have started).

    Returns the extended list of DtRows.
    """
    col = TENOR_COL[tenor_str]
    merged: list[DtRow] = []
    started = False
    i_dt = i_si = 0

    while i_dt < len(dts) and i_si < len(si.ts):
        d_dt = int(dts[i_dt][0])
        d_si = int(si.ts[i_si][0])

        if d_dt < d_si:
            if started:
                merged.append(dts[i_dt] + [None])
            i_dt += 1
        elif d_dt > d_si:
            if started:
                merged.append([d_si] + [None] * (len(dts[0]) - 1) + [si.ts[i_si][col]])
            i_si += 1
        else:
            v = si.ts[i_si][col]
            if v:
                started = True
            if started:
                merged.append(dts[i_dt] + [v])
            i_dt += 1
            i_si += 1

    return merged


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def dump_ts(
    output_path: Path,
    dts: list[DtRow],
    series_list: list[CurveSeries],
    tenor_str: str,
) -> None:
    """Write the merged time series to *output_path* as CSV."""
    header = tenor_str + "," + ",".join(s.label for s in series_list)
    with output_path.open("w") as f:
        f.write(header + "\n")
        for row in dts:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def load_curve(label: str, data_dir: Path, linear: bool, correction_path: Optional[Path] = None) -> CurveSeries:
    """Scan, load, optionally correct, and fill a single CurveSeries."""
    stem = CURVE_FILES.get(label)
    if stem is None:
        raise ValueError(
            f"Unknown curve '{label}'. Known curves: {list(CURVE_FILES)}"
        )
    filepath = data_dir / f"{stem}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Curve file not found: {filepath}")

    series = CurveSeries(label=label, filepath=filepath)
    series.scan()
    series.load()

    if label == "OIS" and correction_path is not None:
        series.apply_correction(correction_path)

    series.fill_all_tenors(linear)
    return series


def process_tenor(
    tenor_str: str,
    output_stem: str,
    series_list: list[CurveSeries],
    output_dir: Path,
    linear: bool,
) -> None:
    """Build and write the merged spread series for one tenor."""
    if tenor_str not in TENOR_COL:
        raise ValueError(f"Unknown tenor '{tenor_str}'. Known tenors: {TENORS}")

    s1, s2 = series_list[0], series_list[1]
    dts = build_dt(s1, s2, tenor_str)
    print(f"  [{tenor_str}] base TS length: {len(dts)}")
    if dts:
        print(f"    first date: {dts[0][0]}  last date: {dts[-1][0]}")

    for si in series_list[2:]:
        dts = merge_dt(dts, si, tenor_str)
        print(f"  [{tenor_str}] after merging {si.label}: {len(dts)} rows")

    # Fill remaining gaps in the merged series.
    for col_idx in range(1, len(series_list) + 1):
        _fill_missing(dts, col_idx, linear)

    output_path = output_dir / f"{output_stem}.csv"
    dump_ts(output_path, dts, series_list, tenor_str)
    print(f"  [{tenor_str}] written -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build spread time series between financial rate curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--curve",
        required=True,
        help=f"Additional curve to merge. Choices: {list(CURVE_FILES)}",
    )
    parser.add_argument(
        "-i", "--ddir",
        default=".",
        help="Input data directory (default: current directory).",
    )
    parser.add_argument(
        "-d", "--dir",
        default=".",
        help="Output directory (default: same as --ddir).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-t", "--tenor",
        help="Single tenor (e.g. 012M).",
    )
    mode.add_argument(
        "-f", "--file",
        help="Batch file: each line is  tenor,output_stem",
    )
    mode.add_argument(
        "-F", "--full",
        action="store_true",
        help="Generate all tenors automatically.",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file stem (required with -t; e.g. sofr_12m -> sofr_12m.csv).",
    )
    parser.add_argument(
        "-L", "--linear",
        action="store_true",
        help="Use linear interpolation to fill missing values (default: sentinel).",
    )

    args = parser.parse_args(argv)

    # Validate: -t requires -o
    if args.tenor and not args.output:
        parser.error("-t/--tenor requires -o/--output")

    return args


def build_tenor_list(args: argparse.Namespace, curve_label: str) -> list[list[str]]:
    """Return a list of [tenor_str, output_stem] pairs."""
    prefix = curve_label.lower()

    if args.full:
        return [[t, f"{prefix}_{t}"] for t in TENORS]

    if args.file:
        tenor_list = []
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    tenor_list.append(parts)
        return tenor_list

    return [[args.tenor.strip(), args.output.strip()]]


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    data_dir   = Path(args.ddir)
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correction_path = data_dir / CORRECTION_FILE

    # ------------------------------------------------------------------
    # Always load the two base curves.
    # ------------------------------------------------------------------
    print("Loading base curves ...")
    series_list: list[CurveSeries] = [
        load_curve(label, data_dir, args.linear, correction_path)
        for label in BASE_CURVES
    ]
    for s in series_list:
        print(f"  {s}")

    # ------------------------------------------------------------------
    # Load the additional curve if it is not already in the base pair.
    # ------------------------------------------------------------------
    extra_label = args.curve.upper()
    if extra_label not in BASE_CURVES:
        if extra_label not in CURVE_FILES:
            print(
                f"Error: unknown curve '{extra_label}'. "
                f"Known curves: {list(CURVE_FILES)}",
                file=sys.stderr,
            )
            return 1
        print(f"Loading extra curve: {extra_label} ...")
        extra = load_curve(extra_label, data_dir, args.linear, correction_path)
        series_list.append(extra)
        print(f"  {extra}")

    # ------------------------------------------------------------------
    # Process each tenor.
    # ------------------------------------------------------------------
    tenor_list = build_tenor_list(args, extra_label if extra_label not in BASE_CURVES else BASE_CURVES[1])

    print(f"\nProcessing {len(tenor_list)} tenor(s) ...")
    errors: list[str] = []
    for tenor_str, output_stem in tenor_list:
        try:
            process_tenor(tenor_str, output_stem, series_list, output_dir, args.linear)
        except Exception as exc:
            msg = f"  ERROR [{tenor_str}]: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        print(f"\n{len(errors)} tenor(s) failed.", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
