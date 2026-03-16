# data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd


REQUIRED_COLS: list[str] = [
    "date",
    "open_qfq",
    "high_qfq",
    "low_qfq",
    "close_qfq",
    "volume",
    "amount",
    "turnover_rate",
]


@dataclass
class DataReport:
    rows_raw: int
    rows_after_dropna_core: int
    rows_after_filters: int
    duplicate_dates: int
    bad_ohlc_rows: int
    sheet_used: str
    notes: List[str]


def load_stock_excel(
    excel_path: str,
    sheet_name: Optional[str] = None,
    strict_columns: bool = True,
) -> Tuple[pd.DataFrame, DataReport]:
    """
    Load and clean A-share daily data from Excel.

    Expected columns:
        date, open_qfq, high_qfq, low_qfq, close_qfq, volume, amount, turnover_rate
    """
    notes: List[str] = []
    df_obj = pd.read_excel(excel_path, sheet_name=sheet_name)

    sheet_used = ""
    if isinstance(df_obj, dict):
        if sheet_name is not None and sheet_name in df_obj:
            df = df_obj[sheet_name]
            sheet_used = str(sheet_name)
        else:
            first_sheet = next(iter(df_obj.keys()))
            df = df_obj[first_sheet]
            sheet_used = str(first_sheet)
        notes.append(f"Excel has multiple sheets; using sheet: {sheet_used}")
    else:
        df = df_obj
        sheet_used = sheet_name or "default_sheet"

    rows_raw = len(df)

    # Column normalization: strip spaces
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        if strict_columns:
            raise ValueError(msg)
        notes.append(msg)

    keep_cols = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # Parse date (auto-detect)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = int(df["date"].isna().sum())
    if bad_date > 0:
        notes.append(f"Dropped rows with unparseable dates: {bad_date}")
    df = df.dropna(subset=["date"]).copy()

    df = df.sort_values("date").reset_index(drop=True)

    duplicate_dates = int(df["date"].duplicated().sum())
    if duplicate_dates > 0:
        notes.append(f"Found duplicate dates: {duplicate_dates}. Keeping last occurrence per date.")
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # Convert numeric columns
    num_cols = [c for c in REQUIRED_COLS if c != "date" and c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    core = ["open_qfq", "high_qfq", "low_qfq", "close_qfq"]
    rows_before = len(df)
    df = df.dropna(subset=core).copy()
    rows_after_dropna_core = len(df)
    if rows_after_dropna_core < rows_before:
        notes.append(f"Dropped rows with NaN core OHLC: {rows_before - rows_after_dropna_core}")

    # Fill missing non-core fields with 0
    for c in ["volume", "amount", "turnover_rate"]:
        n_missing = int(df[c].isna().sum())
        if n_missing > 0:
            notes.append(f"Filled missing {c} with 0: {n_missing}")
            df[c] = df[c].fillna(0.0)

    # Validity filters
    bad_price = int(
        ((df["open_qfq"] <= 0) | (df["high_qfq"] <= 0) | (df["low_qfq"] <= 0) | (df["close_qfq"] <= 0)).sum()
    )
    if bad_price > 0:
        notes.append(f"Dropped non-positive price rows: {bad_price}")
        df = df[(df["open_qfq"] > 0) & (df["high_qfq"] > 0) & (df["low_qfq"] > 0) & (df["close_qfq"] > 0)].copy()

    bad_vol_amt = int(((df["volume"] < 0) | (df["amount"] < 0)).sum())
    if bad_vol_amt > 0:
        notes.append(f"Dropped negative volume/amount rows: {bad_vol_amt}")
        df = df[(df["volume"] >= 0) & (df["amount"] >= 0)].copy()

    # OHLC consistency
    bad_ohlc_mask = ~(
        (df["high_qfq"] >= df[["open_qfq", "close_qfq", "low_qfq"]].max(axis=1))
        & (df["low_qfq"] <= df[["open_qfq", "close_qfq", "high_qfq"]].min(axis=1))
    )
    bad_ohlc_rows = int(bad_ohlc_mask.sum())
    if bad_ohlc_rows > 0:
        notes.append(f"Found OHLC inconsistent rows: {bad_ohlc_rows}. Dropping them.")
        df = df[~bad_ohlc_mask].copy()

    df = df.sort_values("date").reset_index(drop=True)
    rows_after_filters = len(df)

    if rows_after_filters < 300:
        notes.append(f"Warning: data length {rows_after_filters} < 300 trading days. ML may be unstable.")

    report = DataReport(
        rows_raw=rows_raw,
        rows_after_dropna_core=rows_after_dropna_core,
        rows_after_filters=rows_after_filters,
        duplicate_dates=duplicate_dates,
        bad_ohlc_rows=bad_ohlc_rows,
        sheet_used=sheet_used,
        notes=notes,
    )
    return df, report