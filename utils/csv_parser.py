"""CSV upload parser — validates, normalizes, and converts to pipeline format."""

import base64
import io
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column names recognized as timestamps
_TIME_ALIASES = ("timestamp", "time_tag", "date", "datetime", "time", "ds")


def parse_uploaded_csv(
    contents: str | None,
    filename: str = "upload.csv",
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Parse a Dash/Streamlit-style base64-encoded CSV upload into pipeline format.

    Args:
        contents: Base64-encoded CSV string (``"data:text/csv;base64,<payload>"``).
                  May also be a raw base64 string without the prefix.
        filename: Original filename (for error messages only).

    Returns:
        ``(df, None)`` on success — df has columns ``[timestamp, value]``.
        ``(None, error_message)`` on failure.
    """
    if contents is None:
        return None, "Dosya seçilmedi"

    try:
        # Strip the data-URI prefix if present
        if "," in contents:
            _, content_string = contents.split(",", 1)
        else:
            content_string = contents

        decoded = base64.b64decode(content_string)

        # --- Read CSV ---
        try:
            df = pd.read_csv(io.BytesIO(decoded))
        except pd.errors.EmptyDataError:
            return None, "Dosya boş veya geçersiz CSV formatı"
        except pd.errors.ParserError:
            return None, "CSV parse hatası — dosya formatını kontrol edin"

        if df.empty or len(df.columns) == 0:
            return None, "Dosya boş veya sütun bulunamadı"

        if len(df) < 10:
            return None, f"Çok az veri: {len(df)} satır (minimum 10 gerekli)"

        # --- Resolve timestamp column ---
        time_col = None
        for alias in _TIME_ALIASES:
            if alias in df.columns:
                time_col = alias
                break

        if time_col is None:
            df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="1s")
        else:
            df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
            if df["timestamp"].isna().all():
                return None, f"'{time_col}' sütunu geçerli tarih formatında değil"

        # --- Resolve value column ---
        if "value" not in df.columns:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != time_col]
            if not numeric_cols:
                return (
                    None,
                    f"Sayısal sütun bulunamadı. Mevcut sütunlar: {', '.join(df.columns.tolist())}",
                )
            df["value"] = df[numeric_cols[0]].astype(float)
        else:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # --- Build pipeline DataFrame ---
        result = df[["timestamp", "value"]].copy()
        result = result.dropna(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        if result["value"].isna().all():
            return None, "Tüm değerler NaN — geçerli sayısal veri yok"

        logger.info("CSV parsed: %d rows from %s", len(result), filename)
        return result, None

    except Exception as e:
        return None, f"Beklenmeyen hata: {str(e)}"
