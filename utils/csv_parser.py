"""CSV upload parser — validates, normalizes, and converts to pipeline format."""

import base64
import io
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Column names recognized as timestamps
_TIME_ALIASES = ("timestamp", "time_tag", "date", "datetime", "time", "ds")


def parse_uploaded_csv(
    contents: str | None,
    filename: str = "upload.csv",
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Parse a base64-encoded file upload into pipeline format.

    Supports CSV, TSV, Excel (.xlsx/.xls), and JSON files.
    Auto-detects delimiter for CSV files.

    Args:
        contents: Base64-encoded string (``"data:...;base64,<payload>"``).
        filename: Original filename (used for format detection).

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

        # --- Read file based on extension ---
        filename_lower = filename.lower()

        try:
            if filename_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(decoded))

            elif filename_lower.endswith(".json"):
                df = pd.read_json(io.BytesIO(decoded))

            elif filename_lower.endswith(".tsv"):
                df = pd.read_csv(io.BytesIO(decoded), sep="\t")

            elif filename_lower.endswith((".h5", ".hdf5")):
                try:
                    import h5py
                except ImportError:
                    return None, "HDF5 destegi icin h5py yukleyin: pip install h5py"
                import os
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
                try:
                    tmp.write(decoded)
                    tmp.close()
                    with h5py.File(tmp.name, "r") as f:
                        keys = list(f.keys())
                        if not keys:
                            return None, "HDF5 dosyasinda dataset bulunamadi"
                        ds = f[keys[0]]
                        if hasattr(ds, "shape") and len(ds.shape) <= 2:
                            df = pd.DataFrame(ds[()])
                        else:
                            return None, f"HDF5 dataset boyutu desteklenmiyor: {ds.shape}"
                finally:
                    os.unlink(tmp.name)

            elif filename_lower.endswith(".parquet"):
                try:
                    df = pd.read_parquet(io.BytesIO(decoded))
                except ImportError:
                    return None, "Parquet destegi icin pyarrow yukleyin: pip install pyarrow"

            else:
                # CSV with auto-delimiter detection
                try:
                    import csv as _csv_mod
                    sample = decoded[:2048].decode("utf-8", errors="ignore")
                    dialect = _csv_mod.Sniffer().sniff(sample)
                    df = pd.read_csv(io.BytesIO(decoded), sep=dialect.delimiter)
                except Exception:
                    df = pd.read_csv(io.BytesIO(decoded))

        except pd.errors.EmptyDataError:
            return None, "Dosya boş veya geçersiz CSV formatı"
        except pd.errors.ParserError:
            return None, "CSV parse hatası — dosya formatını kontrol edin"
        except Exception as e:
            return None, f"Dosya okuma hatası: {e}"

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
