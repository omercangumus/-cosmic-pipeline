"""GOES satellite data downloader from NOAA SWPC JSON API."""

import os
import json
import requests
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# Notable solar flare events for reference
SOLAR_FLARE_EVENTS = {
    "X2.2_2024_05_08": "2024-05-08",
    "X1.0_2024_02_22": "2024-02-22",
    "M5.8_2024_03_28": "2024-03-28",
}


@dataclass
class GOESConfig:
    """Configuration for GOES data downloader."""
    api_url: str = "https://services.swpc.noaa.gov/json/goes/primary/differential-protons-1-day.json"
    cache_enabled: bool = True
    timeout_seconds: int = 10


class NetworkError(Exception):
    """Raised when network request fails."""
    pass


def download_goes_realtime(save_path: str = "data/raw/goes_proton.json") -> Optional[str]:
    """
    Download latest GOES proton flux JSON from NOAA SWPC.
    
    Args:
        save_path: Path to save downloaded JSON file
        
    Returns:
        Path to saved file, or None on failure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://services.swpc.noaa.gov/json/goes/primary/differential-protons-1-day.json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, "w") as f:
            f.write(response.text)
        
        print(f"[goes_downloader] Downloaded {len(response.text)} bytes → {save_path}")
        return save_path
        
    except requests.exceptions.RequestException as e:
        print(f"[goes_downloader] Download failed: {e}")
        return None


def parse_goes_json(filepath: str) -> pd.DataFrame:
    """
    Parse NOAA SWPC JSON file into normalized DataFrame.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        DataFrame with columns [timestamp, channel, value]
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    
    records = []
    
    for entry in data:
        try:
            ts = pd.to_datetime(entry.get("time_tag"))
            
            # Try common SWPC field names for flux value
            val = (
                entry.get("flux") or
                entry.get("proton_flux") or
                entry.get("electron_flux") or
                entry.get("p1") or  # differential channel
                entry.get("p2") or
                None
            )
            
            # If no known field, take first numeric value
            if val is None:
                for v in entry.values():
                    if isinstance(v, (int, float)):
                        val = v
                        break
            
            if val is not None:
                records.append({
                    "timestamp": ts,
                    "channel": "proton_flux",
                    "value": float(val)
                })
                
        except (KeyError, TypeError, ValueError):
            continue
    
    df = pd.DataFrame(records)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    
    return df


def get_goes_dataframe() -> pd.DataFrame:
    """
    Convenience wrapper: download → parse → return DataFrame.
    Falls back to synthetic signal if download or parse fails.
    
    Returns:
        DataFrame with GOES data or synthetic fallback
    """
    from data.synthetic_generator import generate_clean_signal, inject_faults
    
    path = download_goes_realtime()
    
    if path is None:
        print("[goes_downloader] Using synthetic fallback")
        df, _ = inject_faults(generate_clean_signal(1440))  # 1 day at 1min resolution
        df["channel"] = "proton_flux_synthetic"
        return df
    
    try:
        df = parse_goes_json(path)
        
        if len(df) == 0:
            raise ValueError("Empty dataframe after parse")
        
        print(f"[goes_downloader] Parsed {len(df)} real GOES records")
        return df
        
    except Exception as e:
        print(f"[goes_downloader] Parse failed: {e}. Using synthetic fallback.")
        df, _ = inject_faults(generate_clean_signal(1440))
        df["channel"] = "proton_flux_synthetic"
        return df
