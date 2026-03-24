"""
GOES satellite data downloader.

This module will be implemented by Ömer (omercangumus) in Task 3.
Downloads GOES proton flux data from NOAA SWPC API.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from datetime import datetime


@dataclass
class GOESConfig:
    """Configuration for GOES data retrieval."""
    api_url: str = "https://services.swpc.noaa.gov/json/goes/primary/"
    cache_enabled: bool = True
    timeout_seconds: int = 10


class GOESDownloader:
    """Downloads GOES satellite proton flux data."""
    
    def download(
        self,
        start_time: datetime,
        end_time: datetime,
        config: GOESConfig
    ) -> pd.DataFrame:
        """
        Download GOES proton flux data.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            config: Download configuration
            
        Returns:
            DataFrame with columns [timestamp, proton_flux]
            
        Raises:
            NetworkError: If download fails and no cache available
        """
        # TODO: Implement in Task 3
        raise NotImplementedError("To be implemented in Task 3")
