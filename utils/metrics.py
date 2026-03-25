"""Metrics calculation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² Score."""
    return r2_score(y_true, y_pred)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    
    Args:
        signal: Clean signal
        noise: Noise component
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_all_metrics(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    ground_truth: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Calculate all available metrics.
    
    Args:
        original: Original corrupted data
        cleaned: Cleaned data
        ground_truth: Optional ground truth clean data
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Remove NaN for comparison
    mask = ~(original['value'].isna() | cleaned['value'].isna())
    orig_clean = original.loc[mask, 'value'].values
    cleaned_clean = cleaned.loc[mask, 'value'].values
    
    if len(orig_clean) > 0:
        metrics['rmse'] = calculate_rmse(orig_clean, cleaned_clean)
        metrics['mae'] = calculate_mae(orig_clean, cleaned_clean)
        
        try:
            metrics['r2_score'] = calculate_r2(orig_clean, cleaned_clean)
        except:
            metrics['r2_score'] = 0.0
        
        # SNR (noise = difference)
        noise = orig_clean - cleaned_clean
        if len(cleaned_clean) > 0:
            metrics['snr'] = calculate_snr(cleaned_clean, noise)
        else:
            metrics['snr'] = 0.0
    
    # If ground truth available
    if ground_truth is not None:
        mask_gt = ~(ground_truth['value'].isna() | cleaned['value'].isna())
        gt_clean = ground_truth.loc[mask_gt, 'value'].values
        cleaned_gt = cleaned.loc[mask_gt, 'value'].values
        
        if len(gt_clean) > 0:
            metrics['rmse_vs_truth'] = calculate_rmse(gt_clean, cleaned_gt)
            metrics['mae_vs_truth'] = calculate_mae(gt_clean, cleaned_gt)
            metrics['r2_vs_truth'] = calculate_r2(gt_clean, cleaned_gt)
    
    return metrics
