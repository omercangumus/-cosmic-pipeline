"""
Pipeline orchestrator - coordinates detection, filtering, and metrics.

This module will be implemented by Ahmet (Day 1).
Expected interface for dashboard integration:

def run_pipeline(df, methods=["classic", "ml"], ground_truth_mask=None):
    '''
    Run the complete pipeline on telemetry data.
    
    Args:
        df: DataFrame with columns [timestamp, value]
        methods: List of methods to use (e.g., ["classic", "ml"])
        ground_truth_mask: Optional dict with keys: seu, tid, gap, noise
        
    Returns:
        dict with structure:
        {
            "classic": {
                "cleaned_df": pd.DataFrame,
                "anomaly_mask": np.ndarray (bool),
                "metrics": {
                    "snr": float,
                    "rmse": float,
                    "precision": float,
                    "recall": float,
                    "f1": float
                }
            },
            "ml": {
                "cleaned_df": pd.DataFrame,
                "anomaly_mask": np.ndarray (bool),
                "metrics": {
                    "snr": float,
                    "rmse": float,
                    "precision": float,
                    "recall": float,
                    "f1": float
                }
            },
            "ensemble": {
                "anomaly_mask": np.ndarray (bool),
                "metrics": dict
            }
        }
    '''
    pass
