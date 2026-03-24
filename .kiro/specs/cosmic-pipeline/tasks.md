# Implementation Plan: Cosmic Pipeline

## Overview

This implementation plan breaks down the cosmic-pipeline satellite telemetry radiation-fault cleaning system into discrete coding tasks. The system uses hybrid DSP/ML methods for detecting and correcting radiation-induced faults (SEU, TID, gaps, noise) in satellite telemetry data.

Tasks are assigned to team members based on their expertise:
- **Ömer (omercangumus)**: Infrastructure, data layer, dashboard, DevOps
- **Ahmet (ahmetsn702)**: Pipeline core, ML models, detection/filtering algorithms

All code must include type hints, docstrings, and graceful error handling. No placeholder code - everything must work.

## Tasks

- [x] 1. Project setup and infrastructure (Ömer)
  - Initialize Git repository with main/develop branches
  - Create project directory structure (data/, pipeline/, models/, dashboard/, config/, utils/, tests/)
  - Create .gitignore (exclude __pycache__, .pytest_cache, *.pyc, .env, models/*.pth, data/cache/)
  - Create requirements.txt (numpy, pandas, scipy, scikit-learn, torch, streamlit, plotly, pyyaml, pytest, requests)
  - Create Makefile with targets: install, test, lint, run-dashboard
  - Create PR template with review checklist
  - Create README.md with project overview and quickstart instructions
  - _Requirements: 13.3, 14.1, 14.2, 14.3, 14.5_

- [x] 2. Implement synthetic telemetry generator (Ömer)
  - [x] 2.1 Create data/synthetic_generator.py with FaultConfig dataclass
    - Implement FaultConfig with seu_probability, tid_drift_rate, gap_probability, gap_size_range, noise_snr_db
    - _Requirements: 1.2, 1.3, 1.4, 1.5_
  
  - [x] 2.2 Implement SyntheticGenerator.generate() method
    - Generate clean sinusoidal signal with configurable duration and sampling_rate
    - Inject SEU bit-flips at random locations based on seu_probability
    - Inject TID drift as gradual baseline shift using exponential function
    - Inject data gaps at random intervals with sizes from gap_size_range
    - Add Gaussian noise based on noise_snr_db parameter
    - Return (timestamps, corrupted_signal, ground_truth_labels) as numpy arrays
    - Include type hints and docstrings
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 13.1, 13.2_
  
  - [ ]* 2.3 Write unit tests for synthetic generator
    - Test signal generation with various durations and sampling rates
    - Test fault injection probabilities match configuration
    - Test ground truth labels correctly identify fault locations
    - Test edge case: zero fault probability produces clean signal
    - _Requirements: 12.2_

- [x] 3. Implement GOES satellite data downloader (Ömer)
  - [x] 3.1 Create data/goes_downloader.py with GOESConfig dataclass
    - Implement GOESConfig with api_url, cache_enabled, timeout_seconds
    - _Requirements: 2.2_
  
  - [x] 3.2 Implement GOESDownloader.download() method
    - Make HTTP GET request to NOAA SWPC JSON API with timeout
    - Parse JSON response into pandas DataFrame with [timestamp, proton_flux] columns
    - Validate data completeness (no missing timestamps in range)
    - Implement file-based caching to avoid redundant API calls
    - Handle network errors gracefully - raise NetworkError with descriptive message
    - Include type hints and docstrings
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 13.1, 13.2, 15.2_
  
  - [ ]* 3.3 Write unit tests for GOES downloader
    - Test successful data retrieval with mock API responses
    - Test network failure handling (should raise NetworkError)
    - Test caching behavior (second call should use cache)
    - Test data validation (reject incomplete data)
    - _Requirements: 12.2, 12.6_

- [ ] 4. Implement DSP anomaly detectors (Ahmet)
  - [ ] 4.1 Create pipeline/detectors/dsp_detector.py with DSPMethod enum and DSPConfig dataclass
    - Implement DSPMethod enum with ZSCORE, IQR, ISOLATION_FOREST values
    - Implement DSPConfig with method, zscore_threshold, iqr_multiplier, iforest_contamination
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [ ] 4.2 Implement DSPDetector.detect() method
    - Implement Z-score detection: compute rolling mean/std, flag points exceeding threshold
    - Implement IQR detection: compute Q1/Q3, flag points outside [Q1-k*IQR, Q3+k*IQR]
    - Implement Isolation Forest detection using sklearn.ensemble.IsolationForest
    - Return (binary_labels, anomaly_scores) as numpy arrays
    - Ensure throughput of at least 1000 samples/second
    - Include type hints and docstrings
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 13.1, 13.2_
  
  - [ ]* 4.3 Write unit tests for DSP detectors
    - Test Z-score detection on synthetic signal with known outliers
    - Test IQR detection on synthetic signal with known outliers
    - Test Isolation Forest detection on synthetic signal
    - Test throughput requirement (1000 samples/second)
    - _Requirements: 12.2_

- [ ] 5. Implement LSTM Autoencoder model (Ahmet)
  - [ ] 5.1 Create models/lstm_autoencoder.py with LSTMConfig dataclass
    - Implement LSTMConfig with hidden_dim, num_layers, window_size, threshold_percentile, use_gpu
    - _Requirements: 4.1_
  
  - [ ] 5.2 Implement LSTMAutoencoder PyTorch module
    - Define encoder: nn.LSTM with input_dim -> hidden_dim
    - Define decoder: nn.LSTM with hidden_dim -> input_dim
    - Implement forward() method: encode then decode input sequences
    - Include type hints and docstrings
    - _Requirements: 4.1, 13.1, 13.2_
  
  - [ ] 5.3 Implement LSTMDetector class
    - Implement __init__() to initialize model and device (GPU if available, else CPU)
    - Implement train() method: train autoencoder on clean signals using MSE loss and Adam optimizer
    - Implement detect() method: compute reconstruction error, threshold at configured percentile
    - Implement save() and load() methods for model persistence
    - Handle GPU unavailability gracefully (fall back to CPU)
    - Include type hints and docstrings
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 13.1, 13.2, 15.4_
  
  - [ ]* 5.4 Write unit tests for LSTM detector
    - Test model training on synthetic clean signals
    - Test anomaly detection on signals with injected faults
    - Test model save/load functionality
    - Test GPU fallback to CPU when CUDA unavailable
    - _Requirements: 12.2_

- [ ] 6. Implement ensemble voting (Ahmet)
  - [ ] 6.1 Create pipeline/ensemble_voter.py with VotingConfig dataclass
    - Implement VotingConfig with min_agreement, weight_by_confidence
    - _Requirements: 5.2_
  
  - [ ] 6.2 Implement EnsembleVoter.vote() method
    - Align detector outputs to same length (handle different window sizes)
    - Compute majority vote: count detectors agreeing at each time point
    - Apply min_agreement threshold to produce binary labels
    - Compute confidence scores based on agreement level (fraction of detectors agreeing)
    - Optionally weight votes by detector confidence scores
    - Return (unified_labels, confidence_scores) as numpy arrays
    - Include type hints and docstrings
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 13.1, 13.2_
  
  - [ ]* 6.3 Write unit tests for ensemble voter
    - Test majority voting with 3 detectors (2 agree -> anomaly)
    - Test min_agreement threshold behavior
    - Test confidence score computation
    - Test alignment of different-length detector outputs
    - _Requirements: 12.2_

- [ ] 7. Implement classic signal filters (Ahmet)
  - [ ] 7.1 Create pipeline/filters/classic_filter.py with FilterMethod enum and FilterConfig dataclass
    - Implement FilterMethod enum with MEDIAN, SAVITZKY_GOLAY, WAVELET values
    - Implement FilterConfig with method, median_window, sg_window, sg_polyorder, wavelet_family, wavelet_level
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ] 7.2 Implement ClassicFilter.filter() method
    - Implement median filtering using scipy.signal.medfilt with configurable window
    - Implement Savitzky-Golay filtering using scipy.signal.savgol_filter
    - Implement wavelet denoising using pywt.wavedec and pywt.waverec with soft thresholding
    - Apply filtering only to regions marked by anomaly_mask
    - Preserve signal in non-anomalous regions
    - Interpolate data gaps using scipy.interpolate.interp1d (linear or cubic)
    - Return filtered signal as numpy array
    - Include type hints and docstrings
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 13.1, 13.2_
  
  - [ ]* 7.3 Write unit tests for classic filters
    - Test median filter on signal with spike noise
    - Test Savitzky-Golay filter on signal with smooth noise
    - Test wavelet denoising on signal with high-frequency noise
    - Test preservation of non-anomalous regions
    - Test gap interpolation
    - _Requirements: 12.3_

- [ ] 8. Implement ML-based signal reconstructor (Ahmet)
  - [ ] 8.1 Create pipeline/filters/ml_reconstructor.py with ReconstructorConfig dataclass
    - Implement ReconstructorConfig with confidence_threshold, blend_window, fallback_to_interpolation
    - _Requirements: 7.1_
  
  - [ ] 8.2 Implement MLReconstructor class
    - Implement __init__() to accept trained LSTM model and config
    - Implement reconstruct() method: use LSTM to predict clean values for anomalous segments
    - Implement boundary blending: smooth transition between original and reconstructed segments
    - Implement confidence estimation based on reconstruction error
    - Fall back to linear interpolation when confidence < confidence_threshold
    - Return (reconstructed_signal, confidence_scores) as numpy arrays
    - Include type hints and docstrings
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 13.1, 13.2_
  
  - [ ]* 8.3 Write unit tests for ML reconstructor
    - Test reconstruction of anomalous segments
    - Test boundary blending smoothness
    - Test fallback to interpolation when confidence is low
    - Test confidence score computation
    - _Requirements: 12.3_

- [ ] 9. Implement configuration management (Ahmet)
  - [ ] 9.1 Create config/config.py with all configuration dataclasses
    - Implement DataSourceConfig, PipelineConfig (aggregate all component configs)
    - _Requirements: 9.1, 9.3_
  
  - [ ] 9.2 Create config/parser.py with ConfigParser class
    - Implement parse() method: read YAML/JSON file, validate schema, construct PipelineConfig
    - Implement format() method: serialize PipelineConfig to YAML/JSON file
    - Handle invalid files gracefully with descriptive ConfigError messages
    - Include type hints and docstrings
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 13.1, 13.2, 15.1_
  
  - [ ]* 9.3 Write property-based tests for config parser
    - **Property 1: Round-trip consistency**
    - **Validates: Requirements 9.4**
    - Use hypothesis to generate random valid PipelineConfig objects
    - Test that parse(format(config)) produces equivalent config
    - _Requirements: 12.5_

- [ ] 10. Implement validation utilities (Ahmet)
  - [ ] 10.1 Create utils/validation.py with ValidationError exception and DataValidator class
    - Implement validate_signal() method: check data type, shape, NaN/inf values, minimum length
    - Implement timestamp validation: check monotonically increasing, consistent sampling rate
    - Raise ValidationError with descriptive messages on failure
    - Include type hints and docstrings
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 13.1, 13.2, 15.5_
  
  - [ ]* 10.2 Write unit tests for data validator
    - Test validation of valid signals (should pass)
    - Test rejection of signals with NaN values
    - Test rejection of signals with infinite values
    - Test rejection of signals below minimum length
    - Test rejection of non-monotonic timestamps
    - _Requirements: 12.2_

- [ ] 11. Implement metrics utilities (Ahmet)
  - [ ] 11.1 Create utils/metrics.py with MetricsCalculator class
    - Implement compute_snr() method: calculate 10*log10(signal_power/noise_power)
    - Implement compute_detection_metrics() method: calculate precision, recall, F1 score from predictions and ground truth
    - Return metrics as dictionary with descriptive keys
    - Include type hints and docstrings
    - _Requirements: 13.1, 13.2_
  
  - [ ]* 11.2 Write unit tests for metrics calculator
    - Test SNR computation with known signal and noise
    - Test detection metrics with known predictions and ground truth
    - Test edge case: perfect detection (precision=recall=F1=1.0)
    - Test edge case: no detections (handle division by zero)
    - _Requirements: 12.2_

- [ ] 12. Implement pipeline orchestrator (Ahmet)
  - [ ] 12.1 Create pipeline/pipeline.py with PipelineResult dataclass
    - Implement PipelineResult with cleaned_signal, anomaly_labels, confidence_scores, metrics, processing_time
    - _Requirements: 10.4_
  
  - [ ] 12.2 Implement Pipeline class
    - Implement __init__() to initialize all components from PipelineConfig
    - Implement _initialize_components() helper to instantiate detectors, filters, etc.
    - Implement process() method: orchestrate detection -> voting -> correction pipeline
    - Validate input data using DataValidator
    - Execute all DSP detectors in parallel (if possible)
    - Execute LSTM detector
    - Execute ensemble voting
    - Execute classic filters and ML reconstruction
    - Compute performance metrics (SNR improvement, detection metrics if ground truth available)
    - Log processing steps and timing information
    - Ensure 10000-sample signals complete within 5 seconds
    - Return PipelineResult
    - Include type hints and docstrings
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 13.1, 13.2_
  
  - [ ]* 12.3 Write integration tests for pipeline
    - Test end-to-end pipeline on synthetic data with known faults
    - Test pipeline with GOES data (mock API response)
    - Test performance requirement (10000 samples in <5 seconds)
    - Test graceful handling of missing model files
    - _Requirements: 12.4_

- [ ] 13. Checkpoint - Core pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [-] 14. Implement Streamlit dashboard (Ömer)
  - [x] 14.1 Create dashboard/app.py with Streamlit interface
    - Set page config with dark theme (theme.base="dark")
    - Create three tabs: "Synthetic Data", "GOES Data", "Comparison"
    - _Requirements: 8.1, 8.10_
  
  - [x] 14.2 Implement Synthetic Data tab
    - Add sliders for fault injection parameters (SEU probability, TID drift, gap probability, noise SNR)
    - Add button to generate synthetic signal
    - Display interactive Plotly time-series plot with original, corrupted, and cleaned signals
    - Highlight detected anomaly regions with distinct colors (e.g., red shading)
    - Display SNR improvement metric
    - Display precision, recall, F1 metrics (ground truth available)
    - Add CSV export button for cleaned signal
    - Add JSON export button for metrics
    - _Requirements: 8.2, 8.3, 8.5, 8.6, 8.7, 8.8, 8.9_
  
  - [x] 14.3 Implement GOES Data tab
    - Add date range picker for GOES data retrieval
    - Add button to download GOES data
    - Display interactive Plotly time-series plot with original and cleaned proton flux
    - Highlight detected anomaly regions
    - Display SNR improvement metric
    - Add CSV export button for cleaned data
    - Handle network errors gracefully (display error message, don't crash)
    - _Requirements: 8.2, 8.3, 8.5, 8.6, 8.8, 15.2_
  
  - [-] 14.4 Implement Comparison tab
    - Add side-by-side comparison of different detector methods
    - Display detection results from each detector (DSP methods, LSTM)
    - Display ensemble voting results
    - Add sliders for detection thresholds (Z-score, IQR, LSTM percentile)
    - Update plots dynamically when thresholds change
    - _Requirements: 8.4, 8.5_
  
  - [-] 14.5 Implement session state initialization
    - Initialize all session state variables with default values on first load
    - Handle missing session state gracefully (no crashes)
    - _Requirements: 8.11_
  
  - [ ]* 14.6 Write UI tests for dashboard
    - Test dashboard loads without errors
    - Test synthetic data generation and display
    - Test GOES data download and display (mock API)
    - Test export functionality
    - _Requirements: 12.2_

- [ ] 15. Implement logging utilities (Ahmet)
  - [ ] 15.1 Create utils/logging.py with structured logging setup
    - Configure Python logging with INFO level
    - Format log messages with timestamp, level, module, message
    - Log to both console and file (logs/pipeline.log)
    - Include type hints and docstrings
    - _Requirements: 10.5, 15.1_

- [ ] 16. Implement model training script (Ahmet)
  - [ ] 16.1 Create models/train.py with training script
    - Load synthetic clean signals for training
    - Initialize LSTMAutoencoder with config from command-line arguments
    - Train model for configurable number of epochs
    - Save trained model to models/lstm_ae.pth
    - Log training progress (loss per epoch)
    - Include command-line interface using argparse
    - Include type hints and docstrings
    - _Requirements: 4.2, 13.1, 13.2_
  
  - [ ]* 16.2 Write unit tests for training script
    - Test training completes without errors
    - Test model file is saved correctly
    - Test trained model can be loaded and used for inference
    - _Requirements: 12.2_

- [x] 17. Create example configuration files (Ömer)
  - [x] 17.1 Create config/default.yaml with default pipeline configuration
    - Include all component configurations with reasonable defaults
    - Add comments explaining each parameter
    - _Requirements: 9.5, 13.3_
  
  - [x] 17.2 Create config/fast.yaml with fast processing configuration
    - Disable LSTM detector for speed
    - Use only Z-score and median filter
    - _Requirements: 9.5_
  
  - [x] 17.3 Create config/accurate.yaml with high-accuracy configuration
    - Enable all detectors
    - Use ensemble voting with min_agreement=2
    - Use both classic filters and ML reconstruction
    - _Requirements: 9.5_

- [ ] 18. Update documentation (Both)
  - [ ] 18.1 Update README.md with complete usage guide (Ömer)
    - Add installation instructions (pip install -r requirements.txt)
    - Add quickstart example (train model, run dashboard)
    - Add configuration guide
    - Add API documentation links
    - _Requirements: 13.3_
  
  - [ ] 18.2 Add inline comments to complex algorithms (Ahmet)
    - Add comments explaining wavelet denoising logic
    - Add comments explaining LSTM reconstruction logic
    - Add comments explaining ensemble voting logic
    - _Requirements: 13.4_
  
  - [ ] 18.3 Generate API documentation (Both)
    - Use pydoc or sphinx to generate HTML documentation from docstrings
    - _Requirements: 13.5_

- [ ] 19. Final integration and testing (Both)
  - [ ] 19.1 Run complete test suite
    - Execute pytest with coverage report
    - Ensure at least 80% code coverage
    - Ensure all tests complete within 60 seconds
    - _Requirements: 12.1, 12.7_
  
  - [ ] 19.2 Test end-to-end workflow
    - Train LSTM model using models/train.py
    - Run dashboard with streamlit run dashboard/app.py
    - Test synthetic data generation and cleaning
    - Test GOES data download and cleaning (if network available)
    - Test configuration file loading
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [ ] 19.3 Fix any remaining issues
    - Address test failures
    - Fix linting errors
    - Improve error messages
    - _Requirements: 15.1, 15.6_

- [ ] 20. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Ömer focuses on infrastructure, data layer, and dashboard (tasks 1, 2, 3, 14, 17, 18.1)
- Ahmet focuses on pipeline core, ML models, and algorithms (tasks 4-13, 15, 16, 18.2)
- Both team members collaborate on final integration and testing (tasks 18.3, 19)
- All code must include type hints and docstrings (no exceptions)
- All paths must be relative to project root
- Graceful error handling is mandatory (network failures, missing models, invalid inputs)
- Dark theme is required for dashboard UI
- No placeholder code - everything must work immediately
