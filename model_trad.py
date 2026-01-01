"""
EMG Gesture Classification - Baseline Models
Clean implementation for GrabMyo and Ninapro datasets.

GrabMyo: 32 channels total, 28 used (16 forearm + 12 wrist), 2048 Hz, 16 gestures
Ninapro DB5: 16 channels (2x Myo armbands), 200 Hz, 52 movements + rest

Key differences between datasets:
- GrabMyo: Higher sample rate (2048 Hz), more channels (28), fewer classes (16)
  Window: 256 samples = 125ms, 512 samples = 250ms
- Ninapro: Lower sample rate (200 Hz), fewer channels (16), more classes (52)
  Window: 40 samples = 200ms, 50 samples = 250ms
"""

import numpy as np
import os
import argparse
import csv
import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import mode
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump

# ==================== GLOBALS ====================

SUPPORTED_DATASETS = ["grabmyo", "ninapro"]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# GrabMyo channel indices (0-indexed):
# Forearm: 0-7 (ring1), 8-15 (ring2) = 16 channels
# Wrist: 17-22 (ring3), 25-30 (ring4) = 12 channels  
# Unused: 16, 23, 24, 31
GRABMYO_FOREARM_CHANNELS = list(range(0, 16))  # 16 channels
GRABMYO_WRIST_CHANNELS = [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30]  # 12 channels
GRABMYO_ALL_CHANNELS = GRABMYO_FOREARM_CHANNELS + GRABMYO_WRIST_CHANNELS  # 28 channels
GRABMYO_FS = 2048  # Hz

# Ninapro: all 16 channels used
NINAPRO_CHANNELS = list(range(16))
NINAPRO_FS = 200  # Hz

# Dataset-specific defaults (window size in samples for ~200ms windows)
DATASET_DEFAULTS = {
    'grabmyo': {
        'fs': 2048,
        'channels': GRABMYO_ALL_CHANNELS,
        'window_size': 400,  # ~195ms at 2048 Hz
        'n_channels': 28,
        'n_classes': 16,
    },
    'ninapro': {
        'fs': 200,
        'channels': NINAPRO_CHANNELS,
        'window_size': 40,   # 200ms at 200 Hz
        'n_channels': 16,
        'n_classes': 52,     # Can vary by exercise
    }
}

def time_to_samples(time_ms: float, fs: int) -> int:
    """Convert time in milliseconds to number of samples."""
    return int(time_ms * fs / 1000)

def samples_to_time(samples: int, fs: int) -> float:
    """Convert number of samples to time in milliseconds."""
    return samples * 1000 / fs

# ==================== FEATURE FUNCTIONS ====================

def mean_absolute_value(x):
    """Mean Absolute Value - measures average muscle activation level."""
    return np.mean(np.abs(x), axis=1)

def root_mean_square(x):
    """RMS - related to constant force and non-fatiguing contraction."""
    return np.sqrt(np.mean(x**2, axis=1))

def waveform_length(x):
    """Waveform Length - cumulative length of the waveform."""
    return np.sum(np.abs(np.diff(x, axis=1)), axis=1)

def zero_crossings(x, threshold_ratio=0.01):
    """Zero Crossings - frequency information measure."""
    zc = np.zeros(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        s = x[i]
        thr = threshold_ratio * np.std(s)
        zc[i] = np.sum((s[:-1] * s[1:] < 0) & (np.abs(s[:-1] - s[1:]) > thr))
    return zc

def slope_sign_changes(x, threshold_ratio=0.01):
    """Slope Sign Changes - frequency information measure."""
    ssc = np.zeros(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        s = x[i]
        thr = threshold_ratio * np.std(s)
        diff = np.diff(s)
        ssc[i] = np.sum((diff[:-1] * diff[1:] < 0) & (np.abs(diff[:-1] - diff[1:]) > thr))
    return ssc

def variance(x):
    """Variance - power of the EMG signal."""
    return np.var(x, axis=1)

def integrated_emg(x):
    """Integrated EMG (IEMG) - sum of absolute values."""
    return np.sum(np.abs(x), axis=1)

def simple_square_integral(x):
    """Simple Square Integral (SSI) - energy of the signal."""
    return np.sum(x**2, axis=1)

def log_detector(x):
    """Log Detector - exponential of mean of log of absolute values.
    Good for detecting low-level muscle activity."""
    # Add small epsilon to avoid log(0)
    return np.exp(np.mean(np.log(np.abs(x) + 1e-10), axis=1))

def willison_amplitude(x, threshold_ratio=0.05):
    """Willison Amplitude (WAMP) - counts amplitude changes exceeding threshold.
    Related to firing of motor unit action potentials."""
    wamp = np.zeros(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        s = x[i]
        thr = threshold_ratio * np.max(np.abs(s))
        wamp[i] = np.sum(np.abs(np.diff(s)) > thr)
    return wamp

def myopulse_percentage_rate(x, threshold_ratio=0.1):
    """Myopulse Percentage Rate (MYOP) - percentage of samples above threshold.
    Indicates level of muscle activity."""
    myop = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        s = x[i]
        thr = threshold_ratio * np.max(np.abs(s))
        myop[i] = np.mean(np.abs(s) > thr)
    return myop

def mean_absolute_value_slope(x):
    """MAV Slope - difference between adjacent MAV segments.
    Captures changes in muscle contraction."""
    # Split into 4 segments and compute MAV for each
    n_samples = x.shape[1]
    seg_size = n_samples // 4
    if seg_size == 0:
        return np.zeros(x.shape[0])
    
    mavs = []
    for i in range(4):
        start = i * seg_size
        end = start + seg_size
        mavs.append(np.mean(np.abs(x[:, start:end]), axis=1))
    
    # Return slopes (3 values per channel, flattened)
    slopes = np.column_stack([mavs[i+1] - mavs[i] for i in range(3)])
    return slopes.flatten() if len(slopes.shape) > 1 else slopes

def hjorth_parameters(x):
    """Hjorth Parameters - Activity, Mobility, Complexity.
    Efficient time-domain descriptors of signal shape."""
    # Activity = variance
    activity = np.var(x, axis=1)
    
    # First derivative
    dx = np.diff(x, axis=1)
    activity_dx = np.var(dx, axis=1)
    
    # Second derivative  
    ddx = np.diff(dx, axis=1)
    activity_ddx = np.var(ddx, axis=1)
    
    # Mobility = sqrt(var(dx) / var(x))
    mobility = np.sqrt(activity_dx / (activity + 1e-10))
    
    # Complexity = mobility(dx) / mobility(x)
    mobility_dx = np.sqrt(activity_ddx / (activity_dx + 1e-10))
    complexity = mobility_dx / (mobility + 1e-10)
    
    return np.column_stack([activity, mobility, complexity])

# ==================== SPECTRAL FEATURES ====================

def spectral_features(x, fs):
    """Compute key spectral features: Mean Freq, Median Freq, Spectral Entropy.
    
    Args:
        x: array of shape (channels, samples)
        fs: sampling frequency in Hz
    
    Returns:
        features: array of shape (channels, 3) - [mean_freq, median_freq, spectral_entropy]
    """
    n_channels, n_samples = x.shape
    
    # Compute FFT for all channels at once
    fft_vals = np.fft.rfft(x, axis=1)
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    
    # Normalize PSD to get probability distribution
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_norm = psd / (psd_sum + 1e-10)
    
    # Mean Frequency: weighted average of frequencies
    mean_freq = np.sum(freqs * psd_norm, axis=1)
    
    # Median Frequency: frequency at which cumulative PSD reaches 50%
    cumsum_psd = np.cumsum(psd_norm, axis=1)
    median_freq = np.zeros(n_channels)
    for ch in range(n_channels):
        idx = np.searchsorted(cumsum_psd[ch], 0.5)
        median_freq[ch] = freqs[min(idx, len(freqs) - 1)]
    
    # Spectral Entropy: randomness of frequency distribution
    # Higher entropy = more uniform/noisy, lower = more concentrated/periodic
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
    # Normalize by max possible entropy (log2 of number of frequency bins)
    max_entropy = np.log2(psd.shape[1])
    spectral_entropy = spectral_entropy / max_entropy
    
    return np.column_stack([mean_freq, median_freq, spectral_entropy])

# Feature set definitions
FEATURE_SETS = {
    'basic': {
        'description': '6 Hudgins features (MAV, RMS, WL, ZC, SSC, VAR) - fastest',
        'features_per_channel': 6
    },
    'extended': {
        'description': 'Basic + Hjorth parameters (9 features) - balanced',
        'features_per_channel': 9
    },
    'full': {
        'description': 'All time-domain + spectral features (17 features) - most comprehensive',
        'features_per_channel': 17
    }
}

def extract_features(emg, window_size, overlap, fs, feature_set='basic'):
    """Extract features from EMG signal.
    
    Args:
        emg: array of shape (channels, samples)
        window_size: number of samples per window
        overlap: overlap ratio (0.0 to 1.0)
        fs: sampling frequency (Hz)
        feature_set: 'basic' (6), 'extended' (9), or 'full' (17) features per channel
    
    Returns:
        features: array of shape (n_windows, n_features)
    """
    step = int(window_size * (1 - overlap))
    n_samples = emg.shape[1]
    n_windows = max(0, (n_samples - window_size) // step + 1)
    
    if n_windows == 0:
        return np.array([])
    
    features = []
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = emg[:, start:end]
        
        # Basic features (6 per channel) - always included
        feat_parts = [
            mean_absolute_value(window),      # 1 per channel
            root_mean_square(window),         # 1 per channel
            waveform_length(window),          # 1 per channel
            zero_crossings(window),           # 1 per channel
            slope_sign_changes(window),       # 1 per channel
            variance(window),                 # 1 per channel
        ]
        
        if feature_set in ['extended', 'full']:
            # Add Hjorth parameters (3 per channel)
            feat_parts.append(hjorth_parameters(window).flatten())
        
        if feature_set == 'full':
            # Add extra time-domain features (5 per channel)
            feat_parts.extend([
                integrated_emg(window),
                simple_square_integral(window),
                log_detector(window),
                willison_amplitude(window),
                myopulse_percentage_rate(window),
            ])
            # Add spectral features (3 per channel)
            feat_parts.append(spectral_features(window, fs).flatten())
        
        feat = np.concatenate(feat_parts)
        features.append(feat)
    
    return np.array(features)

# ==================== NORMALIZATION ====================

def normalize_features(X_train, X_test):
    """Standard normalization fitted on training data."""
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm, scaler

# ==================== FEATURE CACHE ====================

# Cache version - increment when preprocessing/feature extraction changes
CACHE_VERSION = "v2"  # v2: added per-subject/trial z-score normalization

def make_cache_key(dataset, window_size, overlap, feature_set='basic', **kwargs):
    """Generate a unique cache key based on all relevant parameters.
    
    This ensures different parameter combinations don't share caches.
    """
    import hashlib
    
    # Build a string with all parameters that affect the data
    parts = [
        f"ver={CACHE_VERSION}",
        f"dataset={dataset}",
        f"ws={window_size}",
        f"ov={overlap}",
        f"feat={feature_set}",
    ]
    
    # Add dataset-specific parameters
    if dataset == 'grabmyo':
        sessions = sorted(kwargs.get('sessions', [1, 2, 3]))
        participants = sorted(kwargs.get('participants', list(range(1, 44))))
        channels = kwargs.get('channels', 'all')
        parts.append(f"sessions={sessions}")
        parts.append(f"participants={participants}")
        parts.append(f"channels={channels}")
    elif dataset == 'ninapro':
        subjects = sorted(kwargs.get('subjects', list(range(1, 11))))
        exercises = sorted(kwargs.get('exercises', [1, 2, 3]))
        include_rest = kwargs.get('include_rest', False)
        parts.append(f"subjects={subjects}")
        parts.append(f"exercises={exercises}")
        parts.append(f"include_rest={include_rest}")
    
    # Create a hash for the full config (to keep filenames short)
    config_str = "_".join(parts)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Human-readable prefix + hash
    return f"{dataset}_ws{window_size}_ov{overlap}_{feature_set}_{config_hash}"

def get_cache_paths(cache_dir, cache_key):
    """Generate cache file paths for a given cache key."""
    return {
        'X': os.path.join(cache_dir, f"{cache_key}_X.npy"),
        'y': os.path.join(cache_dir, f"{cache_key}_y.npy"),
        'groups': os.path.join(cache_dir, f"{cache_key}_groups.npy"),
        'meta': os.path.join(cache_dir, f"{cache_key}_meta.txt")
    }

def save_cache(cache_dir, cache_key, X, y, groups, meta_info=""):
    """Save extracted features to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = get_cache_paths(cache_dir, cache_key)
    
    np.save(paths['X'], X.astype(np.float32))
    np.save(paths['y'], y.astype(np.int16))
    np.save(paths['groups'], groups.astype(np.int16))
    
    # Save metadata for human reference
    with open(paths['meta'], 'w') as f:
        f.write(f"Cache key: {cache_key}\n")
        f.write(f"Created: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Classes: {len(np.unique(y))}\n")
        f.write(f"Groups: {len(np.unique(groups))}\n")
        f.write(f"\n{meta_info}\n")
    
    print(f"Saved cache: {cache_key}")
    print(f"  -> {paths['X']}")

def load_cache(cache_dir, cache_key):
    """Load cached features if available."""
    paths = get_cache_paths(cache_dir, cache_key)
    
    # Check all required files exist
    required = ['X', 'y', 'groups']
    if not all(os.path.exists(paths[k]) for k in required):
        return None
    
    print(f"Loading cached features: {cache_key}")
    X = np.load(paths['X'])
    y = np.load(paths['y'])
    groups = np.load(paths['groups'])
    
    # Print metadata if available
    if os.path.exists(paths['meta']):
        with open(paths['meta'], 'r') as f:
            print(f"  Cache info: {f.readline().strip()}")
    
    return X, y, groups

# ==================== DATA LOADERS ====================

def load_grabmyo(data_path, sessions, participants, window_size, overlap, channels='all', feature_set='basic'):
    """Load GrabMyo dataset.
    
    Args:
        data_path: path to GrabMyo data root (contains Session1, Session2, Session3)
        sessions: list of session numbers [1, 2, 3]
        participants: list of participant numbers [1, ..., 43]
        window_size: samples per window
        overlap: overlap ratio
        channels: 'all' (28), 'forearm' (16), or 'wrist' (12)
        feature_set: 'basic', 'extended', or 'full'
    """
    try:
        import wfdb
    except ImportError:
        raise RuntimeError("wfdb package required for GrabMyo. Install with: pip install wfdb")
    
    # Select channels
    if channels == 'forearm':
        ch_idx = GRABMYO_FOREARM_CHANNELS
    elif channels == 'wrist':
        ch_idx = GRABMYO_WRIST_CHANNELS
    else:
        ch_idx = GRABMYO_ALL_CHANNELS
    
    X_all, y_all, groups_all = [], [], []
    fs = 2048  # GrabMyo sampling rate
    
    for session in sessions:
        session_dir = os.path.join(data_path, f"Session{session}")
        if not os.path.isdir(session_dir):
            print(f"Warning: Session directory not found: {session_dir}")
            continue
            
        for participant in participants:
            # Try both naming conventions
            subj_dir = os.path.join(session_dir, f"session{session}_participant{participant}")
            if not os.path.isdir(subj_dir):
                subj_dir = os.path.join(session_dir, f"session{session}_subject{participant}")
            if not os.path.isdir(subj_dir):
                continue
            
            for gesture in range(1, 17):  # 16 gestures
                for trial in range(1, 8):  # 7 trials
                    # Try different file naming patterns
                    base_names = [
                        f"session{session}_participant{participant}_gesture{gesture}_trial{trial}",
                        f"session{session}_subject{participant}_gesture{gesture}_trial{trial}"
                    ]
                    
                    record = None
                    for base in base_names:
                        fpath = os.path.join(subj_dir, base)
                        if os.path.exists(fpath + ".hea"):
                            try:
                                record = wfdb.rdrecord(fpath)
                                break
                            except Exception:
                                continue
                    
                    if record is None:
                        continue
                    
                    # Get EMG data: (samples, channels) -> (channels, samples)
                    emg = record.p_signal.T[ch_idx]
                    
                    # Per-trial z-score normalization (reduces inter-subject variability)
                    emg_mean = emg.mean(axis=1, keepdims=True)
                    emg_std = emg.std(axis=1, keepdims=True)
                    emg_std = np.where(emg_std < 1e-8, 1.0, emg_std)
                    emg = (emg - emg_mean) / emg_std
                    
                    # Extract features
                    feats = extract_features(emg, window_size, overlap, fs, feature_set)
                    if len(feats) == 0:
                        continue
                    
                    X_all.extend(feats)
                    y_all.extend([gesture] * len(feats))
                    groups_all.extend([participant] * len(feats))
        
        print(f"Session {session}: loaded {len([g for g in groups_all if g in participants])} windows so far")
    
    return np.array(X_all), np.array(y_all), np.array(groups_all)

def load_ninapro(data_path, subjects, exercises, window_size, overlap, include_rest=False, feature_set='basic'):
    """Load Ninapro DB5 dataset.
    
    Args:
        data_path: path to Ninapro data root (contains s1, s2, ..., s10)
        subjects: list of subject numbers [1, ..., 10]
        exercises: list of exercise numbers [1, 2, 3]
        window_size: samples per window
        overlap: overlap ratio
        include_rest: whether to include rest class (label 0)
        feature_set: 'basic', 'extended', or 'full'
    
    Note: Labels are offset per exercise to avoid collision:
        E1: 1-12 (basic finger movements)
        E2: 13-29 (wrist/hand configurations) 
        E3: 30-52 (grasping/functional movements)
        Rest: 0 (if include_rest=True)
    """
    try:
        import scipy.io
    except ImportError:
        raise RuntimeError("scipy required for Ninapro. Install with: pip install scipy")
    
    # Label offsets per exercise (E1: 0, E2: 12, E3: 29)
    # This makes labels unique across exercises
    EXERCISE_LABEL_OFFSET = {1: 0, 2: 12, 3: 29}
    
    X_all, y_all, groups_all = [], [], []
    fs = 200  # Ninapro DB5 sampling rate
    
    for subject in subjects:
        subj_dir = os.path.join(data_path, f"s{subject}")
        if not os.path.isdir(subj_dir):
            continue
        
        print(f"Loading Ninapro subject {subject}...")
        
        for exercise in exercises:
            mat_file = os.path.join(subj_dir, f"S{subject}_E{exercise}_A1.mat")
            if not os.path.exists(mat_file):
                continue
            
            try:
                mat = scipy.io.loadmat(mat_file)
            except Exception as e:
                print(f"Warning: Could not load {mat_file}: {e}")
                continue
            
            # Check required fields
            if not all(k in mat for k in ['emg', 'restimulus', 'rerepetition']):
                continue
            
            emg = mat['emg']  # (samples, channels)
            labels = mat['restimulus'].flatten()
            reps = mat['rerepetition'].flatten()
            
            # Per-subject z-score normalization of raw EMG
            # This reduces inter-subject variability significantly
            emg_mean = emg.mean(axis=0, keepdims=True)
            emg_std = emg.std(axis=0, keepdims=True)
            emg_std = np.where(emg_std < 1e-8, 1.0, emg_std)
            emg = (emg - emg_mean) / emg_std
            
            # Get label offset for this exercise
            label_offset = EXERCISE_LABEL_OFFSET.get(exercise, 0)
            
            # Process each movement and repetition
            for label in np.unique(labels):
                if label == 0 and not include_rest:
                    continue
                
                # Apply offset to non-rest labels
                global_label = label + label_offset if label > 0 else 0
                
                label_mask = labels == label
                for rep in np.unique(reps[label_mask]):
                    seg_mask = label_mask & (reps == rep)
                    seg_emg = emg[seg_mask].T  # (channels, samples)
                    
                    if seg_emg.shape[1] < window_size:
                        continue
                    
                    feats = extract_features(seg_emg, window_size, overlap, fs, feature_set)
                    if len(feats) == 0:
                        continue
                    
                    X_all.extend(feats)
                    y_all.extend([global_label] * len(feats))  # Use offset label
                    groups_all.extend([subject] * len(feats))
    
    return np.array(X_all), np.array(y_all), np.array(groups_all)

# ==================== CLASSIFIERS ====================

def get_classifiers(selected=None):
    """Return dictionary of classifiers to evaluate.
    
    Args:
        selected: list of model names to include, or None for all
    """
    all_classifiers = {
        # LDA variants
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'LDA_eigen': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        # QDA - per-class covariance, can capture more complex boundaries
        'QDA': QuadraticDiscriminantAnalysis(reg_param=0.1),
        'QDA_reg': QuadraticDiscriminantAnalysis(reg_param=0.3),
        # Optimized for 52-class Ninapro
        'LogReg': LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1, C=0.5, penalty='l2'),
        'LogRegL1': LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1, C=0.1, penalty='l1'),
        # NaiveBayes variants with different smoothing
        'NaiveBayes': GaussianNB(),
        'NaiveBayes_smooth': GaussianNB(var_smoothing=1e-7),  # More smoothing
        'NaiveBayes_less': GaussianNB(var_smoothing=1e-11),   # Less smoothing
        # Multiple SVM configurations
        'LinearSVM': LinearSVC(max_iter=10000, dual='auto', C=0.1),
        'LinearSVM_C1': LinearSVC(max_iter=10000, dual='auto', C=1.0),
    }
    
    if selected is None:
        return all_classifiers
    return {k: v for k, v in all_classifiers.items() if k in selected}

def train_and_evaluate(X_train, y_train, X_test, y_test, selected_models=None, save_dir=None, prefix="", use_pca=None, select_k_best=None):
    """Train selected classifiers and return results.
    
    Args:
        use_pca: None for no PCA, or float (0-1) for variance ratio, or int for n_components
        select_k_best: None or int - number of best features to select using ANOVA F-test
    """
    
    # Normalize features
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)
    
    # Encode labels (needed for feature selection)
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Optional feature selection (before PCA if both used)
    selector = None
    if select_k_best is not None:
        k = min(select_k_best, X_train_norm.shape[1])
        selector = SelectKBest(f_classif, k=k)
        X_train_norm = selector.fit_transform(X_train_norm, y_train_enc)
        X_test_norm = selector.transform(X_test_norm)
        print(f"Feature selection: {k} best features (from {scaler.n_features_in_})")
    
    # Optional PCA dimensionality reduction
    pca = None
    if use_pca is not None:
        if isinstance(use_pca, float) and 0 < use_pca < 1:
            pca = PCA(n_components=use_pca, random_state=42)
        else:
            pca = PCA(n_components=int(use_pca), random_state=42)
        X_train_norm = pca.fit_transform(X_train_norm)
        X_test_norm = pca.transform(X_test_norm)
        print(f"PCA: {pca.n_components_} components, {pca.explained_variance_ratio_.sum():.1%} variance")
    
    classifiers = get_classifiers(selected_models)
    results = {}
    
    for name, clf in classifiers.items():
        print(f"Training {name}...", end=" ", flush=True)
        try:
            clf.fit(X_train_norm, y_train_enc)
            y_pred = clf.predict(X_test_norm)
            
            acc = accuracy_score(y_test_enc, y_pred)
            f1 = f1_score(y_test_enc, y_pred, average='macro')
            
            results[name] = {'accuracy': acc, 'f1': f1}
            print(f"acc={acc:.4f}, f1={f1:.4f}")
            
            # Save model if requested
            if save_dir:
                os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(save_dir, 'models', f"{prefix}_{name}_{ts}.joblib")
                dump({'model': clf, 'scaler': scaler, 'label_encoder': le, 'pca': pca, 'selector': selector}, model_path)
                results[name]['model_path'] = model_path
                
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'accuracy': float('nan'), 'f1': float('nan')}
    
    return results

# ==================== MAIN ====================

def main(args):
    # Print feature set info
    feat_info = FEATURE_SETS[args.features]
    n_channels = 28 if args.dataset == 'grabmyo' and args.channels == 'all' else \
                 16 if args.dataset == 'grabmyo' and args.channels == 'forearm' else \
                 12 if args.dataset == 'grabmyo' and args.channels == 'wrist' else 16
    n_features = n_channels * feat_info['features_per_channel']
    
    print(f"Dataset: {args.dataset}")
    print(f"Features: {args.features} ({feat_info['description']})")
    print(f"  -> {n_channels} channels × {feat_info['features_per_channel']} features = {n_features} total")
    print(f"Window size: {args.window_size}, Overlap: {args.overlap}")
    print(f"Split: {args.split}, Train ratio: {args.train_ratio}")
    print(f"Models: {args.models}")
    print("-" * 50)
    
    # Build cache key with all relevant parameters
    if args.dataset == 'grabmyo':
        cache_kwargs = {
            'sessions': args.sessions,
            'participants': args.participants,
            'channels': args.channels
        }
        meta_info = f"Feature set: {args.features}\nSessions: {args.sessions}\nParticipants: {args.participants}\nChannels: {args.channels}"
    else:
        cache_kwargs = {
            'subjects': args.subjects,
            'exercises': args.exercises,
            'include_rest': args.include_rest
        }
        meta_info = f"Feature set: {args.features}\nSubjects: {args.subjects}\nExercises: {args.exercises}\nInclude rest: {args.include_rest}"
    
    cache_key = make_cache_key(args.dataset, args.window_size, args.overlap, args.features, **cache_kwargs)
    print(f"Cache key: {cache_key}")
    
    # Try to load from cache (unless --no-cache specified)
    cached = None
    if not args.no_cache:
        cached = load_cache(args.cache_dir, cache_key)
    
    if cached is not None:
        X, y, groups = cached
    else:
        # Load and extract features
        if args.dataset == 'grabmyo':
            X, y, groups = load_grabmyo(
                args.data_path,
                args.sessions,
                args.participants,
                args.window_size,
                args.overlap,
                channels=args.channels,
                feature_set=args.features
            )
        else:
            X, y, groups = load_ninapro(
                args.data_path,
                args.subjects,
                args.exercises,
                args.window_size,
                args.overlap,
                include_rest=args.include_rest,
                feature_set=args.features
            )
        
        if len(X) == 0:
            raise RuntimeError("No samples loaded! Check data path and parameters.")
        
        # Save to cache
        save_cache(args.cache_dir, cache_key, X, y, groups, meta_info)
    
    print(f"Total samples: {len(X)}, Features per sample: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}, Subjects: {len(np.unique(groups))}")
    print("-" * 50)
    
    # Early exit if cache-only mode
    if args.cache_only:
        print("\n--cache-only mode: Features cached, skipping training.")
        print(f"Cache location: {args.cache_dir}/{cache_key}_*.npy")
        return
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    results_all = []
    
    if args.split == 'random':
        # Simple random train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.train_ratio, 
            random_state=RANDOM_SEED, stratify=y
        )
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        results = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            selected_models=args.models,
            save_dir=args.save_dir,
            prefix=f"{args.dataset}_{args.features}_random",
            use_pca=args.pca,
            select_k_best=args.select_k
        )
        
        for model, res in results.items():
            results_all.append(['random', model, res.get('accuracy', ''), 
                              res.get('f1', ''), res.get('model_path', '')])
    
    else:  # LOSO cross-validation
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"\n--- Fold {fold + 1} ---")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            results = train_and_evaluate(
                X_train, y_train, X_test, y_test,
                selected_models=args.models,
                save_dir=args.save_dir,
                prefix=f"{args.dataset}_{args.features}_fold{fold+1}",
                use_pca=args.pca,
                select_k_best=args.select_k
            )
            
            for model, res in results.items():
                results_all.append([f'fold_{fold+1}', model, res.get('accuracy', ''),
                                  res.get('f1', ''), res.get('model_path', '')])
    
    # Build descriptive run name: dataset_features_models_split_timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    models_str = "-".join(sorted(args.models))
    run_name = f"{args.dataset}_{args.features}_{models_str}_{args.split}_{ts}"

    # Save results with metadata header
    results_file = os.path.join(args.save_dir, f"results_{run_name}.csv")

    # Build metadata dictionary
    meta = {
        'dataset': args.dataset,
        'features': args.features,
        'window_size': args.window_size,
        'overlap': args.overlap,
        'split': args.split,
        'train_ratio': args.train_ratio,
        'models': ",".join(sorted(args.models)),
        'pca': args.pca,
        'select_k': args.select_k,
        'cache_key': cache_key,
    }

    # Add dataset-specific metadata
    if args.dataset == 'grabmyo':
        meta.update({'sessions': args.sessions, 'participants': args.participants, 'channels': args.channels})
    else:
        meta.update({'subjects': args.subjects, 'exercises': args.exercises, 'include_rest': args.include_rest})

    def _write_results_with_meta(path, meta_info, header, rows):
        # Write metadata lines (commented) followed by the CSV header and rows
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            # metadata
            for k, v in meta_info.items():
                f.write(f"# {k}={v}\n")
            f.write(f"# created={datetime.datetime.now().isoformat()}\n")
            # csv header and rows
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    _write_results_with_meta(results_file, meta, ['split', 'model', 'accuracy', 'f1', 'model_path'], results_all)

    print(f"\nResults saved to: {results_file}")

# ==================== CLI ====================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EMG Gesture Classification Baselines")
    
    # Dataset selection
    p.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="grabmyo",
                   help="Dataset to use")
    p.add_argument("--data-path", required=True,
                   help="Path to dataset root directory")
    
    # GrabMyo specific
    p.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 3],
                   help="GrabMyo sessions to use (1-3)")
    p.add_argument("--participants", nargs="+", type=int, default=list(range(1, 44)),
                   help="GrabMyo participants to use (1-43)")
    p.add_argument("--channels", choices=['all', 'forearm', 'wrist'], default='all',
                   help="GrabMyo channels: all (28), forearm (16), or wrist (12)")
    
    # Ninapro specific
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)),
                   help="Ninapro subjects to use (1-10)")
    p.add_argument("--exercises", nargs="+", type=int, default=[1, 2, 3],
                   help="Ninapro exercises to use (1-3)")
    p.add_argument("--include-rest", action="store_true",
                   help="Include rest class for Ninapro")
    
    # Feature extraction
    p.add_argument("--features", choices=list(FEATURE_SETS.keys()), default="basic",
                   help="Feature set: basic (6/ch, fastest), extended (9/ch), full (17/ch)")
    p.add_argument("--window-size", type=int, default=256,
                   help="Window size in samples (default: 256)")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Window overlap ratio (default: 0.5)")
    
    # Training
    p.add_argument("--split", choices=["random", "loso"], default="random",
                   help="Train/test split method")
    p.add_argument("--train-ratio", type=float, default=0.7,
                   help="Training set ratio for random split")
    p.add_argument("--models", nargs="+", 
                   choices=["LDA", "LDA_eigen", "QDA", "QDA_reg", "LogReg", "LogRegL1", 
                            "NaiveBayes", "NaiveBayes_smooth", "NaiveBayes_less", 
                            "LinearSVM", "LinearSVM_C1"],
                   default=["LDA", "LogReg", "NaiveBayes", "LinearSVM"],
                   help="Models to train")
    p.add_argument("--pca", type=float, default=None,
                   help="PCA variance ratio (0-1, e.g. 0.95) or n_components (>1)")
    p.add_argument("--select-k", type=int, default=None,
                   help="Select K best features using ANOVA F-test (e.g. 100, 150)")
    
    # Caching and output
    p.add_argument("--cache-dir", default="./feature_cache",
                   help="Directory for feature cache")
    p.add_argument("--save-dir", default="./results",
                   help="Directory for results and models")
    p.add_argument("--use-cache", action="store_true",
                   help="Use cached features if available")
    p.add_argument("--no-cache", action="store_true",
                   help="Don't use cache, force reload data")
    p.add_argument("--cache-only", action="store_true",
                   help="Only extract features and save to cache, skip training")
    
    args = p.parse_args()
    
    # cache-only implies use-cache (but we'll skip loading from cache to force extraction)
    if args.cache_only:
        args.use_cache = True  # Enable cache saving
        args.no_cache = True   # Force re-extraction (don't load existing)
    
    main(args)
