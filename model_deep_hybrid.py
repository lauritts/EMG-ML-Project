#!/usr/bin/env python3
"""
Clean Hybrid CNN-RNN for EMG Gesture Classification
====================================================
Operates on RAW EMG windows (not precomputed features).

Architecture (per proposal section 2.4.1):
  1. 1D CNN layers: learnable feature extractors along time dimension
  2. RNN layers (GRU/LSTM): model long-range temporal dependencies  
  3. Classifier head: fully-connected layers for final classification

Input shape: (batch, channels, time_samples)
  - GrabMyo: (batch, 28, window_size) at 2048 Hz
  - Ninapro: (batch, 16, window_size) at 200 Hz

Author: Clean implementation
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# Try to import wfdb for raw data loading
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    print("Warning: wfdb not installed. Raw data loading may fail.")

# ==============================================================================
# CONSTANTS
# ==============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GrabMyo: 28 usable channels out of 32
# Forearm (F1-F16): indices 0-15
# Wrist group 1 (W1-W6): indices 17-22  
# Wrist group 2 (W7-W12): indices 25-30
# Unused (U1-U4): indices 16, 23, 24, 31
GRABMYO_CHANNELS = list(range(0, 16)) + list(range(17, 23)) + list(range(25, 31))  # 28 channels
GRABMYO_SAMPLE_RATE = 2048  # Hz
GRABMYO_GESTURES = 17  # 0=rest + 16 gestures (1-16)
GRABMYO_TRIALS = 7
GRABMYO_DURATION = 5  # seconds per trial

# Ninapro DB5: 16 channels (2 Myo armbands × 8 channels each)
NINAPRO_CHANNELS = list(range(0, 16))  # 16 channels
NINAPRO_SAMPLE_RATE = 200  # Hz


# ==============================================================================
# RAW DATA LOADERS
# ==============================================================================

class RawGrabMyoLoader:
    """Load raw EMG windows from GrabMyo dataset."""
    
    def __init__(self, data_path: str, channels: List[int] = None):
        self.data_path = Path(data_path)
        self.channels = channels if channels is not None else GRABMYO_CHANNELS
        
    def load_recording(self, session: int, participant: int, gesture: int, trial: int) -> Optional[np.ndarray]:
        """Load a single recording, return shape (channels, time_samples)."""
        session_dir = self.data_path / f"Session{session}"
        if not session_dir.exists():
            return None
        
        # Actual structure: Session{s}/session{s}_participant{p}/session{s}_participant{p}_gesture{g}_trial{t}.dat
        participant_dir = session_dir / f"session{session}_participant{participant}"
        record_name = f"session{session}_participant{participant}_gesture{gesture}_trial{trial}"
        record_path = participant_dir / record_name
        
        if not (participant_dir / f"{record_name}.dat").exists():
            return None
        
        try:
            if HAS_WFDB:
                record = wfdb.rdrecord(str(record_path))
                data = record.p_signal.T  # (channels, samples)
                # Select only the channels we want
                data = data[self.channels, :]
                return data.astype(np.float32)
        except Exception as e:
            return None
        return None

    def load_windows(self, sessions: List[int], participants: List[int], 
                     window_size: int, overlap: float,
                     gestures: List[int] = None, trials: List[int] = None,
                     include_rest: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw EMG data and segment into windows.
        
        Returns:
            windows: (n_windows, n_channels, window_size)
            labels: (n_windows,) gesture labels
            groups: (n_windows,) participant IDs
        """
        if gestures is None:
            gestures = list(range(0 if include_rest else 1, GRABMYO_GESTURES))
        if trials is None:
            trials = list(range(1, GRABMYO_TRIALS + 1))
            
        all_windows = []
        all_labels = []
        all_groups = []
        
        step = int(window_size * (1 - overlap))
        if step <= 0:
            step = 1
            
        for session in sessions:
            for participant in participants:
                for gesture in gestures:
                    for trial in trials:
                        data = self.load_recording(session, participant, gesture, trial)
                        if data is None:
                            continue
                        
                        # Segment into windows
                        n_samples = data.shape[1]
                        n_windows = (n_samples - window_size) // step + 1
                        
                        for i in range(n_windows):
                            start = i * step
                            end = start + window_size
                            window = data[:, start:end]
                            
                            if window.shape[1] == window_size:
                                all_windows.append(window)
                                all_labels.append(gesture)
                                all_groups.append(participant)
        
        if len(all_windows) == 0:
            return np.zeros((0, len(self.channels), window_size)), np.array([]), np.array([])
        
        windows = np.stack(all_windows, axis=0).astype(np.float32)
        labels = np.array(all_labels, dtype=np.int64)
        groups = np.array(all_groups, dtype=np.int64)
        
        return windows, labels, groups


class RawNinaproLoader:
    """Load raw EMG windows from Ninapro DB5 dataset.
    
    Structure: data_path/s{subject}/S{subject}_E{exercise}_A1.mat
    Each .mat contains continuous EMG with per-sample labels (restimulus).
    """
    
    def __init__(self, data_path: str, channels: List[int] = None):
        self.data_path = Path(data_path)
        self.channels = channels if channels is not None else NINAPRO_CHANNELS
        
    def load_subject_exercise(self, subject: int, exercise: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load data for one subject and exercise.
        
        Returns:
            emg: (samples, channels) raw EMG
            labels: (samples,) per-sample movement labels (restimulus)
            reps: (samples,) per-sample repetition numbers (rerepetition)
        """
        subject_dir = self.data_path / f"s{subject}"
        mat_file = subject_dir / f"S{subject}_E{exercise}_A1.mat"
        
        if not mat_file.exists():
            return None
            
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(mat_file))
            
            # Use restimulus (refined labels) and rerepetition
            if not all(k in mat for k in ['emg', 'restimulus', 'rerepetition']):
                print(f"Warning: Missing required fields in {mat_file}")
                return None
            
            emg = mat['emg'][:, self.channels]  # (samples, channels)
            labels = mat['restimulus'].flatten()
            reps = mat['rerepetition'].flatten()
            
            return emg.astype(np.float32), labels.astype(np.int64), reps.astype(np.int64)
            
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            return None
    
    def load_windows(self, subjects: List[int], window_size: int, overlap: float,
                     exercises: List[int] = None,
                     include_rest: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw EMG data and segment into windows.
        
        Windows are extracted from continuous segments where all samples have
        the same movement label (from restimulus). This ensures clean labels.
        
        Labels are offset per exercise to avoid collision:
            E1: 1-12 (basic finger movements)
            E2: 13-29 (wrist/hand configurations) 
            E3: 30-52 (grasping/functional movements)
            Rest: 0 (if include_rest=True)
        
        Args:
            subjects: List of subject numbers [1-10]
            window_size: Window size in samples (e.g., 40 for 200ms at 200Hz)
            overlap: Overlap ratio (0.0 to 1.0)
            exercises: List of exercises [1, 2, 3] or None for all
            include_rest: Whether to include rest class (label 0)
        
        Returns:
            windows: (n_windows, n_channels, window_size)
            labels: (n_windows,) gesture labels  
            groups: (n_windows,) subject IDs
        """
        if exercises is None:
            exercises = [1, 2, 3]
        
        # Label offsets per exercise (E1: 0, E2: 12, E3: 29)
        EXERCISE_LABEL_OFFSET = {1: 0, 2: 12, 3: 29}
            
        all_windows = []
        all_labels = []
        all_groups = []
        
        step = int(window_size * (1 - overlap))
        if step <= 0:
            step = 1
            
        for subject in subjects:
            for exercise in exercises:
                result = self.load_subject_exercise(subject, exercise)
                if result is None:
                    continue
                
                emg, labels, reps = result
                emg = emg.T  # (channels, samples)
                
                # Get label offset for this exercise
                label_offset = EXERCISE_LABEL_OFFSET.get(exercise, 0)
                
                # Process each unique (movement, repetition) segment
                for label in np.unique(labels):
                    if label == 0 and not include_rest:
                        continue
                    
                    # Apply offset to non-rest labels
                    global_label = label + label_offset if label > 0 else 0
                    
                    label_mask = labels == label
                    for rep in np.unique(reps[label_mask]):
                        if rep == 0:  # Skip non-movement periods
                            continue
                        seg_mask = label_mask & (reps == rep)
                        seg_indices = np.where(seg_mask)[0]
                        
                        if len(seg_indices) < window_size:
                            continue
                        
                        # Get contiguous segment
                        seg_start = seg_indices[0]
                        seg_end = seg_indices[-1] + 1
                        seg_emg = emg[:, seg_start:seg_end]
                        
                        # Extract windows from this segment
                        n_samples = seg_emg.shape[1]
                        for i in range(0, n_samples - window_size + 1, step):
                            window = seg_emg[:, i:i + window_size]
                            all_windows.append(window)
                            all_labels.append(global_label)  # Use offset label
                            all_groups.append(subject)
            
            print(f"  Subject {subject}: {len([g for g in all_groups if g == subject])} windows")
        
        if len(all_windows) == 0:
            return np.zeros((0, len(self.channels), window_size)), np.array([]), np.array([])
        
        windows = np.stack(all_windows, axis=0).astype(np.float32)
        labels = np.array(all_labels, dtype=np.int64)
        groups = np.array(all_groups, dtype=np.int64)
        
        return windows, labels, groups


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def per_subject_normalize(windows: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Z-score normalize per subject (group) to reduce inter-subject variability.
    
    Args:
        windows: (n_windows, n_channels, window_size)
        groups: (n_windows,) subject/participant IDs
        
    Returns:
        Normalized windows with same shape
    """
    normalized = windows.copy()
    
    for g in np.unique(groups):
        mask = groups == g
        subject_data = windows[mask]  # (n_subject_windows, channels, time)
        
        # Compute mean and std across all windows and time for this subject
        # Shape: (channels,)
        mean = subject_data.mean(axis=(0, 2), keepdims=True)
        std = subject_data.std(axis=(0, 2), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
        
        normalized[mask] = (subject_data - mean) / std
    
    return normalized


def bandpass_filter(windows: np.ndarray, lowcut: float = 20.0, highcut: float = 450.0,
                    fs: int = 2048, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to EMG windows."""
    try:
        from scipy.signal import butter, filtfilt
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)  # Ensure below Nyquist
        
        b, a = butter(order, [low, high], btype='band')
        
        # Filter each window and channel
        filtered = np.zeros_like(windows)
        for i in range(windows.shape[0]):
            for c in range(windows.shape[1]):
                filtered[i, c, :] = filtfilt(b, a, windows[i, c, :])
        
        return filtered
    except ImportError:
        print("Warning: scipy not available for filtering")
        return windows


# ==============================================================================
# PYTORCH DATASET
# ==============================================================================

class EMGWindowDataset(Dataset):
    """PyTorch Dataset for raw EMG windows."""
    
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        Args:
            windows: (n_samples, n_channels, window_size)
            labels: (n_samples,)
        """
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class HybridCNNRNN(nn.Module):
    """
    Hybrid CNN-RNN for EMG gesture classification.
    
    Architecture:
        1. 1D CNN layers: Extract local temporal features
        2. RNN layers (GRU/LSTM): Capture long-range dependencies
        3. Optional attention: Weight temporal positions
        4. Classifier: Dense layers for classification
    
    Input: (batch, channels, time_samples)
    Output: (batch, num_classes)
    """
    
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 # CNN params
                 cnn_channels: List[int] = [64, 128, 256],
                 cnn_kernels: List[int] = [7, 5, 3],
                 cnn_dropout: float = 0.2,
                 # RNN params
                 rnn_type: str = 'gru',
                 rnn_hidden: int = 128,
                 rnn_layers: int = 2,
                 rnn_dropout: float = 0.3,
                 bidirectional: bool = True,
                 # Attention
                 use_attention: bool = True,
                 # Classifier
                 fc_hidden: int = 128,
                 fc_dropout: float = 0.5):
        super().__init__()
        
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.rnn_hidden = rnn_hidden
        
        # =====================
        # 1D CNN Feature Extractor
        # =====================
        cnn_layers = []
        prev_channels = in_channels
        
        for i, (out_ch, kernel) in enumerate(zip(cnn_channels, cnn_kernels)):
            cnn_layers.extend([
                nn.Conv1d(prev_channels, out_ch, kernel_size=kernel, padding=kernel//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(cnn_dropout),
            ])
            prev_channels = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_channels[-1]
        
        # =====================
        # RNN Sequence Modeler
        # =====================
        RNN = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM
        self.rnn = RNN(
            input_size=self.cnn_out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        rnn_out_size = rnn_hidden * (2 if bidirectional else 1)
        
        # =====================
        # Attention (optional)
        # =====================
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(rnn_out_size, rnn_out_size // 2),
                nn.Tanh(),
                nn.Linear(rnn_out_size // 2, 1, bias=False)
            )
        
        # =====================
        # Classifier Head
        # =====================
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_samples)
        Returns:
            logits: (batch, num_classes)
        """
        # CNN: (batch, channels, time) -> (batch, cnn_channels, time//8)
        z = self.cnn(x)
        
        # Reshape for RNN: (batch, time', features)
        z = z.permute(0, 2, 1)
        
        # RNN: (batch, time', rnn_out_size)
        rnn_out, _ = self.rnn(z)
        
        # Aggregate temporal dimension
        if self.use_attention:
            # Attention: compute weighted sum
            attn_scores = self.attention(rnn_out)  # (batch, time', 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (attn_weights * rnn_out).sum(dim=1)  # (batch, rnn_out_size)
        else:
            # Mean pooling
            context = rnn_out.mean(dim=1)
        
        # Classify
        logits = self.classifier(context)
        
        return logits


# ==============================================================================
# TRAINING
# ==============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device,
                clip_grad: float = 1.0) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return loss, acc, f1, all_preds, all_labels


def encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Encode labels to contiguous integers starting from 0."""
    unique = np.unique(labels)
    label_map = {orig: enc for enc, orig in enumerate(unique)}
    encoded = np.array([label_map[l] for l in labels], dtype=np.int64)
    return encoded, label_map


# ==============================================================================
# CACHING
# ==============================================================================

def get_cache_key(dataset: str, window_size: int, overlap: float, 
                  sessions: List[int] = None, participants: List[int] = None,
                  subjects: List[int] = None, exercises: List[int] = None,
                  include_rest: bool = False) -> str:
    """Generate unique cache key based on parameters."""
    params = {
        'dataset': dataset,
        'window_size': window_size,
        'overlap': overlap,
        'include_rest': include_rest,
    }
    if sessions:
        params['sessions'] = sorted(sessions)
    if participants:
        params['participants'] = sorted(participants)
    if subjects:
        params['subjects'] = sorted(subjects)
    if exercises:
        params['exercises'] = sorted(exercises)
    
    param_str = json.dumps(params, sort_keys=True)
    hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    return f"{dataset}_raw_ws{window_size}_ov{overlap}_{hash_suffix}"


def load_cache(cache_dir: str, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load cached windows, labels, groups."""
    cache_path = Path(cache_dir)
    X_path = cache_path / f"{cache_key}_X.npy"
    y_path = cache_path / f"{cache_key}_y.npy"
    g_path = cache_path / f"{cache_key}_g.npy"
    
    if X_path.exists() and y_path.exists() and g_path.exists():
        try:
            X = np.load(X_path)
            y = np.load(y_path)
            g = np.load(g_path)
            print(f"Loaded cache: {cache_key}")
            print(f"  Windows: {X.shape}, Labels: {y.shape}, Groups: {g.shape}")
            return X, y, g
        except Exception as e:
            print(f"Cache load failed: {e}")
    return None


def save_cache(cache_dir: str, cache_key: str, X: np.ndarray, y: np.ndarray, g: np.ndarray):
    """Save windows, labels, groups to cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    np.save(cache_path / f"{cache_key}_X.npy", X)
    np.save(cache_path / f"{cache_key}_y.npy", y)
    np.save(cache_path / f"{cache_key}_g.npy", g)
    
    # Save metadata
    meta = {
        'cache_key': cache_key,
        'X_shape': list(X.shape),
        'y_shape': list(y.shape),
        'g_shape': list(g.shape),
        'created': datetime.now().isoformat()
    }
    with open(cache_path / f"{cache_key}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved cache: {cache_key}")


# ==============================================================================
# MAIN
# ==============================================================================

def main(args):
    print("=" * 60)
    print("Hybrid CNN-RNN for EMG Gesture Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Window size: {args.window_size}, Overlap: {args.overlap}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # =====================
    # Load Data
    # =====================
    cache_key = None
    cached = None
    
    if args.use_cache:
        if args.dataset == 'grabmyo':
            cache_key = get_cache_key('grabmyo', args.window_size, args.overlap,
                                       sessions=args.sessions, participants=args.participants,
                                       include_rest=args.include_rest)
        else:
            cache_key = get_cache_key('ninapro', args.window_size, args.overlap,
                                       subjects=args.subjects, exercises=args.exercises,
                                       include_rest=args.include_rest)
        cached = load_cache(args.cache_dir, cache_key)
    
    if cached is not None:
        windows, labels, groups = cached
    else:
        print("\nLoading raw EMG data...")
        
        if args.dataset == 'grabmyo':
            loader = RawGrabMyoLoader(args.data_path, channels=GRABMYO_CHANNELS)
            windows, labels, groups = loader.load_windows(
                sessions=args.sessions,
                participants=args.participants,
                window_size=args.window_size,
                overlap=args.overlap,
                include_rest=args.include_rest
            )
            sample_rate = GRABMYO_SAMPLE_RATE
        else:  # ninapro
            loader = RawNinaproLoader(args.data_path, channels=NINAPRO_CHANNELS)
            windows, labels, groups = loader.load_windows(
                subjects=args.subjects,
                window_size=args.window_size,
                overlap=args.overlap,
                exercises=args.exercises,
                include_rest=args.include_rest
            )
            sample_rate = NINAPRO_SAMPLE_RATE
        
        if len(windows) == 0:
            print("ERROR: No data loaded! Check paths and parameters.")
            sys.exit(1)
        
        print(f"Loaded {len(windows)} windows")
        print(f"  Shape: {windows.shape}")
        print(f"  Labels: {len(np.unique(labels))} classes")
        print(f"  Groups: {len(np.unique(groups))} participants")
        
        # Optional: bandpass filter
        if args.bandpass:
            print("Applying bandpass filter...")
            windows = bandpass_filter(windows, fs=sample_rate)
        
        # Save cache
        if args.use_cache and cache_key:
            save_cache(args.cache_dir, cache_key, windows, labels, groups)
    
    # Early exit if cache-only mode
    if args.cache_only:
        print("\n--cache-only mode: Data cached, skipping training.")
        print(f"Cache location: {args.cache_dir}/{cache_key}_*.npy")
        return
    
    # =====================
    # Preprocessing
    # =====================
    print("\nPreprocessing...")
    
    # Per-subject normalization
    windows = per_subject_normalize(windows, groups)
    
    # Encode labels
    labels_enc, label_map = encode_labels(labels)
    num_classes = len(label_map)
    print(f"Label map: {label_map}")
    print(f"Number of classes: {num_classes}")
    
    # =====================
    # Train/Test Split
    # =====================
    print(f"\nSplitting data (mode: {args.split})...")
    
    if args.split == 'loso':
        # Leave-One-Subject-Out: handled in training loop
        unique_groups = np.unique(groups)
        print(f"LOSO with {len(unique_groups)} folds")
        
        all_results = []
        all_preds = []
        all_trues = []
        
        for fold_idx, test_group in enumerate(unique_groups):
            print(f"\n{'='*40}")
            print(f"LOSO Fold {fold_idx + 1}/{len(unique_groups)} - Test participant: {test_group}")
            print('='*40)
            
            train_mask = groups != test_group
            test_mask = groups == test_group
            
            X_train, y_train = windows[train_mask], labels_enc[train_mask]
            X_test, y_test = windows[test_mask], labels_enc[test_mask]
            
            if len(X_test) == 0 or len(X_train) == 0:
                print("Skipping fold (empty split)")
                continue
            
            # Train
            fold_result = train_model(X_train, y_train, X_test, y_test, 
                                      num_classes, args, fold=fold_idx+1)
            all_results.append(fold_result)
            all_preds.extend(fold_result['preds'])
            all_trues.extend(fold_result['trues'])
        
        # Aggregate LOSO results
        f1_scores = [r['test_f1'] for r in all_results]
        acc_scores = [r['test_acc'] for r in all_results]
        
        print(f"\n{'='*60}")
        print("LOSO Results Summary")
        print('='*60)
        print(f"Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Mean Acc: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
        
        # Save results
        results = {
            'mode': 'loso',
            'folds': all_results,
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'mean_acc': float(np.mean(acc_scores)),
            'std_acc': float(np.std(acc_scores)),
        }
        
    else:
        # Random or group-based split
        if args.split == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                windows, labels_enc, test_size=1-args.train_ratio, 
                random_state=args.seed, stratify=labels_enc
            )
        else:  # group
            gss = GroupShuffleSplit(n_splits=1, test_size=1-args.train_ratio, random_state=args.seed)
            train_idx, test_idx = next(gss.split(windows, labels_enc, groups))
            X_train, X_test = windows[train_idx], windows[test_idx]
            y_train, y_test = labels_enc[train_idx], labels_enc[test_idx]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train
        result = train_model(X_train, y_train, X_test, y_test, num_classes, args)
        results = {
            'mode': args.split,
            'train_ratio': args.train_ratio,
            **result
        }
    
    # =====================
    # Save Results
    # =====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Descriptive filename
    split_str = f"loso" if args.split == 'loso' else f"{args.split}_{int(args.train_ratio*100)}-{int((1-args.train_ratio)*100)}"
    result_name = f"results_{args.dataset}_hybrid_{split_str}_{timestamp}"
    
    result_path = Path(args.save_dir) / f"{result_name}.json"
    with open(result_path, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\nResults saved to: {result_path}")


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                num_classes: int, args, fold: int = None) -> Dict[str, Any]:
    """Train and evaluate the hybrid model."""
    
    # Create datasets
    train_ds = EMGWindowDataset(X_train, y_train)
    test_ds = EMGWindowDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True if DEVICE.type == 'cuda' else False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    # Create model
    in_channels = X_train.shape[1]
    model = HybridCNNRNN(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=[64, 128, 256],
        cnn_kernels=[7, 5, 3],
        cnn_dropout=args.cnn_dropout,
        rnn_type=args.rnn_type,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        bidirectional=args.bidirectional,
        use_attention=args.attention,
        fc_hidden=128,
        fc_dropout=args.fc_dropout
    ).to(DEVICE)
    
    # Class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                            DEVICE, clip_grad=args.clip_grad)
        test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, DEVICE)
        
        scheduler.step(test_f1)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })
        
        if epoch % args.log_interval == 0 or epoch == 1:
            fold_str = f"[Fold {fold}] " if fold else ""
            print(f"{fold_str}Epoch {epoch:3d}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, test_f1={test_f1:.4f}")
        
        # Early stopping
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            if args.save_model:
                fold_str = f"_fold{fold}" if fold else ""
                model_path = Path(args.save_dir) / f"best_model{fold_str}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_f1': test_f1,
                    'test_acc': test_acc,
                }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation
    test_loss, test_acc, test_f1, preds, trues = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\nBest epoch: {best_epoch}, Best F1: {best_f1:.4f}")
    print(f"Final: test_acc={test_acc:.4f}, test_f1={test_f1:.4f}")
    
    return {
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'best_f1': float(best_f1),
        'best_epoch': best_epoch,
        'history': history,
        'preds': preds.tolist(),
        'trues': trues.tolist()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid CNN-RNN for EMG Classification')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='grabmyo', choices=['grabmyo', 'ninapro'])
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    
    # GrabMyo specific
    parser.add_argument('--sessions', type=int, nargs='+', default=[1])
    parser.add_argument('--participants', type=int, nargs='+', default=list(range(1, 44)))
    
    # Ninapro specific  
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--exercises', type=int, nargs='+', default=[1, 2, 3],
                        help='Ninapro exercises: 1=basic fingers, 2=wrist, 3=grasps')
    
    # Common
    parser.add_argument('--include-rest', action='store_true', help='Include rest class')
    parser.add_argument('--window-size', type=int, default=256, 
                        help='Window size in samples. Defaults: GrabMyo=256 (~125ms), Ninapro=40 (~200ms)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap (0-1)')
    parser.add_argument('--bandpass', action='store_true', help='Apply bandpass filter')
    
    # Split
    parser.add_argument('--split', type=str, default='random', choices=['random', 'group', 'loso'])
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model architecture
    parser.add_argument('--rnn-type', type=str, default='gru', choices=['gru', 'lstm'])
    parser.add_argument('--rnn-hidden', type=int, default=128)
    parser.add_argument('--rnn-layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', action='store_true')
    parser.add_argument('--attention', action='store_true', default=True)
    parser.add_argument('--no-attention', action='store_true')
    
    # Dropout
    parser.add_argument('--cnn-dropout', type=float, default=0.2)
    parser.add_argument('--rnn-dropout', type=float, default=0.3)
    parser.add_argument('--fc-dropout', type=float, default=0.5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--log-interval', type=int, default=5)
    
    # Output
    parser.add_argument('--save-dir', type=str, default='results_hybrid')
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--no-save-model', action='store_true')
    
    # Cache
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--cache-dir', type=str, default='./raw_cache')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only load/extract data and save to cache, skip training')
    
    args = parser.parse_args()
    
    # Handle negation flags
    if args.no_bidirectional:
        args.bidirectional = False
    if args.no_attention:
        args.attention = False
    if args.no_save_model:
        args.save_model = False
    if args.no_cache:
        args.use_cache = False
    
    # cache-only implies use-cache
    if args.cache_only:
        args.use_cache = True
    
    main(args)
