import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Workload archetype labels
SPIKE = "SPIKE"
PERIODIC = "PERIODIC"
RAMP = "RAMP"
STATIONARY_NOISY = "STATIONARY_NOISY"
ABSTAIN = None

class WindowedLabelingFunctions:
    """Labeling functions optimized for 60-minute windows."""
    
    @staticmethod
    def lf_spike_by_kurtosis(features: Dict[str, float]) -> Optional[str]:
        """High kurtosis indicates heavy tails (spikes) in the window."""
        if features.get('kurtosis', 0) > 10:  # Lower threshold for windows
            return SPIKE
        return ABSTAIN
    
    @staticmethod
    def lf_spike_by_peak_ratio(features: Dict[str, float]) -> Optional[str]:
        """Large max to median ratio indicates spikes."""
        if features.get('max_to_median_ratio', 0) > 20 and features.get('median', 0) > 0:
            return SPIKE
        return ABSTAIN
    
    @staticmethod
    def lf_spike_by_sudden_activity(features: Dict[str, float]) -> Optional[str]:
        """Low baseline with sudden high activity."""
        if (features.get('active_ratio', 0) < 0.3 and 
            features.get('max', 0) > features.get('mean', 0) * 10):
            return SPIKE
        return ABSTAIN
    
    @staticmethod
    def lf_periodic_by_spectral_entropy(features: Dict[str, float]) -> Optional[str]:
        """Low spectral entropy indicates regular pattern in window."""
        if features.get('spectral_entropy', 1) < 0.5:
            return PERIODIC
        return ABSTAIN
    
    @staticmethod
    def lf_periodic_by_autocorrelation(features: Dict[str, float]) -> Optional[str]:
        """Strong autocorrelation at short lags for 60-min window."""
        # Check 5, 10, 30 minute autocorrelations
        if (features.get('autocorr_lag_5', 0) > 0.7 or 
            features.get('autocorr_lag_10', 0) > 0.7 or
            features.get('autocorr_lag_30', 0) > 0.6):
            return PERIODIC
        return ABSTAIN
    
    @staticmethod
    def lf_periodic_by_regular_peaks(features: Dict[str, float]) -> Optional[str]:
        """Regular peaks indicate periodic pattern."""
        peaks_per_hour = features.get('peaks_per_hour', 0)
        if 2 <= peaks_per_hour <= 12:  # 2-12 peaks per hour suggests periodicity
            return PERIODIC
        return ABSTAIN
    
    @staticmethod
    def lf_ramp_by_trend(features: Dict[str, float]) -> Optional[str]:
        """Strong linear trend in the window."""
        # Normalize slope by mean to make it scale-invariant
        mean_val = features.get('mean', 1)
        normalized_slope = abs(features.get('linear_trend_slope', 0)) / (mean_val + 1e-6)
        if normalized_slope > 0.02:  # 2% change per minute
            return RAMP
        return ABSTAIN
    
    @staticmethod
    def lf_ramp_by_consistent_change(features: Dict[str, float]) -> Optional[str]:
        """Consistent directional change."""
        mean_change = features.get('mean_change', 0)
        std_change = features.get('std_change', 1)
        if std_change > 0 and abs(mean_change) / std_change > 0.8:
            return RAMP
        return ABSTAIN
    
    @staticmethod
    def lf_stationary_by_low_variance(features: Dict[str, float]) -> Optional[str]:
        """Low relative variance with no trend."""
        cv = features.get('cv', 0)
        if (0.1 < cv < 1.0 and 
            abs(features.get('linear_trend_slope', 0)) < 0.01 and
            features.get('kurtosis', 0) < 3):
            return STATIONARY_NOISY
        return ABSTAIN
    
    @staticmethod
    def lf_stationary_by_high_entropy(features: Dict[str, float]) -> Optional[str]:
        """High entropy and low autocorrelation indicates noise."""
        if (features.get('spectral_entropy', 0) > 0.8 and 
            features.get('autocorr_lag_10', 1) < 0.3 and
            features.get('active_ratio', 0) > 0.5):
            return STATIONARY_NOISY
        return ABSTAIN


class WindowedWeakSupervisionLabeler:
    """Weak supervision labeler for windowed time series data."""
    
    def __init__(self):
        self.labeling_functions = self._initialize_labeling_functions()
        
    def _initialize_labeling_functions(self) -> List[Callable]:
        """Initialize all labeling functions."""
        lfs = [
            # Spike LFs
            WindowedLabelingFunctions.lf_spike_by_kurtosis,
            WindowedLabelingFunctions.lf_spike_by_peak_ratio,
            WindowedLabelingFunctions.lf_spike_by_sudden_activity,
            
            # Periodic LFs
            WindowedLabelingFunctions.lf_periodic_by_spectral_entropy,
            WindowedLabelingFunctions.lf_periodic_by_autocorrelation,
            WindowedLabelingFunctions.lf_periodic_by_regular_peaks,
            
            # Ramp LFs
            WindowedLabelingFunctions.lf_ramp_by_trend,
            WindowedLabelingFunctions.lf_ramp_by_consistent_change,
            
            # Stationary Noisy LFs
            WindowedLabelingFunctions.lf_stationary_by_low_variance,
            WindowedLabelingFunctions.lf_stationary_by_high_entropy
        ]
        
        return lfs
    
    def label_windows(self, windowed_features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply weak supervision to label time windows.
        
        Args:
            windowed_features_df: DataFrame with windowed features
            
        Returns:
            DataFrame with added 'label' and 'confidence' columns
        """
        labels = []
        confidences = []
        label_matrix = []
        
        # Get feature columns (exclude metadata)
        metadata_cols = ['function_hash', 'window_idx', 'window_start_minute', 
                        'window_end_minute', 'day', 'hour_of_day']
        feature_cols = [col for col in windowed_features_df.columns if col not in metadata_cols]
        
        for idx, row in windowed_features_df.iterrows():
            # Extract features as dictionary
            features = row[feature_cols].to_dict()
            
            # Apply all labeling functions
            lf_outputs = []
            for lf in self.labeling_functions:
                try:
                    label = lf(features)
                    if label is not None:
                        lf_outputs.append(label)
                except Exception as e:
                    logger.debug(f"LF {lf.__name__} failed: {e}")
            
            label_matrix.append(lf_outputs)
            
            # Aggregate labels (majority voting)
            if lf_outputs:
                label_counts = Counter(lf_outputs)
                majority_label, count = label_counts.most_common(1)[0]
                confidence = count / len(lf_outputs)
                
                labels.append(majority_label)
                confidences.append(confidence)
            else:
                # Fallback labeling
                labels.append(self._fallback_labeling(features))
                confidences.append(0.3)
        
        # Add labels to dataframe
        result_df = windowed_features_df.copy()
        result_df['label'] = labels
        result_df['confidence'] = confidences
        
        # Log label distribution
        label_dist = result_df['label'].value_counts()
        logger.info(f"Window label distribution:\n{label_dist}")
        
        return result_df
    
    def _fallback_labeling(self, features: Dict[str, float]) -> str:
        """Fallback labeling when no LF fires."""
        # Use simple heuristics
        if features.get('kurtosis', 0) > 5:
            return SPIKE
        elif features.get('autocorr_lag_10', 0) > 0.5:
            return PERIODIC
        elif abs(features.get('linear_trend_slope', 0)) > 0.05:
            return RAMP
        else:
            return STATIONARY_NOISY
    
    def get_labeling_statistics(self, labeled_df: pd.DataFrame) -> Dict:
        """Get statistics about the labeling process."""
        stats = {
            'total_windows': len(labeled_df),
            'label_distribution': labeled_df['label'].value_counts().to_dict(),
            'avg_confidence': labeled_df['confidence'].mean(),
            'high_confidence_ratio': (labeled_df['confidence'] >= 0.6).mean(),
            'windows_per_label': {}
        }
        
        # Statistics per label
        for label in labeled_df['label'].unique():
            label_data = labeled_df[labeled_df['label'] == label]
            stats['windows_per_label'][label] = {
                'count': len(label_data),
                'avg_confidence': label_data['confidence'].mean(),
                'unique_functions': label_data['function_hash'].nunique()
            }
        
        return stats


if __name__ == "__main__":
    # Test with synthetic windowed data
    print("Testing windowed weak supervision...")
    
    # Create synthetic windowed features
    test_data = pd.DataFrame([
        # Spike windows
        {
            'function_hash': 'func1', 'window_idx': 0,
            'kurtosis': 15, 'max_to_median_ratio': 30, 'active_ratio': 0.2,
            'spectral_entropy': 0.9, 'autocorr_lag_10': 0.1, 'linear_trend_slope': 0.01,
            'cv': 3.5, 'mean': 10, 'median': 2, 'max': 60,
            'window_start_minute': 0, 'window_end_minute': 60, 'day': 0, 'hour_of_day': 0
        },
        # Periodic windows
        {
            'function_hash': 'func2', 'window_idx': 0,
            'kurtosis': 2, 'max_to_median_ratio': 2, 'active_ratio': 0.9,
            'spectral_entropy': 0.3, 'autocorr_lag_10': 0.8, 'linear_trend_slope': 0.005,
            'cv': 0.3, 'peaks_per_hour': 6, 'mean': 50, 'median': 48,
            'window_start_minute': 0, 'window_end_minute': 60, 'day': 0, 'hour_of_day': 0
        },
        # Ramp window
        {
            'function_hash': 'func3', 'window_idx': 0,
            'kurtosis': 0, 'max_to_median_ratio': 3, 'active_ratio': 0.7,
            'spectral_entropy': 0.6, 'autocorr_lag_10': 0.4, 'linear_trend_slope': 0.5,
            'cv': 0.5, 'mean_change': 1.0, 'std_change': 0.8, 'mean': 20,
            'window_start_minute': 0, 'window_end_minute': 60, 'day': 0, 'hour_of_day': 0
        }
    ])
    
    # Apply labeling
    labeler = WindowedWeakSupervisionLabeler()
    labeled_df = labeler.label_windows(test_data)
    
    print("\nLabeling results:")
    print(labeled_df[['function_hash', 'window_idx', 'label', 'confidence']])
    
    # Get statistics
    stats = labeler.get_labeling_statistics(labeled_df)
    print("\nLabeling statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")