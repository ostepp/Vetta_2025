"""Utilities for processing and analyzing stance phases from sensor and treadmill data.

This module provides functionality to detect, process, and analyze stance phases from
accelerometer and treadmill data. It includes methods for peak detection, stance
segmentation, and visualization of the results.
"""

from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from .config import PipelineConfig

class SignalProcessor:
    """Processes and analyzes sensor signals."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def find_accel_strike_indices(
        self,
        accel: pd.Series,
        accel_unfiltered: pd.Series
    ) -> Tuple[List[int], List[int]]:
        """Find strikes in accelerometer signal.
        
        Args:
            accel: Filtered accelerometer signal
            accel_unfiltered: Raw accelerometer signal
            
        Returns:
            Tuple of (raw strike indices, refined strike indices)
        """
        if len(accel) < self.config.min_stance_size:
            return [], []
        
        # Step 1: Find all peaks in filtered acceleration signal
        accel_peaks, _ = signal.find_peaks(accel, **self.config.accel_peak_params)

        # Step 2: Filter peaks to find initial contact points
        strike_indices = []
        for i, curr_peak in enumerate(accel_peaks[1:-1], 1):  # Skip first and last peaks
            prev_peak, next_peak = accel_peaks[i-1], accel_peaks[i+1]
            time_before, time_after = curr_peak - prev_peak, next_peak - curr_peak

            # Initial contact has shorter time before peak than after (asymmetric pattern)
            if time_before < time_after:  # _^__ pattern (initial contact)
                strike_indices.append(curr_peak)

        # Step 3: Refine initial contact points using jerk analysis
        refined_strike_indices = []
        for peak in strike_indices:
            if peak == 0:
                continue

            # Calculate jerk in window around contact point
            accel_window = accel_unfiltered[peak:peak + self.config.jerk_window_size]
            jerk_signal = np.diff(accel_window)
            jerk_peaks, _ = signal.find_peaks(jerk_signal, **self.config.jerk_peak_params)

            if len(jerk_peaks) == 0:
                refined_strike_indices.append(peak)
            else:
                # Adjust contact point to first jerk peak
                refined_strike_indices.append(peak + jerk_peaks[0] + 1)

        return strike_indices, refined_strike_indices

    def find_vgrf_strike_indices(self, vgrf: pd.Series) -> Tuple[List[int], List[int]]:
        """Find strikes in vGRF signal.
        
        Args:
            vgrf: vGRF force signal
            
        Returns:
            Tuple of (start indices, end indices)
        """
        _, properties = signal.find_peaks(vgrf, **self.config.vgrf_peak_params)
        # print("left_bases:", properties['left_bases'])
        # print("right_bases:", properties['right_bases'])

        return properties['left_bases'], properties['right_bases']
    

class StanceProcessor:
    """Processes and normalizes stance phases."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def normalize_stance(self, step: pd.Series) -> pd.Series:
        """Normalize a step to 100 frames using linear interpolation.
        
        Args:
            step: Raw step data
            
        Returns:
            Normalized stance with 100 frames
        """
        # Normalize to 100 frames
        old_x = np.arange(0, len(step))
        new_x = np.arange(0, 100)
        return pd.Series(np.interp(new_x, old_x, step))

    def filter_stance(
        self,
        stance: pd.Series,
        filters: List[Tuple[int, int, float]]
    ) -> Optional[pd.Series]:
        """Apply bandpass filters to stance phase.
        
        Args:
            stance: Stance phase to filter
            filters: List of (low_freq, high_freq, weight) tuples
            
        Returns:
            Filtered stance phase or None if filtering fails
        """
        if stance.shape[0] < self.config.min_stance_size:
            return None
        if stance.shape[0] > self.config.max_stance_size:
            return None
        for start, end, threshold in filters:
            window = stance[start:end]
            if np.mean(window) < threshold:
                return None
        return stance


class StanceAnalyzer:
    """Main class for analyzing and matching stances."""
    
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.signal_processor = SignalProcessor(config)
        self.stance_processor = StanceProcessor(config)
    
    def extract_sensor_stances(
        self,
        leg_accel: pd.Series,
        leg_accel_unfiltered: pd.Series,
         waist_accel: pd.Series
         #start_idx: List[int]
    ) -> Tuple[List[int], List[pd.Series]]:
        """Extract stances from sensor data.
        
        Args:
            leg_accel: Filtered leg accelerometer signal
            leg_accel_unfiltered: Raw leg accelerometer signal
            waist_accel: Waist accelerometer signal
            
        Returns:
            Tuple of (strike indices, stance phases)
        """
        _, refined_strike_indices = self.signal_processor.find_accel_strike_indices(
            leg_accel, leg_accel_unfiltered
        )

        # Split waist into slices defined by strike indices
        waist_slices = [
            waist_accel[i:j]
            for i, j in zip(refined_strike_indices, refined_strike_indices[1:] + [len(waist_accel)])
        ]
            #for i, j in zip(start_idx, start_idx[1:] + [len(waist_accel)])
        #]

        stances = [self.stance_processor.normalize_stance(slice) for slice in waist_slices]
        # print(waist_slices)
        return refined_strike_indices, stances

    def extract_treadmill_stances(self, vgrf: pd.Series) -> Tuple[List[int], List[pd.Series]]:
        """Extract stances from treadmill vGRF data.
        
        Args:
            vgrf: vGRF force signal
            
        Returns:
            Tuple of (strike indices, stance phases)
        """
        start_idx, end_idx = self.signal_processor.find_vgrf_strike_indices(vgrf)
        slices = [vgrf[i:j] for i, j in zip(start_idx, end_idx)]
        stances = [self.stance_processor.normalize_stance(slice) for slice in slices]
        return start_idx, stances

    def parse_and_match_stances(
        self,
        leg_accel: pd.Series,
        leg_accel_unfiltered: pd.Series,
        waist_accel: pd.Series,
        vgrf: pd.Series
    ) -> Tuple[List[int], List[pd.Series], Dict]:
        """Extract and match stance phases from sensors and treadmill.
        
        Args:
            leg_accel: Filtered leg accelerometer signal
            leg_accel_unfiltered: Raw leg accelerometer signal
            waist_accel: Waist accelerometer signal
            vgrf: vGRF signal
            
        Returns:
            Tuple of (strike indices, matched stance phases, debug info)
        """
        #start_idx, _ = self.extract_treadmill_stances(vgrf)
        accel_strikes, accel_stances = self.extract_sensor_stances(
            leg_accel, leg_accel_unfiltered, waist_accel
        )
        print("ACCEL STRIKES:", accel_strikes)
        print("ACCEL STANCES:", accel_stances)
        #start_idx, accel_stances = self.extract_sensor_stances(
        #    leg_accel, leg_accel_unfiltered, waist_accel, start_idx
        #)
        #accel_strikes = start_idx
        if len(accel_strikes) != len(accel_stances):
            raise ValueError(f"Number of strikes ({len(accel_strikes)}) != number of stances ({len(accel_stances)}) in accel data")

        vgrf_strikes, vgrf_stances = self.extract_treadmill_stances(vgrf)
        print("VGRF STRIKES:", vgrf_strikes)
        print("VGRF STANCES:", vgrf_stances)
        if len(vgrf_strikes) != len(vgrf_stances):
            raise ValueError(f"Number of strikes ({len(vgrf_strikes)}) != number of stances ({len(vgrf_stances)}) in vgrf data")

        matched_strikes = []
        matched_stances = []
        # for accel_idx, accel_stance in zip(accel_strikes, accel_stances): # NOTE: This is a quick fix to segment accel data using vgrf strikes
        for accel_idx, accel_stance in zip(vgrf_strikes, vgrf_stances):
            #print("ACCEL STANCE:", accel_stance)  # <-- Add this line
            #print(f"ACCEL STANCE INDEX: {accel_idx}")
            #print(accel_stance)

            # Plot the stance
            #plt.figure()
            #accel_stance.plot()
            #plt.title(f"Accelerometer Stance at Index {accel_idx}")
            #plt.xlabel("Sample")
            #plt.ylabel("Acceleration (units?)")
            #plt.grid(True)
            #plt.show()
            for vgrf_idx, vgrf_stance in zip(vgrf_strikes, vgrf_stances):
                if abs(accel_idx - vgrf_idx) <= self.config.stance_matching_time_threshold:
                    if self.stance_processor.filter_stance(accel_stance, self.config.accel_filters) is None:
                        continue
                    if self.stance_processor.filter_stance(vgrf_stance, self.config.vgrf_filters) is None:
                        continue
                    matched_stances.append((accel_stance, vgrf_stance))
                    matched_strikes.append((accel_idx, vgrf_idx))

        debug = {
            # 'accel_strikes': accel_strikes,   # NOTE: This is a quick fix to segment accel data using vgrf strikes
            'accel_strikes': vgrf_strikes,
            'vgrf_strikes': vgrf_strikes,
            'matched_strikes': matched_strikes,
            # 'accel_stances': accel_stances,   # NOTE: This is a quick fix to segment accel data using vgrf strikes
            'accel_stances': vgrf_stances,
            'vgrf_stances': vgrf_stances,
            'matched_stances': matched_stances
        }
        return matched_strikes, matched_stances, debug


# For backward compatibility
def parse_and_match_stances(
    leg_accel: pd.Series,
    leg_accel_unfiltered: pd.Series,
    waist_accel: pd.Series,
    vgrf: pd.Series,
    config: PipelineConfig = PipelineConfig()
) -> Tuple[List[int], List[pd.Series], Dict]:
    """Legacy function for backward compatibility."""
    analyzer = StanceAnalyzer(config)
    return analyzer.parse_and_match_stances(
        leg_accel, leg_accel_unfiltered, waist_accel, vgrf
    )
