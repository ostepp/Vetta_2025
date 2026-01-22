"""Configuration parameters for stance analysis pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Peak detection parameters
ACCEL_PEAK_PARAMS = {
    'height': 1.0,
    'prominence': 0.5,
    'width': 5.0,
    'distance': 10
}

JERK_PEAK_PARAMS = {
    'height': 0.0,
    'prominence': 0.1
}

VGRF_PEAK_PARAMS = {
    'height': 0.5,
    'prominence': 0.8,
    'width': 10.0,
    'distance': 50,
    'rel_height': 0.95
}

# Filter parameters
ACCEL_FILTERS = [
    (10, 20, 0.5),
    (20, 30, 0.5),
    (30, 40, 0.5),
]

VGRF_FILTERS = [
    (10, 20, 0.5),
    (20, 30, 0.5),
    (30, 40, 0.5),
    (45, 55, 0.5),
    (65, 75, 0.5),
    (75, 85, 0.5)
]

# Stance parameters
JERK_WINDOW_SIZE = 50
MIN_STANCE_SIZE = 80  # 0.8s assuming 100 hz
MAX_STANCE_SIZE = 140  # 1.4s assuming 100 hz
STANCE_MATCHING_TIME_THRESHOLD = 5

@dataclass
class PipelineConfig:
    """Configuration for the stance analysis pipeline.
    
    Attributes:
        accel_peak_params: Parameters for accelerometer peak detection
        jerk_peak_params: Parameters for jerk peak detection
        jerk_window_size: Size of window for jerk analysis
        vgrf_peak_params: Parameters for vGRF peak detection
        min_stance_size: Minimum number of frames for a valid stance
        max_stance_size: Maximum number of frames for a valid stance
        stance_matching_time_threshold: Time threshold for matching stances
        accel_filters: List of (start, end, threshold) tuples for accelerometer filtering
        vgrf_filters: List of (start, end, threshold) tuples for vGRF filtering
    """
    accel_peak_params: Dict[str, float] = field(default_factory=lambda: ACCEL_PEAK_PARAMS.copy())
    jerk_peak_params: Dict[str, float] = field(default_factory=lambda: JERK_PEAK_PARAMS.copy())
    jerk_window_size: int = JERK_WINDOW_SIZE
    vgrf_peak_params: Dict[str, float] = field(default_factory=lambda: VGRF_PEAK_PARAMS.copy())
    min_stance_size: int = MIN_STANCE_SIZE
    max_stance_size: int = MAX_STANCE_SIZE
    stance_matching_time_threshold: int = STANCE_MATCHING_TIME_THRESHOLD
    accel_filters: List[Tuple[int, int, float]] = field(default_factory=lambda: ACCEL_FILTERS.copy())
    vgrf_filters: List[Tuple[int, int, float]] = field(default_factory=lambda: VGRF_FILTERS.copy()) 