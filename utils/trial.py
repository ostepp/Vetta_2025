from typing import List
import pandas as pd
import os
import json

from utils.sensor import load_sensors_data
from utils.treadmill import load_treadmill_data

# Constants
SENSORS_FILENAME = 'raw-sensors.txt'
TREADMILL_FILENAME = 'raw-treadmill.mot'
# SENSORS_FILENAME = 'IMU_data_RevB_v3_ncbc_s17_tm_p9_t1.csv'
# TREADMILL_FILENAME = 'ncbc_s17_tm_p9_t1_1.forces'

class Trial:
    """Container for all data from a single trial.
    
    This class manages the loading and processing of sensor and treadmill data
    for a single trial, including stance phase detection and storage.
    
    Attributes:
        path: Path to the trial directory
        name: Trial identifier
        subject_name: Name of the subject
        subject_weight: Weight of the subject in kg
    """
    def __init__(self, path: str, 
                 subject_name: str, 
                 subject_weight: float, 
                 subject_info: dict) -> None:
        """Initialize trial with path and subject information.
        
        Args:
            path: Path to trial directory
            subject_name: Name of the subject
            subject_weight: Weight of the subject in kg
            subject_info: Additional subject information from info.json
        """
        self.path = path
        self.name = os.path.basename(path)
        self.subject_name = subject_name
        self.subject_weight = subject_weight

        # assign sensors based on info.json if available, otherwise use defaults
        if 'left_sensor' in subject_info: 
            self.left_sensor = subject_info['left_sensor']
        else:
            self.left_sensor = None
        if 'right_sensor' in subject_info:
            self.right_sensor = subject_info['right_sensor']
        else: 
            self.right_sensor = None
        if 'waist_sensor' in subject_info:
            self.waist_sensor = subject_info['waist_sensor']
        else: 
            self.waist_sensor = None
        if self.left_sensor and self.right_sensor and self.waist_sensor:
            self.sensor_ids = {'left': self.left_sensor, 
                               'right': self.right_sensor, 
                               'waist': self.waist_sensor}
            print(f"Assigned sensor IDs for trial {self.name}: {self.sensor_ids}")
        else: 
            self.sensor_ids = None

        if 'left_vgrf_offsets' in subject_info:
            self.left_vgrf_offsets = subject_info['left_vgrf_offsets']
        if 'right_vgrf_offsets' in subject_info:
            self.right_vgrf_offsets = subject_info['right_vgrf_offsets']


    def load_sensors(self) -> pd.DataFrame:
        """Load and process sensor data from the trial directory.
        
        Returns:
            DataFrame containing processed sensor data
        """
        try:
            sensor_data = load_sensors_data(os.path.join(self.path, SENSORS_FILENAME), self.sensor_ids)
        except:
            fn_options = [x for x in os.listdir(self.path) if 
                          (
                              '.csv' in x and \
                              'IMU_data' in x and \
                              self.subject_name in x
                          )
                          ]
            if len(fn_options) == 1:
                SENSORS_FILENAME = fn_options[0]
                pass
            else:

                fn_options = [x for x in os.listdir(self.path) if 'raw-sensors.txt' in x]
                if len(fn_options) == 1:
                    SENSORS_FILENAME = fn_options[0]
                    pass

                elif len(fn_options) == 0:
                    raise FileNotFoundError(f"No sensor data found.")
                else:
                    raise FileNotFoundError(f"Too many sensor data files found: {fn_options}")
            
            sensor_data = load_sensors_data(os.path.join(self.path, SENSORS_FILENAME), self.sensor_ids)

        return sensor_data

    def load_treadmill(self) -> pd.DataFrame:
        """Load and process treadmill data from the trial directory.
        
        Returns:
            DataFrame containing processed treadmill data
        """
        return load_treadmill_data(os.path.join(self.path, TREADMILL_FILENAME), self.subject_weight)

def load_trials(subject_path: str) -> List[Trial]:
    """Load all trials for a subject.
    
    Args:
        subject_path: Path to the subject directory
        
    Returns:
        List of trial data objects
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If subject info is invalid
    """
    # Get subject name from path
    subject_name = os.path.basename(subject_path)

    # Load subject info
    info_path = os.path.join(subject_path, 'info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"info.json is missing from {subject_path}")

    with open(info_path, 'r', encoding='utf-8') as fp:
        subject_info = json.load(fp)
        if 'weight' not in subject_info:
            raise ValueError("info.json is missing the weight field")
        subject_weight = float(subject_info['weight'])

    # Load trials
    trials_dir = os.path.join(subject_path, 'trials')
    if not os.path.exists(trials_dir):
        raise FileNotFoundError(f"trials directory is missing from {subject_path}")

    results = []
    for trial_name in os.listdir(trials_dir):
        trial_dir = os.path.join(trials_dir, trial_name)
        if trial_name.startswith('.') or not os.path.isdir(trial_dir):
            continue
        contents = list(os.listdir(trial_dir))

        # Check for required files
        if SENSORS_FILENAME not in contents:
            raise FileNotFoundError(f"{SENSORS_FILENAME} missing from {trial_dir}")
        if TREADMILL_FILENAME not in contents:
            raise FileNotFoundError(f"{TREADMILL_FILENAME} missing from {trial_dir}")

        # Create trial object
        trial = Trial(trial_dir, subject_name, subject_weight, subject_info)
        results.append(trial)

    return results

def load_trial(subjects_path: str, subject_name: str, trial_name: str) -> Trial:
    """Load a single trial for a subject.
    
    Args:
        subjects_path: Path to subjects directory
        subject_name: Name of the subject
        trial_name: Name of the trial
        
    Returns:
        Trial object containing the trial data
    """
    trial_path = os.path.join(subjects_path, subject_name, 'trials', trial_name)
    if not os.path.exists(trial_path):
        raise ValueError(f"Trial path does not exist: {trial_path}")

    # Load subject info
    subject_info_path = os.path.join(subjects_path, subject_name, 'info.json')
    with open(subject_info_path, 'r', encoding='utf-8') as f:
        subject_info = json.load(f)

    

    return Trial(trial_path, subject_name, subject_info['weight'], subject_info)
