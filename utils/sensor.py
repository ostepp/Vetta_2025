"""Sensor data processing utilities for accelerometer data.

This module provides functionality to load, process, and filter accelerometer data
from sensor files. It handles data from multiple sensors (left, right, and waist)
and provides methods for resampling and filtering the data.
"""

from typing import Tuple, List
from dataclasses import dataclass
import os
import json
import numpy as np
import pandas as pd
from scipy import signal

# Physical constants
GRAVITY = 9.81  # m/s²

@dataclass
class SensorData:
    """Container for processed sensor data from multiple sensors.
    DataFrame columns are accel, accel_filtered, time.
    
    Attributes:
        left_df: Processed accelerometer data from left sensor
        right_df: Processed accelerometer data from right sensor
        waist_df: Processed accelerometer data from waist sensor
    """
    left_df: pd.DataFrame
    right_df: pd.DataFrame
    waist_df: pd.DataFrame

    # determine effective sampling rate for each sensor
    

def _resample_accel(df: pd.DataFrame, resample_freq: int) -> pd.DataFrame:
    """Resample accelerometer data to a specified frequency.
    
    Args:
        df: DataFrame containing 'time' and 'accel' columns
        resample_freq: Target sampling frequency in Hz
        
    Returns:
        DataFrame with resampled accelerometer data
    """
    # print('Resampling to: ', resample_freq)
    try:
        start, end = np.nanmin(df['time']), np.nanmax(df['time'])
        interval = 1 / resample_freq

        out_df = pd.DataFrame({
            'time': np.arange(start, end, interval),
            'accel': np.interp(
                np.arange(start, end, interval),
                df['time'],

                df['accel']
            )
        })
    except:
        print(df)
        raise StopIteration
    
    return out_df


def _read_sensors_json(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and parse sensor data from JSON file.
    
    Args:
        file_path: Path to the JSON file containing sensor data
        
    Returns:
        Tuple of DataFrames (left, right, waist) containing raw sensor data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    def organize(df: pd.DataFrame) -> pd.DataFrame:
        """Sort and clean sensor DataFrame."""
        return df.drop('id', axis=1).sort_values(by='time').reset_index(drop=True)

    with open(file_path, 'r', encoding='utf-8') as fp:
        # Parse double-encoded JSON data
        rows = [json.loads(json.loads(r)) for r in fp.readlines()]
        df = pd.DataFrame(rows)[['id', 'accel', 'time']]

        # Extract data for each sensor
        sensor_data = {
            'waist': df[df['id'] == '3a'],
            'left': df[df['id'] == '39'],
            'right': df[df['id'] == '38']
        }

        return tuple(organize(df) for df in sensor_data.values())
    

def _read_sensors_csv(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and parse sensor data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing sensor data
        
    Returns:
        Tuple of DataFrames (left, right, waist) containing raw sensor data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    def organize(df: pd.DataFrame) -> pd.DataFrame:
        """Sort and clean sensor DataFrame."""
        cols_to_drop = ['DeviceID', 'PacketType','PayloadLen']
        return df.drop(cols_to_drop, axis=1).sort_values(by='Timestamp').reset_index(drop=True)

    df = pd.read_csv(file_path)

    # Extract data for each sensor
    sensor_data = {
        'waist': df[df['DeviceID'] == 2],
        'left': df[df['DeviceID'] == 3],
        'right': df[df['DeviceID'] == 1]
    }

    return tuple(organize(df) for df in sensor_data.values())
    

def _read_sensors_txt(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and parse sensor data from txt file.
    
    Args:
        file_path: Path to the txt file containing sensor data
        
    Returns:
        Tuple of DataFrames (left, right, waist) containing raw sensor data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    def organize(df: pd.DataFrame) -> pd.DataFrame:
        """Sort and clean sensor DataFrame."""
        cols_to_drop = ['DeviceID']
        return df.drop(cols_to_drop, axis=1).sort_values(by='time').reset_index(drop=True)

    df = pd.DataFrame(columns=['DeviceID', 'accel', 'time'])
    with open(file_path, 'r', encoding='utf-8') as fp:
        # Parse double-encoded JSON data
        for line in fp:
            row = json.loads(json.loads(line))
            df.loc[len(df)] = [row['id'], row['accel'], row['time']]

    # Extract data for each sensor
    sensor_data = {
        'waist': df[df['DeviceID'] == '3a'],
        'left': df[df['DeviceID'] == '39'],
        'right': df[df['DeviceID'] == '38']
    }

    # log raw sampling rates
    for sensor in [sensor_data['waist'], sensor_data['left'], sensor_data['right']]:
        elapsed_time = sensor.time.iloc[-1] - sensor.time.iloc[0]
        num_samples = len(sensor)
        samp_rate = num_samples / elapsed_time
        print(f'Raw sampling rate: {samp_rate}')

    return tuple(organize(df) for df in sensor_data.values())
    

def input_df(df) -> dict: #Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and parse sensor data from JSON file.
    
    Args:
        file_path: Path to the JSON file containing sensor data
        
    Returns:
        Tuple of DataFrames (left, right, waist) containing raw sensor data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    def organize(df: pd.DataFrame) -> pd.DataFrame:
        """Sort and clean sensor DataFrame."""
        return df.sort_values(by='time').reset_index(drop=True)

    # Extract data for each sensor
    sensor_data = {
        'waist': df[df['DeviceID'] == 1],
        'left': df[df['DeviceID'] == 2],
        'right': df[df['DeviceID'] == 3]
        # 'waist': organize(df[df['DeviceID'] == 1]),
        # 'left': organize(df[df['DeviceID'] == 2]),
        # 'right': organize(df[df['DeviceID'] == 3])
    }

    return sensor_data #tuple(organize(df) for df in sensor_data.values())

def _process_sensor_df(
    df: pd.DataFrame,
    resample_freq: int = 100,
    butter_ord: int = 4,
    butter_cutoff: float = 20
) -> pd.DataFrame:
    """Process sensor data with normalization, resampling, and filtering.
    
    Args:
        df: Raw sensor DataFrame
        resample_freq: Target sampling frequency in Hz (100 Hz default)
        butter_ord: Order of Butterworth filter (4th order default)
        butter_cutoff: Cutoff frequency for Butterworth filter (0.2 default)

    Returns:
        Processed DataFrame with normalized, resampled, and filtered data
    """
    # Normalize acceleration vectors
    if 'accel' in df.columns:
        norm = df['accel'].apply(np.linalg.norm)
        df['accel'] = norm
    elif 'AccelX' in df.columns:
        norm = np.linalg.norm(df[['AccelX','AccelY','AccelZ']].to_numpy(), axis=1)
        df['accel'] = norm

    # convert time column
    if 'time' not in df.columns:
        start_time = df['Timestamp'][0]
        time_factor = 1000
        time_data = (df['Timestamp'] - start_time) / time_factor
        df['time'] = time_data

        # log raw sampling rates
        elapsed_time = df.time.iloc[-1] - df.time.iloc[0]
        num_samples = len(df)
        samp_rate = num_samples / elapsed_time
        print(f'Raw sampling rate: {samp_rate}')


    # Resample
    # print('shape before resampling: ', df.shape)
    df = _resample_accel(df, resample_freq)
    # print(df.head())
    # print('shape after resampling: ', df.shape)

    # convert to Gs (if needed)
    # df['accel'] /= GRAVITY 

    # Apply Butterworth filter
    try:
        sos = signal.butter(butter_ord, butter_cutoff, fs=resample_freq, output='sos')
        df['accel_filtered'] = signal.sosfiltfilt(sos, df['accel'])
    except:
        df['accel_filtered'] = df['accel']
        print('Could not filter acceleration signal')

    # print(df.iloc[5000:7000])


    # plot pre and post processed 
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(df['time'], df['accel'], label='resampled acc')
    # plt.plot(df['time'], df['accel_filtered'], label='filtered acc')
    # plt.xlim([50, 52])
    # plt.xlabel('Time (s)')
    # plt.ylabel('g')
    # plt.legend()

    # raise StopIteration

    return df


def load_sensors_data(
    file_path: str,
    resample_freq: int = 100,
    butter_ord: int = 4,
    butter_cutoff: float = 12
) -> SensorData:
    """Load and process sensor data from a JSON or CSV file.
    
    Args:
        file_path: Path to the file containing sensor data
        resample_freq: Target sampling frequency in Hz
        butter_ord: Order of Butterworth filter
        butter_cutoff: Cutoff frequency for Butterworth filter
        
    Returns:
        SensorData object containing processed data from all sensors
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If resample_freq is not positive
    """
    print(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if resample_freq <= 0:
        raise ValueError(f"resample_freq must be positive, got {resample_freq}")

     # Load sensor data
    if '.json' in file_path:
        unprocessed_sensors = _read_sensors_json(file_path)

    elif '.csv' in file_path:
        unprocessed_sensors = _read_sensors_csv(file_path)
    
    elif '.txt' in file_path:
        unprocessed_sensors = _read_sensors_txt(file_path)

    else:
        raise ValueError(f"file type not recognized: {file_path}")

    # process sensor data
    processed_sensors = [
            _process_sensor_df(df, resample_freq, butter_ord, butter_cutoff)
            for df in unprocessed_sensors
        ]

    return SensorData(*processed_sensors)
