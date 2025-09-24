"""Utilities for predicting treadmill stances from accelerometer data."""

import onnxruntime
import numpy as np
import pandas as pd

# Model path for stance prediction
MODEL_PATH = "models/tfmodel_5_RP.onnx"

# Global session variable for lazy loading
_session = None

def predict_stance(input_stance: pd.Series) -> pd.Series:
    """Predict treadmill stance from accelerometer stance using trained model.
    
    Args:
        input_stance: Normalized accelerometer stance phase (100 frames)
        
    Returns:
        Predicted treadmill stance phase (100 frames)
    """
    global _session

    # Load model if not already loaded
    if _session is None:
        _session = onnxruntime.InferenceSession(MODEL_PATH)

    # Convert input to numpy array with correct shape and type
    input_array = input_stance.to_numpy(dtype=np.float32)[np.newaxis, :]

    # Run prediction
    prediction = _session.run(
        None,
        {'dense_24_input': input_array}
    )[0][0]

    return pd.Series(prediction)
