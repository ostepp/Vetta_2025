# UNC Applied Biomechanics — vGRF Estimation

This project estimates vertical ground reaction forces (vGRF) during walking from
wearable accelerometer (IMU) data. Accelerations from the waist and ankles are run
through a trained MLP model (`models/tfmodel_5_RP.onnx`) to predict per-step vGRF
waveforms, which can be validated against treadmill force-plate measurements.

There are two ways to use the project:

1. **Offline trial processing** — process previously collected trial data (sensors +
   treadmill) in a notebook, align the signals, parse gait cycles, and compare predicted
   vGRF against the treadmill reference.
2. **Real-time estimation** — stream live IMU data from the sensor hardware over a serial
   port (or replay a recorded CSV), detect steps, and predict vGRF on the fly.

#### Setup

This project uses uv to manage dependencies. Follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

Next set up a virtual environment and install dependencies:

```bash
uv venv
uv pip install -e .
```

Real-time **serial** mode additionally needs `pyserial` (not in the default
dependencies), so install it if you plan to stream from hardware:

```bash
uv pip install pyserial
```

#### Project Structure

```
run_realtime.py                       - Entry point for real-time vGRF estimation (serial or CSV replay)
Realtime vGRF Estimation.ipynb        - Notebook for exploring/visualizing real-time estimation
process-trial_vGRF_alignment.ipynb    - Notebook: align, parse, and predict on a trial (.csv/.txt firmware)
process-trial_sensor_vGRF_alignment.ipynb - Variant of the alignment/processing notebook
models/tfmodel_5_RP.onnx              - The trained MLP model (ONNX)
subjects/                             - Collected subject/trial data (see layout below)
results/                              - Per-subject/trial output (peaks + waveforms)
utils/                                - Helper modules used by the scripts and notebooks
scripts/                              - Standalone scripts (e.g. clean_and_zip.py)
```

Key `utils/` modules:

```
utils/trial.py        - Trial/subject loading; reads info.json (weight, sensor IDs, vGRF offsets)
utils/sensor.py       - Load & process accelerometer data from .json / .csv / .txt; resample + Butterworth filter
utils/treadmill.py    - Load & process treadmill vGRF (.mot)
utils/stance.py       - Step detection and stance (gait-cycle) extraction (StanceAnalyzer)
utils/config.py       - PipelineConfig: peak-detection, filter, and stance-size parameters
utils/predict.py      - Run the ONNX model to predict a vGRF stance from a waist-accel stance
utils/viz.py          - Dash-based tool for visually aligning sensor and treadmill signals
utils/output.py       - Assemble peak/waveform results for export
utils/read_packets.py - Binary packet parsing + real-time processing loop for streaming data
```

#### Subject Data Layout

Organize `subjects/` as follows:

```
subjects/
        ├── ncbc_s17/
        │   ├── info.json
        │   └── trials/
        │       ├── p9/
        │       │   ├── raw-sensors.txt   (or an IMU_data_*.csv for newer firmware)
        │       │   └── raw-treadmill.mot
        │       ├── norm/
        │       └── p6/
        ├── ncbc_s15/
        │   └── ...
        └── ...
```

The `example-subjects/` folder shows the structure (its raw data files are empty
placeholders and won't process).

##### info.json

`info.json` holds subject metadata. Only `weight` (kg) is required. For trials collected
with the new multi-sensor firmware, you can also specify which device ID maps to each
body location and the per-trial alignment offsets:

```json
{
    "weight": 77,
    "left_sensor": 2,
    "right_sensor": 1,
    "waist_sensor": 3,
    "left_vgrf_offsets":  { "norm": 1238, "p3": 600, "p6": 2750, "p9": 595 },
    "right_vgrf_offsets": { "norm": 1238, "p3": 600, "p6": 2750, "p9": 595 }
}
```

#### Usage — Offline Trial Processing

Open `process-trial_vGRF_alignment.ipynb` and work through the steps:

1. **Load data** — set `SUBJECTS_DIR`, `SUBJECT_NAME`, and `TRIAL_NAME`, then load the
   sensors and (unaligned) treadmill data.
2. **Align signals** — use the Dash alignment tool (`utils.viz.visually_align_signal`) to
   visually find the offset between each ankle sensor and its treadmill force signal. Sensor
   signals carry absolute timestamps; treadmill signals start at zero, so they must be aligned.
   Record the offsets (and optionally save them into `info.json` as `*_vgrf_offsets`).
3. **Apply offsets** and extract left/right strike indices and stances using a
   `PipelineConfig`.
4. **Predict and compare** — run each parsed waist-acceleration stance through the model and
   compare predicted vGRF against the treadmill reference. Results are written under
   `results/SUBJECT_NAME/.../` as peak (`*_peaks.xlsx`) and waveform (`*_waveforms_*.xlsx`)
   files.

#### Usage — Real-time Estimation

`run_realtime.py` reads streaming IMU packets, detects steps, and predicts vGRF live.

```bash
# Stream from the device over serial (set PORT/BAUDRATE in run_realtime.py)
python run_realtime.py            # main(run='serial')

# Or replay a recorded CSV as if it were arriving in real time
# (set main(run='csv') at the bottom of run_realtime.py)
```

- **serial** — connects to the configured `PORT` (e.g. `COM3`) at `BAUDRATE`, spawns a
  reader thread, and parses binary header/payload/footer packets (with CRC8 checks) in
  `utils/read_packets.py`. Accelerations are scaled to Gs, buffered, resampled, filtered, and
  fed to `process_packet`, which detects steps and runs the model.
- **csv** — replays a recorded stream file row-by-row through the same `process_packet`
  pipeline, useful for testing without hardware.

In both modes the run saves three CSVs on exit: the raw stream (`ESP_stream_data.csv`),
detected steps with parsed waist accelerations (`Parsed_steps_data.csv`), and predicted vGRF
per step (`Predicted_vGRF_data.csv`).

#### Processing Procedure (details)

##### Parse and filter acceleration data
1. Divide the raw data into 3 signals (waist, left ankle, right ankle), using the sensor IDs
   from `info.json` when present.
2. Normalize acceleration vectors into scalar magnitudes.
3. Resample each signal to 100 Hz.
4. Convert to Gs (divide by gravity / detect m/s² inputs).
5. Apply a forward-backward Butterworth low-pass filter (≈20 Hz cutoff).

##### Parse and filter vertical ground force (vGRF) data
1. Divide the raw treadmill data into 2 signals (left and right).
2. Normalize vGRF vectors into scalar magnitudes.
3. Resample each signal to 100 Hz.
4. Normalize by subject weight (body weights, BW).

##### Align signals
Sensor signals have synchronized, absolute timestamps; treadmill signals start at zero. The
Dash alignment tool lets you visually offset the treadmill signal so both share an absolute
time base. Offsets can be stored per trial in `info.json`.

##### Create stances
1. Use `scipy.signal.find_peaks()` to find peaks/peak widths in the left and right signals.
2. Split each signal into stances bounded by successive peaks.
3. Resample each stance to 100 frames and run it through the model.

#### TODO

- [ ] Validate the alignment UI
- [ ] Verify the accel and vGRF normalization procedure
- [ ] Fix the tiny timestamp alignment issue
- [ ] Verify the procedure for retrieving stances
- [ ] Tune real-time step detection across sampling rates
- [ ] Stress test with a large batch of subjects and the MLP model
```
