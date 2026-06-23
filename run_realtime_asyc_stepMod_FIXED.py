# run_realtime_SL8_AK526_acquisition_udp_async_dedup.py
# Asynchronous two-process live biofeedback version.
#
# Main change from SL6:
#   The serial reader thread no longer runs step detection or vGRF prediction.
#   It only reads/parses COM3 packets and places accepted data packets into
#   packet_queue. A separate processing worker consumes that queue and performs
#   rolling-buffer step detection, vGRF prediction, saving, and UDP feedback.
#   This prevents slow model/processing code from blocking COM3 reads.
#
# Two-process live biofeedback version.
#
# This script handles ONLY acquisition, step detection, vGRF prediction, and saving.
# It does NOT open a live biofeedback GUI. Instead, whenever a new predicted
# first-peak vGRF is available, it sends a small UDP message to a separate
# display script running on the same computer.
#
# Why this design:
#   1) The previous one-process GUI versions caused severe delay/freezing on Windows.
#   2) The non-feedback acquisition code worked reliably.
#   3) Separating acquisition from visualization protects COM3/serial acquisition if
#      the display window freezes or crashes.
#
# Start this script in one terminal and feedback_display_udp.py in a second terminal.

# ============================================================
# Imports and external dependencies
# ============================================================
import os
import serial
import struct
import time
import threading
import queue
import traceback
import warnings
import socket
import json
import pandas as pd
import utils.read_packets as RP
import matplotlib.pyplot as plt
import numpy as np
from utils.config import PipelineConfig

from utils.stance import StanceAnalyzer
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
from utils.sensor import SensorData, input_df, _process_sensor_df



# ============================================================
# Packet protocol and serial communication settings
# ============================================================
CRC_POLYNOMIAL = 0xAB

HEADER_FORMAT = '<BBBI'
FOOTER_FORMAT = '<B'
DATAPACKET_FORMAT = '<I9hBB'

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # Calculate expected header size
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)  # Calculate expected footer size

PORT = "COM3" # Com port of the device
BAUDRATE = 921600

serial_lock = threading.Lock()
serial_comm = None
# serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
running = True
stop_requested = False
imu_packets = []

csv_filename = 'ESP_stream_data.csv'  # Update this to match the subject and trial name
steps_filename = 'Parsed_steps_data.csv'
vgrf_filename = 'Predicted_vGRF_data.csv'
plot_filename = 'acc_steps_vgrfs.png'
csv_file = csv_filename
steps_file = steps_filename
vgrf_file = vgrf_filename

# ============================================================
# Biofeedback and UDP feedback configuration
# ============================================================
# -----------------------------
# Biofeedback settings
# -----------------------------
ENABLE_BIOFEEDBACK = False  # Keep False: the live GUI is now a separate UDP display process.
ENABLE_UDP_FEEDBACK = True
UDP_FEEDBACK_HOST = "127.0.0.1"
UDP_FEEDBACK_PORT = 5055
PRINT_PREDICTIONS_TO_CONSOLE = False
TARGET_VGRF = 1.15                 # BW
ROLLING_WINDOW_STEPS = 2           # Adjustable: 2, 3, 4, etc.
FIRST_PEAK_WINDOW_FRACTION = 0.50  # First 50% of predicted waveform
BIOFEEDBACK_YMIN = 0.90
BIOFEEDBACK_YMAX = 1.30
feedback_display = None
GENERATE_FINAL_PLOT = True       # Keep False for live testing to avoid any Matplotlib window at shutdown.
SHOW_FINAL_PLOT = True           # If GENERATE_FINAL_PLOT=True, save final plot without opening a window.

# ============================================================
# Asynchronous processing infrastructure
# ============================================================
# Queue used to pass accepted data packets from the serial-reading thread
# to the processing worker. The serial thread must stay lightweight so COM3
# is not blocked by pandas filtering, step detection, or ONNX prediction.
packet_queue = queue.Queue(maxsize=100000)
packet_queue_drop_count = 0
processed_packet_count = 0

# Running step-count display state
last_reported_left_steps = -1
last_reported_right_steps = -1

# Optional diagnostic switch. Keep True for normal operation. Set False only to
# test whether live step detection is stable without ONNX vGRF prediction.
ENABLE_VGRF_PREDICTION = True

# UDP socket used to send predicted step values to the separate UDP feedback display.
# UDP is connectionless and non-blocking for this small local message use case; if
# the display is not running, acquisition continues normally.
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# Rolling processing buffer settings
# -----------------------------
# ESPData / stream_df still stores the FULL trial and is saved at the end.
# For speed, step detection and vGRF prediction are run only on the most recent
# PROCESSING_WINDOW_SECONDS of data. This avoids re-filtering and re-detecting
# steps from the beginning of the trial every 50 packets.
PROCESSING_WINDOW_SECONDS = 8.0       # Recent window used for live processing; shorten if processing is slow.
MIN_PROCESSING_SECONDS = 2.0          # Lower startup delay so predictions begin sooner
PROCESS_EVERY_N_ROWS = 50             # Run analysis every N incoming packets
RESAMPLED_SENSOR_HZ = 100             # _process_sensor_df resamples to ~100 Hz

# Step de-duplication for rolling-window processing.
#
# The same physical step can appear in several overlapping rolling windows.
# The original approach used an exact rounded end-time key, but the detected
# end time can shift by a few samples from one window to the next. That can
# create duplicate saved steps such as R1, R2, R3 for one actual right step.
#
# Here we keep the recent saved end times by side and reject a new step if it
# occurs too close to a previously saved same-side step. For walking, the same
# foot normally contacts again roughly every 0.8-1.2 s, so 0.55 s is a
# conservative duplicate-rejection threshold. Adjust if needed for running or
# unusually high cadence.
processed_step_end_times = {"left": [], "right": []}
SAME_SIDE_MIN_STEP_INTERVAL = 0.55      # seconds; reject same-side steps closer than this
DUPLICATE_STEP_TIME_TOLERANCE = 0.25    # seconds; reject if close to any recent same-side step

# -----------------------------
# Packet debugging settings
# -----------------------------
# These counters are printed during live acquisition so we can tell whether
# the ESP32 is sending actual data packets after S and A. They do not change
# saved outcomes; they only help diagnose the stream.
DEBUG_PACKETS = False
PACKET_DEBUG_PRINT_EVERY = 5000
packet_type_counts = {}
seen_devices = set()  # Track sensors that have started streaming
valid_data_packet_count = 0
crc_fail_count = 0
payload_fail_count = 0
unknown_packet_count = 0
last_debug_total = 0

packet_analysis = True
analysis_window = 1000
Analysis_DF = pd.DataFrame(columns=['DeviceID', 'PacketID', 'Timestamp'])
duplicate_counter = 0
ooo_counter = 0
loss_counter = 0

# Based on configuration
accel_scale = 8.0 / 32768.0
gyro_scale = 250.0 / 32768.0
magneto_scale = 4.0 / 32768.0

#def stop_program(event=None):
    #"""Callback for the STOP button in the biofeedback figure.

    #This stops the Python acquisition loop. It does not automatically send
    #D or Q to the ESP32, so for a normal planned stop use D, Q, then exit.
    #"""
    #global stop_requested, running
    #stop_requested = True
    #running = False
    #print("STOP button pressed")
    

# ============================================================
# UDP feedback messaging
# ============================================================
def send_feedback_udp(side, first_peak_vgrf, step_id=None, acquisition_time=None):
    """Send a predicted step value to the separate UDP UDP feedback display.

    This function intentionally does not block acquisition. The display script
    listens on UDP_FEEDBACK_HOST:UDP_FEEDBACK_PORT and draws the bars. If the
    display is not running, UDP messages are simply dropped by the OS and the
    acquisition script continues saving data.

    Message format:
        {
            "type": "step",
            "side": "left" or "right",
            "value": first_peak_vgrf in BW,
            "step_id": e.g., "L3" or "R3",
            "time": acquisition time in seconds
        }
    """
    if not ENABLE_UDP_FEEDBACK:
        return

    try:
        payload = {
            "type": "step",
            "side": str(side).lower(),
            "value": float(first_peak_vgrf),
            "step_id": step_id,
            "time": None if acquisition_time is None else float(acquisition_time),
        }
        udp_socket.sendto(
            json.dumps(payload).encode("utf-8"),
            (UDP_FEEDBACK_HOST, UDP_FEEDBACK_PORT),
        )
    except Exception as e:
        # Do not let display messaging stop serial acquisition.
        print(f"WARNING: UDP feedback send failed: {e}")

# ============================================================
# Packet parsing and validation
# ============================================================
def handle_info_packet(packet_bytes):
    info_packet = RP.parse_info_packet(packet_bytes)

    # check crc here
    crc_caclulated = RP.calculate_crc8(packet_bytes[:HEADER_SIZE+info_packet['PayloadLen']])
    if(crc_caclulated != info_packet['CRC']):
        # Packet is corrupted. Discard the Data
        return

    # Display information
    print(f"Info Packet=>Time:{info_packet['Timestamp']}, Info:{info_packet['Info']}")

def handle_data_packet(packet_bytes):
    global Analysis_DF, duplicate_counter, ooo_counter, loss_counter, imu_packets
    global valid_data_packet_count, crc_fail_count
    # Parse the data
    data_packet = RP.parse_data_packet(packet_bytes)    

    # check crc here
    crc_caclulated = RP.calculate_crc8(packet_bytes[:HEADER_SIZE+data_packet['PayloadLen']])
    # assert crc_caclulated == data_packet['CRC'], "CRC Check Failed - data packet."
    if crc_caclulated != data_packet['CRC']:
        # Packet is corrupted. Discard the data, but count failures so we can
        # tell whether data packets are arriving but being rejected.
        crc_fail_count += 1
        if DEBUG_PACKETS and crc_fail_count % 100 == 0:
            print(f"CRC failures so far: {crc_fail_count}")
        return

    valid_data_packet_count += 1

    # Report when sensors first begin streaming
    global seen_devices
    device_id = data_packet['DeviceID']
    if device_id not in seen_devices:
        seen_devices.add(device_id)
        print(f"Sensor {device_id} is now streaming.")
        if len(seen_devices) == 3:
            print("All three sensors are streaming and ready.")

    if DEBUG_PACKETS and valid_data_packet_count % PACKET_DEBUG_PRINT_EVERY == 0:
        print(f"Valid data packets received: {valid_data_packet_count}")
    
    # Update the packet with meaningful values
    # Accelerometer conversion
    data_packet['AccelX'] =  data_packet['AccelX'] * accel_scale
    data_packet['AccelY'] =  data_packet['AccelY'] * accel_scale
    data_packet['AccelZ'] =  data_packet['AccelZ'] * accel_scale

    # Gyroscope conversion
    data_packet['GyroX'] =  data_packet['GyroX'] * gyro_scale
    data_packet['GyroY'] =  data_packet['GyroY'] * gyro_scale
    data_packet['GyroZ'] =  data_packet['GyroZ'] * gyro_scale

    # Magnetometer conversion
    data_packet['MagX'] =  data_packet['MagX'] * magneto_scale
    data_packet['MagY'] =  data_packet['MagY'] * magneto_scale
    data_packet['MagZ'] =  data_packet['MagZ'] * magneto_scale

    # Store the packet
    imu_packets.append(data_packet.copy())
    newdata = pd.DataFrame([data_packet])
    # newdata.to_csv(csv_file, mode='a', index=False, header=False)

    if(packet_analysis):
        # Check for Duplicate packet
        duplicate = not Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID']) & (Analysis_DF['PacketID'] == data_packet['PacketID'])].empty
        if(duplicate):
            duplicate_counter+=1
            # print(f"Duplicate Packet (DeviceID={data_packet['DeviceID']}, PacketID={data_packet['PacketID']}) | Counter = {duplicate_counter}")

        # # Check for OOO Packets
        # ooo_packets = not Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID']) & (data_packet['PacketID'] < Analysis_DF['PacketID'])].empty
        # if(ooo_packets):
        #     ooo_counter+=1
        #     print(f"Out of order Packet (DeviceID={data_packet['DeviceID']}, PacketID={data_packet['PacketID']}) | Counter = {ooo_counter}")

        # # Check for Missing packets
        # difference = data_packet['PacketID'] - Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID'])]['PacketID'].tail(1)
        # if not difference.empty and difference.iloc[0] > 1:
        #     print("Difference : ", difference.iloc[0])

        # Append new row after checking everything
        new_row = {'DeviceID':data_packet['DeviceID'], 'PacketID':data_packet['PacketID'], 'Timestamp':data_packet['Timestamp']}
        Analysis_DF = pd.concat([Analysis_DF, pd.DataFrame([new_row])], ignore_index=True)
        # delete older rows
        if len(Analysis_DF) > analysis_window:
            Analysis_DF = Analysis_DF.iloc[-analysis_window:].reset_index(drop=True)

    return data_packet

def parse_header(data):
    return struct.unpack(HEADER_FORMAT, data)

# ============================================================
# Serial acquisition thread
# ============================================================
# Read packets from the ESP32 and place valid data into the processing queue.
def read_serial():
    global running, stop_requested, serial_comm
    global packet_type_counts, payload_fail_count, unknown_packet_count, last_debug_total
    global packet_queue, packet_queue_drop_count
    while running and not stop_requested:
        # print("READ THREAD: Attempting to read header...")
        # Read just the header
        header_bytes = serial_comm.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            continue

        try:
            packet_type, payload_len, device_id, timestamp = parse_header(header_bytes)

            # Count every parsed packet header. PacketType 0x01 = info packet;
            # PacketType 0x02 = data packet. If type 2 never appears after S/A,
            # the issue is upstream of step detection.
            if DEBUG_PACKETS:
                packet_type_counts[packet_type] = packet_type_counts.get(packet_type, 0) + 1
                total_seen = sum(packet_type_counts.values())
                if total_seen - last_debug_total >= PACKET_DEBUG_PRINT_EVERY:
                    print(f"Packet type counts so far: {packet_type_counts}")
                    last_debug_total = total_seen
            # print(f"DEBUG: Received Packet Type: {hex(packet_type)}")

            # ADDED SYNCHRONIZATION LOGIC HERE (OS 12/15/25)
            if packet_type not in [0x01, 0x02]:
                unknown_packet_count += 1
                if DEBUG_PACKETS and unknown_packet_count % 100 == 0:
                    print(f"Unknown packet headers skipped: {unknown_packet_count}")
                serial_comm.read(1) # Discard the first byte and shift the window
                continue

            remaining_bytes = serial_comm.read(payload_len + FOOTER_SIZE)
            
            # Read the payload and footer based on packet_type
            if len(remaining_bytes) == payload_len + FOOTER_SIZE:
                if packet_type == 0x01:
                    if payload_len > 64: # Added to prevent reading massive strings OS 12/17/2025
                        print(f"PROTOCOL ERROR: Info payload too large ({payload_len} bytes). Resyncing.")
                        serial_comm.reset_input_buffer()
                        continue
                    handle_info_packet(header_bytes + remaining_bytes)
                elif packet_type == 0x02:
                    # Data packets MUST have a payload length of 24 bytes
                    if payload_len != 24:
                        payload_fail_count += 1
                        print(
                            f"PROTOCOL ERROR: Data payload size mismatch "
                            f"(Expected 24, Got {payload_len}). Count={payload_fail_count}. Resyncing."
                        )
                        serial_comm.reset_input_buffer()
                        continue
                    packet = handle_data_packet(header_bytes + remaining_bytes)

                    if packet is not None: # Added lines OS 12/16/2025
                        # Success: enqueue the packet for the processing worker.
                        # IMPORTANT: do not run step detection or vGRF prediction
                        # in this serial thread. Those operations are too expensive
                        # and can block COM3 reading, creating backlog/delay.
                        try:
                            packet_queue.put_nowait(packet)
                        except queue.Full:
                            packet_queue_drop_count += 1
                            if packet_queue_drop_count % 100 == 0:
                                print(f"WARNING: packet_queue full; dropped packets={packet_queue_drop_count}")
                    else:
                        serial_comm.reset_input_buffer()
                        print("STREAM RESYNC: Cleared input buffer after data unpacking error.")
                        time.sleep(0.001) #Sleep for 1 millisecond (prevents CPU overload)

                    # TODO: here is where I expect we need to process packets in real time. These are global 
                    # variables, so they should still be present for exporting at the end. 
                    # ESPData, steps_df, vgrf_df, axes = process_packet(packet, ESPData, steps_df, vgrf_df, axes)
                else:
                    # Received unknown packets
                    pass
        
        except struct.error:
            print("Error unpacking data")

# ============================================================
# Command transmission to ESP32
# ============================================================
# Send user commands (S/A/D/Q) to the ESP32.
def send_data(message):
    """Send a command string to the ESP32 over serial.

    Commands are interpreted by the ESP32 firmware, not by Python. The device
    advertises S=START, Q=QUIT, A=ACTIVATE, D=DEACTIVATE.
    """
    global running, stop_requested
    with serial_lock:
        if serial_comm is None or not serial_comm.is_open:
            print("Serial port is not open; command was not sent.")
            return
        try:
            data_bytes = message.strip().encode() + b'\n'
            serial_comm.write(data_bytes)
            print(f"Sent: {data_bytes}")
        except serial.SerialTimeoutException as e:
            print(f"Serial write timeout while sending {message!r}: {e}")
            running = False
            stop_requested = True
        except serial.SerialException as e:
            print(f"Serial write failed while sending {message!r}: {e}")
            running = False
            stop_requested = True


# ============================================================
# User command thread
# ============================================================
# Listen for terminal commands without blocking acquisition.
def command_input_loop():
    """Read user commands without blocking the main GUI/update loop.

    This runs in a daemon thread. The main thread remains free to update the
    biofeedback figure safely.
    """
    global running, stop_requested
    while running and not stop_requested:
        try:
            user_input = input(">>").strip()
        except (EOFError, KeyboardInterrupt):
            running = False
            stop_requested = True
            break

        if user_input == "":
            continue

        if user_input.lower() == 'exit':
            running = False
            stop_requested = True
            break

        send_data(user_input)
        time.sleep(0.05)



# ============================================================
# Background processing worker
# ============================================================
# Process queued packets, detect steps, predict vGRFs, and send feedback.
def processing_worker():
    """Consume accepted data packets and run the expensive processing pipeline.

    This worker is intentionally separate from read_serial(). The serial thread
    should only read and parse COM3 packets; this worker handles pandas storage,
    rolling-buffer step detection, ONNX vGRF prediction, CSV dataframe updates,
    and UDP feedback messaging.

    If vGRF prediction is slow, the packet queue may grow, but COM3 reading will
    continue. This lets us diagnose processing delay without destabilizing the
    serial port.
    """
    global running, stop_requested, ESPData, steps_df, vgrf_df, axes
    global packet_queue, processed_packet_count

    last_report = time.time()

    while running or not packet_queue.empty():
        try:
            packet = packet_queue.get(timeout=0.10)
        except queue.Empty:
            continue

        try:
            ESPData, steps_df, vgrf_df, axes = process_packet(
                packet,
                ESPData,
                steps_df,
                vgrf_df,
                axes,
                feedback_events=None
            )
            processed_packet_count += 1

            # Low-rate health report to see whether processing is falling behind.
            now = time.time()
            if now - last_report > 5.0:
                print(
                    f"Processing health: processed={processed_packet_count}, "
                    f"queue_backlog={packet_queue.qsize()}, "
                    f"steps={len(steps_df)}, vGRF={len(vgrf_df)}"
                )
                last_report = now

        except Exception as e:
            print(f"ERROR during packet processing worker: {e}")
            traceback.print_exc()
            running = False
            stop_requested = True
            break

        finally:
            try:
                packet_queue.task_done()
            except Exception:
                pass


# ============================================================
# Step detection helper functions
# ============================================================
def safe_extract_sensor_stances(SA, side_name, leg_filtered, leg_raw, waist_filtered):
    """Run StanceAnalyzer.extract_sensor_stances with a safety guard.

    During live/rolling-window processing, a detected strike can occur near the
    edge of the current window and create an empty waist slice. The original
    normalize_stance() call then fails with "array of sample points is empty".
    Rather than crashing the serial thread, skip that update and wait for the
    next buffer.
    """
    try:
        return SA.extract_sensor_stances(leg_filtered, leg_raw, waist_filtered)
    except ValueError as e:
        if 'array of sample points is empty' in str(e):
            print(f"WARNING: Empty {side_name} stance slice in rolling window; skipping this update.")
            return [], []
        raise


def should_save_step(side, end_time):
    """Return True if this detected step should be saved.

    Because live processing uses overlapping rolling windows, the same physical
    step may be detected repeatedly with slightly different end times. Saving
    all of those detections makes the feedback bars change too quickly and
    creates apparent sequences like R, R, R before the next L.

    This function rejects:
      1) any same-side step close to a previously saved same-side end time, and
      2) any same-side step occurring sooner than SAME_SIDE_MIN_STEP_INTERVAL
         after the last saved step for that side.

    This does not change the vGRF calculation for accepted steps. It only
    prevents duplicate predictions from being saved/sent for the same physical
    step.
    """
    global processed_step_end_times

    side = str(side).lower()
    end_time = float(end_time)

    if side not in processed_step_end_times:
        processed_step_end_times[side] = []

    recent_times = processed_step_end_times[side]

    # Reject if this end time is near any recent saved same-side end time.
    for prev_time in recent_times[-10:]:
        if abs(end_time - prev_time) <= DUPLICATE_STEP_TIME_TOLERANCE:
            return False

    # Reject if it is too soon after the last saved same-side step.
    if recent_times and (end_time - recent_times[-1]) < SAME_SIDE_MIN_STEP_INTERVAL:
        return False

    recent_times.append(end_time)
    return True

# ============================================================
# Main gait analysis and vGRF prediction pipeline
# ============================================================
# Main gait-processing pipeline for step detection and vGRF estimation.
def process_packet(packet, stream_df, steps_df, vgrf_df, axes, feedback_events=None, verbose=False, plot=True):
    '''
    Process a single incoming packet and append to general and step dataframes.

    Args:
        packet: Dictionary containing packet data
        stream_df: DataFrame to append raw packet data
        steps_df: DataFrame to append detected steps
        vgrf_df: DataFrame to append vertical ground reaction forces
        axes: Array of Axes objects for plotting
        verbose: If True, print debug information
        plot: If True, update plots with new data

    Returns:
        Updated stream_df, steps_df, vgrf_df, and figure object
    '''

    # append packet to dataframe
    # if type(stream_df) is not pd.DataFrame:
    #     return stream_df, steps_df, vgrf_df, axes
    
    # if packet == None:
    #     return stream_df, steps_df, vgrf_df, axes

    if len(stream_df) == 0:
        elapsed_time = 0.0
    else:
        elapsed_time = (packet['Timestamp'] - stream_df['Timestamp'].min()) / 1000.0 # in seconds

    # print('Processing packets')
    stream_df.loc[len(stream_df)] = [packet['PacketType'], packet['PayloadLen'], packet['DeviceID'], 
                       packet['Timestamp'], packet['PacketID'], 
                    #    [packet['accel'][0], packet['accel'][1], packet['accel'][2]],
                       [packet['AccelX'], packet['AccelY'], packet['AccelZ']], # combine accel into list for norm calcs later
                    #    packet['GyroX'], packet['GyroY'], packet['GyroZ'],  # omit gyroscope
                    #    packet['MagX'], packet['MagY'], packet['MagZ'],  # omit magnetometer
                       packet['Flags'], packet['Battery'], packet['CRC'], elapsed_time]
    

    # ------------------------------------------------------------------
    # Live processing schedule
    # ------------------------------------------------------------------
    # The full stream_df is kept intact for saving, but analysis is not run
    # on the entire growing dataframe. Instead, analysis is run every
    # PROCESS_EVERY_N_ROWS packets using only a recent rolling window.
    # This keeps processing speed roughly constant throughout long trials.
    if len(stream_df) % PROCESS_EVERY_N_ROWS != 0:
        return stream_df, steps_df, vgrf_df, axes

    if elapsed_time < MIN_PROCESSING_SECONDS:
        return stream_df, steps_df, vgrf_df, axes

    processing_start_time = max(0.0, elapsed_time - PROCESSING_WINDOW_SECONDS)
    processing_df = stream_df[stream_df['time'] >= processing_start_time].copy()

    # If the current rolling buffer does not contain all three sensors yet,
    # skip this update. This can happen immediately after starting streaming
    # or if a sensor temporarily drops out.
    if processing_df.empty or processing_df['DeviceID'].nunique() < 3:
        return stream_df, steps_df, vgrf_df, axes

    # process sensor data (parse, resample, and filter)
    Sensors = input_df(processing_df)
    if verbose:
        print(Sensors.keys(), Sensors['left'].shape, Sensors['right'].shape, Sensors['waist'].shape)

    ProcessedSensors = {}
    SensorNames = ['left', 'right', 'waist']
    for s in SensorNames:

        # Added check for data analysis crash OS 12/16/2025
        if Sensors[s].empty: 
            print(f"DEBUG: Sensor data for {s} is empty. Skipping analysis for this packet.")
            return stream_df, steps_df, vgrf_df, axes
        
        ProcessedSensors[s] = _process_sensor_df(Sensors[s])
    if any(ProcessedSensors[s] is None for s in SensorNames):
        print("Error processing sensor data")
        return stream_df, steps_df, vgrf_df, axes
    # print(ProcessedSensors)

    # set gait cycle identification parameters
    config = PipelineConfig(
        accel_peak_params={
            'height': 1.0,
            'prominence': 0.5,
            'width': 5.0,
            'distance': 10
        },
        jerk_peak_params={
            'height': 0.0,
            'prominence': 0.1
        },
        jerk_window_size=50,
        stance_matching_time_threshold=50,
        accel_filters=[],
        vgrf_filters=[],
        min_stance_size=60,
        max_stance_size=140
        )

    # identify steps and parse gait cycles
    # print('Searching for steps...')
    SA = StanceAnalyzer(config)
    left_strikes, left_stances = safe_extract_sensor_stances(
        SA,
        'left',
        ProcessedSensors['left']['accel_filtered'],
        ProcessedSensors['left']['accel'],
        ProcessedSensors['waist']['accel_filtered'],
    )
        
    right_strikes, right_stances = safe_extract_sensor_stances(
        SA,
        'right',
        ProcessedSensors['right']['accel_filtered'],    
        ProcessedSensors['right']['accel'],
        ProcessedSensors['waist']['accel_filtered'],
    )
    left_strike_times = [ProcessedSensors['left']['time'][i] for i in left_strikes]
    right_strike_times = [ProcessedSensors['right']['time'][i] for i in right_strikes]

    if elapsed_time > 5: # and float(str(elapsed_time).split('.')[1]) < 0.1:
        global last_reported_left_steps, last_reported_right_steps

        left_steps = len(steps_df[steps_df['Side'] == 'left'])
        right_steps = len(steps_df[steps_df['Side'] == 'right'])
        total_steps = left_steps + right_steps

        if (left_steps != last_reported_left_steps or
                right_steps != last_reported_right_steps):

            print(
                f"{elapsed_time:.1f} s | "
                f"Frames: {len(stream_df)} | "
                f"Left: {left_steps} | "
                f"Right: {right_steps} | "
                f"Total: {total_steps}"
            )

            last_reported_left_steps = left_steps
            last_reported_right_steps = right_steps

        # print(f"Left stances: {len(left_stances)}  Right stances: {len(right_stances)}")


    # log output in steps df
    new_l_inds = []
    new_r_inds = []
    # print(f"DEBUG: Checking strikes. Left: {len(left_strikes)}, Right: {len(right_strikes)}")
    if len(left_strikes) > 2 and len(left_stances) > 0:
        side = 'left'
        left_times = np.asarray(ProcessedSensors['left']['time'])
        start_time = float(left_times[left_strikes[-2]])
        end_time = float(left_times[left_strikes[-1]])

        if should_save_step(side, end_time):
            # Save frame indices as approximate global 100-Hz frames. This is
            # more stable than using local rolling-window indices.
            start_frame = int(round(start_time * RESAMPLED_SENSOR_HZ))
            end_frame = int(round(end_time * RESAMPLED_SENSOR_HZ))
            id = 'L' + str(len(steps_df[steps_df['Side'] == 'left']))
            waist_data = list(left_stances[-1])
            new_l_inds.append(len(steps_df))
            steps_df.loc[len(steps_df)] = [elapsed_time, side, start_frame, end_frame, id] + waist_data

    if len(right_strikes) > 2 and len(right_stances) > 0:
        side = 'right'
        right_times = np.asarray(ProcessedSensors['right']['time'])
        start_time = float(right_times[right_strikes[-2]])
        end_time = float(right_times[right_strikes[-1]])

        if should_save_step(side, end_time):
            # Save frame indices as approximate global 100-Hz frames. This is
            # more stable than using local rolling-window indices.
            start_frame = int(round(start_time * RESAMPLED_SENSOR_HZ))
            end_frame = int(round(end_time * RESAMPLED_SENSOR_HZ))
            id = 'R' + str(len(steps_df[steps_df['Side'] == 'right']))
            waist_data = list(right_stances[-1])
            new_r_inds.append(len(steps_df))
            steps_df.loc[len(steps_df)] = [elapsed_time, side, start_frame, end_frame, id] + waist_data


    # send to model for prediction
    for l_ind in new_l_inds:
        waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]

        # predict_stance() expects one step as a 1D 100-point input.
        # Passing a one-row DataFrame causes utils.predict to add another
        # batch dimension, producing a 3D ONNX input.
        l_waist_acc = steps_df.loc[l_ind, waist_acc_cols].astype(float)
        l_waist_acc.index = range(0, 100)

        if ENABLE_VGRF_PREDICTION:
            from utils.predict import predict_stance
            pred_start = time.perf_counter()
            l_output = predict_stance(l_waist_acc)
            pred_ms = (time.perf_counter() - pred_start) * 1000.0
            l_output = np.asarray(l_output).flatten()

            # First peak from the first 50% of the predicted waveform
            peak_end = int(len(l_output) * FIRST_PEAK_WINDOW_FRACTION)
            first_peak_vgrf = l_output[:peak_end].max()
        else:
            # Diagnostic mode: no ONNX inference. Save NaNs so step detection can
            # be tested without the prediction model.
            l_output = np.full(100, np.nan)
            first_peak_vgrf = np.nan
            pred_ms = 0.0

        # save output to vgrfs_df
        vgrf_df.loc[len(vgrf_df)] = [steps_df['ID'].iloc[l_ind], first_peak_vgrf] + list(l_output)

        # Send predicted step value to the separate UDP feedback display.
        # This does not affect saving: vgrf_df above still stores the peak and full waveform.
        send_feedback_udp('left', first_peak_vgrf, step_id=steps_df['ID'].iloc[l_ind], acquisition_time=elapsed_time)
        if PRINT_PREDICTIONS_TO_CONSOLE:
            print(f"UDP feedback: LEFT {steps_df['ID'].iloc[l_ind]} FirstPeakvGRF={first_peak_vgrf:.3f} BW | ONNX={pred_ms:.1f} ms")

    for r_ind in new_r_inds:
        waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]

        # predict_stance() expects one step as a 1D 100-point input.
        # Passing a one-row DataFrame causes utils.predict to add another
        # batch dimension, producing a 3D ONNX input.
        r_waist_acc = steps_df.loc[r_ind, waist_acc_cols].astype(float)
        r_waist_acc.index = range(0, 100)

        if ENABLE_VGRF_PREDICTION:
            from utils.predict import predict_stance
            pred_start = time.perf_counter()
            r_output = predict_stance(r_waist_acc)
            pred_ms = (time.perf_counter() - pred_start) * 1000.0
            r_output = np.asarray(r_output).flatten()

            # First peak from the first 50% of the predicted waveform
            peak_end = int(len(r_output) * FIRST_PEAK_WINDOW_FRACTION)
            first_peak_vgrf = r_output[:peak_end].max()
        else:
            # Diagnostic mode: no ONNX inference. Save NaNs so step detection can
            # be tested without the prediction model.
            r_output = np.full(100, np.nan)
            first_peak_vgrf = np.nan
            pred_ms = 0.0

        # save output to vgrfs_df
        vgrf_df.loc[len(vgrf_df)] = [steps_df['ID'].iloc[r_ind], first_peak_vgrf] + list(r_output)

        # Send predicted step value to the separate UDP feedback display.
        # This does not affect saving: vgrf_df above still stores the peak and full waveform.
        send_feedback_udp('right', first_peak_vgrf, step_id=steps_df['ID'].iloc[r_ind], acquisition_time=elapsed_time)
        if PRINT_PREDICTIONS_TO_CONSOLE:
            print(f"UDP feedback: RIGHT {steps_df['ID'].iloc[r_ind]} FirstPeakvGRF={first_peak_vgrf:.3f} BW | ONNX={pred_ms:.1f} ms")


    # plot results
    # if time > 10 and plot:

    #     ax1 = axes[0]  # Top subplot
    #     ax2 = axes[1]  # Middle subplot
    #     ax3 = axes[2]  # Bottom subplot
    #     A1 = 0.5
    #     A2 = 0.3

    #     # realtime plot of input acceleration data
    #     ax1.plot(ProcessedSensors['left']['time'], ProcessedSensors['left']['accel_filtered'], 
    #              color='C0', label='Left Accel Filtered', alpha=A1)
    #     ax1.plot(ProcessedSensors['right']['time'], ProcessedSensors['right']['accel_filtered'], 
    #              color='C1', label='Right Accel Filtered', alpha=A1)
    #     ax1.plot(ProcessedSensors['waist']['time'], ProcessedSensors['waist']['accel_filtered'], 
    #              color='C2', label='Waist Accel Filtered', alpha=A1)
    #     for x in left_strike_times:
    #         ax1.axvline(x=x, color='C0', linestyle='--', alpha=A1)
    #     for x in right_strike_times:
    #         ax1.axvline(x=x, color='C1', linestyle='--', alpha=A1)
    #     # ax1.legend(fontsize='small')
    #     ax1.set_title('Streaming Acceleration Data')
    #     ax1.set_ylabel('Accel (g)')
    #     ax1.set_xlabel('Time (s)')

    #     # extracted & parsed waist acceleration to send to model
    #     waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]
    #     l_waist_acc = steps_df[waist_acc_cols].iloc[new_l_inds]
    #     l_waist_acc.columns = range(0, 100)
    #     r_waist_acc = steps_df[waist_acc_cols].iloc[new_r_inds]
    #     r_waist_acc.columns = range(0, 100)
    #     ax2.plot(l_waist_acc.T, color='C0', alpha=A2, lw=2)
    #     ax2.plot(r_waist_acc.T, color='C1', alpha=A2, lw=2)
    #     ax2.set_title('Step-Parsed Waist Accelerations')
    #     ax2.set_ylabel('Accel (g)')
    #     ax2.set_xlabel('% Gait Cycle')

    #     # model predicted vGRF
    #     vgrf_cols = [col for col in vgrf_df.columns if 'vGRF_' in col]
    #     l_vgrf = vgrf_df[vgrf_cols].iloc[new_l_inds]
    #     l_vgrf.columns = range(0, 100)
    #     r_vgrf = vgrf_df[vgrf_cols].iloc[new_r_inds]
    #     r_vgrf.columns = range(0, 100)
    #     ax3.plot(l_vgrf.T, color='C0', alpha=A2, lw=2)
    #     ax3.plot(r_vgrf.T, color='C1', alpha=A2, lw=2)
    #     ax3.set_title('Step-Predicted vGRFs')
    #     ax3.set_ylabel('vGRF (BW)')
    #     ax3.set_xlabel('% Gait Cycle')

    #     # plt.show(block=False) # Commented out OS 12/16/2025
    #     plt.tight_layout()
    #     plt.savefig('acc_steps_preds.png')

        
    return stream_df, steps_df, vgrf_df, axes

# ============================================================
# Final plotting and summary output
# ============================================================
# Generate and save the end-of-trial summary figure.
def generate_final_plot(ESPData, steps_df, vgrf_df, folder_path): # Taken out of process_packets for stability OS 12/17/2025
    """
    Generates the final summary plot using the original logic from process_packet.

    This function is intentionally conservative: if no valid data packets or no
    steps were saved, it skips plotting rather than raising an error at shutdown.
    """

    if ESPData.empty:
        print("Final plot skipped: ESPData is empty, so no valid data packets were saved.")
        return

    if steps_df.empty or vgrf_df.empty:
        print("Final plot skipped: no parsed steps or predicted vGRFs were saved.")
        return

    # Re-process full dataset to get filtered lines for the top plot
    Sensors = input_df(ESPData.copy())
    ProcessedSensors = {}
    for s in ['left', 'right', 'waist']:
        ProcessedSensors[s] = _process_sensor_df(Sensors[s])

    # Setup the figure
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
    ax1 = axes[0]  # Top subplot
    ax2 = axes[1]  # Middle subplot
    ax3 = axes[2]  # Bottom subplot
    A1 = 0.5
    A2 = 0.3

    # Plot 1: acceleration data
    ax1.plot(ProcessedSensors['left']['time'], ProcessedSensors['left']['accel_filtered'], 
                color='C0', label='Left Accel Filtered', alpha=A1)
    ax1.plot(ProcessedSensors['right']['time'], ProcessedSensors['right']['accel_filtered'], 
                color='C1', label='Right Accel Filtered', alpha=A1)
    ax1.plot(ProcessedSensors['waist']['time'], ProcessedSensors['waist']['accel_filtered'], 
                color='C2', label='Waist Accel Filtered', alpha=A1)
    # Vertical lines for all detected steps
    left_strike_times = steps_df[steps_df['Side'] == 'left']['Timestamp']
    right_strike_times = steps_df[steps_df['Side'] == 'right']['Timestamp']
    for x in left_strike_times:
        ax1.axvline(x=x, color='C0', linestyle='--', alpha=A1)
    for x in right_strike_times:
        ax1.axvline(x=x, color='C1', linestyle='--', alpha=A1)
    # ax1.legend(fontsize='small')
    ax1.set_title('Streaming Acceleration Data')
    ax1.set_ylabel('Accel (g)')
    ax1.set_xlabel('Time (s)')

    # Plot 2: extracted & parsed waist acceleration to send to model
    waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]
    l_waist_acc = steps_df[steps_df['Side'] == 'left'][waist_acc_cols]
    r_waist_acc = steps_df[steps_df['Side'] == 'right'][waist_acc_cols]
    l_waist_acc.columns = range(0, 100)
    r_waist_acc.columns = range(0, 100)
    ax2.plot(l_waist_acc.T, color='C0', alpha=A2, lw=2)
    ax2.plot(r_waist_acc.T, color='C1', alpha=A2, lw=2)
    ax2.set_title('Step-Parsed Waist Accelerations')
    ax2.set_ylabel('Accel (g)')
    ax2.set_xlabel('% Gait Cycle')

    # Plot 3: model predicted vGRF
    vgrf_cols = [col for col in vgrf_df.columns if 'vGRF_' in col]
    l_vgrf = vgrf_df[vgrf_df['ID'].str.startswith('L')][vgrf_cols]
    r_vgrf = vgrf_df[vgrf_df['ID'].str.startswith('R')][vgrf_cols]
    l_vgrf.columns = range(0, 100)
    r_vgrf.columns = range(0, 100)
    ax3.plot(l_vgrf.T, color='C0', alpha=A2, lw=2)
    ax3.plot(r_vgrf.T, color='C1', alpha=A2, lw=2)
    ax3.set_title('Step-Predicted vGRFs')
    ax3.set_ylabel('vGRF (BW)')
    ax3.set_xlabel('% Gait Cycle')

    plt.tight_layout()
    save_path = os.path.join(folder_path, plot_filename)
    plt.savefig(save_path)
    print(f"plot saved as '{save_path}'")
    plt.close(fig)

# ============================================================
# Program entry point and runtime control
# ============================================================
# Initialize acquisition, worker threads, saving, and shutdown logic.
def main(run):
    """Main function to run real-time data processing from serial port or CSV file.
    args:
        run: 
            'serial' to read sensor data in real time from serial port, 
            'csv' to read from CSV file and play through the data as if real-time.
    """
    global serial_comm, running, ESPData, steps_df, vgrf_df, axes, stop_requested
    global csv_file, steps_file, vgrf_file, imu_packets, feedback_display
    global processed_step_end_times, packet_queue, packet_queue_drop_count, processed_packet_count
    global packet_type_counts, valid_data_packet_count, crc_fail_count
    global payload_fail_count, unknown_packet_count, last_debug_total

    # Reset all run-level state each time the script starts. This prevents
    # previous failed attempts from carrying counters or packet lists forward.
    running = True
    stop_requested = False
    imu_packets = []
    processed_step_end_times = {"left": [], "right": []}
    packet_queue = queue.Queue(maxsize=100000)
    packet_queue_drop_count = 0
    processed_packet_count = 0
    packet_type_counts = {}
    valid_data_packet_count = 0
    crc_fail_count = 0
    payload_fail_count = 0
    unknown_packet_count = 0
    last_debug_total = 0

    # 1. Prompt for Folder Name (generalized)
    input_num = input("What would you like to name this session? ")
    folder_name = f"{input_num}"

    # 1. Prompt for NCBC Study
    # sub_num = input("Enter Subject #: ")
    # trial_num = input("Enter Trial #: ")
    # folder_name = f"ncbc_s{sub_num}_{trial_num}"

    # 2. Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

    # 3. Update file paths to be inside that folder
    csv_file = os.path.join(folder_name, csv_filename)
    steps_file = os.path.join(folder_name, steps_filename)
    vgrf_file = os.path.join(folder_name, vgrf_filename)

    # Initialize dataframes to store results ONCE at the start (moved outside of while True OS 12/17/2025)
    w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
    steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
    vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
    vgrf_df = pd.DataFrame(columns=['ID','FirstPeakvGRF'] + vgrf_cols)
    ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
                                'Timestamp', 'PacketID', 'accel', 
                                'Flags', 'Battery', 'CRC', 'time'])
    
    axes = None

    # No live GUI is created in this acquisition script. Biofeedback is shown by
    # feedback_display_udp.py, which receives UDP messages from send_feedback_udp().
    feedback_display = None

    # # create figure and subplots for showing data
    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # fig.set_figheight(10) 
    # fig.set_figwidth(8)

    try:
        if run == 'serial':
            # running in serial mode
            if serial_comm is None:
                try:
                    print(f"Connecting to {PORT} at {BAUDRATE} baud...")
                    # write_timeout prevents the program from hanging indefinitely
                    # if the device disconnects or the port stops accepting data.
                    serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1, write_timeout=1)
                    serial_comm.reset_input_buffer() # buffer clearing
                    print("Serial buffers reset.")
                except serial.SerialException as e:
                    print(f"Could not open serial port {PORT}: {e}")
                    print("Check the COM port, close other serial programs, and unplug/replug the ESP32.")
                    return

            reader_thread = threading.Thread(target=read_serial, daemon=True)
            reader_thread.start()

            processing_thread = threading.Thread(target=processing_worker, daemon=True)
            processing_thread.start()

            command_thread = threading.Thread(target=command_input_loop, daemon=True)
            command_thread.start()

            print("Reader thread started.")
            print("Processing worker started.")
            print("Commands: S=START, A=ACTIVATE, D=DEACTIVATE, Q=QUIT, exit=save/close Python.")

            # Main thread loop: serial reading and data processing are now in
            # separate threads. This loop only monitors thread health.
            while running and not stop_requested:
                time.sleep(0.10)
                if not reader_thread.is_alive():
                    print("Serial reader thread stopped.")
                    running = False
                    break
                if not processing_thread.is_alive():
                    print("Processing worker stopped.")
                    running = False
                    break
                # # print('running: ', running)

                # else:
                #     print('serial port connected, continuing...')


                    # w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
                    # steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
                    # vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
                    # vgrf_df = pd.DataFrame(columns=['ID','FirstPeakvGRF'] + vgrf_cols)
                    # ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
                    #                         'Timestamp', 'PacketID', 'accel', 
                    #                         'Flags', 'Battery', 'CRC', 'time'])
                    # fig, axes = plt.subplots(nrows=3, ncols=1)
                    # fig.set_figheight(10) 
                    # fig.set_figwidth(8)


                # plt.tight_layout()
                # plt.show()
                # fig.savefig('acc_steps_preds.png')


        elif run == 'csv':
            # load data to simulate real-time processing loop
            # fn = 'IMU_data_RevB_v3_09112025_Walk1.csv'
            fn = input("Enter the input CSV filename (e.g., my_data.csv): ")
            if not os.path.exists(fn):
                print(f"Error: The file '{fn}' was not found.")
                return 
            
            data = pd.read_csv(fn)
            print(f'loading csv data from: {fn}')
            print(data.head())

            # w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
            # steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
            # vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
            # vgrf_df = pd.DataFrame(columns=['ID','FirstPeakvGRF'] + vgrf_cols)

            # ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
            #                         'Timestamp', 'PacketID', 'accel', 
            #                         'Flags', 'Battery', 'CRC', 'time'])
            
            # create figure and subplots for showing data
            # fig, axes = plt.subplots(nrows=3, ncols=1)
            # fig.set_figheight(10) 
            # fig.set_figwidth(8)

            # simulate data reception from an existing CSV as fast as possible.
            # This mode is only for offline testing/debugging. It no longer delays
            # rows based on CSV timestamps because live feedback will be driven by
            # the sensors in serial mode.
            print('\nStarting CSV simulation as fast as possible...')

            for index, row in data.iterrows():
                if stop_requested:
                    print("Stopping simulation early...")
                    break
                
                packet = row.to_dict()

                if len(packet) == 0:
                    continue

                if index % 500 == 0:
                    print(f"Processing row {index}...")

                ESPData, steps_df, vgrf_df, axes = process_packet(
                    packet,
                    ESPData,
                    steps_df,
                    vgrf_df,
                    axes,
                    feedback_events=None
                )

            print("\nSimulation complete. Processing final results...")

                # if index > 14000:
                #     print("\nEnding simulation.")
                #     print('Streaming DF: \n', ESPData.tail())
                #     print('Steps DF: \n', steps_df.tail())
                #     print('vGRF DF: \n', vgrf_df.tail())

                #     ax1 = axes[0] 
                #     window = 10
                #     if max(ESPData['time']) > window:
                #         ax1.set_xlim([max(ESPData['time']) - window, max(ESPData['time'])])

                #     plt.tight_layout()
                #     plt.close(fig)
                #     fig.savefig('acc_steps_preds.png')

                #     print('Ending Simulation early')
                #     break

                # break # exit csv running when all rows ran through
                
        else:
            raise ValueError("Invalid run mode. Use 'serial' or 'csv'.")
            
            
    except KeyboardInterrupt:
        pass

    finally:
    # if not running:
        if run == 'serial':
            running = False
            stop_requested = True
            if serial_comm is not None and serial_comm.is_open:
                try:
                    serial_comm.close()
                    print("Serial port closed.")
                    serial_comm = None
                except serial.SerialException as e:
                    print(f"Serial port close failed: {e}")

        if DEBUG_PACKETS:
            print("Packet debug summary:")
            print(f"  Packet queue backlog at shutdown: {packet_queue.qsize() if 'packet_queue' in globals() else 'NA'}")
            print(f"  Packet queue dropped: {packet_queue_drop_count if 'packet_queue_drop_count' in globals() else 'NA'}")
            print(f"  Packets processed by worker: {processed_packet_count if 'processed_packet_count' in globals() else 'NA'}")
            print(f"  Packet type counts: {packet_type_counts}")
            print(f"  Valid data packets: {valid_data_packet_count}")
            print(f"  CRC failures: {crc_fail_count}")
            print(f"  Payload length failures: {payload_fail_count}")
            print(f"  Unknown packet headers: {unknown_packet_count}")

        # csv_file = 'ESP_stream_data.csv' # moved to beginning of script OS 12/16/2025
        # steps_file = 'Parsed_steps_data.csv'
        # vgrf_file = 'Predicted_vGRF_data.csv'

        # Save CSV Data
        ESPData.to_csv(csv_file, index=False)
        steps_df.to_csv(steps_file, index=False)
        vgrf_df.to_csv(vgrf_file, index=False)
        print(f"Data saved to {csv_file} and {steps_file} and {vgrf_file}")

        # Save IMU file    
        if imu_packets:
            print("Saving expanded IMU data...")
            imu_df = pd.DataFrame(imu_packets)
            target_cols = ['PacketType', 'PayloadLen', 'DeviceID', 'Timestamp', 'PacketID', 
                           'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 
                           'MagX', 'MagY', 'MagZ', 'Flags', 'Battery', 'CRC']
            imu_df = imu_df[[c for c in target_cols if c in imu_df.columns]]

            # Save to the folder
            imu_path = os.path.join(folder_name, 'IMU_data_RevB_v3.csv')
            imu_df.to_csv(imu_path, index=False)
            print(f"Expanded IMU data saved to {imu_path}")

        if GENERATE_FINAL_PLOT:
            print("Generating final summary plot...")
            try:
                generate_final_plot(ESPData, steps_df, vgrf_df, folder_name)
            except Exception as e:
                print(f"Could not generate plot: {e}")
        else:
            print("Skipping final summary plot because GENERATE_FINAL_PLOT = True.")
        # Close UDP socket at shutdown. This does not affect the display process.
        try:
            udp_socket.close()
        except Exception:
            pass
        

if __name__ == "__main__":
    main(run='serial') # use 'csv' only for offline testing/debugging