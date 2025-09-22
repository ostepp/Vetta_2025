import serial
import struct
import time
import threading
import pandas as pd
import numpy as np
from .config import PipelineConfig
import matplotlib.pyplot as plt

from utils.stance import StanceAnalyzer
from utils.sensor import SensorData, input_df, _process_sensor_df

# written by: Ratan Gundami, 2025

CRC_POLYNOMIAL = 0xAB

HEADER_FORMAT = '<BBBI'
FOOTER_FORMAT = '<B'
DATAPACKET_FORMAT = '<I9hBB'

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # Calculate expected header size
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)  # Calculate expected footer size

PORT = "COM8" # Com port of the device
BAUDRATE = 921600

serial_lock = threading.Lock()
serial_comm = None
running = True

packet_analysis = True
analysis_window = 1000 # n frames to hold in buffer for step analysis and vgrf predictions
Analysis_DF = pd.DataFrame(columns=['DeviceID', 'PacketID', 'Timestamp'])
duplicate_counter = 0
ooo_counter = 0
loss_counter = 0

# Based on configuration
accel_scale = 2.0 / 32768.0
gyro_scale = 250.0 / 32768.0
magneto_scale = 4.0 / 32768.0

csv_file = "IMU_data_RevB_v3.csv"
ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 'Timestamp', 'PacketID', 
                                'AccelX', 'AccelY', 'AccelZ', 
                                'GyroX', 'GyroY', 'GyroZ', 
                                'MagX', 'MagY', 'MagZ', 
                                'Flags', 'Battery', 'CRC'])
ESPData.to_csv(csv_file, index=False)


def calculate_crc8(data):
    crc = CRC_POLYNOMIAL
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc

def parse_header(data):
    return struct.unpack(HEADER_FORMAT, data)

def parse_footer(data):
    return struct.unpack(FOOTER_FORMAT, data)

def parse_info_packet(data):
    # parse header
    header = data[:HEADER_SIZE]
    packet_type, payload_len, device_id, timestamp = parse_header(header)

    # Parse footer
    footer = data[HEADER_SIZE+payload_len:HEADER_SIZE+payload_len+FOOTER_SIZE]
    crc = parse_footer(footer)[0]

    # parse payload
    payload = data[HEADER_SIZE:HEADER_SIZE+payload_len]
    info_str = payload.decode('utf-8', errors='ignore').rstrip('\x00')

    return {'PacketType': packet_type, 'PayloadLen': payload_len, 'DeviceID': device_id,
        'Timestamp': timestamp, 'CRC': crc, 'Info': info_str}

def parse_data_packet(data):
    # parse header
    header = data[:HEADER_SIZE]
    packet_type, payload_len, device_id, timestamp = parse_header(header)

    # Parse footer
    footer = data[HEADER_SIZE+payload_len:HEADER_SIZE+payload_len+FOOTER_SIZE]
    crc = parse_footer(footer)[0]

    # Parse Payload
    payload = data[HEADER_SIZE : HEADER_SIZE+payload_len]
    data_payload = struct.unpack(DATAPACKET_FORMAT, payload)
    return {
        'PacketType': packet_type, 'PayloadLen': payload_len, 'DeviceID': device_id,
        'Timestamp': timestamp,  'PacketID': data_payload[0],
        'AccelX': data_payload[1], 'AccelY': data_payload[2], 'AccelZ': data_payload[3],
        'GyroX': data_payload[4], 'GyroY': data_payload[5], 'GyroZ': data_payload[6],
        'MagX': data_payload[7], 'MagY': data_payload[8], 'MagZ': data_payload[9],
        'Flags': data_payload[10], 'Battery': data_payload[11], 'CRC': crc}

def handle_info_packet(packet_bytes):
    info_packet = parse_info_packet(packet_bytes)

    # check crc here
    crc_caclulated = calculate_crc8(packet_bytes[:HEADER_SIZE+info_packet['PayloadLen']])
    if(crc_caclulated != info_packet['CRC']):
        # Packet is corrupted. Discard the Data
        return

    # Display information
    print(f"Info Packet=>Time:{info_packet['Timestamp']}, Info:{info_packet['Info']}")

def handle_data_packet(packet_bytes):
    global Analysis_DF, duplicate_counter, ooo_counter, loss_counter
    # Parse the data
    data_packet = parse_data_packet(packet_bytes)    

    # check crc here
    crc_caclulated = calculate_crc8(packet_bytes[:HEADER_SIZE+data_packet['PayloadLen']])
    # assert crc_caclulated == data_packet['CRC'], "CRC Check Failed - data packet."
    if(crc_caclulated != data_packet['CRC']):
        # Packet is corrupted. Discard the Data
        return
    
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
    newdata = pd.DataFrame([data_packet])
    newdata.to_csv(csv_file, mode='a', index=False, header=False)

    if(packet_analysis):
        # Check for Duplicate packet
        duplicate = not Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID']) & (Analysis_DF['PacketID'] == data_packet['PacketID'])].empty
        if(duplicate):
            duplicate_counter+=1
            print(f"Duplicate Packet (DeviceID={data_packet['DeviceID']}, PacketID={data_packet['PacketID']}) | Counter = {duplicate_counter}")

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


def read_serial():
    global running, serial_comm
    while running:
        # Read just the header
        header_bytes = serial_comm.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            continue

        try:
            packet_type, payload_len, device_id, timestamp = parse_header(header_bytes)
            remaining_bytes = serial_comm.read(payload_len + FOOTER_SIZE)
            # Read the payload and footer based on packet_type
            if len(remaining_bytes) == payload_len + FOOTER_SIZE:
                if packet_type == 0x01:
                    handle_info_packet(header_bytes + remaining_bytes)
                elif packet_type == 0x02:
                    handle_data_packet(header_bytes + remaining_bytes)
                else:
                    # Received unknown packet
                    pass
        
        except struct.error:
            print("Error unpacking data")


def send_data(message):
    with serial_lock:
        data_bytes = message.encode() + b'\n'
        serial_comm.write(data_bytes)
        print(f"Sent: {data_bytes}")


def process_packet(packet, df, steps_df):
    # handle_data_packet(struct.pack(DATAPACKET_FORMAT, packet['PacketID'],
    #                                int(packet['AccelX']/accel_scale), int(packet['AccelY']/accel_scale), int(packet['AccelZ']/accel_scale),
    #                                int(packet['GyroX']/gyro_scale), int(packet['GyroY']/gyro_scale), int(packet['GyroZ']/gyro_scale),
    #                                int(packet['MagX']/magneto_scale), int(packet['MagY']/magneto_scale), int(packet['MagZ']/magneto_scale),
    #                                packet['Flags'], packet['Battery']) + struct.pack(HEADER_FORMAT, 0x02, 20, packet['DeviceID'], packet['Timestamp']) + struct.pack(FOOTER_FORMAT, packet['CRC']))

    # append packet to dataframe
    # print(df.head())
    if len(df) == 0:
        time = 0.0
    else:
        time = (packet['Timestamp'] - df['Timestamp'].min()) / 1000.0 # in seconds

    df.loc[len(df)] = [packet['PacketType'], packet['PayloadLen'], packet['DeviceID'], 
                       packet['Timestamp'], packet['PacketID'], 
                       [packet['AccelX'], packet['AccelY'], packet['AccelZ']], 
                    #    packet['GyroX'], packet['GyroY'], packet['GyroZ'], 
                    #    packet['MagX'], packet['MagY'], packet['MagZ'], 
                       packet['Flags'], packet['Battery'], packet['CRC'], time]
    
    # df['time'] = (df['Timestamp'] - df['Timestamp'].min()) / 1000.0 # in seconds
    # print(df.head())

    Sensors = input_df(df.copy())
    # print(Sensors.keys(), Sensors['left'].shape, Sensors['right'].shape, Sensors['waist'].shape)

    # ensure enough data for analysis - set buffer size (est_samp_freq * search_seconds)
    est_samp_freq = 200
    search_seconds = 10
    start = len(df) - est_samp_freq * search_seconds
    # print('Start index for analysis: ', start, '   Total Samples: ', len(df))
    if start < 0 or len(df) < est_samp_freq * search_seconds:
        # print("Not enough data for analysis")
        return df, steps_df
    
    # search for steps in the recent buffer of data
    # print('Searching for steps...')
    # for s in Sensors.keys():
    #     print(s)
    #     Sensors[s] = Sensors[s].iloc[start:, :].reset_index(drop=True)
    #     if len(Sensors[s]) == 0:
    #         # print(Sensors)
    #         print(f"No data for sensor: {s}")
    #         return df, steps_df
    # Sensors = [Sensors[s].iloc[start:, :].reset_index(drop=True) for s in Sensors]


    # process sensor data: Parse, resample, and filter
    ProcessedSensors = {}
    SensorNames = ['left', 'right', 'waist']
    for s in SensorNames:
        ProcessedSensors[s] = _process_sensor_df(Sensors[s])
    if any(ProcessedSensors[s] is None for s in SensorNames):
        print("Error processing sensor data")
        return df, steps_df
    # print(ProcessedSensors)


    # identify steps
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

    print('Searching for steps...')
    SA = StanceAnalyzer(config)
    left_strikes, left_stances = SA.extract_sensor_stances(
        ProcessedSensors['left']['accel_filtered'],
        ProcessedSensors['left']['accel'],
        ProcessedSensors['waist']['accel_filtered'],
        # config # <--- Uncomment to use your own config
    )
        
    right_strikes, right_stances = SA.extract_sensor_stances(
        ProcessedSensors['right']['accel_filtered'],    
        ProcessedSensors['right']['accel'],
        ProcessedSensors['waist']['accel_filtered'],
        # config # <--- Uncomment to use your own config
    )

    if time > 4:
        print(f"{time} {len(df)} Left steps found: {left_strikes}  Right steps found: {right_strikes}")
        print(f"Left stances: {len(left_stances)}  Right stances: {len(right_stances)}")
        
    if time > 7:
        plt.figure()
        plt.plot(ProcessedSensors['left']['accel_filtered'], label='Left Accel Filtered')
        plt.plot(ProcessedSensors['right']['accel_filtered'], label='Right Accel Filtered')
        plt.plot(ProcessedSensors['waist']['accel_filtered'], label='Waist Accel Filtered')
        for x in left_strikes:
            plt.axvline(x=x, color='blue', linestyle='--')
        for x in right_strikes:
            plt.axvline(x=x, color='orange', linestyle='--')
        plt.legend()
        plt.show()
        raise StopIteration


    # if len(left_strikes) > 0 or len(right_strikes) > 0:
    #     print(time, left_strikes, right_strikes)
    #     # add to steps_df
    #     raise StopIteration
        

    # parse gait cycles


    # send to model for prediction


    # log results for saving later

    return df, steps_df


# def main():
#     global serial_comm, running
#     serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
#     print(f"Connecting to {PORT} at {BAUDRATE} baud...")
    
#     reader_thread = threading.Thread(target=read_serial, daemon=True)
#     reader_thread.start()

#     try:
#         while True:
#             user_input = input("")
#             if user_input.lower() == 'exit':
#                 break
#             send_data(user_input)

#     except KeyboardInterrupt:
#         pass
#     finally:
#         running = False
#         serial_comm.close()
#         print("Serial port closed.")

# if __name__ == "__main__":
#     main()
