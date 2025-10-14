import serial
import struct
import time
import threading
import pandas as pd

CRC_POLYNOMIAL = 0xAB

HEADER_FORMAT = '<BBBI'
FOOTER_FORMAT = '<B'
DATAPACKET_FORMAT = '<I9hBB'

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # Calculate expected header size
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)  # Calculate expected footer size

PORT = "COM3" # Com port of the device (default was 8)
BAUDRATE = 921600

serial_lock = threading.Lock()
serial_comm = None
running = True

packet_analysis = False
analysis_window = 1000
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

def main():
    global serial_comm, running
    serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"Connecting to {PORT} at {BAUDRATE} baud...")
    
    reader_thread = threading.Thread(target=read_serial, daemon=True)
    reader_thread.start()

    try:
        while True:
            user_input = input("")
            if user_input.lower() == 'exit':
                break
            send_data(user_input)
            
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        serial_comm.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
