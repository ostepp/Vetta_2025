import serial
import struct
# import time
import threading
import pandas as pd
import utils.read_packets as rp

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
analysis_window = 1000
Analysis_DF = pd.DataFrame(columns=['DeviceID', 'PacketID', 'Timestamp'])
duplicate_counter = 0
ooo_counter = 0
loss_counter = 0

# Based on configuration
accel_scale = 2.0 / 32768.0
gyro_scale = 250.0 / 32768.0
magneto_scale = 4.0 / 32768.0

# csv_file = "IMU_data_RevB_v3.csv"
ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
                                'Timestamp', 'PacketID', 
                                'accel', # 'AccelX', 'AccelY', 'AccelZ', 
                                # 'GyroX', 'GyroY', 'GyroZ', 
                                # 'MagX', 'MagY', 'MagZ', 
                                'Flags', 'Battery', 'CRC', 'time'])
# ESPData.to_csv(csv_file, index=False)


def main():
    # global serial_comm, running
    # serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
    # print(f"Connecting to {PORT} at {BAUDRATE} baud...")
    
    # reader_thread = threading.Thread(target=rp.read_serial, daemon=True)
    # reader_thread.start()
    w_acc_cols = [f'w_accel_{i}' for i in range(0, 100)]
    vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
    steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 
                                     'PeakvGRF'] + w_acc_cols + vgrf_cols)
    # steps_df.head()

    # pretend to run real-time processing loop
    data = pd.read_csv('IMU_data_RevB_v3_09112025_Walk1.csv')
    # print(data.head())
    # received_packets = 0

    for index, row in data.iterrows():
        # simulate real-time data reception
        packet = row.to_dict()

        if len(packet) == 0:
            continue

        # print(index)
        # print('Packet received:   ', packet)
        steps_df = rp.process_packet(packet, ESPData, steps_df)

        if index > 5000:
            raise StopIteration


    # try:
    #     while True:

    #         user_input = input("")
    #         if user_input.lower() == 'exit':
    #             break
    #         rp.send_data(user_input)

 

            
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     running = False
    #     serial_comm.close()
    #     print("Serial port closed.")


if __name__ == "__main__":
    main()