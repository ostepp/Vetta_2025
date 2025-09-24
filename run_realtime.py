import serial
import struct
# import time
import threading
import pandas as pd
import utils.read_packets as RP
import matplotlib.pyplot as plt

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


def main(run):
    """Main function to run real-time data processing from serial port or CSV file.
    args:
        run: 
            'serial' to read sensor data in real time from serial port, 
            'csv' to read from CSV file and play through the data as if real-time.
    """
    
    try:
        while True:

            global serial_comm, running, ESPData, steps_df, vgrf_df, axes

            # initialize dataframes to store results
            w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
            steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
            vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
            vgrf_df = pd.DataFrame(columns=['ID','PeakvGRF'] + vgrf_cols)
            ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
                                        'Timestamp', 'PacketID', 'accel', 
                                        'Flags', 'Battery', 'CRC', 'time'])
            
            # create figure and subplots for showing data
            fig, axes = plt.subplots(nrows=3, ncols=1)
            fig.set_figheight(10) 
            fig.set_figwidth(8)


            if run == 'serial':
                # running in serial mode
                # global serial_comm, running, ESPData, steps_df, vgrf_df
                serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
                print(f"Connecting to {PORT} at {BAUDRATE} baud...")
                reader_thread = threading.Thread(target=RP.read_serial, daemon=True)
                reader_thread.start()

                user_input = input("")
                if user_input.lower() == 'exit':
                    break
                RP.send_data(user_input)


            elif run == 'csv':
                # load data to simulate real-time processing loop
                fn = 'IMU_data_RevB_v3_09112025_Walk1.csv'
                data = pd.read_csv(fn)
                print(f'loading csv data from: {fn}')
                print(data.head())

                # simulate real-time data reception
                print('\nStarting simulation...')
                for index, row in data.iterrows():
                    packet = row.to_dict()

                    if len(packet) == 0:
                        continue

                    # print(index)
                    # print('Packet received:   ', packet)
                    ESPData, steps_df, vgrf_df, axes = RP.process_packet(packet, ESPData, steps_df, vgrf_df, axes)

                    if index > 14000:
                        print("\nEnding simulation.")
                        print('Streaming DF: \n', ESPData.tail())
                        print('Steps DF: \n', steps_df.tail())
                        print('vGRF DF: \n', vgrf_df.tail())

                        ax1 = axes[0] 
                        window = 10
                        if max(ESPData['time']) > window:
                            ax1.set_xlim([max(ESPData['time']) - window, max(ESPData['time'])])

                        plt.tight_layout()
                        plt.show()
                        fig.savefig('acc_steps_preds.png')

                        print('Ending Simulation early')
                        break

                break # exit csv running when all rows ran through
                    
            else:
                raise ValueError("Invalid run mode. Use 'serial' or 'csv'.")
            
            
    except KeyboardInterrupt:
        pass

    finally:
        if run == 'serial':
            running = False
            serial_comm.close()
            print("Serial port closed.")

        csv_file = 'ESP_stream_data.csv'
        steps_file = 'Parsed_steps_data.csv'
        vgrf_file = 'Predicted_vGRF_data.csv'
        ESPData.to_csv(csv_file, index=False)
        steps_df.to_csv(steps_file, index=False)
        vgrf_df.to_csv(vgrf_file, index=False)
        print(f"Data saved to {csv_file} and {steps_file} and {vgrf_file}")


if __name__ == "__main__":
    main(run='csv') # change to serial to run with sensors connected to the serial port