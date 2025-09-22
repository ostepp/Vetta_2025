import serial
import time
from enum import Enum
import struct
import numpy as np
from scipy import signal
import tensorflow as tf

class Sample:
    def __init__(self):
        self.id = -1
        self.time = -1
        self.accel = []
        self.gyro = []
        self.mag = []
        self.flag = -1
        self.battery = -1
        self.battery_percent = -1
    def __init__(self, new_id, new_time, new_accel, new_gyro, new_mag, new_flag, new_battery, battery_percent):
        self.id = new_id
        self.time = new_time
        self.accel = new_accel
        self.gyro = new_gyro
        self.mag = new_mag
        self.flag = new_flag
        self.battery = new_battery
        self.battery_percent = battery_percent
    def __str__(self):
        return 'Sensor: ' + str(self.id) + '\nAccel: ' + str(self.accel[0]) + ',' + str(self.accel[1]) + ',' + str(self.accel[2]) + '\nGyro: ' + str(self.gyro[0]) + ',' + str(self.gyro[1]) + ',' + str(self.gyro[2]) + '\nMag: ' + str(self.mag[0]) + ',' + str(self.mag[1]) + ',' + str(self.mag[2]) + '\nStatus: ' + str(self.flag)

class HubSample:
    def __init__(self):
        self.time = -1
        self.connectionStatus = HubStatus.DISCONNECTED
        self.lastSensorSampleTime = -1
        self.connectedSensors = []
    def __init__(self,new_time,new_status,new_SampleTime,new_sensors):
        self.time = new_time
        self.connectionStatus = new_status
        self.lastSensorSampleTime = new_SampleTime
        self.connectedSensors = new_sensors

class HubStatus(int,Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    LOW_CONNECTIVITY = 2

def GetSerialPorts():
    print("Looking for Serial Devices")
    myports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
    return myports

def FindTargetDevice(target):
    ports = GetSerialPorts()
    for candidate in ports:
        if target in candidate[1]:
            print(candidate[0])
            #Construct a Serial Object using the found port id
            serial_port = serial.Serial(port=str(candidate[0]),\
                baudrate=230400,\
                parity=serial.PARITY_NONE,\
                stopbits=serial.STOPBITS_ONE,\
                bytesize=serial.EIGHTBITS,\
                timeout=0)
            return serial_port
    return None
    
#Read an individual packet from the hub and convert it to Sample instance
def ProcessPacket(packetBin):
    # timestamp
    new_time = time.time()

    # Decode Sensor ID
    # This assumes less than 8 sensors to cut down on conversionss
    new_id = str(packetBin[2][2:])
    #print(new_id)

    # Decode Accel, Gyro, and Mag as 3-element float tuples
    new_accel = []
    new_gyro = []
    new_mag = []

    # Accel
    for x in range(3):
        start = 4 + (4 * x)
        new_float = bytes()
        for y in range(start,start + 4):
            hex_string = packetBin[y][2:]
            #print(hex_string)
            if len(hex_string) < 2:
                hex_string = '0' + hex_string
            new_float+=(bytes.fromhex(hex_string))
        #print("\n")
        value = struct.unpack('f', (new_float))[0]
        new_accel.append(value)
    #print(new_accel)

    # Gyro
    for x in range(3):
        start = 16 + (4 * x)
        new_float = bytes()
        for y in range(start,start + 4):
            hex_string = packetBin[y][2:]
           #print(hex_string)
            if len(hex_string) < 2:
                hex_string = '0' + hex_string
            new_float+=(bytes.fromhex(hex_string))
        #print("\n")
        value = struct.unpack('f', (new_float))[0]
        new_gyro.append(value)
    #print(new_gyro)

    # Mag
    for x in range(3):
        start = 28 + (4 * x)
        new_float = bytes()
        for y in range(start,start + 4):
            hex_string = packetBin[y][2:]
            #print(hex_string)
            if len(hex_string) < 2:
                hex_string = '0' + hex_string
            new_float+=(bytes.fromhex(hex_string))
        #print("\n")
        value = struct.unpack('f', (new_float))[0]
        new_mag.append(value)
    #print(new_mag)

    # Flag
    try:
        new_flag = int(packetBin[40][2:])
    except ValueError:
        new_flag = 0
    #print(new_flag)

    # Battery
    new_float = bytes()
    hex_string = packetBin[41][2:]
    new_battery = int(hex_string, 16)
    max_battery = 220
    battery_percent = round(new_battery / max_battery, 3)
    
    # package updates into sample class
    new_sample = Sample(new_id, new_time, new_accel, new_gyro, new_mag, new_flag, new_battery, battery_percent)
    # print(new_sample)
    # print('\n')
    return new_sample



#Actually passes values to the model
def PredictPeakVGRF(waistSamples,id,side, model):

    global vgrfWaveForms

    #These parameters will have to be doublechecked
    height = .8
    distance = 10
    prominence = .15
    width = 2
    # if model == None:
    #     print("No Model Loaded")
    #     return

    magnitudes = []
    print('getting samples')
    # print(waistSamples)
    for sample in waistSamples:
        magnitudes.append(GetMagnitude(sample.accel))
    #print(len(magnitudes))
    print(magnitudes)
    inter_magnitudes = signal.resample(magnitudes,100)
    # inter_magnitudes = signal.resample_poly(magnitudes, 100, len(magnitudes))
    # print(len(inter_magnitudes))
    print(inter_magnitudes)
    # print(type(inter_magnitudes))
    # print(inter_magnitudes.shape)
    waist = list([float(x) for x in inter_magnitudes])
    print('input')

    input = tf.TensorSpec.from_numpy(np.asarray(waist))
    input.name = 'dense_24_input'
    print(input)
    # vgrf = model.predict([inter_magnitudes])[0]
    try:
    #     # vgrf = session.run(['dense_26'], {'dense_24_input': [np.float32([1]), 
    #     #                                                      np.asarray(waist, dtype=np.float32)]})
    #     vgrf = session.run(['dense_26'], {'dense_24_input': np.asarray(waist, dtype=np.float32)})

    #     # vgrf = session.run(['dense_26'], {'dense_24_input': np.asarray(waist, dtype=np.float32)})
        vgrf = model()
        print('output')
        print(vgrf)
    except Exception as e:
        print(f"{type(e)}: {e}")
    # T = torch.from_numpy(np.asarray(waist))
    # try:
    #     vgrf = session.run(None, {'input': T})
    # except:
    #     raise 'error in running model'
    #print(vgrf)

    #save full wave form
    vgrfWaveForm = VGRFWaveForm(id,time.time(),side,vgrf)
    jsonData = str(vgrfWaveForm)
    vgrfWaveForms.append(jsonData)

    #Grab peak vgrf for stimulus
    peaks,properties = signal.find_peaks(vgrf, height = height, prominence = prominence, width = width, distance = distance)
    peakSample = VGRFSample(id,time.time(),side,properties['peak_heights'][0])
    return peakSample

def LoadModel():
    return joblib.load(modelFile) 

def GetMagnitude(sample):
    return np.sqrt(sample[0] ** 2 + sample[1] ** 2 + sample[2] ** 2)

#Derived from Ricky's example
# VMJerk values: Initialize once
cutoff = 6
gain = cutoff / np.sqrt(2)
sos = signal.butter(2, gain, fs=100, output='sos')

def VectorMagJerk(samples):
    global sos
    x = []
    y = []
    z = []
    for sample in samples:
        x.append(sample.accel[0])
        y.append(sample.accel[1])
        z.append(sample.accel[2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    shankValues = np.array([x,y,z])
    # Vector Norm Jerk
    jerk = np.diff(np.linalg.norm(shankValues,axis=0))

    F = signal.sosfiltfilt(sos,shankValues.T, axis = 0)
    #Filt = pd.DataFrame(data=F)
    xF = F[0]
    yF = F[1]
    zF = F[2]

    VMF = np.sqrt(np.multiply(xF,xF) + np.multiply(yF,yF) + np.multiply(zF,zF)).tolist()
    return VMF

def GetVMAJ(samples):
    # return vector magnitude acceleration and jerk from list-based input
    #global SampleCols 

    x = []
    y = []
    z = []
    for sample in samples:
        x.append(sample.accel[0])
        y.append(sample.accel[1])
        z.append(sample.accel[2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    shankValues = np.array([x,y,z])

    VMA = np.linalg.norm(shankValues,axis=0)
    jerk = np.diff(VMA)     # Vector Norm Jerk
    
    return VMA, jerk

def FindHeelStrikes(jerk):
    ht = 3
    dis = 50
    IClocs, ICprops = signal.find_peaks(jerk, height=ht, distance=dis)
    ICpks = [jerk[x] for x in IClocs]

    return IClocs, ICpks


#Currently no references.  Initial and Final event detection.  FindHeelStrikes is what's currently used
def FindGaitEvents(jerk):
    prom = 5 # specify promimence for small peak
    [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
    FCpks = [VMF[x] for x in FClocs]
    
    # get initial contact times
    prom = (1, 5)  # specify promimence for large peak
    wid = (5, 20)
    [IClocs, ICprops] = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
    ICpks = [VMF[x] for x in IClocs]

    return FCpks,ICpks



#def FindHeelStrikes(VMF):
#    prom = (.5, 4)  # specify promimence for large peak
#    wid = (5, 30)
#    IClocs, ICprops = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
#    ICpks = [VMF[x] for x in IClocs]
#    #peaks,_ = scipy.signal.find_peaks(jerk,height = 5,distance=10)
#    return ICpks

#Need to test height band parameters tomorrow morning.
def FindToeOffs(VMF):
    prom = 5 # specify promimence for small peak
    [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
    FCpks = [VMF[x] for x in FClocs]
    return FCpks


class VGRFWaveForm:
    def __init__(self):
        self.id = -1
        self.time = -1
        self.side = ""
        self.values = []
    def __init__(self,new_id,new_time,new_side,new_values):
        self.id = new_id
        self.time = new_time
        self.side = new_side
        self.values = new_values
    def __str__(self):
        return str(self.id) +","+str(self.time)+","+str(self.side)+","+str(self.values).replace("[","").replace("]","")

class VGRFSample:
    def __init__(self):
        self.id = -1
        self.time = -1
        self.side = ""
        self.peakValue = -1
    def __init__(self,new_id,new_time,new_side,new_peakValue):
        self.id = new_id
        self.time = new_time
        self.side = new_side
        self.peakValue = new_peakValue