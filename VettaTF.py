import serial
import serial.tools.list_ports
import struct
import time
import json
import sys
import numpy as np
import scipy
from scipy import signal
import joblib
import os
from os.path import exists
from enum import Enum
import warnings
import tensorflow as tf

import utils.sensor_realtime as SRT
import utils.read_packets as RP

# if needed, install hub driver for Hub: Adafruit HUZZAH32 - ESP32 Feather 
# https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=overview


# Constants
modelFile = 'models/tfmodel_5.onnx'

targetDeviceString = "Silicon Labs CP210x USB to UART Bridge"
serial_port = SRT.FindTargetDevice(targetDeviceString)

if (serial_port != None):
    targetDevicePresent = True
    hubStatus = SRT.HubStatus.CONNECTED
    print("Hub Connected!")
    serial_port.write(b'S')  # start recording command into serial port
else:
    targetDevicePresent = False
    hubStatus = SRT.HubStatus.DISCONNECTED

# How frequently to send a new HubSample in whole seconds
hubUpdateTime = 1
lastHubUpdate = time.time()

lastSensorSampleTime = -1
connectedSensors = set()
sensorSampleTimes = {}
# In whole seconds
sensorTimeout = .3
connectionStablizeDuration = 10
connectionStabilizeTime = -1

line = []
last_byte = ''
newPacket = False
running = True
maxSamples = 20
count = 0

# For saving to a file
samples = []
vgrfSamples = []
vgrfWaveForms = []
testVRGFId = 0
testVGRFDelta = .5
testLastVGRFUpdate = -1
side = "Left"

# Load the Model
model = tf.saved_model.load('Models/tfmodel_5')

# Initialize Step Detection Variables
lStepTime = -1
rStepTime = -1
samples_gait_event = 100
shankSampleTarget = 100

# Define sensor unicode id values
waistID = "38"
leftShankID = "39"
rightShankID = "3a"

# Main Real-time Processing Loop
while running:
    try:
     
        # If the device is connected, read data
        if targetDevicePresent:

            for c in serial_port.read():

                # Parse Packet Data
                if newPacket:
                    line.append(hex(c))
                    if c == 204 and last_byte == 170:
                        newPacket = True
                        line = []
                        line.append(hex(last_byte))
                        line.append(hex(c))
                    
                    # A new packet/ End of the packet is detected
                    elif c == 51 and last_byte == 85:
                        #print("Line: ",line)
                        newPacket = False
                        count+=1
                        #print(time.time())
                        #Convert bytes to object
                        try:
                            new_sample = SRT.ProcessPacket(line)
                        except:
                            print("Error Creating Sample")
                            continue

                        #Create Shank Signals for step counting
                        if new_sample.id == leftShankID:
                            if len(leftShankSamples) > 300:
                                leftShankSamples = []
                                leftWaistSamples =[]
                            leftShankSamples.append(new_sample)
                        elif new_sample.id == rightShankID:
                            if len(leftShankSamples) > 300:
                                rightShankSamples = []
                                rightWaistSamples = []
                            rightShankSamples.append(new_sample)

                        #Create Waist Signals for predictions
                        if new_sample.id == waistID:
                            leftWaistSamples.append(new_sample)
                            rightWaistSamples.append(new_sample)
            
                        sampleCount += 1
                        
                        # Update sensor status metrics
                        connectedSensors.add(new_sample.id)
                        sensorSampleTimes[new_sample.id] = new_sample.time
                        lastSensorSampleTime = new_sample.time

                        # print(sampleCount)
                        # print(new_sample)

                        if sampleCount > 20:
                            raise StopIteration
                        
                        # StepCheck
                        print('Detecting Steps ...')
                       
                        if len(leftShankSamples) >= shankSampleTarget: # if sufficient samples to check for a gait event
                            print('sufficient samples')
                            VMA, jerk = SRT.GetVMAJ(leftShankSamples)
                            Lind, pks = SRT.FindHeelStrikes(jerk)

                            if len(Lind) > 0:
                                #LStepTimes.append(leftShankSamples[Lind[0]][1])
                                print("Left Step Found!")
                                if not leftStance:
                                    #print("First Step!")
                                    leftStance = True
                                    leftWaistSamples = []
                                    lStepTime = time.time()
                                else:
                                    currentTime = time.time()
                                    if currentTime-lStepTime<2000:
                                        print("Ending Step!")
                                        stopIndex = int(len(leftWaistSamples)*.6)+1
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore")
                                            print('Predicting vGRF')
                                            leftVGRFSample = SRT.PredictPeakVGRF(leftWaistSamples[:stopIndex],leftPeakId,"Left", model)
                                            print('Predicting vGRF')
                                            jsonData = json.dumps(leftVGRFSample.__dict__)
                                            print(jsonData)
                                            #leftWaistSamples = []
                                            vgrfSamples.append(jsonData)
                                            leftPeakId+=1
                                        #Get Waist Samples to Model
                                    #else:
                                        #print("Orphaned Step! Discarding!")
                                    leftStance = False
                            leftShankSamples = []
                        if len(rightShankSamples) >= shankSampleTarget: # if sufficient samples to check for a gait event
                            VMA, jerk = SRT.GetVMAJ(rightShankSamples)
                            Rind, pks = SRT.FindHeelStrikes(jerk)

                            if len(Rind) > 0:
                                #LStepTimes.append(leftShankSamples[Lind[0]][1])
                                #print("Right Step Found!")
                                if not leftStance:
                                    #print("First Step!")
                                    rightStance = True
                                    rightWaistSamples = []
                                    rStepTime = time.time()
                                else:
                                    currentTime = time.time()
                                    if currentTime-rStepTime<2000:
                                        stopIndex = int(len(rightWaistSamples)*.6)+1
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore")
                                            rightVGRFSample = SRT.PredictPeakVGRF(rightWaistSamples[:stopIndex],rightPeakId,"Right", session)
                                            jsonData = json.dumps(rightVGRFSample.__dict__)
                                            print(jsonData)
                                            rightPeakId+=1
                                            vgrfSamples.append(jsonData)
                                        
                                    rightStance = False
                            rightShankSamples = []

                        # Save sensor json
                        jsonData = json.dumps(new_sample.__dict__)
                        #Send to STDOut
                        sampleCount = 0
                        samples.append(jsonData)
                        line = []

                elif c == 204 and last_byte == 170:
                    newPacket = True
                    line.append(hex(last_byte))
                    line.append(hex(c))
                last_byte = c

        # await a matching USB device connection
        else:
            # Sleep thread a set time
            # time.sleep(1)
            # Check for an attached device
            serial_port = SRT.FindTargetDevice(targetDeviceString)
            if (serial_port != None):
                targetDevicePresent = True
                hubStatus = SRT.HubStatus.CONNECTED
    
            # Send a status update
            if (targetDevicePresent):
                print("Hub Connected!")
            else:
                print("Waiting for device!")
        
        cur_time = time.time()
        # Check For Low Connectivity/Dropped Sensors
        for key in sensorSampleTimes:
            if cur_time - sensorSampleTimes[key] > sensorTimeout:
                if key in connectedSensors:
                    connectedSensors.remove(key)
        if hubStatus != SRT.HubStatus.DISCONNECTED:
            if len(sensorSampleTimes) != len(connectedSensors):
                hubStatus = SRT.HubStatus.LOW_CONNECTIVITY
            else:
                if connectionStabilizeTime == -1:
                    connectionStabilizeTime = cur_time
                elif cur_time - connectionStabilizeTime >= connectionStablizeDuration:
                    connectionStabilizeTime = -1
                    hubStatus = SRT.HubStatus.CONNECTED


        # Send HubSample
        if cur_time - lastHubUpdate >= hubUpdateTime:
            jsonData = json.dumps(SRT.HubSample(cur_time,hubStatus,lastSensorSampleTime,sorted(connectedSensors)).__dict__)
            print(jsonData)
            #print(HubSample(cur_time,hubStatus,lastSensorSampleTime,connectedSensors))
            lastHubUpdate = cur_time
                
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        running = False
        pass

    except serial.SerialException as e:
        # There is no new data from serial port
        print("Serial Exception!")
        targetDevicePresent = False
        hubStatus = SRT.HubStatus.DISCONNECTED
        pass

    except BaseException as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        running = False
        pass


print("Saving Recorded Sensor Data")
# save raw data
fileCount = "new_Raw"
savePath = 'Output/' + str(fileCount) + ".json"
counter = 0
while exists(savePath):
    counter+=1
    savePath = savePath.replace(".json","_"+str(counter)+".json")
with open(savePath, "w") as f:
    f.write(json.dumps(samples,indent=4, sort_keys=True))
# save step loading peaks
stepsFile = "new_Steps"
savePath = 'Output/' + stepsFile + ".json"
counter = 0
while exists(savePath):
    counter+=1
    savePath = savePath.replace(".json","_"+str(counter)+".json")
with open(savePath, "w") as f:
    f.write(json.dumps(vgrfSamples,indent=4, sort_keys=True))
# save vGRF waveforms
waveFormFile = "new_WaveForms"
savePath = 'Output/' + waveFormFile + ".json"
counter = 0
while exists(savePath):
    counter+=1
    savePath = savePath.replace(".json","_"+str(counter)+".json")
with open(savePath, "w") as f:
    f.write(json.dumps(vgrfWaveForms,indent=4, sort_keys=True))

serial_port.close()



#%%


#Sample lists
leftWaistSamples = []
rightWaistSamples = []
leftShankSamples = []
rightShankSamples = []

# Should be true on first event.  Set to false and samples are cleared on second event
#"Stance" is probably a misnomer at this point due to changes in event detection.
leftStance = False
rightStance = False

minShankSampleCount = 20

#Used for waveform id
leftPeakId = 0
rightPeakId = 0

sampleCount = 0



