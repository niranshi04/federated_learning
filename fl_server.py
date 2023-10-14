import serial
from serial.tools.list_ports import comports
import sys
import struct
import time
import numpy as np
import time
import json
import os
import random
import matplotlib.pyplot as plt
from queue import Queue

random.seed(4321)
np.random.seed(4321)

experiment = "train-test"
fl_epochs = 1   # No. of rounds of FL

model_size = 139 # No. of weights
model_weights = np.random.uniform(-2,2, model_size).astype('float32')
model_received = [0] # flag array for peripheral devices ( edit the size according to number of devices)

def testModel(device, deviceIndex, successes_queue, suc):
    for i in range (0,3):
        for j in range(0,2):
            print(f"Testing for device {deviceIndex+1}")
            predicted = trainOrTest(device, deviceIndex, successes_queue, i, True)
            print(f"{i+1} - {predicted}")
            success = predicted == i+1
            suc.put(success)

def trainOrTest(device, deviceIndex, successes_queue, button, only_forward = False,):
    ini_time = time.time() * 1000
    device.write(struct.pack('B', deviceIndex)) # Sending device index to central device
    k = device.readline().decode()
    while ( k == '' ) :
        k = device.readline().decode()
    device.write(struct.pack('B', 5))
    k = device.readline().decode()
    while ( k == '' ) :
        k = device.readline().decode()
    print(f"Sample recorded confirmation:", k)

    k = device.readline().decode()
    while ( k == '' ) :
        k = device.readline().decode()
    print(f"Testing done confirmation:", k)
    num_button_predicted = recordOutput(device, deviceIndex, only_forward, successes_queue, button)
        
    print(f'Testing done in: {(time.time()*1000)-ini_time} milliseconds)')
    return num_button_predicted

def recordOutput(device, deviceIndex, only_forward, successes_queue, button):
    outputs = [] # final ouput array
    for i in range(0,3): # 3 ouput classes 
        out = []
        for j in range(0,4):
            k = device.readline().decode()
            while ( k == '' ) :
                k = device.readline().decode()
            out.append(int(k)) # Appending bytes received
        [output] = struct.unpack('f',bytes(out)) #unpacking the bytes
        outputs.append(output)
    print(f'Outputs: {outputs}') 
    successes_queue.append(outputs[button])
    bpred = outputs.index(max(outputs))+1 # Predicted class
    print(f'Predicted button: {bpred}')
    
    return outputs.index(max(outputs)) + 1

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except:
            print("ERROR: Not a number")

# Read data from serial port 
def read_port(msg):
    while True:
        try:
            port = input(msg)
            ser = serial.Serial(port, 9600)
            ser.close()
            ser.open()
            return ser
        except Exception as e: 
            print(e)

# Connection to central device
def getDevice(): 
    global device
    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)
    device = read_port(f"Port device: ") # Connected via serial port

# Receive models from devices after training
def FlGetModel(d, device_index, devices_model_weights): 
    global model_size, model_received
    d.reset_input_buffer()
    d.reset_output_buffer()
    d.timeout = 5

    print(f'Starting connection to {device_index+1} ...') # Hanshake
    d.write(struct.pack('B', device_index))
    y = d.readline()
    k = y.decode()
    while ( k == '' ) :
        y = d.readline()
        k = y.decode()
        
    if(y == b'0\r\n'):
        print('Device not ready.')
    else: 
        d.write(struct.pack('B', 4))
        print(f"hi")
        y = d.readline()
        k = y.decode()
        while ( k == '' ) :
            y = d.readline()
            k = y.decode()
        if(y == b'0\r\n'):
            print('Device not ready.')
        else :
            print(f'Receiving model from {device_index+1} ...')
            # Receiving weights
            for i in range(model_size):
                out = []
                for j in range(0,4):
                    k = d.readline().decode()
                    while ( k == '' ) :
                        k = d.readline().decode()
                    out.append(int(k))
                [float_num] = struct.unpack('f',bytes(out))
                devices_model_weights[device_index][i] = float_num
                print(f"{i} {float_num}")
            model_received[device_index] = 1
            print(f'Model received from {device_index+1}')

# Sending updated model
def sendModel(d, device_index, model_weights):
    ini_time = time.time()
    d.write(struct.pack('B', device_index))
    k = d.readline().decode()
    while ( k == '' ) :
        k = d.readline().decode()
    d.write(struct.pack('B', 3))
    k = d.readline().decode()
    while ( k == '' ) :
        k = d.readline().decode()
    for i in range(model_size): 
        float_num = model_weights[i]
        print(f"{i} {float_num}")
        data = list(struct.pack('f', float_num))
        for j in range(0,4):
            d.write(struct.pack('B', device_index))
            k = d.readline().decode()
            while ( k == '' ) :
                k = d.readline().decode()
            d.write(struct.pack('B', data[j]))
    k = d.readline().decode()
    while ( k == '' ) :
        k = d.readline().decode()
    print(f'Model sent to {device_index+1} ({time.time()-ini_time} seconds)')


def startFL():
    suc = Queue()
    global model_weights, fl_epochs, model_received
    print('Starting Federated Learning')
    devices_model_weights = np.empty((1, model_size), dtype='float32')
    model_received = [0]

    ##################
    # Receiving models
    ##################
    while(1):
        l = 1
        for j in range(0,1):
            if(not(model_received[j])):
                time.sleep(4)
                FlGetModel(device, j, devices_model_weights)
            l = (l and model_received[j])
        if(l):
            break

    
    ####################
    # Processing models
    ####################
    ini_time = time.time() * 1000
    model_weights = np.average(devices_model_weights, axis=0, weights=[1])
    print(f'Average millis: {(time.time()*1000)-ini_time} milliseconds)')


    #################
    # Sending models
    #################
    for j in range(0,1):
        successes_queue = []
        print(f'Sending model to device {j+1} ...')
        sendModel(device, j, model_weights)
        device.write(struct.pack('B', j))
        k = device.readline().decode()
        while ( k == '' ) :
            k = device.readline().decode()
        device.write(struct.pack('B', 9))
        device.write(struct.pack('B', j))
        k = device.readline().decode()
        while ( k == '' ) :
            k = device.readline().decode()
        device.write(struct.pack('B', 1))
        testModel(device, j, successes_queue, suc)
        device.write(struct.pack('B', j)) # Sending device index to central device
        k = device.readline().decode()
        while ( k == '' ) :
            k = device.readline().decode()
        print(f"{k}")
        device.write(struct.pack('B', 6))
    test_accuracy = sum(suc.queue)/len(suc.queue)
    print(f"Testing accuracy: {test_accuracy}")
    print(f"{test_accuracy}, ")
    fl_epochs-=1
    if fl_epochs!=0: 
        startFL()
getDevice()
startFL()
