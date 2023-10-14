#include <Arduino_LSM9DS1.h>
#include "neural_network.h"
#include <ArduinoBLE.h>

// Device ID
const char* deviceServiceUuid = "19b10030-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10031-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceReadCharacteristicUuid = "19b10032-e8f2-537e-4f6c-d104768a1214";

// Defining services for the BLE device
BLEService gestureService(deviceServiceUuid); 
BLEByteCharacteristic gestureCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite);
BLEByteCharacteristic readCharacteristic(deviceServiceReadCharacteristicUuid, BLERead | BLEWrite | BLENotify | BLEBroadcast);

static NN myNetwork;
bool v3=0, v1 =0;
int i=0;

void train( bool only_forward) {
  // In this function the model is trained with the captured training data
  if(!only_forward){ // training flag
    myNetwork.train();
  }else {
    myNetwork.test();
  readCharacteristic.writeValue(5); //sending signal to FL Server that training is done
  readCharacteristic.broadcast();

    // Print outputs
    for (uint16_t i = 0; i < 3; i++) {
      union {
    float float_variable;
    byte temp_array[4];
  } u;
    u.float_variable = myNetwork.output_data[i];
      for(int j=0;j<4;j++){
        delay(100);
        readCharacteristic.writeValue(u.temp_array[j]); // Converting output values into float and sending to central device
        readCharacteristic.broadcast();
      }
    }
  }
}

void setup() {
  Serial.begin(115200);

  Serial.println("Nano 33 BLE (Peripheral Device)");
  Serial.println(" ");
  myNetwork.init_network_model(); // Initialize the weights of the model

  if (!IMU.begin()) { // Initialize IMU sensors
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  }

uint16_t trained = 0; // Training flag
int total = 0; // Count flag of the samples
int ble_on = 0; // BLE flag

// Receving weights from FL server
void receive_weights(byte x) {
((char*)myNetwork.FlatWeights)[i++] = x;
if(i == (myNetwork.weight_number*4)){ // if all the weights received
 // Serial.print("fun3 i ");
  Serial.println(i);
     i=0;
     v3=0;
     trained = 0;
     myNetwork.modify_Weights(myNetwork.FlatWeights); // Modify the weights
     readCharacteristic.writeValue(byte(8));
     readCharacteristic.broadcast();
     Serial.println(8);
     delay(50);
  }
}

// Initializes device to get updated weights
void init_receive_weights() {
  readCharacteristic.writeValue(0);
  readCharacteristic.broadcast();
  trained = 1;
  v3 = 1;
    Serial.println("fun6");

}

 // Sending weights
void SendWeights() {
  if(!trained){
    readCharacteristic.writeValue(0); // Sending signal to FL server if not trained
    readCharacteristic.broadcast();
  }else {
    readCharacteristic.writeValue(1); // Signal if trained
    readCharacteristic.broadcast();
  for (uint16_t i = 0; i < myNetwork.weight_number; ++i) {
    union {
    float float_variable;
    byte temp_array[4];
  } u;
  u.float_variable = *((float*)myNetwork.FlatWeights+i);
    for(int i=0;i<4;i++){
      delay(100);
      readCharacteristic.writeValue(u.temp_array[i]); // Sending byte values of the weights
      readCharacteristic.broadcast();
      Serial.println(u.temp_array[i]);
    }
  }
  }
    //Serial.println("fun7");

}

// Testing
void test() {
  trained = 1;
   myNetwork.recordTestData(); // Record the test samples 
    readCharacteristic.writeValue((byte)1); // Inform if recording done
    readCharacteristic.broadcast();
    delay(100);
    train(1); // test ( only forward falg==1 )
      //Serial.println("fun8");

}

// Indicates testing done and turn off BLE
void testing_done() {
  trained = 0;
  ble_on = 0;
  BLE.end();
   // Serial.println("fun9");

}

// Decides whether testing is to be done or not
void test_or_not(byte x) {
  v1 = 0;
  if( x == 0){
  ble_on = 0;
  BLE.end();
  }
    //Serial.println("fun11");

}

// indicates testing is to be done
void init_test() {
  v1 = 1;
  //Serial.println("fun10");
    pin_mode(LED_GREEN, OUTPUT);

}

void loop() {
delay(200);

  if( ble_on){
  BLEDevice central = BLE.central();
  if (central) {
    if (central.connected()) { // If connected to central
      if(gestureCharacteristic.written()){
        byte x = gestureCharacteristic.value();
        Serial.println(x);
      if(v3==1){
        receive_weights(x); // Receiving updated weights
      }else if (v1 == 1){
        test_or_not(x); //decides whether testing is to be done or not
      }
      else if ( x == 3 ) {
        init_receive_weights(); // Initializes device to get updated weights
      }else if (x == 4) {
        SendWeights(); //sending weights
      } else if ( x == 5){
        test();     //testing
      } else if ( x == 6){
        testing_done(); // Indicates testing done and turn off BLE
      } else if(x==9){
        init_test(); // indicates testing is to be done
      }}
    }
  }
  }
  if(!trained){ // Training
    bool recorded = myNetwork.recordTrainData(); // Record data

    if (recorded) {
      if(myNetwork.training_record == NUMBER_OF_DATA){
      train(0);
      trained = 1;
      total = 0;
      ble_on = 1; // BLE on after training
       if (!BLE.begin()) {
    Serial.println("- Starting Bluetooth® Low Energy module failed!");
    while (1);
  }
// Defining BLE name, services and characteristics
  BLE.setLocalName("Arduino Nano 33 BLE (Peripheral)");
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureCharacteristic);
  gestureService.addCharacteristic(readCharacteristic);
  BLE.addService(gestureService);
  BLE.advertise();
pin_mode(LED_RED, OUTPUT);
  Serial.println("Nano 33 BLE (Peripheral Device)");
  Serial.println(" ");
    }
  }
  }
}

