#include <ArduinoBLE.h>
#include <Arduino_APDS9960.h>
#include <HardwareSerial.h>

// UUIDs for the peripheral devices
const char *deviceServiceUUIds[4] = {"19b10000-e8f2-537e-4f6c-d104768a1214", "19b10010-e8f2-537e-4f6c-d104768a1214", "19b10020-e8f2-537e-4f6c-d104768a1214", "19b10030-e8f2-537e-4f6c-d104768a1214"};
const char *deviceServiceCharUUIds[4] = {"19b10001-e8f2-537e-4f6c-d104768a1214", "19b10011-e8f2-537e-4f6c-d104768a1214", "19b10021-e8f2-537e-4f6c-d104768a1214", "19b10031-e8f2-537e-4f6c-d104768a1214"};
const char *deviceServiceReadCharUUIds[4] = {"19b10002-e8f2-537e-4f6c-d104768a1214", "19b10012-e8f2-537e-4f6c-d104768a1214", "19b10022-e8f2-537e-4f6c-d104768a1214", "19b10032-e8f2-537e-4f6c-d104768a1214"};

uint8_t flag=0;   
uint8_t device_index =0;
uint8_t total_devices = 4; // total peripheral devices

void setup()
{
  Serial.begin(115200);
  while (!Serial)
  ;

  if (!BLE.begin())
  {
    while (1);
  }

  BLE.setLocalName("Nano 33 BLE (Central)"); // Set a local name for central device
  BLE.advertise();
}
    BLECharacteristic gestureCharacteristic;
    BLECharacteristic readCharacteristic;
void loop()
{
  connectToPeripheral(); // Connect to peripheral devices
  Serial.flush(); // clear serial buffer
}

void connectToPeripheral()
{
  BLEDevice peripherals[total_devices];
 
    controlPeripheral(peripherals);
  
}
void controlPeripheral(BLEDevice peripherals[])
{

    do
    {
      byte start = 0;
      if (Serial.available())
      {
        start = Serial.read(); //read the device index given from FL server
       // Serial.println(start);
        if(flag==0){
          device_index = start;

          if(!(peripherals[device_index].connected())){ // Connect to only one device at a time
          //  Serial.println("hi1");
            for(int j=0;j<total_devices;j++){
              if(peripherals[j].connected()){
                peripherals[j].disconnect();
              }
            }
            int k=0;
            do
          {
            BLE.scanForUuid(deviceServiceUUIds[device_index]);
            peripherals[device_index] = BLE.available();
            //Serial.println("hello");
            if(peripherals[device_index])break;
         // Serial.flush();
          k++;
          } while (k<400);
          BLE.stopScan();
            int y = peripherals[device_index].connect();
            if(y){
            peripherals[device_index].discoverAttributes();
            gestureCharacteristic = peripherals[device_index].characteristic(deviceServiceCharUUIds[device_index]); 
            readCharacteristic = peripherals[device_index].characteristic(deviceServiceReadCharUUIds[device_index]); 
            readCharacteristic.subscribe();
              Serial.println(byte(1));
              //Serial.flush();
              flag = 1;
            } else {
              flag = 0;
              Serial.println(byte(0));
             // Serial.flush();
            }
          } else {
            Serial.println(byte(1));
           // Serial.flush();
            flag = 1;
          }
        }
        else 
        {
            if(gestureCharacteristic.canWrite()){
              delay(50);
              gestureCharacteristic.writeValue(start);
            }
        flag=0;
        }
      }
      byte x = 0;
            if(readCharacteristic.valueUpdated()){
              readCharacteristic.readValue(x);
              Serial.println(x);
              Serial.flush(); //Clear buffer after the data transmission
            }
    } while (1);
}