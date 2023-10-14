#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK
#include <aifes.h>
#include <Arduino_LSM9DS1.h>


/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/
static const int NUM_SAMPLES=10;  // No. of readings taken in one sample
static const int NUM_AXES=3;     // x,y,z dimension
static const int TRUNCATE=20;     
static const float ACCEL_THRESHOLD=3.0; // Min. acc. required to start recording
static const int INTERVAL=30;

static const int SAMPLES_PER_CLASS=5;

static const int NUMBER_OF_DATA=15; // Total no. of samples of three classes

// static const ints for AIfES FNN-
static const int FNN_3_LAYERS=3;
static const int INPUTS=NUM_SAMPLES * NUM_AXES;
static const int NEURONS=4; 
static const int OUTPUTS=3; // 3 gestures
static const int PRINT_INTERVAL=10;

class NN{
    public:
float output_data[OUTPUTS];
float labels[NUMBER_OF_DATA][OUTPUTS];
float *FlatWeights;
uint32_t weight_number;
int training_record = 0;
void train();
void test();
void init_network_model();
void recordIMU();
void printAndSafeFeatures();
void recordTestData();
bool recordTrainData();
void modify_Weights(float* flatWeights);
bool motionDetected(float ax, float ay, float az);
        
// AIfES-Express model parameter
uint32_t FNN_structure[FNN_3_LAYERS] = {INPUTS,NEURONS,OUTPUTS};
AIFES_E_activations FNN_activations[FNN_3_LAYERS - 1]={AIfES_E_sigmoid, AIfES_E_softmax};
AIFES_E_model_parameter_fnn_f32 FNN;
float baseline[NUM_AXES];
// Features (inputs) of a single gesture
float features[INPUTS];
// 2D array for training (inputs)
// Here the features of the gestures from the data recording are stored
float training_data[NUMBER_OF_DATA][INPUTS];
};


#endif