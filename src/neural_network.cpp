#include <arduino.h>
#include "neural_network.h"

void NN::train () {                                          // Counting variable
  // -------------------------------- Create tensors needed for training ---------------------
  // Create the input tensor for training, contains all samples
  uint16_t input_shape[] = {NUMBER_OF_DATA, INPUTS};        // Definition of the shape of the tensor, here: {# of total samples (i.e. samples per object * 3 objects), 3 (i.e. for each sample we have 3 RGB values)}
  aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, training_data);                 // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // Create the target tensor for training, contains the desired output for the corresponding sample to train the ANN
  uint16_t target_shape[] = {NUMBER_OF_DATA, OUTPUTS};            // Definition of the shape of the tensor, here: {# of total samples (i.e. samples per object * 3 objects), 3 (i.e. for each sample we have 3 possible output classes)}
  aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, labels);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // Create an output tensor for training, here the results of the ANN are saved and compared to the target tensor during training
  float output_data[NUMBER_OF_DATA][OUTPUTS];                     // Array for storage of the output data
  uint16_t output_shape[] = {NUMBER_OF_DATA, OUTPUTS};            // Definition of the shape of the tensor, here: {# of total samples (i.e. samples per object * 3 objects), 3 (i.e. for each sample we have 3 possible output classes)}
  aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // -------------------------------- init weights settings ----------------------------------
    
  AIFES_E_init_weights_parameter_fnn_f32  FNN_INIT_WEIGHTS;
 FNN_INIT_WEIGHTS.init_weights_method = AIfES_E_init_no_init;

  /* init methods
      AIfES_E_init_uniform
      AIfES_E_init_glorot_uniform
      AIfES_E_init_no_init        //If starting weights are already available or if you want to continue training
  */

  FNN_INIT_WEIGHTS.min_init_uniform = -2; // only for the AIfES_E_init_uniform
  FNN_INIT_WEIGHTS.max_init_uniform = 2;  // only for the AIfES_E_init_uniform
  // -------------------------------- set training parameter ----------------------------------
  AIFES_E_training_parameter_fnn_f32  FNN_TRAIN;
  FNN_TRAIN.optimizer = AIfES_E_adam;
  /* optimizers
      AIfES_E_adam
      AIfES_E_sgd
  */
  FNN_TRAIN.loss = AIfES_E_crossentropy;
  /* loss
      AIfES_E_mse,
      AIfES_E_crossentropy
  */
  FNN_TRAIN.learn_rate = 0.1f;                           // Learning rate is for all optimizers
  FNN_TRAIN.sgd_momentum = 0.0;                           // Only interesting for SGD
  FNN_TRAIN.batch_size = NUMBER_OF_DATA;                        // Here a full batch
  FNN_TRAIN.epochs = 1000;                                // Number of epochs
  FNN_TRAIN.epochs_loss_print_interval = PRINT_INTERVAL;  // Print the loss every x times

  //You can enable early stopping, so that learning is automatically stopped when a learning target is reached
  FNN_TRAIN.early_stopping = AIfES_E_early_stopping_on;
  /* early_stopping
      AIfES_E_early_stopping_off,
      AIfES_E_early_stopping_on
  */
  //Define your target loss
  FNN_TRAIN.early_stopping_target_loss = 0.09;

  int8_t error = 0;

  // -------------------------------- do the training ----------------------------------
  // In the training function, the FNN is set up, the weights are initialized and the training is performed.
  error = AIFES_E_training_fnn_f32(&input_tensor,&target_tensor,&FNN,&FNN_TRAIN,&FNN_INIT_WEIGHTS,&output_tensor);  
error = AIFES_E_inference_fnn_f32(&input_tensor,&FNN,&output_tensor);
}

void NN::test() {
  // Tensor for the input RGB
  // Those values are the input of the ANN
  uint16_t input_shape[] = {1, INPUTS};                          // Definition of the shape of the tensor, here: {1 (i.e. 1 sample), 3 (i.e. the sample contains 3 RGB values)}
  aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, features);                 // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // Tensor for the output with 3 classes
  // Output values of the ANN are saved here
  uint16_t output_shape[] = {1, OUTPUTS};                         // Definition of the shape of the tensor, here: {1 (i.e. 1 sample), 3 (i.e. the sample contains predictions for 3 classes/objects)}
  aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version  

  // ----------------------------------------- Run the AIfES model to detect the object --------------------------
  // Run the inference with the trained AIfES model
  int8_t error = 0;
  error = AIFES_E_inference_fnn_f32(&input_tensor,&FNN,&output_tensor);
}

void NN::init_network_model() {
  weight_number = AIFES_E_flat_weights_number_fnn_f32(FNN_structure,FNN_3_LAYERS);
  FlatWeights = (float *)malloc(sizeof(float)*weight_number); 
    FNN_activations[0] = AIfES_E_sigmoid;
    FNN_activations[1] = AIfES_E_softmax;
    float d[259] = {-0.685828, -1.783316, -1.662144, 0.007397, -1.26468, 0.880594, 1.506442, -1.056673, 0.911511, -1.358651, 0.742572, -0.786404, 
-1.538248, 1.119841, 1.021483, -0.747776, -1.059391, 1.047655, 1.439222, 1.768513, 0.860348, 1.031046, -0.31657, 0.887487, 1.379159, 1.194778, -0.286546, 1.145276, -1.140127, 0.74759, 1.846244, 1.543005, -1.03593, -1.347267, -0.478292, 0.146889, -0.1355, -1.643763, 1.529396, 1.125585, -0.936335, 0.531918, -1.40858, 1.502412, 1.706732, 1.944178, 0.595888, 0.027556, 0.435109, 0.904502, -0.6, 0.806599, -1.555702, -0.251561, 0.441761, 0.633928, -0.294682, 1.901297, -0.117716, -0.34301, 1.501639, -0.459149, 
-1.085106, 0.509815, -0.076292, -1.825645, -0.878401, -0.285667, -0.973905, 1.702089, -0.473812, -0.195854, 1.030902, -0.23743, 1.836152, 0.932677, -0.217369, -1.513342, -0.606478, 0.092171, -1.118352, 0.304731, 1.762923, -0.781572, -1.010713, 0.47253, -0.057626, -1.212591, -0.049466, 0.409326, 1.378153, -0.147912, -1.830962, 0.807485, 1.778709, 1.148437, 1.336444, -0.626313, 1.18212, 1.090561, 0.838691, 1.733266, 1.65856, -0.679741, -0.735633, 1.889345, 1.453843, -0.48953, 0.288775, -0.876139, -0.94535, 0.145867, 1.482743, -0.811676, 1.208287, -1.215583, -0.475352, 0.04181, -1.026048, 1.097727, 1.44085, 1.328545, 0.174555, -1.230956, -1.604802, -1.157031, 1.121899, -0.838204, 1.557785, 1.700055, 0.584268, -0.157079, -1.154642, 0.176789, 0.049657, -0.568396, 0.76951, -0.165968, 0.580107, 1.558497, 0.830264, 1.885593, -0.300888, -1.803999, 0.899994, -0.092194, 1.653176, 1.226909, 0.472983, 0.779822, 0.669358, -1.729947, -0.065065, -1.862363, 1.385236, -1.621143, -1.983732, 1.296745, -0.83562, -0.597723, -1.905384, -1.233282, 1.071487, 1.273428, -1.305178, 0.010369, 1.079521, -0.483533, -1.89841, -0.948521, -1.434779, 0.272822, -1.62974, 0.169968, 1.907548, 0.238479, -0.670298, -0.723143, 0.463842, 0.921393, -0.274879, -1.991192, -1.106884, 1.14167, 1.935365, -0.721393, 1.135385, 0.321156, -0.523563, 1.276757, -1.890005, -1.210051, -1.806205, 0.52246, -0.225913, -0.72243, -0.373347, -0.62108, 1.529796, -1.964588, 1.877574, -0.041507, 0.007568, -0.243433, 0.205793, -1.704157, 1.926195, 0.32147, 1.258647, 1.614577, 0.20677, 0.108292, 0.576643, -1.368981, -0.560642, -0.443808, -0.154759, 0.621089, 1.932626, -0.432992, 1.09485, 0.777421, -1.752534, -1.347984, -0.123887, 0.210173, -0.461681, 0.575582, 1.005342, 0.491713, 0.590812, 1.822129, 0.000967, -1.821118, -1.1301, 0.232781, 0.519491, 0.461378, -0.281785, -1.141741, -1.26511, -0.266933, 1.355551, -1.679045, -1.94919, -1.333919, 0.770498, 1.404865, 1.077812, 1.269928, 1.952954, -0.508618, -0.2804, 0.874999, 0.524199, 0.724194, 1.517417, -0.910454, 0.828882};
    for (uint16_t i = 0; i < weight_number; ++i) {
        char a[sizeof(float)];
        memcpy(a, &d[i], sizeof(float));
        for (int n = 0; n < 4; n++) {
            ((char*)FlatWeights)[i*4+n] = a[n];
        }
    }
    FNN.layer_count = FNN_3_LAYERS;
    FNN.fnn_structure = FNN_structure;
    FNN.fnn_activations = FNN_activations;
    FNN.flat_weights = FlatWeights;

      for(int i=0;i<5;i++){
    labels[i][0] = 1;
    labels[i][1] = 0;
    labels[i][2] = 0;
  }
   for(int i=5;i<10;i++){
    labels[i][0] = 0;
    labels[i][1] = 1;
    labels[i][2] = 0;
  }
   for(int i=10;i<15;i++){
    labels[i][0] = 0;
    labels[i][1] = 0;
    labels[i][2] = 1;
  }
}

void NN::modify_Weights(float* flatWeights){
  FNN.flat_weights = flatWeights;
}


void NN::recordTestData(){
bool motion_detected = false;
        float ax, ay, az;
        while(!motion_detected){        
        if (IMU.accelerationAvailable()) {
          IMU.readAcceleration(ax, ay, az);
        }
        
        ax = constrain(ax - baseline[0], -TRUNCATE, TRUNCATE);
        ay = constrain(ay - baseline[1], -TRUNCATE, TRUNCATE);
        az = constrain(az - baseline[2], -TRUNCATE, TRUNCATE);
  
    
        if (motionDetected(ax, ay, az)) {
          motion_detected = true;
        }
        else{
          delay(10);
        }
    }
    recordIMU();
    motion_detected = false;
}

bool NN::recordTrainData() {
  float ax, ay, az; 
  if (IMU.accelerationAvailable()) {
          IMU.readAcceleration(ax, ay, az);
        }
        
        ax = constrain(ax - baseline[0], -TRUNCATE, TRUNCATE);
        ay = constrain(ay - baseline[1], -TRUNCATE, TRUNCATE);
        az = constrain(az - baseline[2], -TRUNCATE, TRUNCATE);

    if (motionDetected(ax, ay, az)) {
      Serial.println("hi");
      recordIMU();
      printAndSafeFeatures();
      training_record = training_record + 1;
      return true;
    }
    return false;
}


bool NN::motionDetected(float ax, float ay, float az) {
    return (abs(ax) + abs(ay) + abs(az)) > ACCEL_THRESHOLD;
}


void NN::recordIMU() {
    float ax, ay, az;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        if (IMU.accelerationAvailable()) {
          IMU.readAcceleration(ax, ay, az);
  
        }

        ax = constrain(ax - baseline[0], -TRUNCATE, TRUNCATE);
        ay = constrain(ay - baseline[1], -TRUNCATE, TRUNCATE);
        az = constrain(az - baseline[2], -TRUNCATE, TRUNCATE);

        features[i * NUM_AXES + 0] = ax;
        features[i * NUM_AXES + 1] = ay;
        features[i * NUM_AXES + 2] = az;

        delay(INTERVAL);
    }
}

void NN::printAndSafeFeatures() {
    const uint16_t numFeatures = sizeof(features) / sizeof(float);
     
    for (int i = 0; i < numFeatures; i++) {
        training_data[training_record][i] = features[i];
    }
}