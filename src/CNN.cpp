#include <Arduino.h>
#include "CNN.h"
#include <math.h>

void NeuralNetwork::initialize(float LearningRate, float Momentum) { // Initialize learning rate and momentum
    this->LearningRate = LearningRate;
    this->Momentum = Momentum;
}
 
void NeuralNetwork::initWeights() {  // Initialize random weights
        for (int h = 0; h < 2; h++){
            for (int w = 0; w < 2; w++){
                ChangeHiddenWeights[h * 2 + w] = 0.0;
                float Rando = float(random(100))/100;
                HiddenWeights[h*2+w] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
            }
        }
        for(int i = 0 ; i < Hidden1Nodes ; i++ ) {    
        for(int j = 0 ; j <= HiddenNodes ; j++ ) { 
            ChangeHidden1Weights[j*Hidden1Nodes + i] = 0.0 ;
            float Rando = float(random(100))/100;
            Hidden1Weights[j*Hidden1Nodes + i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    for(int i = 0 ; i < OutputNodes ; i++ ) {    
        for(int j = 0 ; j <= Hidden1Nodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = 0.0 ;  
            float Rando = float(random(100))/100;        
            OutputWeights[j*OutputNodes + i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
}


float NeuralNetwork::forward(const float Input[], const float Target[]){ // Forward propagation
    /**********************
    * Compute hidden layer activations and calculate activations
    **********************/
   float error = 0;
   /**********************
    * Compute convolution
    **********************/
    for (int h = 0; h < n_H; h++){
        for(int w = 0 ; w < n_W ; w++ ) {  
                
                float partial_sum = 0.0;

                partial_sum += HiddenWeights[0] * Input[(h * Stride) * n_W_prev + w*Stride ];// h = 0, w = 0
                partial_sum += HiddenWeights[1] * Input[(h * Stride) * n_W_prev + w*Stride + 1];// h = 0, w = 1

                partial_sum += HiddenWeights[2] * Input[(h * Stride + 1) * n_W_prev + w*Stride ];// h = 1, w = 0
                partial_sum += HiddenWeights[3] * Input[(h * Stride + 1) * n_W_prev + w*Stride + 1];// h = 1, w = 1
                
                if(partial_sum<0)
                {
                    Hidden[h * n_W + w]=0;
                }
                else
                {
                Hidden[h * n_W + w] = partial_sum;
                }
        }
    }

  //  float error = 0;

    /**********************
    * Compute hidden layer 1 activations
    **********************/
    for (int i = 0; i < Hidden1Nodes; i++) {
        float Accum = Hidden1Weights[HiddenNodes*Hidden1Nodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * Hidden1Weights[j*Hidden1Nodes + i];
        }
        Hidden1[i] = 1.0 / (1.0 + exp(-Accum));
    }

    /**********************
    * Compute output layer activations and calculate errors
    **********************/
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[Hidden1Nodes*OutputNodes + i];
        for (int j = 0; j < Hidden1Nodes; j++) {
            Accum += Hidden1[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]); //(output - expected) * transfer_derivative(output)
        error += 0.33333 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
   return error;
}

void NeuralNetwork::forward(const float Input[]){
    /**********************
    * Compute hidden layer activations and calculate activations
    **********************/
    for (int h = 0; h < n_H; h++){
        for(int w = 0 ; w < n_W ; w++ ) {  
                
                float partial_sum = 0.0;

                partial_sum += HiddenWeights[0] * Input[(h * Stride) * n_W_prev + w*Stride ];// h = 0, w = 0
                partial_sum += HiddenWeights[1] * Input[(h * Stride) * n_W_prev + w*Stride + 1];// h = 0, w = 1

                partial_sum += HiddenWeights[2] * Input[(h * Stride + 1) * n_W_prev + w*Stride ];// h = 1, w = 0
                partial_sum += HiddenWeights[3] * Input[(h * Stride + 1) * n_W_prev + w*Stride + 1];// h = 1, w = 1

                
                Hidden[h * n_W + w] = partial_sum;
        }
    }

    float error = 0;

    /**********************
    * Compute hidden layer 1 activations
    **********************/
    for (int i = 0; i < Hidden1Nodes; i++) {
        float Accum = Hidden1Weights[HiddenNodes*Hidden1Nodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * Hidden1Weights[j*Hidden1Nodes + i];
        }
        Hidden1[i] = 1.0 / (1.0 + exp(-Accum));
    }

    /**********************
    * Compute output layer activations and calculate errors
    **********************/
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[Hidden1Nodes*OutputNodes + i];
        for (int j = 0; j < Hidden1Nodes; j++) {
            Accum += Hidden1[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        //OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        //error += 0.33333 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
}

void NeuralNetwork::backward(const float Input[]){
// Backward
    /**********************
    * Backpropagate errors to hidden layer
    **********************/
    for(int i = 0 ; i < Hidden1Nodes ; i++ ) {    
        float Accum = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            Accum += OutputWeights[i*OutputNodes + j] * OutputDelta[j] ; //weighted error
        }
        // Serial.print("accum");
        // Serial.print(Accum);
        // Serial.print(" ");
        // Serial.print("hid");
        // Serial.print( Hidden1[i]);
        // Serial.print(" ");
        Hidden1Delta[i] = Accum * Hidden1[i] * (1.0 - Hidden1[i]) ; //weighted error*derivative
        // Serial.print("hidd");
        // Serial.print( Hidden1Delta[i]);
        // Serial.print(" ");
    }

    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        float Accum = 0.0 ;
        for(int j = 0 ; j < Hidden1Nodes ; j++ ) {
            Accum += Hidden1Weights[i*Hidden1Nodes + j] * Hidden1Delta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
    }

//Apply Momentum update base
        for (int f = 0; f < 4; f++){
                ChangeHiddenWeights[f] = Momentum * ChangeHiddenWeights[f];
        }
// Calculating delta for the kernel
for (int h = 0; h < 2; h++){
        for(int w = 0 ; w < 2; w++ ) {  
                
                float partial_sum = 0.0;
                int size =2;
                partial_sum += HiddenDelta[0] * Input[(h * Stride) * n_W_prev + w*Stride ];
                partial_sum += HiddenDelta[1] * Input[(h * Stride) * n_W_prev + w*Stride + 1];
                partial_sum += HiddenDelta[2] * Input[(h * Stride) * n_W_prev + w*Stride + 2];
                partial_sum += HiddenDelta[3] * Input[(h * Stride) * n_W_prev + w*Stride + 3];
                partial_sum += HiddenDelta[4] * Input[(h * Stride) * n_W_prev + w*Stride + 4];
                partial_sum += HiddenDelta[5] * Input[(h * Stride) * n_W_prev + w*Stride + 5];
                partial_sum += HiddenDelta[6] * Input[(h * Stride) * n_W_prev + w*Stride + 6];
                partial_sum += HiddenDelta[7] * Input[(h * Stride) * n_W_prev + w*Stride + 7];
                partial_sum += HiddenDelta[8] * Input[(h * Stride) * n_W_prev + w*Stride + 8];

                partial_sum += HiddenDelta[9] * Input[(h * Stride + 1) * n_W_prev + w*Stride+1 ];
                partial_sum += HiddenDelta[10] * Input[(h * Stride + 1) * n_W_prev + w*Stride+2 ];
                partial_sum += HiddenDelta[11] * Input[(h * Stride + 1) * n_W_prev + w*Stride+3 ];
                partial_sum += HiddenDelta[12] * Input[(h * Stride + 1) * n_W_prev + w*Stride+4 ];
                partial_sum += HiddenDelta[13] * Input[(h * Stride + 1) * n_W_prev + w*Stride+5 ];
                partial_sum += HiddenDelta[14] * Input[(h * Stride + 1) * n_W_prev + w*Stride+6 ];
                partial_sum += HiddenDelta[15] * Input[(h * Stride + 1) * n_W_prev + w*Stride+7 ];
                partial_sum += HiddenDelta[16] * Input[(h * Stride + 1) * n_W_prev + w*Stride+8 ];
                partial_sum += HiddenDelta[17] * Input[(h * Stride + 1) * n_W_prev + w*Stride+9 ];

                
                    KernelDelta[h*size + w] = partial_sum;
                
        }
    }
    //Calculate Updates
    for (int h = 0; h < 2; h++){
        for(int w = 0 ; w < 2; w++ ) { 
            int size=2; 
                ChangeHiddenWeights[h*size + w] += (1 - Momentum)* LearningRate * KernelDelta[h*size + w];
                }
    }

    //Apply Updates
        for (int f = 0; f < 4; f++){
                HiddenWeights[f] -= ChangeHiddenWeights[f];
        }
    /**********************
    * Update Inner-->Hidden Weights
    **********************/
    for(int i = 0 ; i < Hidden1Nodes ; i++ ) {     
        ChangeHidden1Weights[HiddenNodes*Hidden1Nodes + i] = LearningRate * Hidden1Delta[i] + Momentum * ChangeHidden1Weights[HiddenNodes*Hidden1Nodes + i] ;
        Hidden1Weights[HiddenNodes*Hidden1Nodes + i] += ChangeHidden1Weights[HiddenNodes*Hidden1Nodes + i] ; // Bias Update
        for(int j = 0 ; j < HiddenNodes ; j++ ) { 
            ChangeHidden1Weights[j*Hidden1Nodes + i] = LearningRate * Hidden[j] * Hidden1Delta[i] + Momentum * ChangeHidden1Weights[j*Hidden1Nodes + i];
            Hidden1Weights[j*Hidden1Nodes + i] += ChangeHidden1Weights[j*Hidden1Nodes + i] ; // Weight Update
        }
    }

    /**********************
    * Update Hidden-->Output Weights
    **********************/
    for(int i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[Hidden1Nodes*OutputNodes + i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[Hidden1Nodes*OutputNodes + i] ;
        OutputWeights[Hidden1Nodes*OutputNodes + i] += ChangeOutputWeights[Hidden1Nodes*OutputNodes + i] ;
        for(int j = 0 ; j < Hidden1Nodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = LearningRate * Hidden1[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j*OutputNodes + i] ;
            OutputWeights[j*OutputNodes + i] += ChangeOutputWeights[j*OutputNodes + i] ;
        }
    }
    


        
}


float* NeuralNetwork::get_output(){
    return Output;
}

float* NeuralNetwork::get_HiddenWeights(){
    return HiddenWeights;
}

float* NeuralNetwork::get_Hidden1Weights(){
    return Hidden1Weights;
}

float* NeuralNetwork:: get_OutputWeights(){
    return OutputWeights;
}

float* NeuralNetwork::get_Error(){
    return Error;
}