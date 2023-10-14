#ifndef NEURAL_NETWORK

#define NEURAL_NETWORK


/**********************
 * Network Configuration - customized per network 
 **********************/

static const int InputNodes = 30;
static const int n_H_prev = 3;
static const int n_W_prev = 10;
static const int Stride = 1;
static const int n_H = (n_H_prev - 2) / Stride + 1;
static const int n_W = (n_W_prev - 2) / Stride + 1;
static const int HiddenNodes = n_H * n_W; // Convolution ouput
static const int Hidden1Nodes = 5; // dense layer
static const int OutputNodes = 3; // ouput layer
static const float InitialWeightMax = 0.5;
static const int total_weights = (HiddenNodes+1) * Hidden1Nodes + (Hidden1Nodes+1) * OutputNodes + 4;


class NeuralNetwork {
    public:

        void initialize(float LearningRate, float Momentum);
        // ~NeuralNetwork();

        void initWeights();
        void forward(const float Input[]);
        void backward(const float Input[]);
        float forward(const float Input[], const float Target[]);

        float* get_output();
        float* get_OutputWeights();

        float* get_HiddenWeights();

        float* get_Hidden1Weights();
        
        float* get_Error();
        
        float Hidden[HiddenNodes] = {}; // Ouput of the convolution
        // float Hidden[]
        float HiddenWeights[4] = {}; // filter
        float Error[HiddenNodes] = {};
        float KernelDelta[4]={}; 
        float HiddenDelta[HiddenNodes] = {}; // derivative of hidden layer
        float Hidden1Delta[Hidden1Nodes] = {};  // derivative of hidden1 layer
        float OutputDelta[OutputNodes] = {}; // derivative ouput of layer
        float LearningRate = 0.05;
        float Momentum = 0.6;
        float Hidden1[Hidden1Nodes] = {}; // dense layer
        float Output[OutputNodes] = {}; // Ouput layer
        float Hidden1Weights[(HiddenNodes+1) * Hidden1Nodes] = {};
        float OutputWeights[(Hidden1Nodes+1) * OutputNodes] = {};
        float ChangeHiddenWeights[4] = {}; 
        float ChangeHidden1Weights[(HiddenNodes+1) * Hidden1Nodes] = {};
        float ChangeOutputWeights[(Hidden1Nodes+1) * OutputNodes] = {}; 
        
};


#endif