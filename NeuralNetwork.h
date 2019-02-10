#pragma once

#include "Matrix.h"
#include "Dictionary.h"
#include "layer.h"

#ifndef NEURALNETWORK_H_INCLUDED
#define NEURALNETWORK_H_INCLUDED

typedef Dictionary< string, matrix<float> > dictionary;
typedef matrix<float> Matrix;

class NeuralNetwork
{
private:
     layer* layers;               // Array of layers holding the number of neurons at each layer and its activation
     dictionary parameters;       // Dictionary containing weights and biases of the network
     dictionary cache;            // Dictionary containing temporary internal activations of layers
     dictionary grades;
     string ErrorType;            // Type of cost function or performance index used
     string optimizer;            // Type of algorithm used
	 int numOfLayers;             // Number of layers
     bool momentum;               // Indicates whether momentum is used or not
     float maxErr;

public:
    NeuralNetwork(layer* mylayers,int L);                               // Constructor that initializes the weighs and biases of the network randomly based on the architecture of the network
    void test(Matrix X_test, Matrix Y, string path,bool batchNorm);     // Function that outputs the input training set X, their associated targets Y and the final activations Y_hat into a text file
    void print();                                                       // Prints all parameters of the network
    void train(const Matrix& X, Matrix& Y, float alpha, int numOfEpochs, int minibatchSize, string optimizer1,int Numprint, string ET,float lambda,bool batchNorm);
    // Function that trains the network based on the following arguments:
    // X: input training set
    // Y: target associated with the input training set
    // alpha: learning rate or damping ratio in case of LM optimizer
    // numOfEpochs: maximum number of iterations
    // minibatchsize: size of mini-batch (don't care if LM algorithm)
    // optimizer: the algorithm used to train the network. It's either "GradientDescent", "Adam" or "LM"
    // Numprint: the number of iterations after which the cost is outputted on the screen
    // ET: the cost function or performance index. It's either "CrossEntropy" or "SquareErr"


private:
    // Feed forward
    Matrix feedforward(const Matrix& x, layer* layers,int L,bool batchNorm);
    // Back propagation
    void calGrads(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, layer* layers, int L,float lambda,bool batchNorm);
    void updateParameters(float& alpha, layer* layers, int L, int iteration, Matrix& Q, Matrix& g, int m,bool batchNorm);
    void BackProp(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, float& alpha, layer* layers, int L, int iteration, Matrix& Q, Matrix& g, int m,float lambda,bool batchNorm);
    // Cost
    float CostFunc(const Matrix& y,Matrix& yhat,float lambda,int L);
    Matrix costMul(Matrix Y, Matrix Y_hat);
    // Classify
    Matrix classify(Matrix Y_hat);
    Matrix Classify_Train(Matrix Y_hat);
    void AccuracyTest(const Matrix& Y,Matrix& Y_hat_classified, const Matrix& X, layer*layers, int L);
    // Error
    float AbsErr(Matrix* Y_hat, Matrix* Y);
    float numOfErrs(Matrix* Y_hat, Matrix* Y);
    // Store data
    void storedata(Matrix X, Matrix Y, Matrix Yhat, Matrix yhat_classified, string path);
};
#endif // NEURALNETWORK_H_INCLUDED
