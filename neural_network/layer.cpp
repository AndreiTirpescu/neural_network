#include "layer.h"

neural_network::layer::layer(const layer_descriptor &descriptor)
        : inputVector(new math::vector_d(descriptor.numberOfInputs)),
          weightMatrix(new math::matrix_d(descriptor.weightGenerator(descriptor.weightRowCount, descriptor.weightColCount))),
          activationFunction(descriptor.activation),
          derivativeFunction(descriptor.activation_derivative),
          derivativeWithRespectToWeights(new math::matrix_d(weightMatrix->rowCount(), weightMatrix->columnCount())),
          biases(new math::vector_d(math::vector_d{descriptor.weightColCount})){}

math::vector_dPtr neural_network::layer::inputs() const {
    return inputVector;
}

math::matrix_dPtr neural_network::layer::weights() const {
    return weightMatrix;
}

math::vector_dPtr neural_network::layer::bias() const {
    return biases;
}

void neural_network::layer::updateInputs(const math::vector_d& updated) {
    for (int i = 0; i < inputVector->size(); ++i) {
        inputVector->at(i) = updated.at(i);
    }
}

const neural_network::activation_function &neural_network::layer::derivative() const {
    return derivativeFunction;
}

void neural_network::layer::activate() {
    for (int i = 0; i < inputVector->size(); ++i) {
        inputVector->at(i) = activationFunction(inputVector->at(i));
    }
}

void neural_network::layer::updateDerivativeAt(int i, int j, double val) {
    derivativeWithRespectToWeights->at(i, j) = val;
}

void neural_network::layer::updateWeightAt(int i, int j, double learningRate, int trainingDataSetCount) {

    double average =  ( (1.0 / (double)trainingDataSetCount) * derivativeWithRespectToWeights->at(i, j));

    weightMatrix->at(i, j) = weightMatrix->at(i, j) + learningRate *  average;
}

double neural_network::layer::getDerivativeAt(int i, int j) const {
    return derivativeWithRespectToWeights->at(i, j);
}

void neural_network::layer::clearDerivatives() {
    for (int i = 0; i < derivativeWithRespectToWeights->rowCount(); ++i) {
        for (int j = 0; j < derivativeWithRespectToWeights->columnCount(); ++j) {
            derivativeWithRespectToWeights->at(i, j) = 0.0;
        }
    }
}

