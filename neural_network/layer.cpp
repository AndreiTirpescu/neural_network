//
// Created by andrei on 29.03.2022.
//

#include "layer.h"

neural_network::layer::layer(const math::vector_d& inputs, const math::matrix_d& weights, const math::vector_d& bias)
        : inputVector(new math::vector_d(inputs)),
          weightMatrix(new math::matrix_d(weights)),
          biases(new math::vector_d(bias)),
          derivativeWithRespectToWeights(new math::matrix_d(weightMatrix->rowCount(), weightMatrix->columnCount())) {}

neural_network::layer::layer(const math::vector_d& inputs, const math::matrix_d& weights, const math::vector_d& bias,
                             neural_network::activation_function activation,
                             neural_network::derivative_function derivative)
        : inputVector(new math::vector_d(inputs)),
          weightMatrix(new math::matrix_d(weights)),
          biases(new math::vector_d(bias)),
          derivativeWithRespectToWeights(new math::matrix_d(weightMatrix->rowCount(), weightMatrix->columnCount())),
          activationFunction(std::move(activation)),
          derivativeFunction(std::move(derivative)) {}

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

void neural_network::layer::updateWeightAt(int i, int j, double learningRate) {
    weightMatrix->at(i, j) = weightMatrix->at(i, j) +  learningRate *  derivativeWithRespectToWeights->at(i, j);
}
