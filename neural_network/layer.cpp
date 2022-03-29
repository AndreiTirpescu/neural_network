//
// Created by andrei on 29.03.2022.
//

#include "layer.h"

neural_network::layer::layer(math::vector_d inputs, math::matrix_d weights, math::vector_d bias)
        : inputVector(std::move(inputs)),
          weightMatrix(std::move(weights)),
          biases(std::move(bias)),
          derivativeWithRespectToWeights(math::matrix_d(weightMatrix.rowCount(), weightMatrix.columnCount())) {}

neural_network::layer::layer(math::vector_d inputs, math::matrix_d weights, math::vector_d bias,
                             neural_network::activation_function activation,
                             neural_network::derivative_function derivative)
        : inputVector(std::move(inputs)),
          weightMatrix(std::move(weights)),
          biases(std::move(bias)),
          activationFunction(std::move(activation)),
          derivativeFunction(std::move(derivative)),
          derivativeWithRespectToWeights(math::matrix_d(weightMatrix.rowCount(), weightMatrix.columnCount())) {}

const math::vector_d &neural_network::layer::inputs() const {
    return inputVector;
}

const math::matrix_d &neural_network::layer::weights() const {
    return weightMatrix;
}

const math::vector_d &neural_network::layer::bias() const {
    return biases;
}

void neural_network::layer::updateInputs(const math::vector_d &updated) {
    for (int i = 0; i < inputVector.size(); ++i) {
        inputVector[i] = updated[i];
    }
}

const neural_network::activation_function &neural_network::layer::derivative() const {
    return derivativeFunction;
}

void neural_network::layer::activate() {
    for (int i = 0; i < inputVector.size(); ++i) {
        inputVector[i] = activationFunction(inputVector[i]);
    }
}

void neural_network::layer::updateDerivativeAt(int i, int j, double val) {
    derivativeWithRespectToWeights[{i, j}] = val;
}

void neural_network::layer::updateWeightAt(int i, int j, double learningRate) {
    weightMatrix[{i, j}] = weightMatrix[{i, j}] +  learningRate *  derivativeWithRespectToWeights[{i, j}];
}
