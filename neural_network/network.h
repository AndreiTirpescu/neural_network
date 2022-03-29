#pragma once

#include "math/vector.h"
#include "math/matrix.h"

#include <utility>
#include <vector>

#include <iostream>

namespace neural_network {
    using activation_function = std::function<double(double)>;
    using derivative_function = std::function<double(double)>;

    class layer {
    public:
        layer(math::vector_d inputs, math::matrix_d weights, math::vector_d bias)
                : inputVector(std::move(inputs)),
                  weightMatrix(std::move(weights)),
                  biases(std::move(bias)),
                  derivativeWithRespectToWeights(math::matrix_d(weightMatrix.rowCount(), weightMatrix.columnCount())) {}

        layer(math::vector_d inputs, math::matrix_d weights, math::vector_d bias, activation_function activation,
              derivative_function derivative)
                : inputVector(std::move(inputs)),
                  weightMatrix(std::move(weights)),
                  biases(std::move(bias)),
                  activationFunction(std::move(activation)),
                  derivativeFunction(std::move(derivative)),
                  derivativeWithRespectToWeights(math::matrix_d(weightMatrix.rowCount(), weightMatrix.columnCount())) {}

        [[nodiscard]] const math::vector_d &inputs() const {
            return inputVector;
        }

        [[nodiscard]] const math::matrix_d &weights() const {
            return weightMatrix;
        }

        [[nodiscard]] const math::vector_d &bias() const {
            return biases;
        }

        void updateInputs(const math::vector_d &updated) {
            for (int i = 0; i < inputVector.size(); ++i) {
                inputVector[i] = updated[i];
            }
        }

        [[nodiscard]] const activation_function &activation() const {
            return activationFunction;
        }

        [[nodiscard]] const activation_function &derivative() const {
            return derivativeFunction;
        }

        void activate() {
            for (int i = 0; i < inputVector.size(); ++i) {
                inputVector[i] = activationFunction(inputVector[i]);
            }
        }

        void updateDerivativeAt(int i, int j, double val) {
            derivativeWithRespectToWeights[{i, j}] = val;
        }

        void updateWeightAt(int i, int j, double learningRate) {
            weightMatrix[{i, j}] = weightMatrix[{i, j}] +  learningRate *  derivativeWithRespectToWeights[{i, j}];
        }

        [[nodiscard]] const math::matrix_d& derivatives() const {
            return derivativeWithRespectToWeights;
        }

    private:
        math::vector_d inputVector;
        math::vector_d biases;
        math::matrix_d weightMatrix;
        math::matrix_d derivativeWithRespectToWeights;

        activation_function activationFunction;
        derivative_function derivativeFunction;

    };

    class network {

    public:
        network(layer inputs, std::vector<layer> hiddenLayers,
                math::vector_d expected, double learningRate) : layers(std::move(hiddenLayers)),
                                                                expected(std::move(expected)),
                                                                learningRate(learningRate) {
            layers.insert(layers.begin(), std::move(inputs));
        }

        void feedForward() {
            for (int i = 0; i < layers.size() - 2; ++i) {

                layers[i + 1].updateInputs(layers[i].weights() * layers[i].inputs());
                layers[i + 1].activate();
            }

            size_t last = layers.size() - 2;
            size_t outputs = layers.size() - 1;
            layers[outputs].updateInputs(layers[last].weights() * layers[last].inputs() + layers[last].bias());
            layers[outputs].activate();
        }

        void backProp() {
            std::vector<math::vector_d> hiddenDeltas = computeHiddenDeltas(deltaOutputs());

            computeDerivativeWithRespectToWeights(hiddenDeltas);

            updateWeights();
        }

        double error() {
            math::vector_d diff = expected - layers[layers.size() - 1].inputs();
            return 0.5 * diff.sum_squared();
        }

        void output() {
            std::cout << "Network output:\n" << layers[layers.size() - 1].inputs();
        }

    private:
        std::vector<layer> layers;
        math::vector_d expected;
        double learningRate;

        math::vector_d deltaOutputs() {
            size_t outputIndex = layers.size() - 1;

            math::vector_d deltas{layers[outputIndex].inputs().size()};

            for (int i = 0; i < deltas.size(); ++i) {

                double deriv = layers[outputIndex].derivative()(layers[outputIndex].inputs()[i]);

                deltas[i] = (expected[i] - layers[outputIndex].inputs()[i]) * deriv;
            }

            return deltas;
        }

        std::vector<math::vector_d> computeHiddenDeltas(const math::vector_d &outputDeltas) {
            std::vector<math::vector_d> deltas;

            for (auto &layer : layers) {
                deltas.emplace_back(layer.inputs().size());
            }

            deltas[deltas.size() - 1] = outputDeltas;

            for (int layerIndex = (int) layers.size() - 2; layerIndex > 0; --layerIndex) {
                const auto& currentLayer = layers[layerIndex];
                const auto& nextLayer = layers[layerIndex + 1];

                for (int j = 0; j < deltas[layerIndex].size(); ++j) {
                    double sum = 0;
                    for (int k = 0; k < deltas[layerIndex + 1].size(); ++k) {
                        double deriv = currentLayer.derivative()(currentLayer.inputs()[j]);
                        double weight = currentLayer.weights().at(k, j) * deriv;
                        sum += deltas[layerIndex + 1][k] * weight;
                    }

                    deltas[layerIndex][j] = sum;
                }
            }

            return deltas;
        }

        void computeDerivativeWithRespectToWeights(const std::vector<math::vector_d> &hiddenDeltas) {
            for (int layerIndex = (int) layers.size() - 1; layerIndex >= 1; --layerIndex) {
                const auto &layer = layers[layerIndex];
                auto &prevLayer = layers[layerIndex - 1];

                for (int j = 0; j < layer.inputs().size(); ++j) {
                    for (int k = 0; k < prevLayer.inputs().size(); ++k) {
                        double delta = hiddenDeltas[layerIndex][j];
                        double ak = prevLayer.inputs()[k];
                        prevLayer.updateDerivativeAt(j, k, delta * ak);
                    }
                }
            }
        }

        void updateWeights() {
            for (int layerIndex = (int) layers.size() - 2; layerIndex >= 0; --layerIndex) {
                auto &layer = layers[layerIndex];

                for (int i = 0; i < layer.weights().rowCount(); ++i) {
                    for (int j = 0; j < layer.weights().columnCount(); ++j) {
                        layer.updateWeightAt(i, j, learningRate);
                    }
                }
            }
        }
    };

}
