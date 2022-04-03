#include "network.h"

neural_network::network::network(const neural_network::network_descriptor_Ptr &descriptor)
    :   layers({descriptor->networkLayers}),
        trainDataSet(descriptor->networkTrainData),
        learningRate(descriptor->learningRate) {

}

void neural_network::network::feedForward() {
    for (int i = 0; i < layers.size() - 2; ++i) {

        layers[i + 1]->updateInputs((*layers[i]->weights()) * (*layers[i]->inputs()));
        layers[i + 1]->activate();
    }

    size_t last = layers.size() - 2;
    size_t outputs = layers.size() - 1;
    layers[outputs]->updateInputs((*layers[last]->weights()) * (*layers[last]->inputs()) + (*layers[last]->bias()));
    layers[outputs]->activate();
}

void neural_network::network::backProp(const math::vector_dPtr& expected) {
    std::vector<math::vector_d> hiddenDeltas = computeHiddenDeltas(deltaOutputs(expected));

    computeDerivativeWithRespectToWeights(hiddenDeltas);
}

double neural_network::network::error(const math::vector_dPtr& expected) {
    math::vector_d diff = (*expected) - (*layers[layers.size() - 1]->inputs());
    return 0.5 * diff.sum_squared();
}

void neural_network::network::output() {
    std::cout << "Network output:\n" << *layers[layers.size() - 1]->inputs();
}

math::vector_d neural_network::network::deltaOutputs(const math::vector_dPtr& expected) {
    size_t outputIndex = layers.size() - 1;

    math::vector_d deltas{layers[outputIndex]->inputs()->size()};

    for (int i = 0; i < deltas.size(); ++i) {
        double deriv = layers[outputIndex]->derivative()(layers[outputIndex]->inputs()->at(i));

        deltas[i] = (expected->at(i) - layers[outputIndex]->inputs()->at(i)) * deriv;
    }

    return deltas;
}

std::vector<math::vector_d> neural_network::network::computeHiddenDeltas(const math::vector_d &outputDeltas) {
    std::vector<math::vector_d> deltas;

    for (auto &layer : layers) {
        deltas.emplace_back(layer->inputs()->size());
    }

    deltas[deltas.size() - 1] = outputDeltas;

    for (int layerIndex = (int) layers.size() - 2; layerIndex > 0; --layerIndex) {
        const auto &currentLayer = layers[layerIndex];
        const auto &nextLayer = layers[layerIndex + 1];

        for (int j = 0; j < deltas[layerIndex].size(); ++j) {
            double sum = 0;
            for (int k = 0; k < deltas[layerIndex + 1].size(); ++k) {
                double deriv = currentLayer->derivative()(currentLayer->inputs()->at(j));
                double weight = currentLayer->weights()->at(k, j) * deriv;
                sum += deltas[layerIndex + 1][k] * weight;
            }

            deltas[layerIndex][j] = sum;
        }
    }

    return deltas;
}

void neural_network::network::computeDerivativeWithRespectToWeights(const std::vector<math::vector_d> &hiddenDeltas) {
    for (int layerIndex = (int) layers.size() - 1; layerIndex >= 1; --layerIndex) {
        const auto &layer = layers[layerIndex];
        auto &prevLayer = layers[layerIndex - 1];

        for (int j = 0; j < layer->inputs()->size(); ++j) {
            for (int k = 0; k < prevLayer->inputs()->size(); ++k) {
                double delta = hiddenDeltas[layerIndex][j];
                double ak = prevLayer->inputs()->at(k);
                prevLayer->updateDerivativeAt(j, k,  prevLayer->getDerivativeAt(j, k) + delta * ak);
            }
        }
    }
}

void neural_network::network::updateWeights() {
    for (int layerIndex = (int) layers.size() - 2; layerIndex >= 0; --layerIndex) {
        auto &layer = layers[layerIndex];

        for (int i = 0; i < layer->weights()->rowCount(); ++i) {
            for (int j = 0; j < layer->weights()->columnCount(); ++j) {
                layer->updateWeightAt(i, j,  learningRate, trainDataSet.size());
            }
        }
    }
}

void neural_network::network::setInputs(const math::vector_d &inputs) {
    layers.at(0)->updateInputs(inputs);
}

void neural_network::network::train(int epochsCount) {
    for (int i = 0; i < epochsCount; ++i) {
        for (auto & trainingData : trainDataSet) {
            setInputs(*trainingData->input);
            feedForward();

            std::cout << "Error: " << error(trainingData->expected) << "\n";

            backProp(trainingData->expected);
        }

        updateWeights();

        for (auto& layer : layers) {
            layer->clearDerivatives();
        }
    }
}

void neural_network::network::test() {
    for (auto & trainingData : trainDataSet) {
        setInputs(*trainingData->input);
        feedForward();
        std::cout << "Input: " << *trainingData->input;
        std::cout << "Expected: " << *trainingData->expected;
        std::cout << "Output: " << *layers[layers.size() - 1]->inputs() << '\n';
    }
}
