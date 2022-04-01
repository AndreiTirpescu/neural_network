//
// Created by andrei on 01.04.2022.
//

#include "network_builder.h"
#include "math/utils/random_matrix_generator.h"
#include "math/utils/empty_matrix_generator.h"

neural_network::network_builder &
neural_network::network_builder::withTrainData(const math::vector_d &inputs, const math::vector_d &outputs) {
    networkDescriptor->networkTrainData.push_back(std::make_shared<train_data>(train_data{inputs, outputs}));

    return *this;
}

neural_network::network_builder &
neural_network::network_builder::withHiddenLayer(int numberOfNeurons, int numberOfNeuronsNext,
                                                 const neural_network::weight_generator &generator,
                                                 const neural_network::activation_function &activation,
                                                 const neural_network::activation_function &derivative) {

    networkDescriptor->networkLayers.push_back(std::make_shared<layer>(
            layer({
                          numberOfNeurons,
                          numberOfNeuronsNext,
                          numberOfNeurons,
                          generator,
                          activation,
                          derivative
                  })
    ));
    return *this;
}

neural_network::network_builder &
neural_network::network_builder::withRandomWeightsAndSigmoidHiddenLayer(int numberOfNeurons, int numberOfNeuronsNext) {
    networkDescriptor->networkLayers.push_back(std::make_shared<layer>(
            layer({
                          numberOfNeurons,
                          numberOfNeuronsNext,
                          numberOfNeurons,
                          math::utils::random_matrix_generator::generate,
                          math::functions::sigmoid::activate,
                          math::functions::sigmoid::derivative
                  })
    ));

    return *this;
}

neural_network::network_builder &neural_network::network_builder::withLearningRate(double learningRate) {
    networkDescriptor->learningRate = learningRate;

    return *this;
}

neural_network::network_builder &
neural_network::network_builder::withInputLayer(int inputLayerSize, const neural_network::weight_generator &generator) {
    networkDescriptor->networkLayers.insert(networkDescriptor->networkLayers.begin(), std::make_shared<layer>(
            layer({
                          inputLayerSize,
                          networkDescriptor->networkLayers.at(0)->inputs()->size(),
                          inputLayerSize,
                          generator
                  })
    ));

    return *this;
}

neural_network::network_builder &neural_network::network_builder::withOutputLayer() {
    int outputLayerSize = networkDescriptor->networkTrainData.at(0)->expected.size();

    networkDescriptor->networkLayers.push_back(std::make_shared<layer>(
            layer({
                outputLayerSize,
                0,
                0,
                math::utils::empty_matrix_generator::generate,
                math::functions::sigmoid::activate,
                math::functions::sigmoid::derivative
            })
    ));

    return *this;
}

std::shared_ptr<neural_network::network> neural_network::network_builder::build() {
    this->withInputLayer(
            networkDescriptor->networkTrainData.at(0)->input.size(),
            math::utils::random_matrix_generator::generate
    ).withOutputLayer();


    return std::make_shared<network>(networkDescriptor);
}
