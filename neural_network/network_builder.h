#pragma once

#include "train_data.h"
#include "network_descriptor.h"
#include "layer_descriptor.h"
#include "network.h"
#include "math/functions/sigmoid.h"

namespace neural_network {
    class network_builder {
    public:
        network_builder &withTrainData(const math::vector_d &inputs, const math::vector_d &outputs);

        network_builder &withHiddenLayer(int numberOfNeurons, int numberOfNeuronsNext, const weight_generator &generator,
                        const activation_function &activation, const activation_function &derivative);

        network_builder &withRandomWeightsAndSigmoidHiddenLayer(int numberOfNeurons, int numberOfNeuronsNext);

        network_builder &withLearningRate(double learningRate);

        std::shared_ptr<network> build();

    private:
        network_descriptor_Ptr networkDescriptor{new network_descriptor()};

        network_builder &withInputLayer(int inputLayerSize, const weight_generator &generator);

        network_builder &withOutputLayer();
    };
}