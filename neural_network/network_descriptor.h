#pragma once

#include "train_data.h"
#include "layer.h"

namespace neural_network {
    struct network_descriptor {
        std::vector<train_dataPtr> networkTrainData{};
        std::vector<layerPtr> networkLayers{};
        double learningRate{};
    };

    using network_descriptor_Ptr = std::shared_ptr<network_descriptor>;
}