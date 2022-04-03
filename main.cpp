#include <iostream>
#include <neural_network/network_builder.h>
#include "math/vector.h"
#include "math/matrix.h"

#include "neural_network/network.h"

int main() {
    std::shared_ptr<neural_network::network> network = neural_network::network_builder()
            .withTrainData({{1, 1}}, {std::vector<double>{0.0}})
            .withTrainData({{0, 1}}, {std::vector<double>{1.0}})
            .withTrainData({{1, 0}}, {std::vector<double>{1.0}})
            .withTrainData({{0, 0}}, {std::vector<double>{0.0}})
            .withLearningRate(1.0)
            .withRandomWeightsAndSigmoidHiddenLayer(3, 1)
            .build();

    network->train(100000);
    network->test();

    return 0;
}
