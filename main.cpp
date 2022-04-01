#include <iostream>
#include <neural_network/network_builder.h>
#include "math/vector.h"
#include "math/matrix.h"

#include "neural_network/network.h"
#include "math/functions/sigmoid.h"
#include "math/utils/random_matrix_generator.h"

int main() {

    std::shared_ptr<neural_network::network> network = neural_network::network_builder()
            .withTrainData(
                    {{1, 1}}, {std::vector<double>{1.0}}
            )
            .withTrainData(
                    {{0, 1}}, {std::vector<double>{1.0}}
            )
            .withTrainData(
                    {{0, 0}}, {std::vector<double>{0.0}}
            )
            .withTrainData(
                    {{1, 0}}, {std::vector<double>{1.0}}
            )
            .withLearningRate(0.03)
            .withRandomWeightsAndSigmoidHiddenLayer(3, 1)
            .build();


    network->feedForward();
    for (int i = 0; i < 100000; ++i) {
        std::cout << "error[" << i << "]: " << network->error() << '\n';
        network->backProp();
        network->feedForward();
    }

    network->output();

    return 0;
}
