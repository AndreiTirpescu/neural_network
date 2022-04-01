#pragma once

#include "math/vector.h"
#include "math/matrix.h"
#include "layer.h"

#include <utility>
#include <vector>

#include <iostream>

#include "network_descriptor.h"

namespace neural_network {

    class network {

    public:
//        network(const std::vector<layer> &hiddenLayers, const math::vector_d &expected, double learningRate);
        explicit network(const network_descriptor_Ptr& descriptor);

        void feedForward();

        void backProp();

        double error();

        void output();

    private:
        std::vector<layerPtr> layers;
        math::vector_dPtr expected;
        double learningRate;

        math::vector_d deltaOutputs();

        std::vector<math::vector_d> computeHiddenDeltas(const math::vector_d &outputDeltas);

        void computeDerivativeWithRespectToWeights(const std::vector<math::vector_d> &hiddenDeltas);

        void updateWeights();
    };

}
