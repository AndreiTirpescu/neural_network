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
        explicit network(const network_descriptor_Ptr& descriptor);

        void feedForward();

        void backProp(const math::vector_dPtr& expected);

        double error(const math::vector_dPtr& expected);

        void output();

        void setInputs(const math::vector_d& inputs);

        void train(int epochsCount);

        void test();

    private:
        std::vector<layerPtr> layers;
        std::vector<train_dataPtr> trainDataSet;
        double learningRate;

        math::vector_d deltaOutputs(const math::vector_dPtr& expected);

        std::vector<math::vector_d> computeHiddenDeltas(const math::vector_d &outputDeltas);

        void computeDerivativeWithRespectToWeights(const std::vector<math::vector_d> &hiddenDeltas);

        void updateWeights();
    };

}
