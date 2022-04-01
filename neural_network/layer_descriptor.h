#pragma once
#include "math/matrix.h"


namespace neural_network {
    using weight_generator = std::function<math::matrix_d(int, int)>;
    using activation_function = std::function<double(double)>;
    using derivative_function = std::function<double(double)>;


    struct layer_descriptor {
    public:
        int numberOfInputs{};
        int weightRowCount{};
        int weightColCount{};
        weight_generator weightGenerator{};
        activation_function activation{};
        derivative_function activation_derivative{};
    };

    using layer_descriptor_Ptr = std::shared_ptr<layer_descriptor>;
}