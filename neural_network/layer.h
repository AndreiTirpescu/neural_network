#pragma once

#include "math/vector.h"
#include "math/matrix.h"

#include <utility>
#include <functional>

namespace neural_network {
    using activation_function = std::function<double(double)>;
    using derivative_function = std::function<double(double)>;

    class layer {
    public:
        layer(const math::vector_d &inputs, const math::matrix_d &weights, const math::vector_d &bias,
              activation_function activation, derivative_function derivative);

        layer(const math::vector_d &inputs, const math::matrix_d &weights, const math::vector_d &bias);

        [[nodiscard]] math::vector_dPtr inputs() const;

        [[nodiscard]] math::matrix_dPtr weights() const;

        [[nodiscard]] math::vector_dPtr bias() const;

        void updateInputs(const math::vector_d& updated);

        [[nodiscard]] const activation_function &derivative() const;

        void activate();

        void updateDerivativeAt(int i, int j, double val);

        void updateWeightAt(int i, int j, double learningRate);

    private:
        math::vector_dPtr inputVector;
        math::vector_dPtr biases;
        math::matrix_dPtr weightMatrix;
        math::matrix_dPtr derivativeWithRespectToWeights;

        activation_function activationFunction;
        derivative_function derivativeFunction;
    };

    using layerPtr = std::shared_ptr<layer>;
}
