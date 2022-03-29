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
        layer(math::vector_d inputs, math::matrix_d weights, math::vector_d bias);

        layer(math::vector_d inputs, math::matrix_d weights, math::vector_d bias, activation_function activation,
              derivative_function derivative);

        [[nodiscard]] const math::vector_d &inputs() const;

        [[nodiscard]] const math::matrix_d &weights() const;

        [[nodiscard]] const math::vector_d &bias() const;

        void updateInputs(const math::vector_d &updated);

        [[nodiscard]] const activation_function &derivative() const;

        void activate();

        void updateDerivativeAt(int i, int j, double val);

        void updateWeightAt(int i, int j, double learningRate);

    private:
        math::vector_d inputVector;
        math::vector_d biases;
        math::matrix_d weightMatrix;
        math::matrix_d derivativeWithRespectToWeights;

        activation_function activationFunction;
        derivative_function derivativeFunction;
    };
}
