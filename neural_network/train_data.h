#pragma once

#include "math/vector.h"

namespace neural_network {

    struct train_data {
    public:
        train_data(const math::vector_d &input, const math::vector_d &expected) : input(std::make_shared<math::vector_d>(input)),
                                                                                  expected(std::make_shared<math::vector_d>(expected)) {}

        math::vector_dPtr input;
        math::vector_dPtr expected;
    };

    using train_dataPtr = std::shared_ptr<train_data>;
}
