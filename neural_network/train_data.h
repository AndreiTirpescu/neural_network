#pragma once

#include "math/vector.h"

namespace neural_network {

    struct train_data {
    public:
        math::vector_d input;
        math::vector_d expected;
    };

    using train_data_Ptr = std::shared_ptr<train_data>;
}
