#pragma once

#include <cmath>

namespace math::functions {
    class sigmoid {
    public:
        static double activate(double input) {
            return 1 / (1 + exp(-input));
        }

        static double derivative(double input) {
            double sigma = activate(input);

            return sigma * (1 - sigma);
        }
    };
}
