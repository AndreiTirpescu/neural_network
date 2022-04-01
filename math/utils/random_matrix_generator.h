#pragma once

#include <random>

namespace math::utils {
    class random_matrix_generator {
    public:
        static math::matrix_d generate(int numRows, int numColumns) {
            math::matrix_d result{numRows, numColumns};
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            for (int i = 0; i < numRows; ++i) {
                for (int j = 0; j < numColumns; ++j) {
                    result.at(i, j) = dist(mt);
                }
            }

            return result;
        }
    };
}