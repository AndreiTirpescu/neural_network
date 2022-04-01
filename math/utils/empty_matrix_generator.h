#pragma once

namespace math::utils {
    class empty_matrix_generator {
    public:
        static math::matrix_d generate(int numRows, int numColumns) {
            math::matrix_d result{numRows, numColumns};

            return result;
        }
    };
}