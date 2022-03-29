#pragma once

#include <vector>
#include <iostream>


namespace math {
    template <class T>
    class matrix {

    public:
        explicit matrix(int numRows) : numRows(numRows), numColumns(numRows) {
            pElements = new T[numRows * numColumns];
            for (int i = 0; i < numRows; ++i) {
                for (int j = 0; j < numColumns; ++j) {
                    pElements[index(i, j)]  = 0;
                }
            }
        }

        matrix(int numRows, int numCols) : numRows(numRows), numColumns(numCols) {
            pElements = new T[numRows * numColumns];
            for (int i = 0; i < numRows; ++i) {
                for (int j = 0; j < numColumns; ++j) {
                    pElements[index(i, j)]  = 0;
                }
            }
        }

        explicit matrix(std::vector<std::vector<T>> elements) : numRows(elements.size()), numColumns(elements[0].size()) {
            pElements = new T[numRows * numColumns];

            for (int i = 0; i < numRows; ++i) {
                for (int j = 0; j < numColumns; ++j) {
                    pElements[index(i, j)]  = elements[i][j];
                }
            }
        }

        static matrix Identity(int numRows) {
            matrix result{numRows};
            for (int i = 0; i < numRows; ++i) {
                result[{i, i}] = 1;
            }

            return result;
        }

        matrix(const matrix& other) : numRows(other.numRows), numColumns(other.numColumns) {
            pElements = new T[numRows * numColumns];

            std::copy(other.pElements, other.pElements + numRows * numColumns, pElements);
        }

        matrix(matrix&& other) noexcept {
            numRows = std::exchange(other.numRows, 0);
            numColumns = std::exchange(other.numColumns, 0);

            pElements = std::exchange(other.pElements, nullptr);
        }

        matrix& operator=(const matrix& other) {
            if (this == &other) {
                return *this;
            }

            delete[] pElements;

            numRows = other.numRows;
            numColumns = other.numColumns;
            pElements = new T[numRows * numColumns];

            std::copy(other.pElements, other.pElements + numRows * numColumns, pElements);

            return *this;
        }

        matrix& operator=(matrix&& other) noexcept {
            if (this == &other) {
                return *this;
            }

            delete[] pElements;

            numRows = std::exchange(other.numRows, 0);
            numColumns = std::exchange(other.numColumns, 0);

            pElements = std::exchange(other.pElements, nullptr);

            return *this;
        }

        ~matrix() {
            delete[] pElements;

            numRows = 0;
            numColumns = 0;
        }

        friend std::ostream &operator<<(std::ostream &os, const matrix& m) {
            for (int i = 0; i < m.numRows; ++i) {
                for (int j = 0; j < m.numColumns; ++j) {
                    os << m.pElements[m.index(i, j)] << ' ';
                }
                os << '\n';
            }

            return os;
        }

        T& at(int i, int j) const {
            return pElements[index(i, j)];
        }

        T& operator[] (std::pair<int, int> linColPair) const {
            return pElements[index(linColPair.first, linColPair.second)];
        }

        friend matrix operator+(const matrix& m1, const matrix& m2) {
            matrix result = m1;
            for (int i = 0; i < m1.numRows; ++i) {
                for (int j = 0; j < m2.numColumns; ++j) {
                    result[{i, j}] += m2[{i, j}];
                }
            }

            return result;
        }

        friend matrix operator-(const matrix& m1, const matrix& m2) {
            matrix result = m1;
            for (int i = 0; i < m1.numRows; ++i) {
                for (int j = 0; j < m2.numColumns; ++j) {
                    result[{i, j}] -= m2[{i, j}];
                }
            }

            return result;
        }

        friend matrix operator*(const matrix& m1, const matrix& m2) {
            matrix result{m1.numRows, m2.numColumns};
            for (int i = 0; i < result.numRows; ++i) {
                for (int j = 0; j < result.numColumns; ++j) {
                    result[{i, j}] = m1.lineToVector(i).dot(m2.columnToVector(j)) ;
                }
            }

            return result;
        }

        math::vector<T> lineToVector(int lineIndex) const {
            math::vector<T> result(numColumns);
            for (int i = 0; i < numColumns; ++i) {
                result[i] = pElements[index(lineIndex, i)];
            }

            return result;
        }

        math::vector<T> columnToVector(int colIndex) const {
            math::vector<T> result(numRows);
            for (int i = 0; i < numRows; ++i) {
                result[i] = pElements[index(i, colIndex)];
            }

            return result;
        }

        matrix transpose() const {
            matrix result{numColumns, numRows};
            for (int i = 0; i < result.numRows; ++i) {
                for (int j = 0; j < result.numColumns; ++j) {
                    result[{i, j}] = pElements[index(j, i)];
                }
            }

            return result;
        }

        friend math::vector<T> operator *(const matrix& m, const math::vector<T>& x) {
            math::vector<T> result{m.numRows};

            for (int i = 0; i < result.size(); ++i) {
                result[i] = m.lineToVector(i).dot(x);
            }

            return result;
        }

        [[nodiscard]] int columnCount() const {
            return numColumns;
        }

        [[nodiscard]] int rowCount() const {
            return numRows;
        }

    private:
        int numColumns;
        int numRows;
        T* pElements;

        [[nodiscard]] int index(int i, int j) const {
            return numColumns * i + j;
        }
    };

    using matrix_d = math::matrix<double>;
    using matrix_dPtr = std::shared_ptr<matrix_d>;
}