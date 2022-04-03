#pragma once
#include <vector>
#include <ostream>
#include <memory>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace math {
    template<class T>
    class vector {
    public:
        vector(const std::vector<T>& elements) : length(elements.size()) {
            pElements = new T[length];
            for (int i = 0; i < length; ++i) {
                pElements[i] = elements[i];
            }
        }

        vector(int size) : length(size) {
            pElements = new T[size];
            for (int i = 0; i < length; ++i) {
                pElements[i] = 0;
            }
        }

        vector(int size, T* elements) : length(size) {
            pElements = new T[size];
            for (int i = 0; i < length; ++i) {
                pElements[i] = elements[i];
            }
        }



        vector(const vector<T>& other) {
            length = other.length;
            pElements = new T[length];

            std::copy(other.pElements, other.pElements + length, pElements);
        }

        vector(vector<T>&& other) noexcept : length(other.length), pElements(nullptr) {
            length = std::exchange(other.length, 0);
            pElements = std::exchange(other.pElements, nullptr);
        }

        vector& operator=(const vector& other) {
            if (this == &other) {
                return *this;
            }

            delete[] pElements;
            length = other.length;
            pElements = new T[length];

            std::copy(other.pElements, other.pElements + length, pElements);

            return *this;
        }

        vector& operator=(vector&& other) noexcept {
            if (this == &other) {
                return *this;
            }

            delete[] pElements;
            length = std::exchange(other.length, 0);
            pElements = std::exchange(other.pElements, nullptr);

            return *this;
        }

        ~vector() {
            delete[] pElements;
            length = 0;

            pElements = nullptr;
        }

        friend std::ostream &operator<<(std::ostream &os, const vector &vector) {
            for (int index = 0; index < vector.length; ++index) {
                os << vector[index] << ' ';
            }

            return os;
        }

        T& operator[] (int index) const {
            return pElements[index];
        }

        T& at(int index) const {
            return pElements[index];
        }

        friend vector operator+(const vector& x, const vector& y) {
            vector result = x;
            for (int i = 0; i < x.length; ++i) {
                result[i] += y[i];
            }

            return result;
        }

        template <class Scalar>
        friend vector operator*(const vector& x, Scalar a) {
            vector result = x;
            for (int i = 0; i < x.length; ++i) {
                result[i] *= a;
            }

            return result;
        }


        friend vector operator-(const vector& x, const vector& y) {
            vector result = x;
            for (int i = 0; i < x.length; ++i) {
                result[i] -= y[i];
            }

            return result;
        }

        T dot(const vector& other) {
            T result = 0;
            for (int i = 0; i < length; ++i) {
                result += pElements[i] * other[i];
            }

            return result;
        }

        [[nodiscard]] double sum() const {
            return std::accumulate(pElements, pElements+length, 0.0);
        }

        [[nodiscard]] double mean() const {
            return sum() / (double) length;
        }

        [[nodiscard]] double magnitude() const {
            return sqrt(std::inner_product(pElements, pElements+length, pElements, 0.0));
        }

        [[nodiscard]] double squared_distance(const vector& other) const {
            double result = 0;

            for (int i = 0; i < length; ++i) {
                result += pow(pElements[i] - other[i], 2);
            }

            return sqrt(result);
        }

        [[nodiscard]] double sum_squared() const {
            double result = 0;

            for (int i = 0; i < length; ++i) {
                result += pElements[i] * pElements[i];
            }

            return result;
        }

        [[nodiscard]] int size() const {
            return length;
        }

    private:
        T* pElements;
        int length{};
    };

    using vector_d = math::vector<double>;
    using vector_dPtr = std::shared_ptr<vector_d>;
}
