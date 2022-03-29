#include <iostream>
#include "math/vector.h"
#include "math/matrix.h"

#include "neural_network/network.h"
#include "math/functions/sigmoid.h"

int main() {

    neural_network::network network(
            neural_network::layer(
                    math::vector_d({1.0, 0.0}),
                    math::matrix_d({{0.85766315, 0.05015078},
                                    {-0.25091461,  -0.90849043},
                                    {-0.05035739,  -0.52372058}}),
                    math::vector_d({0, 0, 0})
            ),
            {
                    neural_network::layer(
                            math::vector_d({0.0, 0.0, 0.0}),
                            math::matrix_d({
                                                   {-0.15254847, -0.00120349, -0.25596635}
                                           }),
                            math::vector_d({0, 0, 0}),
                            math::functions::sigmoid::activate,
                            math::functions::sigmoid::derivative
                    ),

                    neural_network::layer(
                            math::vector_d(std::vector<double>{0}),
                            math::matrix_d(std::vector<std::vector<double> >{{}}),
                            math::vector_d({0, 0, 0}),
                            math::functions::sigmoid::activate,
                            math::functions::sigmoid::derivative
                    )
            },
            math::vector_d(std::vector<double>{1.0}),
            0.003
    );

    network.feedForward();
    for (int i = 0; i < 100000; ++i) {
        if (i % 50000 == 0) {
            std::cout << "error[" << i << "]: " << network.error() << '\n';
        }
        network.backProp();
        network.feedForward();
    }

    network.output();
    return 0;
}
