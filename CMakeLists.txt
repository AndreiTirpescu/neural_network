cmake_minimum_required(VERSION 3.17)
project(untitled)

set(CMAKE_CXX_STANDARD 17)
include_directories(.)
add_executable(neural_network
        main.cpp
        math/functions/sigmoid.h
        math/matrix.h
        math/vector.h
        neural_network/layer.cpp
        neural_network/layer.h
        neural_network/network.cpp
        neural_network/network.h
)
