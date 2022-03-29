cmake_minimum_required(VERSION 3.17)
project(untitled)

set(CMAKE_CXX_STANDARD 17)
include_directories(.)
add_executable(neural_network
        main.cpp
        math/vector.h
        math/matrix.h
        neural_network/network.h
        neural_network/network.cpp
        math/functions/sigmoid.h
)
