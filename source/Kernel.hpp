#pragma once

#include "Neuron.hpp"

#include <array>
#include <utility>

template <typename type, size_t width, size_t height, activation_t activation>
class Kernel {
public:
    using matrix = std::array<std::array<type, width>, height>;

    matrix weights;
    type bias;
    std::function<type(type)> activation_function;
    matrix outputs;

    std::function<double(type)> activation_delta;
    double delta;

    Kernel();
    Kernel(std::string parameters_dir);

    type calc_output(matrix inputs);

    void calc_delta(double expected_output);
    void update_parameters(matrix inputs, double learning_rate);
};

#include "Kernel.tpp"