#pragma once

#include <array>
#include <functional>
#include <string>

enum activation_t {
    ReLU,
    Sigmoid,
    None,
};

template <typename type, size_t num_inputs, activation_t activation>
class Neuron {
public:
    using vector = std::array<type, num_inputs>;

    vector weights;
    type bias;
    std::function<type(type)> activation_function;
    type output;

    std::function<double(type)> activation_delta;
    double delta;

    Neuron();
    Neuron(std::string parameters_dir);

    type calc_output(vector inputs);

    void calc_delta(double expected_output);
    void update_parameters(vector inputs, double learning_rate);
};

#include "Neuron.tpp"