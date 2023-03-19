#include "Neuron.hpp"

#include <iostream>
#include <random>
#include <fstream>

std::default_random_engine generator;
std::normal_distribution normal_distribution(0.0, 0.5);

template <typename type, size_t num_inputs, activation_t activation>
Neuron<type, num_inputs, activation>::Neuron() {
    switch (activation)
    {
    case ReLU:
        activation_function = [](type x) { return (double)x > 0.0 ? x : 0.0; };
        activation_delta = [](type x) { return (double)x > 0.0 ? x : 1.0; };
        break;
    case Sigmoid:
        activation_function = [](type x) { return 1.0 / (1.0 + exp(-(double)x)); };
        activation_delta = [](type x) { return (double)x * (1.0 - (double)x); };
        break;
    default:
        std::cerr << "Error: Invalid activation function'\n";
        exit(1);
    }
    for (auto& weight : weights) {
        weight = normal_distribution(generator);
    }
    bias = 0;
}

template <typename type, size_t num_inputs, activation_t activation>
Neuron<type, num_inputs, activation>::Neuron(std::string parameters_dir)
    : Neuron<type, num_inputs, activation>() {
    std::string line;
    std::ifstream weights_file(parameters_dir + "/weights.txt");
    for (auto& weight : weights) {
        std::getline(weights_file, line);
        weight = std::stod(line);
    }
    weights_file.close();
    std::ifstream bias_file(parameters_dir + "/bias.txt");
    std::getline(bias_file, line);
    bias = std::stod(line);
    bias_file.close();
}

template <typename type, size_t num_inputs, activation_t activation>
type Neuron<type, num_inputs, activation>::calc_output(std::array<type, num_inputs> inputs) {
    type sum = bias;
    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    return output = activation_function(sum); 
}

template <typename type, size_t num_inputs, activation_t activation>
void Neuron<type, num_inputs, activation>::calc_delta(double error) {
    delta = error * activation_delta(output);
}

template <typename type, size_t num_inputs, activation_t activation>
void Neuron<type, num_inputs, activation>::update_parameters(std::array<type, num_inputs> inputs, double learning_rate) {
    for (size_t i = 0; i < inputs.size(); i++) {
        weights[i] -= type(learning_rate * delta * (double)inputs[i]);
    }
    bias -= type(learning_rate * delta);
}