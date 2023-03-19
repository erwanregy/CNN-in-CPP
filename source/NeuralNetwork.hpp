#pragma once

#include "Layers.hpp"

#include <array>
#include <vector>

// struct LayerBuilder {
//     layer_t type;
//     size_t size;
//     activation_t activation = None;
// };

template <size_t num_layers>
template <typename type, size_t num_inputs, std::array<Layer, num_layers> layers>
class NeuralNetwork {
public:
    template <typename layer_type>
    std::array<layer_type, num_layers> layers;
    std::array<type, layers.last().size> outputs;

    type largest_output = 0;
    type smallest_output = 0;

    NeuralNetwork();

    void feed_forward(std::array<type, num_inputs> inputs);
    size_t predict(std::array<type, num_inputs> inputs);

private:
    void back_propagate(size_t label);
    void update_parameters(std::array<type, num_inputs> inputs, double learning_rate);

public:
    template <size_t num>
    void train(std::array<std::array<type, num_inputs>, num> inputs, std::array<size_t, num> labels, size_t num_epochs, size_t batch_size, double learning_rate);
    template <size_t num>
    void test(std::array<std::array<type, num_inputs>, num> inputs, std::array<size_t, num> labels);

    void save(std::string model_path);
    void load(std::string model_path);
};

#include "NeuralNetwork.tpp"