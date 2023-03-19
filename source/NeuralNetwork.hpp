#pragma once

#include "Layers.hpp"

#include <array>
#include <vector>

struct LayerBuilder {
    layer_t type;
    size_t size;
    activation_t activation = None;
};

template <typename type, size_t num_layers, std::array<LayerBuilder, num_layers> layer_builders>
class NeuralNetwork {
public:
    template <typename layer_type>
    std::array<layer_type, num_layers> layers;
    template <size_t num_outputs>
    std::array<type, num_outputs> outputs;

    type largest_output = 0;
    type smallest_output = 0;

    NeuralNetwork();

    template <size_t num_inputs>
    void feed_forward(std::array<type, num_inputs> inputs);
    template <size_t num_inputs>
    size_t predict(std::array<type, num_inputs> inputs);

private:
    void back_propagate(size_t label);
    template <size_t num_inputs>
    void update_parameters(std::array<type, num_inputs> inputs, double learning_rate);

public:
    template <size_t num_inputs, size_t num_images>
    void train(std::array<std::array<type, num_inputs>, num_images> inputs, std::array<size_t, num_images> labels, size_t num_epochs, size_t batch_size, double learning_rate);
    template <size_t num_inputs, size_t num_images>
    void test(std::array<std::array<type, num_inputs>, num_images> inputs, std::array<size_t, num_images> labels);

    void save(std::string model_path);
    void load(std::string model_path);
};

#include "NeuralNetwork.tpp"