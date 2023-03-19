#pragma once

#include "Neuron.hpp"

#include <array>

enum layer_t {
    Input,
    Dense,
    Convolutional,
    Pooling,
};

template <typename type>
class Layer {
public:
    std::array<Neuron<type, 0, None>, 0> neurons;
    virtual void calc_outputs() = 0;
    virtual void update_parameters() = 0;
};

template <typename type, size_t num_neurons, size_t num_inputs, activation_t activation>
class DenseLayer : public Layer<type> {
public:
    std::array<Neuron<type, num_inputs, activation>, num_neurons> neurons;
    std::array<type, num_neurons> outputs;

    DenseLayer() = default;

    std::array<type, num_neurons> calc_outputs(std::array<type, num_inputs> inputs);

    void update_parameters(std::array<type, num_inputs> inputs, double learning_rate);
};

#include "DenseLayer.tpp"

/* class ConvolutionalLayer : public Layer {
public:
    size_t num_filters;
    size_t filter_size;
    size_t stride;
    std::vector<std::vector<std::vector<double>>> filters;

    ConvolutionalLayer(size_t num_filters, size_t filter_size, size_t stride, size_t num_inputs, const char* activation);

    void calc_outputs(std::vector<double> inputs);
};

#include "ConvolutionalLayer.tpp"

class PoolingLayer : public Layer {
public:
    size_t pool_size;
    size_t stride;

    PoolingLayer(size_t pool_size, size_t stride, size_t num_inputs);
    
    void calc_outputs(std::vector<double> inputs);
};

#include "PoolingLayer.tpp" */