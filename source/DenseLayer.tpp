template <typename type, size_t num_neurons, size_t num_inputs, activation_t activation>
std::array<type, num_neurons> DenseLayer<type, num_neurons, num_inputs, activation>::calc_outputs(std::array<type, num_inputs> inputs) {
    for (size_t i = 0; i < num_neurons; i++) {
        outputs[i] = neurons[i].calc_output(inputs);
    }
    return outputs = neurons;
}

template <typename type, size_t num_neurons, size_t num_inputs, activation_t activation>
void DenseLayer<type, num_neurons, num_inputs, activation>::update_parameters(std::array<type, num_inputs> inputs, double learning_rate) {
    for (auto& neuron : neurons) {
        neuron.update_parameters(inputs, learning_rate);
    }
}