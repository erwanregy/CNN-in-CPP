#include <iostream>
#include <fstream>
#include <filesystem>

template <typename type>
NeuralNetwork<type>::NeuralNetwork() : layers(layers) {}

template <typename type>
void NeuralNetwork<type>::feed_forward(std::vector<type> inputs) {
    for (auto& layer : layers) {
        inputs = layer.calc_outputs(inputs);
        largest_output = std::max(largest_output, *std::max_element(inputs.begin(), inputs.end()));
        smallest_output = std::min(smallest_output, *std::min_element(inputs.begin(), inputs.end(), [](type a, type b) { return a != 0 && a < b; }));
    }
    outputs = inputs;
}

template <typename type>
size_t NeuralNetwork<type>::predict(std::vector<type> input) {
    feed_forward(input);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

template <typename type>
void NeuralNetwork<type>::back_propagate(size_t label) {
    auto& output_layer = layers.back();
    for (size_t n = 0; n < output_layer.neurons.size(); n++) {
        double error = (double)outputs[n] - double(n == label);
        output_layer.neurons[n].calc_delta(error);
    }
    for (size_t l = layers.size() - 2; l != -1; l--) {
        auto& layer = layers[l];
        auto& next_layer = layers[l + 1];
        for (size_t n = 0; n < layer.neurons.size(); n++) {
            double error = 0.0;
            for (auto next_layer_neuron : next_layer.neurons) {
                error += next_layer_neuron.weights[n] * next_layer_neuron.delta;
            }
            layer.neurons[n].calc_delta(error);
        }
    }
}

template <typename type>
void NeuralNetwork<type>::update_parameters(std::vector<type> inputs, double learning_rate) {
    for (auto& layer : layers) {
        layer.update_parameters(inputs, learning_rate);
        inputs = layer.outputs;
    }
}

template <typename type>
template <size_t num>
void NeuralNetwork<type>::train(std::array<std::vector<type>, num> inputs, std::array<size_t, num> labels, size_t num_epochs, size_t batch_size, double learning_rate) {
    for (size_t epoch_num = 1; epoch_num <= num_epochs; epoch_num++) {
        for (size_t _ = 0; _ < batch_size; _++) {
            size_t i = rand() % inputs.size();
            feed_forward(inputs[i]);
            back_propagate(labels[i]);
            update_parameters(inputs[i], learning_rate);
        }
        if (num_epochs < 10 || epoch_num % (num_epochs / 10) == 0 || epoch_num == 1 || epoch_num == num_epochs) {
            std::cout << "Epoch " << epoch_num << "/" << num_epochs << " " << std::flush << "- ";
            test(inputs, labels);
        }
    }
}

template <typename type>
template <size_t num>
void NeuralNetwork<type>::test(std::array<std::vector<type>, num> inputs, std::array<size_t, num> labels) {
    double accuracy = 0.0;
    for (size_t i = 0; i < inputs.size(); i++) {
        accuracy += (predict(inputs[i]) == labels[i]) * 100.0;
    }
    accuracy /= inputs.size();
    std::cout << "Accuracy: " << accuracy << "%\n" << std::flush;
}

template <typename type>
void NeuralNetwork<type>::save(std::string model_path) {
    if (std::filesystem::exists(model_path)) {
        std::cout << "Folder '" << model_path << "' already exists. Overwrite? (y/N): " << std::flush;
        if (std::cin.get() != 'y') {
            return;
        } else {
            std::filesystem::remove_all(model_path);
        }
    }
    std::filesystem::create_directory(model_path);
    for (size_t l = 0; l < layers.size(); l++) {
        std::string layer_path = model_path + "/layer_" + std::to_string(l);
        std::filesystem::create_directory(layer_path);
        for (size_t n = 0; n < layers[l].neurons.size(); n++) {
            std::string neuron_path = layer_path + "/neuron_" + std::to_string(n);
            std::filesystem::create_directory(neuron_path);
            std::ofstream weights_file(neuron_path + "/weights.txt");
            for (auto weight : layers[l].neurons[n].weights) {
                weights_file << weight << '\n';
            }
            std::ofstream bias_file(neuron_path + "/bias.txt");
            bias_file << layers[l].neurons[n].bias << '\n';
            bias_file.close();
        }
    }
}

template <typename type>
void NeuralNetwork<type>::load(std::string model_path) {
    if (not std::filesystem::exists(model_path)) {
        std::cerr << "Error: Folder '" << model_path << "' does not exist\n";
        exit(1);
    }
    for (size_t l = 0; l < layers.size(); l++) {
        std::string layer_path = model_path + "/layer_" + std::to_string(l);
        for (size_t n = 0; n < layers[l].neurons.size(); n++) {
            std::string neuron_path = layer_path + "/neuron_" + std::to_string(n);
            std::ifstream weights_file(neuron_path + "/weights.txt");
            for (auto& weight : layers[l].neurons[n].weights) {
                std::string line;
                std::getline(weights_file, line);
                weight = std::stod(line);
            }
            weights_file.close();
            std::ifstream bias_file(neuron_path + "/bias.txt");
            std::string line;
            std::getline(bias_file, line);
            layers[l].neurons[n].bias = std::stod(line);
            bias_file.close();
        }
    }
}