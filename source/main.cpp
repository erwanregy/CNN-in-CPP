#include "NeuralNetwork.hpp"
#include "mnist.hpp"

#include <iostream>
#include <vector>
#include <cstdint>

// template <typename type>
// void test_neural_network_small() {
//     const size_t NUM_INPUTS = 4;
//     const std::vector<LayerBuilder> LAYERS = {
//         {4, ReLU},
//         {4, ReLU},
//         {4, ReLU}
//     };

//     NeuralNetwork<type> neural_network(NUM_INPUTS, LAYERS);
//     for (auto& dense_layer : neural_network.layers) {
//         for (auto& neuron : dense_layer.neurons) {
//             for (size_t i = 0; i < neuron.weights.size(); i++) {
//                 neuron.weights[i] = i;
//             }
//         }
//     }

//     std::vector<type> inputs(NUM_INPUTS);
//     for (size_t i = 0; i < inputs.size(); i++) {
//         inputs[i] = i;
//     }

//     neural_network.feed_forward(inputs);

//     for (auto output : neural_network.get_outputs()) {
//         std::cout << output << '\n';
//     }
// }

// template <typename type>
// void test_neural_network_old() {
//     NeuralNetwork<type> neural_network(784, {{16, "relu"}, {16, "relu"}, {10, "sigmoid"}});
//     char input;

//     std::cout << "Load parameters? (y/n): ";
//     std::cin >> input;
//     if (input == 'y') {
//         neural_network.load("mnist.txt");
//     }

//     std::cout << "Train network? (y/n): ";
//     std::cin >> input;
//     if (input == 'y') {
//         std::vector<std::vector<type>> train_inputs = extract_inputs<type>("data/mnist/train_images.idx3-ubyte");
//         std::vector<size_t> train_labels = extract_labels("data/mnist/train_labels.idx1-ubyte");
//         neural_network.train(train_inputs, train_labels, 1000, 128, 0.01);
//     }

//     std::cout << "Test network? (y/n): ";
//     std::cin >> input;
//     std::vector<std::vector<type>> test_inputs = extract_inputs<type>("data/mnist/test_images.idx3-ubyte");
//     std::vector<size_t> test_labels = extract_labels("data/mnist/test_labels.idx1-ubyte");
//     if (input == 'y') {
//         neural_network.test(test_inputs, test_labels);
//     }

//     std::cout << "Save parameters? (y/n): ";
//     std::cin >> input;
//     if (input == 'y') {
//         neural_network.save("mnist.txt");
//     }

//     std::cout << "View incorrect predictions? (y/n): ";
//     std::cin >> input;
//     std::cin.get();
//     if (input == 'y') {
//         std::vector<std::vector<std::vector<uint8_t>>> test_images = extract_images("data/mnist/test_images.idx3-ubyte");
//         for (size_t i = 0; i < test_images.size(); i++) {
//             size_t prediction = neural_network.predict(test_inputs[i]);
//             if (prediction != test_labels[i]) {
//                 print_image(test_images[i]);
//                 std::cout << "Predicted: " << neural_network.predict(test_inputs[i]) << '\n';
//                 std::cout << "Expected:  " << test_labels[i] << '\n';
//                 std::cin.get();
//             }
//         }
//     }
// }

// template <typename type>
// void neural_network_shell() {
//     NeuralNetwork<type> neural_network(784, {{16, ReLU}, {16, ReLU}, {10, Sigmoid}});

//     std::cout << "Importing data...";
//     std::vector<std::vector<type>> train_inputs = extract_inputs<type>("data/mnist/train_images.idx3-ubyte");
//     std::vector<size_t> train_labels = extract_labels("data/mnist/train_labels.idx1-ubyte");
//     std::vector<std::vector<std::vector<uint8_t>>> test_images = extract_images("data/mnist/test_images.idx3-ubyte");
//     std::vector<std::vector<type>> test_inputs = extract_inputs<type>("data/mnist/test_images.idx3-ubyte");
//     std::vector<size_t> test_labels = extract_labels("data/mnist/test_labels.idx1-ubyte");
//     std::cout << " Done.\n";

//     std::string input;
//     while (true) {
//         std::cout << "> ";
//         std::getline(std::cin, input);
//         if (input == "help") {
//             std::cout << "help:    Display this message.\n";
//             std::cout << "load:    Load parameters from file.\n";
//             std::cout << "train:   Train the neural network.\n";
//             std::cout << "test:    Test the neural network.\n";
//             std::cout << "save:    Save parameters to file.\n";
//             std::cout << "predict: Predict the label for an image.\n";
//             std::cout << "quit:    Exit the program.\n";
//             std::cout << "clear:   Clear the screen.\n";
//         } else if (input == "quit") {
//             exit(0);
//         } else if (input == "clear") {
//             std::cout << "\033[2J\033[1;1H";
//         } else if (input == "load") {
//             neural_network.load("mnist.txt");
//         } else if (input == "train") {
//             neural_network.train(train_inputs, train_labels, 1000, 128, 0.01);
//         } else if (input == "test") {
//             neural_network.test(test_inputs, test_labels);
//         } else if (input == "save") {
//             neural_network.save("mnist.txt");
//         } else if (input == "predict") {
//             size_t i = rand() % test_images.size();
//             print_image(test_images[i]);
//             std::cout << "Predicted: " << neural_network.predict(test_inputs[i]) << '\n';
//             std::cout << "Expected:  " << test_labels[i] << '\n';
//         } else {
//             std::cout << "Unknown command '" << input << "'. Type 'help' for a list of commands.\n";
//         }
//     }
// }

template <typename type>
void test_neural_network() {
    // Layers
    std::

    NeuralNetwork<type> neural_network(layers);

    neural_network.load("../models/3b1b/");

    auto [train_images, train_inputs, train_labels] = extract<type>("data/mnist/train_images.idx3-ubyte", "data/mnist/train_labels.idx1-ubyte");
    auto [test_images, test_inputs, test_labels] = extract<type>("data/mnist/test_images.idx3-ubyte", "data/mnist/test_labels.idx1-ubyte");

    neural_network.test(test_inputs, test_labels);

    neural_network.train(train_inputs, train_labels, 1000, 128, 0.01);

    // neural_network.test(test_inputs, test_labels);

    for (size_t i = 0; i < test_inputs.size(); i++) {
        size_t prediction = neural_network.predict(test_inputs[i]);
        if (prediction != test_labels[i]) {
            print_image(test_images[i]);
            std::cout << "Predicted: " << prediction << '\n';
            std::cout << "Expected:  " << test_labels[i] << '\n';
            std::cin.get();
        }
    }
}

template <typename type>
void test_neuron() {
    Neuron<type, 784, ReLU> neuron("../models/3b1b/layer_0/neuron_0/");

    auto inputs = extract_inputs<type, 1>("../data/mnist/test_images.idx3-ubyte")[0];

    std::cout << neuron.calc_output(inputs) << '\n';
}

// template <typename type>
// void test_neural_network_largest_output() {
//     NeuralNetwork<type> neural_network(
//         {
//             {Input, 784, None},
//             {Dense, 16, ReLU},
//             {Dense, 16, ReLU},
//             {Dense, 10, Sigmoid}
//         }
//     );

//     neural_network.load("models/3b1b");

//     auto train_inputs = extract_inputs<type>("data/mnist/train_images.idx3-ubyte");
//     auto test_inputs = extract_inputs<type>("data/mnist/test_images.idx3-ubyte");
//     // join train and test inputs
//     auto inputs = train_inputs;
//     inputs.insert(train_inputs.end(), test_inputs.begin(), test_inputs.end());

//     type largest_output = 0;
//     type smallest_output = 1;
//     for (auto input : inputs) {
//         neural_network.feed_forward(input);
//         largest_output = std::max(largest_output, neural_network.largest_output);
//         smallest_output = std::min(smallest_output, neural_network.smallest_output, [](type a, type b) { return a != 0 && a < b; });
//     }
//     std::cout << largest_output << ' ' << smallest_output << '\n';
// }


int main() {
    std::ios_base::sync_with_stdio(false);
    typedef double T;

    test_neural_network<T>();
}
