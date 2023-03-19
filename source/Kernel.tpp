template <typename type, size_t width, size_t height, activation_t activation>
Kernel<type, width, height, activation>::Kernel() {
    // Initialize weights and bias
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            weights[i][j] = (type)rand() / RAND_MAX;
        }
    }
    bias = (type)rand() / RAND_MAX;

    // Initialize activation function
    switch (activation) {
        case activation_t::sigmoid:
            activation_function = [](type x) -> type {
                return 1 / (1 + exp(-x));
            };
            activation_delta = [](type x) -> double {
                return x * (1 - x);
            };
            break;
        case activation_t::relu:
            activation_function = [](type x) -> type {
                return x > 0 ? x : 0;
            };
            activation_delta = [](type x) -> double {
                return x > 0 ? 1 : 0;
            };
            break;
        case activation_t::tanh:
            activation_function = [](type x) -> type {
                return tanh(x);
            };
            activation_delta = [](type x) -> double {
                return 1 - x * x;
            };
            break;
    }
}