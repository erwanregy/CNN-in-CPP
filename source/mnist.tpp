#include <fstream>
#include <iostream>
#include <type_traits>

uint32_t swap_endian(uint32_t word) {
    return ((word & 0xff) << 24) | ((word & 0xff00) << 8) | ((word & 0xff0000) >> 8) | ((word & 0xff000000) >> 24);
}

std::ifstream open_file(std::string path) {
    std::ifstream file(path, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file '" << path << "'\n";
    }

    return file;
}

template <size_t num_dimensions>
size_t extract_header(std::ifstream& file, size_t num) {
    uint32_t temp;
    std::streamsize size = sizeof(temp);
    
    file.read((char*) &temp, size);

    file.read((char*) &temp, size);
    if (num == 0) {
        num = swap_endian(temp);
    }

    for (size_t i = 0; i < num_dimensions - 1; i++) {
        file.read((char*) &temp, size);
    }

    return num;
}

template <typename type, size_t num_images>
std::array<image_t, num_images> extract_images(std::string images_path) {
    std::ifstream file = open_file(images_path);
    
    extract_header<3>(file, num_images);

    std::array<image_t, num_images> images;

    for (auto& image : images) {
        for (auto& row : image) {
            for (auto& pixel : row) {
                uint8_t value = 0;
                file.read((char*) &value, sizeof(value));
                pixel = value;
            }
        }
    }

    return images;
}

template <typename type, size_t num_images>
std::array<input_t<type>, num_images> extract_inputs(std::string images_path) {
    std::ifstream file = open_file(images_path);
    
    extract_header<3>(file, num_images);

    std::array<input_t<type>, num_images> inputs;

    for (auto& input : inputs) {
        for (auto& pixel : input) {
            uint8_t value = 0;
            file.read((char*) &value, sizeof(value));
            pixel = (type)value / (type)255;
        }
    }

    return inputs;
}

template <typename type, size_t num_images>
std::tuple<std::array<image_t, num_images>, std::array<input_t<type>, num_images>> extract_data(std::string images_path) {
    std::ifstream file = open_file(images_path);
    
    extract_header<3>(file, num_images);

    std::array<image_t, num_images> images;
    std::array<input_t<type>, num_images> inputs;

    for (size_t image = 0; image < num_images; image++) {
        for (size_t row = 0; row < 28; row++) {
            for (size_t column = 0; column < 28; column++) {
                uint8_t value;
                file.read((char*) &value, sizeof(value));
                images[image][row][column] = value;
                inputs[image][row * 28 + column] = (type)value / (type)255;
            }
        }
    }

    return std::make_tuple(images, inputs);
}

template <size_t num_labels>
std::array<size_t, num_labels> extract_labels(std::string labels_path) {
    std::ifstream file = open_file(labels_path);
    
    extract_header<1>(file, num_labels);

    std::array<size_t, num_labels> labels;

    for (auto& label : labels) {
        uint8_t value = 0;
        file.read((char*) &value, sizeof(value));
        label = value;
    }

    return labels;
}

template <typename type, size_t num>
std::tuple<std::array<image_t, num>, std::array<input_t<type>, num>, std::array<size_t, num>> extract(std::string images_path, std::string labels_path) {
    std::vector<std::vector<std::vector<uint8_t>>> images;
    std::vector<std::vector<type>> inputs;
    std::vector<size_t> labels;

    std::tie(images, inputs) = extract_data<type, num>(images_path);
    labels = extract_labels<num>(labels_path);

    return std::make_tuple(images, inputs, labels);
}

void print_image(image_t image) {
    std::cout << '+';
    for (size_t i = 0; i < image[0].size(); i++) {
        std::cout << "--";
    }
    std::cout << "+\n";
    for (auto row : image) {
        std::cout << '|';
        for (auto pixel : row) {
            size_t brightness = ((double)pixel / 256.0) * ascii_scale.length();
            std::cout << ascii_scale[brightness] << ascii_scale[brightness];
        }
        std::cout << "|\n";
    }
    std::cout << '+';
    for (size_t i = 0; i < image[0].size(); i++) {
        std::cout << "--";
    }
    std::cout << "+\n" << std::flush;
}

template <typename type>
void print_image(input_t<type> image) {
    std::cout << '+';
    for (size_t i = 0; i < image[0].size(); i++) {
        std::cout << "--";
    }
    std::cout << "+\n";
    for (auto row : image) {
        std::cout << '|';
        for (auto pixel : row) {
            size_t brightness;
            if (not std::is_floating_point<type>::value) {
                std::cerr << "Error: unsupported type '" << typeid(type).name() << "'\n";
                exit(1);
            }
            if (pixel > 1.0) {
                brightness = 255;
            } else if (pixel < 0.0) {
                brightness = 0;
            } else {
                brightness = (size_t)((pixel * 255.0) / 256.0) * ascii_scale.length();
            }
            std::cout << ascii_scale[brightness] << ascii_scale[brightness];
        }
        std::cout << "|\n";
    }
    std::cout << '+';
    for (size_t i = 0; i < image[0].size(); i++) {
        std::cout << "--";
    }
    std::cout << "+\n" << std::flush;
}