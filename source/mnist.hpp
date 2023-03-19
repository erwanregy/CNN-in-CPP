#pragma once

#include <array>
#include <string>
#include <cstdint>
#include <tuple>

using image_t = std::array<std::array<uint8_t, 28>, 28>;

template <typename type>
using input_t = std::array<type, 784>;

template <size_t num_images=0>
std::array<image_t, num_images> extract_images(std::string images_path);

template <typename type, size_t num_images=0>
std::array<input_t<type>, num_images> extract_inputs(std::string images_path);

template <typename type, size_t num_images=0>
std::tuple<std::array<image_t, num_images>, std::array<input_t<type>, num_images>> extract_data(std::string images_path);

template <size_t num_labels=0>
std::array<size_t, num_labels> extract_labels(std::string labels_path);

template <typename type, size_t num=0>
std::tuple<std::array<image_t, num>, std::array<input_t<type>, num>, std::array<size_t, num>> extract(std::string images_path, std::string labels_path);

const std::string ascii_scale = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

void print_image(image_t image);

template <typename type>
void print_image(input_t<type> inputs);

#include "mnist.tpp"