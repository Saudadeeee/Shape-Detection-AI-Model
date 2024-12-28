#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include "cnn_model.h"

std::vector<Image> load_data(const std::string& file_path_X, const std::string& file_path_y) {
    std::vector<Image> dataset;
    std::ifstream file_X(file_path_X, std::ios::binary);
    std::ifstream file_y(file_path_y, std::ios::binary);

    if (!file_X.is_open() || !file_y.is_open()) {
        std::cerr << "Error: Could not open files." << std::endl;
        return dataset;
    }

    while (file_X && file_y) {
        Image img;
        file_X.read(reinterpret_cast<char*>(img.data), IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
        file_y.read(reinterpret_cast<char*>(&img.label), sizeof(img.label));
        if (file_X && file_y) {
            dataset.push_back(img);
        }
    }

    return dataset;
}

void apply_horizontal_flip(float data[IMAGE_SIZE][IMAGE_SIZE]) {
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        std::reverse(data[i], data[i] + IMAGE_SIZE);
    }
}

void apply_random_rotation(float data[IMAGE_SIZE][IMAGE_SIZE]) {
    float temp[IMAGE_SIZE][IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            temp[j][IMAGE_SIZE - 1 - i] = data[i][j];
        }
    }
    std::copy(&temp[0][0], &temp[0][0] + IMAGE_SIZE * IMAGE_SIZE, &data[0][0]);
}

void augment_data(std::vector<Image>& batch) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& img : batch) {
        if (dis(gen) > 0.5) {
            apply_horizontal_flip(img.data);
        }
        if (dis(gen) > 0.5) {
            apply_random_rotation(img.data);
        }
    }
}

#endif