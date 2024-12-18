#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <iostream>
#include <thread>
#include <mutex>
#include "cnn_model.h"

std::vector<Image> load_data(const std::string& file_path_X, const std::string& file_path_y) {
    std::vector<Image> dataset;
    std::ifstream file_X(file_path_X, std::ios::binary);
    std::ifstream file_y(file_path_y, std::ios::binary);

    if (!file_X.is_open() || !file_y.is_open()) {
        std::cerr << "Error: Could not open files." << std::endl;
        return dataset;
    }

    std::vector<std::thread> threads;
    std::mutex dataset_mutex;
    while (file_X && file_y) {
        Image img;
        file_X.read(reinterpret_cast<char*>(img.data), IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
        file_y.read(reinterpret_cast<char*>(&img.label), sizeof(img.label));
        if (file_X && file_y) {
            threads.emplace_back([&dataset, &dataset_mutex, img]() {
                std::lock_guard<std::mutex> lock(dataset_mutex);
                dataset.push_back(img);
            });
        }
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    std::cout << "Loaded " << dataset.size() << " images from " << file_path_X << " and " << file_path_y << std::endl;
    return dataset;
}

#endif 