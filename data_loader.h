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

std::vector<Image> load_data(const std::string& file_path_X, const std::string& file_path_y, size_t batch_size = 100) {
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
        std::vector<Image> batch;
        for (size_t i = 0; i < batch_size && file_X && file_y; ++i) {
            Image img;
            file_X.read(reinterpret_cast<char*>(img.data), IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
            file_y.read(reinterpret_cast<char*>(&img.label), sizeof(img.label));
            if (file_X && file_y) {
                batch.push_back(img);
            }
        }
        if (!batch.empty()) {
            threads.emplace_back([&dataset, &dataset_mutex, batch]() {
                std::lock_guard<std::mutex> lock(dataset_mutex);
                dataset.insert(dataset.end(), batch.begin(), batch.end());
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