#include "ImageProcessor.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<std::vector<int>> ImageProcessor::readImage(const std::string& imagePath) {
    std::vector<std::vector<int>> image;
    std::ifstream file(imagePath);
    if (!file) {
        std::cerr << "Could not open the file: " << imagePath << std::endl;
        return image;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::istringstream iss(line);
        int pixel;
        while (iss >> pixel) {
            row.push_back(pixel);
        }
        image.push_back(row);
    }

    file.close();
    return image;
}

std::vector<std::vector<std::vector<int>>> ImageProcessor::readImages(const std::string& imagePath) {
    std::vector<std::vector<std::vector<int>>> images;
    std::ifstream file(imagePath);
    if (!file) {
        std::cerr << "Could not open the file: " << imagePath << std::endl;
        return images;
    }

    std::string line;
    std::vector<std::vector<int>> image;
    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!image.empty()) {
                images.push_back(image);
                image.clear();
            }
        } else {
            std::vector<int> row;
            std::istringstream iss(line);
            int pixel;
            while (iss >> pixel) {
                row.push_back(pixel);
            }
            image.push_back(row);
        }
    }
    if (!image.empty()) {
        images.push_back(image);
    }

    file.close();
    return images;
}