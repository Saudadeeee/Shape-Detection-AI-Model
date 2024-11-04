#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class ImageProcessor {
public:
    static std::vector<std::vector<std::vector<int>>> readImages(const std::string& imagePath);
};

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

#endif // IMAGE_PROCESSOR_H