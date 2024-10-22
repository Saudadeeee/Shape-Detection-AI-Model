//Đọc và xử lí ảnh
#include "ImageProcessor.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<std::vector<int>> ImageProcessor::readImage(const std::string& imagePath) {
    std::vector<std::vector<int>> image;
    std::ifstream file(imagePath);
    if (!file) {//Dùng cerr để in ra lỗi luôn mà không ảnh hướngr tới cout
        std::cerr << "Khong the mo file, hay thu lai " << imagePath << std::endl;
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