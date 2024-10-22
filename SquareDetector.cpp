#include "SquareDetector.h"
#include <iostream>

SquareDetector::SquareDetector() {
    // Constructor
}

bool SquareDetector::predict(const std::vector<std::vector<int>>& image) {
    return isSquare(image);
}

bool SquareDetector::isSquare(const std::vector<std::vector<int>>& image) {
    int height = image.size();
    int width = image[0].size();

    // Kiểm tra nếu chiều cao và chiều rộng bằng nhau
    if (height != width) {
        return false;
    }

    // Kiểm tra nếu tất cả các pixel đều là 1 (giả sử hình vuông là các pixel 1)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (image[i][j] != 1) {
                return false;
            }
        }
    }

    return true;
}