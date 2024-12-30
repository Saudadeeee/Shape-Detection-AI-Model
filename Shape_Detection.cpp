#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint> 
#include <fstream>

using namespace std;

// Hàm tính toán giá trị grayscale từ giá trị RGB
uint8_t rgbToGrayscale(uint8_t r, uint8_t g, uint8_t b) {
    // Công thức chuyển đổi chuẩn: Y = 0.299*R + 0.587*G + 0.114*B
    return static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
}

vector<uint8_t> resizeImage(const vector<uint8_t>& grayscaleImage, int originalWidth, int originalHeight, int targetWidth, int targetHeight) {
    vector<uint8_t> resizedImage(targetWidth * targetHeight);
    for (int i = 0; i < targetHeight; i++) {
        for (int j = 0; j < targetWidth; j++) {
            int srcX = static_cast<int>(j * originalWidth / targetWidth);
            int srcY = static_cast<int>(i * originalHeight / targetHeight);
            resizedImage[i * targetWidth + j] = grayscaleImage[srcY * originalWidth + srcX];
        }
    }
    return resizedImage;
}

int main() {
    const int originalWidth = 200;
    const int originalHeight = 200;
    const int targetWidth = 64;
    const int targetHeight = 64;

    ifstream file("test.png", ios::binary);
    if (!file) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }

    vector<uint8_t> rgbImage((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    // Assuming the image is 200x200 and in RGB format
    vector<uint8_t> grayscaleImage(originalWidth * originalHeight);
    for (int i = 0; i < originalHeight; i++) {
        for (int j = 0; j < originalWidth; j++) {
            int index = (i * originalWidth + j) * 3;
            uint8_t r = rgbImage[index];
            uint8_t g = rgbImage[index + 1];
            uint8_t b = rgbImage[index + 2];
            grayscaleImage[i * originalWidth + j] = rgbToGrayscale(r, g, b);
        }
    }

    vector<uint8_t> resizedImage = resizeImage(grayscaleImage, originalWidth, originalHeight, targetWidth, targetHeight);

    for (int i = 0; i < targetHeight; i++) {
        for (int j = 0; j < targetWidth; j++) {
            cout << static_cast<int>(resizedImage[i * targetWidth + j]) << " ";
        }
        cout << endl;
    }

    return 0;
}
