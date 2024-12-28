#include <iostream>
#include <fstream>
#include <vector>

const int ORIGINAL_IMAGE_SIZE = 200;

int main() {
    std::vector<unsigned char> buffer(ORIGINAL_IMAGE_SIZE * ORIGINAL_IMAGE_SIZE);

    // Fill the buffer with a gradient
    for (int i = 0; i < ORIGINAL_IMAGE_SIZE; ++i) {
        for (int j = 0; j < ORIGINAL_IMAGE_SIZE; ++j) {
            buffer[i * ORIGINAL_IMAGE_SIZE + j] = static_cast<unsigned char>((i + j) % 256);
        }
    }

    std::ofstream file("test_binary.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not create binary image 'test_binary.bin'" << std::endl;
        return -1;
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::cout << "Generated 'test_binary.bin' with size: " << buffer.size() << " bytes" << std::endl;

    return 0;
}
