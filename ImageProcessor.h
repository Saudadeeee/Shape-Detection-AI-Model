#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <string>

class ImageProcessor {
public:
   static std::vector<std::vector<std::vector<int>>> readImages(const std::string& imagePath);
};

#endif // IMAGE_PROCESSOR_H