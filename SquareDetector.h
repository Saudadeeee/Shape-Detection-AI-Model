#ifndef SQUARE_DETECTOR_H
#define SQUARE_DETECTOR_H

#include <vector>
#include <string>

class SquareDetector {
public:
    SquareDetector();
    bool predict(const std::vector<std::vector<int>>& image);

private:
    bool isSquare(const std::vector<std::vector<int>>& image);
};

#endif // SQUARE_DETECTOR_H