#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <algorithm>
#include <cmath>

std::vector<float> relu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](float x) { return std::max(0.0f, x); });
    return output;
}

std::vector<float> max_pool2d(const std::vector<float>& input, int input_size, int pool_size) {
    int output_size = input_size / pool_size;
    std::vector<float> output(output_size * output_size, 0.0f);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int ki = 0; ki < pool_size; ++ki) {
                for (int kj = 0; kj < pool_size; ++kj) {
                    int idx = (i * pool_size + ki) * input_size + (j * pool_size + kj);
                    max_val = std::max(max_val, input[idx]);
                }
            }
            output[i * output_size + j] = max_val;
        }
    }

    return output;
}

#endif // ACTIVATION_FUNCTIONS_H
