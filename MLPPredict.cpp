#include "MLPPredict.h"
#include "ActivationFunctions.h"

float MLP::predict(const std::vector<float>& inputs) const {
    std::vector<float> current_layer = inputs;

    for (int i = 0; i < hidden_layers.size(); ++i) {
        std::vector<float> next_layer(hidden_layers[i].size(), 0.0);
        for (int j = 0; j < hidden_layers[i].size(); ++j) {
            float sum = biases_hidden[i][j];
            for (int k = 0; k < current_layer.size(); ++k) {
                sum += current_layer[k] * weights_hidden[i][k][j];
            }
            next_layer[j] = relu(sum);
        }
        current_layer = next_layer;
    }

    float output_sum = bias_output;
    for (int i = 0; i < current_layer.size(); ++i) {
        output_sum += current_layer[i] * weights_output[i];
    }
    return sigmoid(output_sum);
}
