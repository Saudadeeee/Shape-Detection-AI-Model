#ifndef MLP_H
#define MLP_H

#include <vector>
using namespace std;

class MLP {
public:
    vector<vector<float>> weights_input_hidden1;
    vector<vector<float>> weights_hidden1_hidden2;
    vector<vector<float>> weights_hidden2_output;
    vector<float> hidden_layer1, hidden_layer2;
    float bias_hidden1, bias_hidden2, bias_output;
    float learning_rate;

    MLP(int input_size, int hidden1_size, int hidden2_size, float lr);

    // Hàm predict sẽ được khai báo ở đây
    float predict(const vector<float>& inputs);

    // Hàm train sẽ được khai báo ở đây
    void train(const vector<vector<float>>& training_data, const vector<float>& labels, int epochs);
};

#endif
