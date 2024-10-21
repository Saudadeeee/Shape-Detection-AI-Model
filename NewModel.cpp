Biên dịch các tệp bằng g++ -o NewModel NewModel.cpp MLP.cpp MLPTrain.cpp MLPPredict.cpp ActivationFunctions.cpp
Sau đó chạy chương trình bằng g++ ./NewModel
#include <iostream>
#include "MLP.h"
#include "ActivationFunctions.h"

using namespace std;

int main() {
    // Dữ liệu huấn luyện (ví dụ đơn giản)
    vector<vector<float>> training_data = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // Nhãn đầu ra tương ứng
    vector<float> labels = {0, 1, 1, 0};  // XOR logic gate

    // Khởi tạo mô hình MLP với 2 input, 3 lớp ẩn (mỗi lớp 3 nơ-ron), tốc độ học 0.1
    vector<int> layer_sizes = {2, 3, 3, 3, 1}; // 2 input, 3 nơ-ron lớp ẩn thứ nhất, 3 nơ-ron lớp ẩn thứ hai, 3 nơ-ron lớp ẩn thứ ba, 1 output
    MLP mlp(layer_sizes, 0.1);

    // Huấn luyện mô hình trong 10000 epochs
    mlp.train(training_data, labels, 10000);

    // Kiểm tra dự đoán
    for (auto& data : training_data) {
        cout << "Du doan cho (" << data[0] << ", " << data[1] << "): " << mlp.predict(data) << endl;
    }

    return 0;
}