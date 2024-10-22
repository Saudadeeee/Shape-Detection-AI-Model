
// Đây là file chính của chương trình, nơi chúng ta sẽ sử dụng mô hình MLP để giải quyết bài toán XOR logic gate
// Biên dịch các tệp bằng g++ -o NewModel NewModel.cpp MLP.cpp MLPTrain.cpp MLPPredict.cpp ActivationFunctions.cpp
// Sau đó chạy chương trình bằng g++ ./NewModel
//Tại vì t lười dùng makefile nên là phải làm thế này
//Nếu dùng makefile thì chỉ cần gõ make và ./NewModel là xong
//Cái này nó rất hay bị lỗi nên là nếu lỗi thì cứ biên dịch lại mấy lần là được
//Nếu không được thì hỏi t




#include <iostream>
#include "MLP.h"
#include "ActivationFunctions.h"
#include "ImageProcessor.h"
#include "MLPTrain.h"
#include "MLPPredict.h"

using namespace std;

int main() {
   // Đọc ảnh và chuyển đổi thành dữ liệu đầu vào
    vector<vector<int>> image = ImageProcessor::readImage("image.txt");

    // Chuyển đổi ảnh thành vector 1 chiều
    vector<float> input;
    for (const auto& row : image) {
        for (int pixel : row) {
            input.push_back(static_cast<float>(pixel));
        }
    }

    // Dữ liệu huấn luyện (ví dụ đơn giản)
    vector<vector<float>> training_data = {
        input, input, input, input
    };
    // Nhãn đầu ra tương ứng
    vector<float> labels = {1, 0, 1, 1};  // Giả sử tất cả đều là hình vuông

     // Khởi tạo mô hình MLP với 2 input, 10 lớp ẩn (mỗi lớp 10 nơ-ron), tốc độ học 0.1
    vector<int> layer_sizes = {static_cast<int>(input.size()), 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1}; // 2 input, 10 lớp ẩn, 1 output
    MLP mlp(layer_sizes, 0.1);

    // Huấn luyện mô hình trong 10000 epochs
    mlp.train(training_data, labels, 10000);

    for (size_t i = 0; i < training_data.size(); ++i) {
        cout << "Du doan cho anh " << i + 1 << ": " << mlp.predict(training_data[i]) << endl;
    }

   return 0;

}