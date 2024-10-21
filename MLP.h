// Định nghĩa lớp MLP với các hàm dự đoán và huấn luyện
//Nếu cập nhật trọng số theo công thức truyền thống, mô hình sẽ không học được. Để giải quyết vấn đề này, chúng ta cần sử dụng hàm kích hoạt ReLU cho các lớp ẩn và hàm kích hoạt Sigmoid cho lớp đầu ra
//
//Thêm phưng thức save và load để lưu và tải mô hình
// Thêm phương thức lưu và tải trọng số và bias
// Thêm phương thức đánh giá mô hình trên tệp dữ liệu kiểm tra




#ifndef MLP_H
#define MLP_H

#include <vector>

class MLP {
public:
    std::vector<std::vector<std::vector<float>>> weights_hidden; //vector 3 chiều để lưu trữ trọng số của các lớp ẩn. Mỗi lớp ẩn có một ma trận trọng số.
    std::vector<std::vector<float>> biases_hidden; //vector 2 chiều để lưu trữ bias của các lớp ẩn. Mỗi lớp ẩn có một vector bias
    std::vector<std::vector<float>> hidden_layers; // vector 2 chiều để lưu trữ giá trị của các nơ-ron ở các lớp ẩn
    std::vector<float> weights_output; //vector để lưu trữ trọng số của lớp đầu ra
    float bias_output; // Bias của lớp đầu ra
    float learning_rate;

    MLP(const std::vector<int>& layer_sizes, float lr); //Nhap vao vector voi kich thuoc cac lop va learning rate

    // Dự đoán đầu ra
    float predict(const std::vector<float>& inputs);//dự đoán đầu ra dựa trên các đầu vào

    // Huấn luyện mô hình
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, int epochs);//huấn luyện mô hình với dữ liệu huấn luyện, nhãn và số lượng epochs.

    //  // Lưu mô hình
    // void save_model(const std::string& filename);

    // // Tải mô hình
    // void load_model(const std::string& filename);

};

#endif // MLP_H