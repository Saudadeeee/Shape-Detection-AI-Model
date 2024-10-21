// Định nghĩa các phương thức của lớp MLP
//file này quan trọng nhất nên động vào khi nào đọc hết comment ở trên

#include "MLP.h"
#include "ActivationFunctions.h"
#include <cstdlib> //Chỉ để sử dụng hàm rand()
#include <ctime>//Tương tụư như trên, bỏ cũng được

//Hàm khởi tạo
MLP::MLP(const std::vector<int>& layer_sizes, float lr) {
    std::srand(std::time(0)); // Khởi tạo seed cho số ngẫu nhiên
    int num_layers = layer_sizes.size();
    weights_hidden.resize(num_layers - 1); //Thay đổi kích thước của vector weights_hidden để chứa trọng số của các lớp ẩn.
    biases_hidden.resize(num_layers - 1);//Thay đổi kích thước của vector biases_hidden để chứa bias của các lớp ẩn.
    hidden_layers.resize(num_layers - 1); //Thay đổi kích thước của vector hidden_layers để chứa giá trị của các nơ-ron ở các lớp ẩn.
    //Lặp qua các lớp ẩn để khởi tạo trọng số và bias
    for (int i = 0; i < num_layers - 1; ++i) {
        weights_hidden[i].resize(layer_sizes[i], std::vector<float>(layer_sizes[i + 1]));//Thay đổi kích thước của ma trận trọng số cho lớp ẩn thứ i
        biases_hidden[i].resize(layer_sizes[i + 1]); //Thay đổi kích thước của vector bias cho lớp ẩn thứ i
        hidden_layers[i].resize(layer_sizes[i + 1], 0.0); //Thay đổi kích thước của vector hidden_layers cho lớp ẩn thứ i
        //Vòng lặp qua các nơ-ron của lớp ẩn thứ i
        for (int j = 0; j < layer_sizes[i]; ++j) {//Vòng lặp qua các nơ-ron của lớp thứ i
            for (int k = 0; k < layer_sizes[i + 1]; ++k) {//Vòng lặp qua các nơ-ron của lớp tiếp theo.
                weights_hidden[i][j][k] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
            }
        }
        // Vòng lặp qua các nơ-ron của lớp tiếp theo.
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            biases_hidden[i][j] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
        }
    }
    //Khởi tạo trọng số và bias cho lớp đầu ra
    weights_output.resize(layer_sizes.back());
    //Vòng lặp qua các nơ-ron của lớp đầu ra.
    for (int i = 0; i < layer_sizes.back(); ++i) {
        weights_output[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
    }
    bias_output = static_cast<float>(std::rand()) / RAND_MAX - 0.5f; // Khởi tạo ngẫu nhiên trong khoảng [-0.5, 0.5]
    learning_rate = lr;
}
//Đã tách hàm dự đoán ra khỏi hàm huấn luyện

//Đã tachs hàm train ra
    