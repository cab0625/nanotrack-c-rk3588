#pragma once
#include <string.h>
#include <string>
#include <vector>
#include "rknn_api.h"

class RKNNModel
{
public:
    RKNNModel();
    ~RKNNModel();

    std::vector<rknn_tensor_attr> outputsAttr;

    int runRKNN(std::vector<std::vector<float>> &output, void *input_data, uint32_t input_size, rknn_tensor_type input_type, bool pass_through = false);
    int runRKNN(std::vector<std::vector<float>> &output, void *input_data1, uint32_t input_size1, void *input_data2, uint32_t input_size2, rknn_tensor_type input_type, bool pass_through);

    int loadRKNN(std::string modelPath, int outputLength, std::string modelName="");

    int releaseRKNN();

private:
    void *pModel;
    rknn_context ctx;
    std::string modelName;  // 添加这个成员变量
};
