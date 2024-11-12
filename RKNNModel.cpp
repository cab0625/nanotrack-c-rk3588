#include "RKNNModel.h"
#include <iostream>
using namespace std;


RKNNModel::RKNNModel():pModel(nullptr),ctx(0) {}
RKNNModel::~RKNNModel()
{
    if (this->pModel)
    {
        free(this->pModel);
        this->pModel = nullptr;
    }
    if (this->ctx > 0)
    {
        rknn_destroy(ctx);
        ctx = 0;
    }
}

int RKNNModel::runRKNN(vector<vector<float>> &output, void *input_data, uint32_t input_size, rknn_tensor_type input_type, bool pass_through)
{
    string logMsg;

    rknn_input rknnInputs[1];
    rknnInputs[0].index = 0;
    rknnInputs[0].buf = input_data;
    rknnInputs[0].size = input_size;
    rknnInputs[0].pass_through = pass_through;
    rknnInputs[0].fmt = RKNN_TENSOR_NHWC;
    // rknn_tensor_type: RKNN_TENSOR_UINT8 / RKNN_TENSOR_FLOAT32
    rknnInputs[0].type = input_type;



    // input
    int ret = rknn_inputs_set(this->ctx, 1, rknnInputs);

    if (ret < 0)
    {
        logMsg = "rknn_input_set failed! ret=" + to_string(ret);
        
        return -1;
    }

    // run
    ret = rknn_run(this->ctx, nullptr);
    if (ret < 0)
    {
        logMsg = "rknn_run failed! ret=" + to_string(ret);
        
        return -1;
    }


    // infer output length
    int outputLength = this->outputsAttr.size();
    if (outputLength < 1)
    {
        logMsg = "outputsAttr is empty!";
        
        return -1;
    }

    // get output
    rknn_output *rknnOutputs = new rknn_output[outputLength];
    memset(rknnOutputs, 0, sizeof(rknn_output)*outputLength);

    // rknn_output rknnOutputs[1];
    for (int out_i = 0; out_i < outputLength; out_i++)
    {
        rknnOutputs[out_i].want_float = true;
        rknnOutputs[out_i].is_prealloc = false;
    }

    ret = rknn_outputs_get(this->ctx, outputLength, rknnOutputs, nullptr);
    if (ret < 0)
    {
        logMsg = "rknn_outputs_get failed! ret=" + to_string(ret);
        
        rknn_outputs_release(this->ctx, outputLength, rknnOutputs);
        delete rknnOutputs;
        return -1;
    }

    // set output
    output.resize(outputLength);
    for (int out_i = 0; out_i < outputLength; out_i++)
    {
        // cout << "n_elems=" << this->outputsAttr[out_i].n_elems << ", size=" << this->outputsAttr[out_i].size << endl;
        if (rknnOutputs[out_i].size == this->outputsAttr[out_i].n_elems * sizeof(float))
        {
            float *out_arr = (float *)rknnOutputs[out_i].buf;
            output[out_i] = vector<float>(out_arr, out_arr + this->outputsAttr[out_i].n_elems);
        }
        else
        {
            output.clear();
            logMsg = "rknn_outputs_get #" + to_string(out_i) + " of " + to_string(outputLength) + " failed! get_outputs_size=" + to_string(this->outputsAttr[out_i].size) + ", but expect " + to_string(this->outputsAttr[out_i].n_elems * sizeof(float));
            
            rknn_outputs_release(this->ctx, outputLength, rknnOutputs);
            delete rknnOutputs;
            return -1;
        }
    }

    // release resources
    rknn_outputs_release(this->ctx, outputLength, rknnOutputs);
    delete rknnOutputs;
    
    return 0;
}


int RKNNModel::runRKNN(vector<vector<float>> &output, void *input_data1, uint32_t input_size1, void *input_data2, uint32_t input_size2,rknn_tensor_type input_type, bool pass_through)
{
    string logMsg;
    string funcName = "runRKNN:" + this->modelName;

    rknn_input rknnInputs[2];
    rknnInputs[0].index = 0;
    rknnInputs[0].buf = input_data1;
    rknnInputs[0].size = input_size1;
    rknnInputs[0].pass_through = pass_through;
    rknnInputs[0].fmt = RKNN_TENSOR_NHWC;
    // rknn_tensor_type: RKNN_TENSOR_UINT8 / RKNN_TENSOR_FLOAT32
    rknnInputs[0].type = input_type;

	rknnInputs[1].index = 1;
    rknnInputs[1].buf = input_data2;
    rknnInputs[1].size = input_size2;
    rknnInputs[1].pass_through = pass_through;
    rknnInputs[1].fmt = RKNN_TENSOR_NHWC;
    // rknn_tensor_type: RKNN_TENSOR_UINT8 / RKNN_TENSOR_FLOAT32
    rknnInputs[1].type = input_type;

    // input
    int ret = rknn_inputs_set(this->ctx, 2, rknnInputs);
    if (ret < 0)
    {
        logMsg = "rknn_input_set failed! ret=" + to_string(ret);
        // 
        return -1;
    }

    // run
    ret = rknn_run(this->ctx, nullptr);
    if (ret < 0)
    {
        logMsg = "rknn_run failed! ret=" + to_string(ret);
        
        return -1;
    }

    // infer output length
    int outputLength = this->outputsAttr.size();
    if (outputLength < 1)
    {
        logMsg = "outputsAttr is empty!";
        
        return -1;
    }

    // get output
    rknn_output *rknnOutputs = new rknn_output[outputLength];

    memset(rknnOutputs, 0, sizeof(rknn_output)*outputLength);

    for (int out_i = 0; out_i < outputLength; out_i++)
    {
        rknnOutputs[out_i].want_float = true;
        rknnOutputs[out_i].is_prealloc = false;
    }
    ret = rknn_outputs_get(this->ctx, outputLength, rknnOutputs, nullptr);
    if (ret < 0)
    {
        logMsg = "rknn_outputs_get failed! ret=" + to_string(ret);
        
        rknn_outputs_release(this->ctx, outputLength, rknnOutputs);
        delete rknnOutputs;
        return -1;
    }

    // set output
    output.resize(outputLength);
    for (int out_i = 0; out_i < outputLength; out_i++)
    {
        // cout << "n_elems=" << this->outputsAttr[out_i].n_elems << ", size=" << this->outputsAttr[out_i].size << endl;
        // for (size_t i = 0; i < this->outputsAttr[out_i].n_dims; i++)
        // {
        //    cout << "dims[" << i << "]=" << this->outputsAttr[out_i].dims[i] << endl;
        // }
        
        if (rknnOutputs[out_i].size == this->outputsAttr[out_i].n_elems * sizeof(float))
        {
            float *out_arr = (float *)rknnOutputs[out_i].buf;
            output[out_i] = vector<float>(out_arr, out_arr + this->outputsAttr[out_i].n_elems);
        }
        else
        {
            output.clear();
            logMsg = "rknn_outputs_get #" + to_string(out_i) + " of " + to_string(outputLength) + " failed! get_outputs_size=" + to_string(this->outputsAttr[out_i].size) + ", but expect " + to_string(this->outputsAttr[out_i].n_elems * sizeof(float));
            
            rknn_outputs_release(this->ctx, outputLength, rknnOutputs);
            delete rknnOutputs;
            return -1;
        }
    }

    // release resources
    rknn_outputs_release(this->ctx, outputLength, rknnOutputs);
    delete rknnOutputs;
    
    return 0;
}


int RKNNModel::loadRKNN(string modelPath, int outputLength, string modelName)
{
    if (modelName != "")
    {
        this->modelName = modelName;
    }
    string logMsg;

    int modelLength = -1;
    try
    {
        // if (!check_exist(modelPath))
        // {
        //     logMsg = "modelPath not exist,  " + modelPath;
            
        //     return RKNN_FILE_INVALID;
        // }
        FILE *modelFP = fopen(modelPath.c_str(), "rb");
        if (modelFP == NULL)
        {
            logMsg = "fopen fail! " + modelPath;
            
            this->releaseRKNN();
            return -1;
        }
        fseek(modelFP, 0, SEEK_END);
        modelLength = ftell(modelFP);
        this->pModel = malloc(modelLength);
        fseek(modelFP, 0, SEEK_SET);
        if (modelLength != fread(this->pModel, 1, modelLength, modelFP))
        {
            logMsg = "fread fail! " + modelPath;
            
            fclose(modelFP);
            this->releaseRKNN();
            return -1;
        }
        fclose(modelFP);
    }
    catch (...)
    {
        logMsg = "load rknn fail! exception caught! " + modelPath;
        
        return -1;
    }

    int ret = rknn_init(&(this->ctx), this->pModel, modelLength, 0, nullptr); //| RKNN_FLAG_COLLECT_PERF_MASK
    ret |= rknn_set_core_mask(this->ctx, RKNN_NPU_CORE_0_1_2);
    // printf("this->ctx: %ld\n",this->ctx);
    if (ret < 0)
    {
        logMsg = "rknn_init fail! ret=" + to_string(ret);
        
        this->releaseRKNN();
        return -1;
    }

    // output attribute setting
    for (int iOut = 0; iOut < outputLength; iOut++)
    {
        rknn_tensor_attr modelOutput;
        memset(&modelOutput, 0, sizeof(rknn_tensor_attr));
        modelOutput.index = iOut;
        ret = rknn_query(this->ctx, RKNN_QUERY_OUTPUT_ATTR, &modelOutput, sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            logMsg = "rknn_query failed #" + to_string(iOut) + " of " + to_string(outputLength) + ", ret=" + to_string(ret);
            
            this->releaseRKNN();
            return -1;
        }
        this->outputsAttr.push_back(modelOutput);

        rknn_tensor_attr *attr = &modelOutput;
        std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
        for (int i = 1; i < attr->n_dims; ++i)
        {
            shape_str += ", " + std::to_string(attr->dims[i]);
        }
    }

    rknn_input_output_num io_num;
    ret = rknn_query(this->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
        }
    }

    

    // get sdk version
    rknn_sdk_version version;
    ret = rknn_query(this->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        logMsg = "rknn_query sdk version failed, , ret=" + to_string(ret);
        
        this->releaseRKNN();
        return -1;
    }
    string api_version(version.api_version);
    string drv_version(version.drv_version);
    cout << modelPath << endl;
    cout << logMsg << endl;

    return 0;
}

int RKNNModel::releaseRKNN()
{
    if (this->ctx > 0)
    {
        int ret = rknn_destroy(this->ctx);
        if (ret < 0)
        {
            return ret;
        }
        this->ctx = 0;
    }
    if (this->pModel)
    {
        free(this->pModel);
        this->pModel = nullptr;
    }
    if (!this->outputsAttr.empty())
        this->outputsAttr.clear();
    this->modelName = "UndefinedModel";
    return 0;
}
