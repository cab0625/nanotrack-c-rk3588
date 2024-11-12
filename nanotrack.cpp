#include <iostream>
#include <cstdlib>
#include <string>
#include "nanotrack.hpp"

using namespace std;

std::vector<float> convert_score(const std::vector<float> &input){
    std::vector<float> exp_values(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        exp_values[i] = std::exp(input[i]);
    }
    std::vector<float> output(256);
    for (size_t i = 256; i < input.size(); ++i) {
        output[i-256] = exp_values[i] / (exp_values[i] + exp_values[i-256]);
    }
    return output;
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i*cols+j] + pad[i*cols+j]) * (h[i*cols+j] + pad[i*cols+j])) / sz;
            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2; 
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * rows + j] / h[i * rows + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2; 
}

NanoTrack::NanoTrack()
{   
    
    
}

NanoTrack::~NanoTrack()
{
    
}

void NanoTrack::init(cv::Mat img, cv::Rect bbox) 
{
    create_window(); 
    create_grids(); 
    cv::Point2f target_pos ={0.f, 0.f}; // cx, cy
    cv::Point2f target_sz = {0.f, 0.f}; //w,h

    target_pos.x = bbox.x + (bbox.width - 1) / 2; 
    target_pos.y = bbox.y + (bbox.height -1) / 2;
    target_sz.x=bbox.width;
    target_sz.y=bbox.height;
    
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));  

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;
    
    z_crop  = get_subwindow_tracking(img, target_pos, cfg.exemplar_size, int(s_z),avg_chans); //cv::Mat BGR order 

    vector<vector<float>> rknnOutputs;
    int ret = module_T127.runRKNN(rknnOutputs, (void*)z_crop.data, 127 * 127 * 3, RKNN_TENSOR_UINT8, false);

    this->result_T = rknnOutputs[0];
    
    this->state.channel_ave=avg_chans;
    this->state.im_h=img.rows;
    this->state.im_w=img.cols;
    this->state.target_pos=target_pos;
    this->state.target_sz= target_sz; 

}

void NanoTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz,  float scale_z, float &cls_score_max)
{


    vector<vector<float>> rknnOutputs;
    int ret = this->module_X255.runRKNN(rknnOutputs, (void*)x_crops.data, 255 * 255 * 3 , RKNN_TENSOR_UINT8, false);
    this->result_X = rknnOutputs[0];


    std::vector<float> result_T_transposedVec(48 * 8 * 8);
    // 原始 shape 为 (48, 8, 8)
    // 目标 shape 为 (8, 8, 48)
    for (int i = 0; i < 48; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                result_T_transposedVec[j * 8 * 48 + k * 48 + i] = result_T[i * 8 * 8 + j * 8 + k];
            }
        }
    }

    std::vector<float> result_X_transposedVec(48 * 16 * 16);
    // 原始 shape 为 (48, 16, 16)
    // 目标 shape 为 (16, 16, 48)
    for (int i = 0; i < 48; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 16; ++k) {
                result_X_transposedVec[j * 16 * 48 + k * 48 + i] = result_X[i * 16 * 16 + j * 16 + k];
            }
        }
    }

    vector<vector<float>> rknnOutputs_2;
    net_head.runRKNN(rknnOutputs_2, (void*)result_T_transposedVec.data(), 48 * 8 * 8*4, (void*)result_X_transposedVec.data(), 48*16*16*4, RKNN_TENSOR_FLOAT32, false);

    vector<float> cls_score_result = rknnOutputs_2[0];

    vector<float> bbox_pred_result = rknnOutputs_2[1];

    float* cls_score_data = (float*) cls_score_result.data();

    int cols = 16; 
    int rows = 16; 
    vector<float> cls_scores = convert_score(cls_score_result);
    std::vector<float> pred_x1(cols*rows, 0), pred_y1(cols*rows, 0), pred_x2(cols*rows, 0), pred_y2(cols*rows, 0);

    float* bbox_pred_data = (float*) bbox_pred_result.data();

    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - bbox_pred_data[i*cols + j];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - bbox_pred_data[i*cols + j + 16*16*1];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + bbox_pred_data[i*cols + j + 16*16*2];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + bbox_pred_data[i*cols + j + 16*16*3];
        }
    }

    // size penalty  
    std::vector<float> w(cols*rows, 0), h(cols*rows, 0); 
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++) 
        {
            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * cfg.penalty_k);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    // int r_max = 0, c_max = 0; 
    float maxScore = 0; 

    int max_idx = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_scores[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence; 
        if (pscore[i] > maxScore) 
        {
            // get max 
            maxScore = pscore[i]; 
            max_idx = i;

        }
    }
    
    // to real size
    float pred_x1_real = pred_x1[max_idx]; 
    float pred_y1_real = pred_y1[max_idx];
    float pred_x2_real = pred_x2[max_idx];
    float pred_y2_real = pred_y2[max_idx];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;
    float diff_xs = pred_xs ;
    float diff_ys = pred_ys ;

    diff_xs /= scale_z; 
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr = penalty[max_idx] * cls_scores[max_idx] * cfg.lr;

    // size rate
    auto res_xs = float (target_pos.x + diff_xs);
    auto res_ys = float (target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    target_pos.x = res_xs;
    target_pos.y = res_ys;
    target_sz.x = res_w;
    target_sz.y = res_h;
    cls_score_max = cls_scores[max_idx];
}

float NanoTrack::track(cv::Mat& im)
{
    cv::Point target_pos = this->state.target_pos;
    cv::Point2f target_sz = this->state.target_sz;
    
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  
    float scale_z = cfg.exemplar_size / s_z;  

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2; 
    float pad = d_search / scale_z; 
    float s_x = s_z + 2*pad;

    cv::Mat x_crop;  
    x_crop  = get_subwindow_tracking(im, target_pos, cfg.instance_size, std::round(s_x),state.channel_ave);

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;

    this->update(x_crop, target_pos, target_sz, scale_z, cls_score_max);
    target_pos.x = std::max(0, min(state.im_w, target_pos.x));
    target_pos.y = std::max(0, min(state.im_h, target_pos.y));
    target_sz.x = float(std::max(10, min(state.im_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, min(state.im_h, int(target_sz.y))));

    state.target_pos = target_pos;
    state.target_sz = target_sz;
    return cls_score_max;
}


void NanoTrack::load_model(std::string T_model_backbone, std::string X_model_backbone, std::string model_head)
{
    this->module_T127.loadRKNN(T_model_backbone, 1, "model_T");
    this->module_X255.loadRKNN(X_model_backbone, 1, "model_X");
    this->net_head.loadRKNN(model_head, 2, "model_head");

}

void NanoTrack::create_window()
{
    int score_size= cfg.score_size;
    std::vector<float> hanning(score_size,0);
    this->window.resize(score_size*score_size, 0);

    for (int i = 0; i < score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    } 
    for (int i = 0; i < score_size; i++)
    {
        for (int j = 0; j < score_size; j++)
        {
            this->window[i*score_size+j] = hanning[i] * hanning[j]; 
        }
    }  

}

// 生成每一个格点的坐标 
void NanoTrack::create_grids()
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = cfg.score_size;   //16x16

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = j*cfg.total_stride-128;   
            this->grid_to_search_y[i*sz+j] = i*cfg.total_stride-128;
        }
    }
}

cv::Mat NanoTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = pos.x - c + 0.5;
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = pos.y - c + 0.5;
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));
    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
       
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path; 
}