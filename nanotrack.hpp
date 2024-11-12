#ifndef NANOTRACK_H
#define NANOTRACK_H

#include <vector> 
#include <map>  
 
#include <opencv2/core/core.hpp> 
// #include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 

#include <string.h>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "RKNNModel.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>

#define _BASETSD_H

#include "RgaUtils.h"

#include "rknn_api.h"

using namespace std;

#define PI 3.1415926 

// using namespace cv;

struct Config{ 
    
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.15;
    float window_influence = 0.455;
    // float lr = 0.38;
    float lr = 0.37;

    int exemplar_size=127;
    int instance_size=255;
    int total_stride=16;
    int score_size=16;
    float context_amount = 0.5;
};

struct State { 
    int im_h; 
    int im_w;  
    cv::Scalar channel_ave; 
    cv::Point target_pos; 
    cv::Point2f target_sz = {0.f, 0.f}; 
    float cls_score_max; 
};

class NanoTrack {

public: 
    
    NanoTrack();
    
    ~NanoTrack(); 

    void init(cv::Mat img, cv::Rect bbox);
    
    void update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, float scale_z, float &cls_score_max);
        
    float track(cv::Mat& im);
    
    void load_model(std::string T_backbone_model, std::string X_backbone_model, std::string model_head);

    vector<float> result_T, result_X;


    int stride=16;
    
    // state  dynamic
    State state;
    
    // config static
    Config cfg; 

    const float mean_vals[3] = { 0.485f*255.f, 0.456f*255.f, 0.406f*255.f };  
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

	RKNNModel module_T127;
	RKNNModel module_X255;
	RKNNModel net_head;

    rknn_context ctx_T;
    rknn_context ctx_X;
    rknn_context ctx_head;



private: 
    void create_grids(); 
    void create_window();  
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
};

#endif 