#include "nanotrack.hpp"
#include "RKNNModel.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video_file_path>" << endl;
        return -1;
    }

    string video_name = argv[1];

    // 加载模型
    NanoTrack tracker;
    tracker.load_model("models/model_T.rknn", "models/model_X.rknn", "models/model_head.rknn");

    // 打开视频文件
    VideoCapture cap(video_name);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open video file " << video_name << endl;
        return -1;
    }

    // 获取视频帧率和帧尺寸
    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // 创建视频写入器
    VideoWriter video_writer("output_video.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height));

    // 读取第一帧并初始化
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: Unable to read first frame from video file " << video_name << endl;
        return -1;
    }

    Rect bbox(1104, 373, 60, 41); // 初始边界框
    tracker.init(frame, bbox);

    // 追踪
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        double t1 = getTickCount();
        float score = tracker.track(frame);
        double t2 = getTickCount();
        double process_time_ms = (t2 - t1) * 1000 / getTickFrequency();
        double fps_value = getTickFrequency() / (t2 - t1);
        cout << "每帧处理时间: " << process_time_ms << " ms, FPS: " << fps_value << endl;

        // 绘制边界框
        rectangle(frame, tracker.state.target_pos, tracker.state.target_sz, Scalar(0, 255, 0), 2);

        // 写入视频
        video_writer.write(frame);

        // 显示追踪结果
        imshow("Tracking", frame);
        if (waitKey(30) == 27) { // 按下Esc键退出
            break;
        }
    }

    // 释放资源
    video_writer.release();
    cap.release();
    destroyAllWindows();

    return 0;
}

