#ifndef IMAGE_KERNELS_CUH
#define IMAGE_KERNELS_CUH
#include <opencv2/opencv.hpp>

void grayscaleGPU(const cv::Mat& input, cv::Mat& output);
void gaussianBlurGPU(const cv::Mat& input, cv::Mat& output);
void sobelGPU(const cv::Mat& input, cv::Mat& output);

#endif