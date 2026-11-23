#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../include/image_kernels.cuh"
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    std::string inputFolder = "dataset/input";
    std::string outputFolder = "output";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            inputFolder = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputFolder = argv[++i];
        }
    }

    if (!fs::exists(inputFolder)) {
        std::cerr << "Input folder does not exist!\n";
        return 1;
    }

    fs::create_directories(outputFolder + "/grayscale");
    fs::create_directories(outputFolder + "/blurred");
    fs::create_directories(outputFolder + "/edges");

    int totalImages = 0;
    int processedImages = 0;

    std::cout << "Starting CUDA Batch Image Processing...\n";

    try {
        for (const auto& entry : fs::directory_iterator(inputFolder)) {
            if (!entry.is_regular_file())
                continue;

            std::string extension = entry.path().extension().string();

            if (extension != ".jpg" && extension != ".jpeg" &&
                extension != ".png" && extension != ".bmp" &&
                extension != ".JPG" && extension != ".PNG")
                continue;

            totalImages++;
            std::string filename = entry.path().stem().string();

            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

            if (image.empty()) {
                std::cout << "Could not load: " << filename << "\n";
                continue;
            }

            cv::Mat grayscale;
            cv::Mat blurred;
            cv::Mat edges;

            grayscaleGPU(image, grayscale);
            gaussianBlurGPU(grayscale, blurred);
            sobelGPU(grayscale, edges);

            cv::imwrite(outputFolder + "/grayscale/" + filename + ".png", grayscale);
            cv::imwrite(outputFolder + "/blurred/" + filename + ".png",blurred);
            cv::imwrite(outputFolder + "/edges/" + filename + ".png", edges);
            processedImages++;
            std::cout << "Processed: " << filename << "\n";
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n========== SUMMARY ==========\n";
    std::cout << "Total images found: " << totalImages << "\n";
    std::cout << "Successfully processed: " << processedImages << "\n";
    std::cout << "Failed to process: " << totalImages - processedImages << "\n";
    std::cout << "Output directory: " << outputFolder << "\n";
    std::cout << "Processing completed!\n";

    return 0;
}