#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "Usage: video_cmp originalVideo compressedVideo\n";
        return EXIT_FAILURE;
    }

    cv::VideoCapture originalCapture(argv[argc - 2]);
    cv::VideoCapture compressedCapture(argv[argc - 1]);

    assert(originalCapture.isOpened());
    assert(compressedCapture.isOpened());

    // Verify dimensions and format
    uint64_t width = originalCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    uint64_t height = originalCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    uint64_t nFrames = originalCapture.get(cv::CAP_PROP_FRAME_COUNT);

    assert(compressedCapture.get(cv::CAP_PROP_FRAME_WIDTH) == width);
    assert(compressedCapture.get(cv::CAP_PROP_FRAME_HEIGHT) == height);
    assert(compressedCapture.get(cv::CAP_PROP_FRAME_COUNT) == nFrames);

    assert(     originalCapture.get(cv::CAP_PROP_CODEC_PIXEL_FORMAT) 
             == compressedCapture.get(cv::CAP_PROP_CODEC_PIXEL_FORMAT));
    
    std::cout << "Video has " << nFrames << " " << width << " x " << height << " frames" << std::endl;

    // Set std::cout fp precision
    std::cout << std::fixed << std::setprecision(6);

    uint64_t frameId = 0;
    while (true)
    {
        cv::Mat originalFrame;
        cv::Mat compressedFrame;

        originalCapture >> originalFrame;
        compressedCapture >> compressedFrame;

        if(originalFrame.empty() || compressedFrame.empty())
        {
            break;
        }

        double squaredError = 0.0;

        for(uint64_t row = 0; row < height; row++)
        {
            for(uint64_t col = 0; col < width; col++)
            {
                cv::Vec3b originalPixel = originalFrame.at<cv::Vec3b>(row, col);
                cv::Vec3b compressedPixel = compressedFrame.at<cv::Vec3b>(row, col);

                squaredError += std::pow(originalPixel[0] - compressedPixel[0], 2);
                squaredError += std::pow(originalPixel[1] - compressedPixel[1], 2);
                squaredError += std::pow(originalPixel[2] - compressedPixel[2], 2);
            }
        }
        
        double meanSquaredError = squaredError / (width * height * 3);

        double psnr = 10 * log10( std::pow(255, 2) / meanSquaredError );

        std::cout << "Frame " << std::setw(4) << frameId++ << " PSNR: " << psnr << " (dB)\n";
    }
    

    return EXIT_SUCCESS;
}