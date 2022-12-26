#pragma once
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "YUV4MPEG2.h"
#include "BitStream.h"
#include "GolombCoder.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class FrameQuantization
{
public:
    static void Quantize(cv::Mat& m, uint16_t nBits)
    {
        assert(m.type() == CV_8UC1);
        assert(nBits < 8);
        
        const double delta = pow(2, 8 - nBits);

        const uint64_t width = m.cols;
        const uint64_t height = m.rows;

        for(uint64_t row = 0; row < height; row++)
        {
            for(uint64_t col = 0; col < width; col++)
            {
                double sample = m.at<uchar>(row, col);
                double result = (std::floor(sample / delta) + 0.5) * delta;

                m.at<uchar>(row, col) = static_cast<uchar>(result) / delta;
            }
        }
    }

    static void DeQuantize(cv::Mat& m, uint16_t nBits)
    {
        assert(m.type() == CV_8UC1);
        assert(nBits < 8);
        
        const double delta = pow(2, 8 - nBits);

        const uint64_t width = m.cols;
        const uint64_t height = m.rows;

        for(uint64_t row = 0; row < height; row++)
        {
            for(uint64_t col = 0; col < width; col++)
            {
                double sample = m.at<uchar>(row, col);

                m.at<uchar>(row, col) = sample * delta;
            }
        }
    }

    template <typename T>
    static void Quantize(std::vector<T>& vec, uint16_t nBits)
    {
        assert(nBits < 16);

        for(uint64_t i = 0; i < vec.size(); i++)
        {
            vec[i] >>= nBits;
        }
    }
    
    template <typename T>
    static void Quantize(std::vector<T>& vec, uint16_t nBits, uint64_t ptr, uint64_t MODE)
    {
        assert(nBits < 16);

        for(uint64_t i = ptr; i < ptr + (MODE * MODE); i++)
        {
            vec[i] >>= nBits;
        }
    }

    template <typename T>
    static void Dequantize(std::vector<T>& vec, uint16_t nBits)
    {
        assert(nBits < 16);

        for(uint64_t i = 0; i < vec.size(); i++)
        {
            vec[i] <<= nBits;
        }
    }

    template <typename T>
    static void Dequantize(std::vector<T>& vec, uint16_t nBits, uint64_t ptr, uint64_t MODE)
    {
        assert(nBits < 16);

        for(uint64_t i = ptr; i < ptr + (MODE * MODE); i++)
        {
            vec[i] <<= nBits;
        }
    }
};