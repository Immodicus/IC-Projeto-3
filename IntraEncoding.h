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

class IntraEncoding
{
public:
    enum Predictor : uint8_t
    {
        Vertical = 0, 
        Horizontal = 1,
        DC = 2,
        DiagDownLeft = 3,
        DiagDownRight = 4,
        VerticalLeft = 5,
        VerticalRight = 6,
        HorizontalDown = 7,
        HorizontalUp = 8,

        NONE = UINT8_MAX
    };

    static void WritePredictor(const Predictor lastPredictor, const Predictor p, BitStream& bitstream)
    {
        if(p == lastPredictor)
        {
            bitstream.WriteBit(true);
        }
        else
        {
            bitstream.WriteBit(false);
            bitstream.WriteBit((p >> 3) & 1);
            bitstream.WriteBit((p >> 2) & 1); 
            bitstream.WriteBit((p >> 1) & 1);    
            bitstream.WriteBit(p & 1);   
        }
    }

    static Predictor ReadPredictor(const Predictor lastPredictor, BitStream& bitstream)
    {      
        bool b = true;
        bitstream.ReadBit(b);

        if(b)
        {
            return lastPredictor;
        }

        uint8_t t = 0;

        bitstream.ReadBit(b);
        t = t | (b << 3);
        bitstream.ReadBit(b);
        t = t | (b << 2);
        bitstream.ReadBit(b);
        t = t | (b << 1);
        bitstream.ReadBit(b);
        t = t | b;

        return (Predictor)t;
    }

    static void LuminanceEncode(const cv::Mat& Y, BitStream& bitstream)
    {
        const constexpr uint64_t MODE = 4;

        Encode(Y, bitstream, MODE);
    }

    static void ChromaEncode(const cv::Mat& Y, BitStream& bitstream)
    {
        const constexpr uint64_t MODE = 8;

        Encode(Y, bitstream, MODE);
    }

    static cv::Mat LuminanceDecode(BitStream& bitstream, const YUV4MPEG2::YUV4MPEG2Description& desc)
    {
        const constexpr uint64_t MODE = 4;

        return Decode(bitstream, desc.width, desc.height, MODE);
    }

    static cv::Mat ChromaDecode(BitStream& bitstream, const YUV4MPEG2::YUV4MPEG2Description& desc)
    {
        const constexpr uint64_t MODE = 8;

        return Decode(bitstream, desc.width / 2, desc.height / 2, MODE);
    }

private:
    static void Encode(const cv::Mat& Y, BitStream& bitstream, const uint64_t MODE)
    {     
        uint64_t ymacroBlocks = Y.rows / MODE;
        uint64_t xmacroBlocks = Y.cols / MODE;

        std::vector<int16_t> residuals(Y.rows * Y.cols);

        std::vector<Predictor> predictors(ymacroBlocks * xmacroBlocks);
        predictors.clear();

        std::vector<int16_t> tmp(MODE * MODE);

        uint64_t ptr = 0;
        for(uint64_t y = 0; y < ymacroBlocks; y++)
        {
            for(uint64_t x = 0; x < xmacroBlocks; x++, ptr += (MODE * MODE))
            {
                uint64_t min = UINT64_MAX;
                Predictor p = NONE;

                if(TestBest(x, y, ptr, min, Y, tmp, residuals, MODE, VerticalEncode)) p = Vertical;
                if(TestBest(x, y, ptr, min, Y, tmp, residuals, MODE, HorizontalEncode)) p = Horizontal;
                if(TestBest(x, y, ptr, min, Y, tmp, residuals, MODE, DCEncode)) p = DC;

                if(MODE == 4)
                {
                    if(TestBest(x, y, xmacroBlocks, ptr, min, Y, tmp, residuals, DiagDownLeftEncode)) p = DiagDownLeft;
                    if(TestBest(x, y, ptr, min, Y, tmp, residuals, DiagDownRightEncode)) p = DiagDownRight;
                    if(TestBest(x, y, xmacroBlocks, ptr, min, Y, tmp, residuals, VerticalLeftEncode)) p = VerticalLeft;
                    if(TestBest(x, y, ptr, min, Y, tmp, residuals, VerticalRightEncode)) p = VerticalRight;
                    if(TestBest(x, y, ptr, min, Y, tmp, residuals, HorizontalDownEncode)) p = HorizontalDown;
                    if(TestBest(x, y, ptr, min, Y, tmp, residuals, HorizontalUpEncode)) p = HorizontalUp;
                }

                assert(p != NONE);
                predictors.push_back(p);
            }
        } 

        uint8_t m = GolombCoder::EstimateMFast(residuals);
        if(m == 0) m = 1;
        bitstream.WriteAlign(false);
        bitstream.Write(m);

        Predictor lastPredictor = NONE;
        for(const auto p : predictors)
        {
            WritePredictor(lastPredictor, p, bitstream);
            lastPredictor = p;
        }

        for(const auto r : residuals)
        {
            bitstream.WriteNBits(GolombCoder::Encode(r, m));
        }
    }

    inline static uint64_t VerticalEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;
        for (uint64_t row = 0; row < MODE; row++)
        {
            for (uint64_t col = 0; col < MODE; col++)
            {
                int16_t pred, pixel;

                if (y == 0) pred = 128;
                else pred = Y.at<uchar>(y * MODE - 1, x * MODE + col);

                pixel = Y.at<uchar>(y * MODE + row, x * MODE + col);

                diff += abs(pixel - pred);

                tmp[ptr++] = pixel - pred;
            }
        }

        return diff;
    }

    inline static uint64_t HorizontalEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;
        for (uint64_t row = 0; row < MODE; row++)
        {
            for (uint64_t col = 0; col < MODE; col++)
            {
                int16_t pred, pixel;

                if (x == 0) pred = 128;
                else pred = Y.at<uchar>(y * MODE + row, x * MODE - 1);

                pixel = Y.at<uchar>(y * MODE + row, x * MODE + col);

                diff += abs(pixel - pred);

                tmp[ptr++] = pixel - pred;
            }
        }

        return diff;
    }

    inline static uint64_t DCEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar A, B, C, D, I, J, K, L;
        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);
        }

        int16_t pred = (A + B + C + D + I + J + K + L + 4) / 8;

        for (uint64_t row = 0; row < MODE; row++)
        {
            for (uint64_t col = 0; col < MODE; col++)
            {
                int16_t pixel = Y.at<uchar>(y * MODE + row, x * MODE + col);

                diff += abs(pixel - pred);

                tmp[ptr++] = pixel - pred;
            }
        }

        return diff;
    }

    inline static uint64_t DiagDownLeftEncode(uint64_t x, uint64_t y, uint64_t xMax, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar A, B, C, D, E, F, G, H;

        if(y == 0)
        {
            A = 128; B = 128, C = 128, D = 128, E = 128, F = 128, G = 128, H = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);

            if(x != xMax)
            {
                E = Y.at<uchar>(y * MODE - 1, x * MODE + 4);
                F = Y.at<uchar>(y * MODE - 1, x * MODE + 5);
                G = Y.at<uchar>(y * MODE - 1, x * MODE + 6);
                H = Y.at<uchar>(y * MODE - 1, x * MODE + 7);
            }
            else 
            {
                E = 128, F = 128, G = 128, H = 128;
            }
        }

        int16_t p1, p2, p3, p4, p5, p6, p7;
        p1 = (A + 2 * B + C + 2) / 4;
        p2 = (B + 2 * C + D + 2) / 4;
        p3 = (C + 2 * D + E + 2) / 4;
        p4 = (D + 2 * E + F + 2) / 4;
        p5 = (E + 2 * F + G + 2) / 4;
        p6 = (F + 2 * G + H + 2) / 4;
        p7 = (G + 3 * H + 2) / 4;

        int16_t pixel = Y.at<uchar>(y * MODE, x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 1);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 2);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 3);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 1);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 2);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 3);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 1);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 2);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 3);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 1);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 2);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 3);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        return diff;
    }

    inline static uint64_t DiagDownRightEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar Q, A, B, C, D, I, J, K, L;

        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);
        }

        if(y == 0 || x == 0)  Q = 128;
        else Q = Y.at<uchar>(y * MODE - 1, x * MODE - 1);
        
        uint16_t p1, p2, p3, p4, p5, p6, p7;

        p1 = (L + 2 * K + J + 2) / 4;
        p2 = (K + 2 * J + I + 2) / 4;
        p3 = (J + 2 * I + Q + 2) / 4;
        p4 = (I + 2 * Q + A + 2) / 4;
        p5 = (Q + 2 * A + B + 2) / 4;
        p6 = (A + 2 * B + C + 2) / 4;
        p7 = (B + 2 * C + D + 2) / 4;

        int16_t pixel = Y.at<uchar>(y * MODE, x * MODE);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 1);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 2);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 3);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 1);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 2);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 3);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 1);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 2);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 3);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 1);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 2);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 3);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        return diff;
    }

    inline static uint64_t VerticalLeftEncode(uint64_t x, uint64_t y, uint64_t xMax, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar A, B, C, D, E, F, G;

        if(y == 0)
        {
            A = 128; B = 128, C = 128, D = 128, E = 128, F = 128, G = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);

            if(x != xMax)
            {
                E = Y.at<uchar>(y * MODE - 1, x * MODE + 4);
                F = Y.at<uchar>(y * MODE - 1, x * MODE + 5);
                G = Y.at<uchar>(y * MODE - 1, x * MODE + 6);
            }
            else 
            {
                E = 128, F = 128, G = 128;
            }
        }

        int16_t p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;

        p1 = (A + B + 1) / 2;
        p2 = (B + C + 1) / 2;
        p3 = (C + D + 1) / 2;
        p4 = (D + E + 1) / 2;
        p5 = (E + F + 1) / 2;
        p6 = (A + 2 * B + C + 2) / 4;
        p7 = (B + 2 * C + D + 2) / 4;
        p8 = (C + 2 * D + E + 2) / 4;
        p9 = (D + 2 * E + F + 2) / 4;
        p10 = (E + 2 * F + G + 2) / 4;

        int16_t pixel = Y.at<uchar>(y * MODE, x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 1);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 2);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 3);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 1);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 2);
        diff += abs(pixel - p8);
        tmp[ptr++] = pixel - p8;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 3);
        diff += abs(pixel - p9);
        tmp[ptr++] = pixel - p9;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 1);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 2);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 3);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 1);
        diff += abs(pixel - p8);
        tmp[ptr++] = pixel - p8;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 2);
        diff += abs(pixel - p9);
        tmp[ptr++] = pixel - p9;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 3);
        diff += abs(pixel - p10);
        tmp[ptr++] = pixel - p10;

        return diff;
    }

    inline static uint64_t VerticalRightEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar Q, A, B, C, D, I, J, K;

        if(x == 0)
        {
            I = 128, J = 128, K = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);
        }

        if(y == 0 || x == 0)  Q = 128;
        else Q = Y.at<uchar>(y * MODE - 1, x * MODE - 1);
        
        uint16_t p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;

        p1 = (Q + A + 1) / 2;
        p2 = (A + B + 1) / 2;
        p3 = (B + C + 1) / 2;
        p4 = (C + D + 1) / 2;
        p5 = (I + 2 * Q + A + 2) / 4;
        p6 = (Q + 2 * A + B + 2) / 4;
        p7 = (A + 2 * B + C + 2) / 4;
        p8 = (B + 2 * C + D + 2) / 4;
        p9 = (Q + 2 * I + J + 2) / 4;
        p10 = (I + 2 * J + K + 2) / 4;

        int16_t pixel = Y.at<uchar>(y * MODE, x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 1);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 2);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 3);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 1);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 2);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 3);
        diff += abs(pixel - p8);
        tmp[ptr++] = pixel - p8;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 1);
        diff += abs(pixel - p9);
        tmp[ptr++] = pixel - p9;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 2);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 3);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE);
        diff += abs(pixel - p10);
        tmp[ptr++] = pixel - p10;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 1);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 2);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 3);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        return diff;
    }

    inline static uint64_t HorizontalDownEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar Q, A, B, C, I, J, K, L;

        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
        }

        if(y == 0 || x == 0)  Q = 128;
        else Q = Y.at<uchar>(y * MODE - 1, x * MODE - 1);

        uint16_t p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;

        p1 = (Q + I + 1) / 2;
        p2 = (I + 2 * Q + A + 2) / 4;
        p3 = (Q + 2 * A + B + 2) / 4;
        p4 = (A + 2 * B + C + 2) / 4;
        p5 = (I + J + 1) / 2;
        p6 = (Q + 2 * I + J + 2) / 4;
        p7 = (J + K + 1) / 2;
        p8 = (I + 2 * J + K + 2) / 4;
        p9 = (K + L + 1) / 2;
        p10 = (J + 2 * K + L + 2) / 4;

        int16_t pixel = Y.at<uchar>(y * MODE, x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 1);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 2);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 3);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 1);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 2);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 3);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 1);
        diff += abs(pixel - p8);
        tmp[ptr++] = pixel - p8;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 2);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 3);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE);
        diff += abs(pixel - p9);
        tmp[ptr++] = pixel - p9;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 1);
        diff += abs(pixel - p10);
        tmp[ptr++] = pixel - p10;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 2);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 3);
        diff += abs(pixel - p8);
        tmp[ptr++] = pixel - p8;

        return diff;
    }

    inline static uint64_t HorizontalUpEncode(uint64_t x, uint64_t y, const cv::Mat& Y, std::vector<int16_t>& tmp, const uint64_t MODE)
    {
        uint64_t ptr = 0;
        uint64_t diff = 0;

        uchar I, J, K, L;
        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        int16_t p1, p2, p3, p4, p5, p6, p7;

        p1 = (I + J + 1) / 2;
        p2 = (I + 2 * J + K + 2) / 4;
        p3 = (J + K + 1) / 2;
        p4 = (J + 2 * K + L + 2) / 4;
        p5 = (K + L + 1) / 2;
        p6 = (K + 2 * L + L + 2) / 4;
        p7 = L;

        int16_t pixel = Y.at<uchar>(y * MODE, x * MODE);
        diff += abs(pixel - p1);
        tmp[ptr++] = pixel - p1;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 1);
        diff += abs(pixel - p2);
        tmp[ptr++] = pixel - p2;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 2);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE, x * MODE + 3);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE);
        diff += abs(pixel - p3);
        tmp[ptr++] = pixel - p3;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 1);
        diff += abs(pixel - p4);
        tmp[ptr++] = pixel - p4;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 2);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 1 , x * MODE + 3);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE);
        diff += abs(pixel - p5);
        tmp[ptr++] = pixel - p5;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 1);
        diff += abs(pixel - p6);
        tmp[ptr++] = pixel - p6;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 2);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 2 , x * MODE + 3);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 1);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 2);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        pixel = Y.at<uchar>(y * MODE + 3 , x * MODE + 3);
        diff += abs(pixel - p7);
        tmp[ptr++] = pixel - p7;

        return diff;
    }

    static cv::Mat Decode(BitStream& bitstream, const uint64_t width, const uint64_t height, const uint64_t MODE)
    {
        cv::Mat Y(height, width, CV_8UC1);
        
        uint64_t ymacroBlocks = Y.rows / MODE;
        uint64_t xmacroBlocks = Y.cols / MODE;

        uint8_t m = 0;
        bitstream.ReadAlign();
        bitstream.Read(m);

        Predictor lastPredictor = NONE;
        std::vector<Predictor> predictors(ymacroBlocks * xmacroBlocks);
        for(uint64_t p = 0; p < ymacroBlocks * xmacroBlocks; p++)
        {
            predictors[p] = lastPredictor = ReadPredictor(lastPredictor, bitstream);
        }

        std::vector<int64_t> residuals = GolombCoder::Decode(bitstream, m, width * height);
        if(residuals.size() == 0) return cv::Mat(0, 0, 0);

        uint64_t ptr = 0;
        uint64_t b = 0;
        for(uint64_t y = 0; y < ymacroBlocks; y++)
        {
            for(uint64_t x = 0; x < xmacroBlocks; x++, b++)
            {   
                Predictor p = predictors[b];
                
                switch (p)
                {
                case Vertical:
                {
                    VerticalDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                case Horizontal:
                {
                    HorizontalDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                case DC:
                {
                    DCDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                case DiagDownLeft:
                {
                    DiagDownLeftDecode(x, y, xmacroBlocks, Y, residuals, ptr, MODE);
                    break;
                }
                case DiagDownRight:
                {
                    DiagDownRightDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                case VerticalLeft:
                {
                    VerticalLeftDecode(x, y, xmacroBlocks, Y, residuals, ptr, MODE);
                    break;
                }
                case VerticalRight:
                {
                    VerticalRightDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                case HorizontalDown:
                {
                    HorizontalDownDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                case HorizontalUp:
                {
                    HorizontalUpDecode(x, y, Y, residuals, ptr, MODE);
                    break;
                }
                default:
                {
                    throw std::runtime_error(std::string("Invalid predictor ") + std::to_string((uint32_t)p) + " detected");
                    break;
                }
                }
            }
        }

        return Y;
    }

    inline static void VerticalDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        for (uint64_t row = 0; row < MODE; row++)
        {
            for (uint64_t col = 0; col < MODE; col++)
            {
                int16_t pred, res;

                if (y == 0) pred = 128;
                else pred = Y.at<uchar>(y * MODE - 1, x * MODE + col);

                res = residuals[ptr++];
                Y.at<uchar>(y * MODE + row, x * MODE + col) = pred + res;
            }
        }
    }

    inline static void HorizontalDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        for (uint64_t row = 0; row < MODE; row++)
        {
            for (uint64_t col = 0; col < MODE; col++)
            {
                int16_t pred, res;

                if (x == 0) pred = 128;
                else pred = Y.at<uchar>(y * MODE + row, x * MODE - 1);

                res = residuals[ptr++];
                Y.at<uchar>(y * MODE + row, x * MODE + col) = pred + res;
            }
        }
    }

    inline static void DCDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar A, B, C, D, I, J, K, L;
        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);
        }

        int16_t pred = (A + B + C + D + I + J + K + L + 4) / 8;
        
        for (uint64_t row = 0; row < MODE; row++)
        {
            for (uint64_t col = 0; col < MODE; col++)
            {
                int16_t res = residuals[ptr++];
                Y.at<uchar>(y * MODE + row, x * MODE + col) = pred + res;
            }
        }
    }

    inline static void DiagDownLeftDecode(uint64_t x, uint64_t y, uint64_t xMax, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar A, B, C, D, E, F, G, H;

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128, E = 128, F = 128, G = 128, H = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);

            if(x != xMax)
            {
                E = Y.at<uchar>(y * MODE - 1, x * MODE + 4);
                F = Y.at<uchar>(y * MODE - 1, x * MODE + 5);
                G = Y.at<uchar>(y * MODE - 1, x * MODE + 6);
                H = Y.at<uchar>(y * MODE - 1, x * MODE + 7);
            }
            else 
            {
                E = 128, F = 128, G = 128, H = 128;
            }
        }

        int16_t p1, p2, p3, p4, p5, p6, p7;
        p1 = (A + 2 * B + C + 2) / 4;
        p2 = (B + 2 * C + D + 2) / 4;
        p3 = (C + 2 * D + E + 2) / 4;
        p4 = (D + 2 * E + F + 2) / 4;
        p5 = (E + 2 * F + G + 2) / 4;
        p6 = (F + 2 * G + H + 2) / 4;
        p7 = (G + 3 * H + 2) / 4;

        Y.at<uchar>(y * MODE, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 1) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 2) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 3) = p4 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 1, x * MODE) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 1) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 2) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 3) = p5 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 2, x * MODE) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 1) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 2) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 3) = p6 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 3, x * MODE) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 1) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 2) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 3) = p7 + residuals[ptr++];
    }

    inline static void DiagDownRightDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar Q, A, B, C, D, I, J, K, L;

        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);
        }

        if(y == 0 || x == 0)  Q = 128;
        else Q = Y.at<uchar>(y * MODE - 1, x * MODE - 1);
        
        uint16_t p1, p2, p3, p4, p5, p6, p7;

        p1 = (L + 2 * K + J + 2) / 4;
        p2 = (K + 2 * J + I + 2) / 4;
        p3 = (J + 2 * I + Q + 2) / 4;
        p4 = (I + 2 * Q + A + 2) / 4;
        p5 = (Q + 2 * A + B + 2) / 4;
        p6 = (A + 2 * B + C + 2) / 4;
        p7 = (B + 2 * C + D + 2) / 4;

        Y.at<uchar>(y * MODE, x * MODE) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 1) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 2) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 3) = p7 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 1, x * MODE) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 1) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 2) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 3) = p6 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 2, x * MODE) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 1) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 2) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 3) = p5 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 3, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 1) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 2) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 3) = p4 + residuals[ptr++];
    }

    inline static void VerticalLeftDecode(uint64_t x, uint64_t y, uint64_t xMax, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar A, B, C, D, E, F, G;

        if(y == 0)
        {
            A = 128; B = 128, C = 128, D = 128, E = 128, F = 128, G = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);

            if(x != xMax)
            {
                E = Y.at<uchar>(y * MODE - 1, x * MODE + 4);
                F = Y.at<uchar>(y * MODE - 1, x * MODE + 5);
                G = Y.at<uchar>(y * MODE - 1, x * MODE + 6);
            }
            else 
            {
                E = 128, F = 128, G = 128;
            }
        }

        int16_t p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;

        p1 = (A + B + 1) / 2;
        p2 = (B + C + 1) / 2;
        p3 = (C + D + 1) / 2;
        p4 = (D + E + 1) / 2;
        p5 = (E + F + 1) / 2;
        p6 = (A + 2 * B + C + 2) / 4;
        p7 = (B + 2 * C + D + 2) / 4;
        p8 = (C + 2 * D + E + 2) / 4;
        p9 = (D + 2 * E + F + 2) / 4;
        p10 = (E + 2 * F + G + 2) / 4;

        Y.at<uchar>(y * MODE, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 1) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 2) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 3) = p4 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 1, x * MODE) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 1) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 2) = p8 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 3) = p9 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 2, x * MODE) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 1) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 2) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 3) = p5 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 3, x * MODE) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 1) = p8 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 2) = p9 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 3) = p10 + residuals[ptr++];
    }

    inline static void VerticalRightDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar Q, A, B, C, D, I, J, K;

        if(x == 0)
        {
            I = 128, J = 128, K = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128, D = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
            D = Y.at<uchar>(y * MODE - 1, x * MODE + 3);
        }

        if(y == 0 || x == 0)  Q = 128;
        else Q = Y.at<uchar>(y * MODE - 1, x * MODE - 1);
        
        uint16_t p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;

        p1 = (Q + A + 1) / 2;
        p2 = (A + B + 1) / 2;
        p3 = (B + C + 1) / 2;
        p4 = (C + D + 1) / 2;
        p5 = (I + 2 * Q + A + 2) / 4;
        p6 = (Q + 2 * A + B + 2) / 4;
        p7 = (A + 2 * B + C + 2) / 4;
        p8 = (B + 2 * C + D + 2) / 4;
        p9 = (Q + 2 * I + J + 2) / 4;
        p10 = (I + 2 * J + K + 2) / 4;

        Y.at<uchar>(y * MODE, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 1) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 2) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 3) = p4 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 1, x * MODE) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 1) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 2) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 3) = p8 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 2, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 1) = p9 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 2) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 3) = p3 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 3, x * MODE) = p10 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 1) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 2) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 3) = p7 + residuals[ptr++];
    }

    inline static void HorizontalDownDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar Q, A, B, C, I, J, K, L;

        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        if(y == 0)
        {
            A = 128, B = 128, C = 128;
        }
        else
        {
            A = Y.at<uchar>(y * MODE - 1, x * MODE);
            B = Y.at<uchar>(y * MODE - 1, x * MODE + 1);
            C = Y.at<uchar>(y * MODE - 1, x * MODE + 2);
        }

        if(y == 0 || x == 0)  Q = 128;
        else Q = Y.at<uchar>(y * MODE - 1, x * MODE - 1);

        uint16_t p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;

        p1 = (Q + I + 1) / 2;
        p2 = (I + 2 * Q + A + 2) / 4;
        p3 = (Q + 2 * A + B + 2) / 4;
        p4 = (A + 2 * B + C + 2) / 4;
        p5 = (I + J + 1) / 2;
        p6 = (Q + 2 * I + J + 2) / 4;
        p7 = (J + K + 1) / 2;
        p8 = (I + 2 * J + K + 2) / 4;
        p9 = (K + L + 1) / 2;
        p10 = (J + 2 * K + L + 2) / 4;

        Y.at<uchar>(y * MODE, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 1) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 2) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 3) = p4 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 1, x * MODE) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 1) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 2) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 3) = p2 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 2, x * MODE) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 1) = p8 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 2) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 3) = p6 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 3, x * MODE) = p9 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 1) = p10 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 2) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 3) = p8 + residuals[ptr++];
    }

    inline static void HorizontalUpDecode(uint64_t x, uint64_t y, cv::Mat& Y, const std::vector<int64_t>& residuals, uint64_t& ptr, const uint64_t MODE)
    {
        uchar I, J, K, L;
        if(x == 0)
        {
            I = 128, J = 128, K = 128, L = 128;
        }
        else
        {
            I = Y.at<uchar>(y * MODE, x * MODE - 1);
            J = Y.at<uchar>(y * MODE + 1, x * MODE - 1);
            K = Y.at<uchar>(y * MODE + 2, x * MODE - 1);
            L = Y.at<uchar>(y * MODE + 3, x * MODE - 1);
        }

        int16_t p1, p2, p3, p4, p5, p6, p7;

        p1 = (I + J + 1) / 2;
        p2 = (I + 2 * J + K + 2) / 4;
        p3 = (J + K + 1) / 2;
        p4 = (J + 2 * K + L + 2) / 4;
        p5 = (K + L + 1) / 2;
        p6 = (K + 2 * L + L + 2) / 4;
        p7 = L;

        Y.at<uchar>(y * MODE, x * MODE) = p1 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 1) = p2 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 2) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE, x * MODE + 3) = p4 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 1, x * MODE) = p3 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 1) = p4 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 2) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 1, x * MODE + 3) = p6 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 2, x * MODE) = p5 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 1) = p6 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 2) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 2, x * MODE + 3) = p7 + residuals[ptr++];

        Y.at<uchar>(y * MODE + 3, x * MODE) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 1) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 2) = p7 + residuals[ptr++];
        Y.at<uchar>(y * MODE + 3, x * MODE + 3) = p7 + residuals[ptr++];
    }


    template<typename F>
    inline static bool TestBest(uint64_t x, uint64_t y, uint64_t ptr, uint64_t& min, const cv::Mat& Y, std::vector<int16_t>& tmp, std::vector<int16_t>& predictions, const uint64_t MODE, F func)
    {
        uint64_t diff = func(x, y, Y, tmp, MODE);

        if(diff < min)
        {
            min = diff;
            
            for(const auto r : tmp)
            {
                predictions[ptr++] = r;
            }

            return true;
        }

        return false;
    }

    template<typename F>
    inline static bool TestBest(uint64_t x, uint64_t y, uint64_t xMax, uint64_t ptr, uint64_t& min, const cv::Mat& Y, std::vector<int16_t>& tmp, std::vector<int16_t>& predictions, F func)
    {
        constexpr const uint64_t MODE = 4;
        
        uint64_t diff = func(x, y, xMax, Y, tmp, MODE);

        if(diff < min)
        {
            min = diff;
            
            for(const auto r : tmp)
            {
                predictions[ptr++] = r;
            }

            return true;
        }

        return false;
    }

    template<typename F>
    inline static bool TestBest(uint64_t x, uint64_t y, uint64_t ptr, uint64_t& min, const cv::Mat& Y, std::vector<int16_t>& tmp, std::vector<int16_t>& predictions, F func)
    {
        constexpr const uint64_t MODE = 4;
        
        uint64_t diff = func(x, y, Y, tmp, MODE);

        if(diff < min)
        {
            min = diff;
            
            for(const auto r : tmp)
            {
                predictions[ptr++] = r;
            }

            return true;
        }

        return false;
    }
};