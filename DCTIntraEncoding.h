#pragma once
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <numeric>
#include <algorithm>

#include "YUV4MPEG2.h"
#include "BitStream.h"
#include "GolombCoder.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class DCTIntraEncoding
{
public:
    enum MODE {Luma, Chroma};

    struct Result
    {
    public:
        struct Block
        {
            int64_t data[8 * 8];

            int64_t& operator[](int i)
            {
                return data[i];
            }

            const int64_t& operator[](int i) const
            {
                return data[i];
            }

            size_t GetZigZagCount()
            {
                size_t cnt = 64;
                for(int i = 63; i >= 0 && data[i] == 0; i--)
                {
                    cnt--;
                }

                return cnt;
            }

            size_t GetZigZagCount() const
            {
                size_t cnt = 64;
                for(int i = 63; i >= 0 && data[i] == 0; i--)
                {
                    cnt--;
                }

                return cnt;
            }

            void Encode()
            {
                int64_t cpy[8 * 8];

                cpy[0] = data[0];

                int x = 1, y = 0;

                size_t ptr = 1;
                while(true)
                {
                    for(y = 0; x >= 0; x--, y++)
                    {
                        cpy[ptr++] = data[y * 8 + x];
                    }

                    if(y == 8) break;

                    for(x = 0; y >= 0; x++, y--)
                    {
                        cpy[ptr++] = data[y * 8 + x];
                    }
                }

                x = 1, y = 7;
                while(true)
                {
                    for(; x < 8; x++, y--)
                    {
                        cpy[ptr++] = data[y * 8 + x];
                    }

                    y += 2;
                    if(y == 8) break;

                    for(x = 7; y < 8; x--, y++)
                    {
                        cpy[ptr++] = data[y * 8 + x];
                    }

                    y--;
                    x += 2;
                }

                memcpy(data, cpy, sizeof(int64_t) * 8 * 8);
            }

            void Decode()
            {
                int64_t cpy[8 * 8];

                cpy[0] = data[0];

                int x = 1, y = 0;

                size_t ptr = 1;
                while(true)
                {
                    for(y = 0; x >= 0; x--, y++)
                    {
                        cpy[y * 8 + x] = data[ptr++];
                    }

                    if(y == 8) break;

                    for(x = 0; y >= 0; x++, y--)
                    {
                        cpy[y * 8 + x] = data[ptr++];
                    }
                }

                x = 1, y = 7;
                while(true)
                {
                    for(; x < 8; x++, y--)
                    {
                        cpy[y * 8 + x] = data[ptr++];
                    }

                    y += 2;
                    if(y == 8) break;

                    for(x = 7; y < 8; x--, y++)
                    {
                        cpy[y * 8 + x] = data[ptr++];
                    }

                    y--;
                    x += 2;
                }

                memcpy(data, cpy, sizeof(int64_t) * 8 * 8);
            }

            void Clear()
            {
                memset(data, 0, sizeof(int64_t) * 8 * 8);
            }
        };

    private:
        uint64_t m_estimatedBits;

        int m_xBlocks;
        int m_yBlocks;

        Block* m_data;

    public:

        Result(int xBlocks, int yBlocks)
        {
            m_xBlocks = xBlocks;
            m_yBlocks = yBlocks;
            m_estimatedBits = 0;

            m_data = new Block[xBlocks * yBlocks];
        }

        Result()
        {
            m_xBlocks = 0;
            m_yBlocks = 0;
            m_estimatedBits = 0;
            m_data = nullptr;
        }

        Block& get(int x, int y)
        {
            return m_data[y * m_xBlocks + x];
        }

        const Block& get(int x, int y) const 
        {
            return m_data[y * m_xBlocks + x];
        }

        int xBlocks() const
        {
            return m_xBlocks;
        }

        int yBlocks() const
        {
            return m_yBlocks;
        }

        ~Result()
        {
            delete[] m_data;
        }
    };

    static std::unique_ptr<Result> Encode(const cv::Mat& m, int quality, MODE mode)
    {
        int xBlocks = m.cols / 8;
        int yBlocks = m.rows / 8;

        assert(m.rows % 8 == 0);
        assert(m.cols % 8 == 0);

        cv::Mat out(m.rows, m.cols, CV_8UC1);

        Result* result = new Result(xBlocks, yBlocks);

        cv::Mat din(8, 8, CV_64FC1);
        cv::Mat dout(8, 8, CV_64FC1);

        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {
                CopyBlockToArray(x, y, m, din);

                cv::InputArray dctIn(din);
                cv::OutputArray dctOut(dout);

                cv::dct(dctIn, dctOut);

                QuantizeBlock(result->get(x, y), dout, quality, mode);

                result->get(x, y).Encode();
            }
        }

        return std::unique_ptr<Result>(result);
    }

    static uint64_t EstimateBits(const Result* result, BitStream& bitstream)
    {
        int xBlocks = result->xBlocks();
        int yBlocks = result->yBlocks();

        uint64_t mSum = 0;

        int64_t lastDC = 0;
        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {
                int count = result->get(x, y).GetZigZagCount();
                
                uint64_t sum = 0;
                sum += std::abs(lastDC - result->get(x, y)[0]);
                for(int i = 1; i < count; i++)
                {
                    sum += std::abs(result->get(x, y)[i]);
                }
                sum += std::abs(count);
                count++;

                if(count == 0) count = 1;

                uint64_t tmpM = std::ceil((double)sum/count);
                if(tmpM == 0) tmpM = 1;

                mSum += tmpM;

                lastDC = result->get(x, y)[0];
            }
        }

        uint64_t m = std::ceil((double)mSum / (yBlocks * xBlocks));

        m = std::clamp(m, (uint64_t)1, (uint64_t)31);
        WriteM(m, bitstream);

        lastDC = 0;

        uint64_t bits = 0;
        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {   
                int count = result->get(x, y).GetZigZagCount();

                bits += GolombCoder::ComputeRequiredBits(count, m, true);
                bits += GolombCoder::ComputeRequiredBits(lastDC - result->get(x, y)[0], m, true);

                for(int i = 1; i < count; i++)
                {
                    bits += GolombCoder::ComputeRequiredBits(result->get(x, y)[i], m, true);
                }

                lastDC = result->get(x, y)[0];
            }
        }

        return bits;
    }

    static void Write(const Result* result, BitStream& bitstream)
    {
        int xBlocks = result->xBlocks();
        int yBlocks = result->yBlocks();

        uint64_t mSum = 0;

        int64_t lastDC = 0;
        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {
                int count = result->get(x, y).GetZigZagCount();
                
                uint64_t sum = 0;
                sum += std::abs(lastDC - result->get(x, y)[0]);
                for(int i = 1; i < count; i++)
                {
                    sum += std::abs(result->get(x, y)[i]);
                }
                sum += std::abs(count);
                count++;

                if(count == 0) count = 1;

                uint64_t tmpM = std::ceil((double)sum/count);
                if(tmpM == 0) tmpM = 1;

                mSum += tmpM;

                lastDC = result->get(x, y)[0];
            }
        }

        uint64_t m = std::ceil((double)mSum / (yBlocks * xBlocks));

        m = std::clamp(m, (uint64_t)1, (uint64_t)31);
        WriteM(m, bitstream);

        lastDC = 0;
        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {   
                int count = result->get(x, y).GetZigZagCount();

                bitstream.WriteNBits(GolombCoder::EncodeFold(count, m));
                bitstream.WriteNBits(GolombCoder::EncodeFold(lastDC - result->get(x, y)[0], m));
                for(int i = 1; i < count; i++)
                {
                    bitstream.WriteNBits(GolombCoder::EncodeFold(result->get(x, y)[i], m));
                }

                lastDC = result->get(x, y)[0];
            }
        }
    }

    static void Decode(cv::Mat& mat, int quality, MODE mode, BitStream& bitstream)
    {
        int xBlocks = mat.cols / 8;
        int yBlocks = mat.rows / 8;

        assert(mat.rows % 8 == 0);
        assert(mat.cols % 8 == 0);

        cv::Mat din(8, 8, CV_64FC1);
        cv::Mat dout(8, 8, CV_64FC1);

        Result::Block block;

        uint64_t m = ReadM(bitstream);
        int64_t count;

        int64_t lastDC = 0;

        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {                
                count = GolombCoder::DecodeOneFold(bitstream, m);

                block.Clear();
                block[0] = lastDC - GolombCoder::DecodeOneFold(bitstream, m);
                for(int i = 1; i < count; i++)
                {
                    block[i] = GolombCoder::DecodeOneFold(bitstream, m);
                }

                block.Decode();

                DeQuantizeBlock(block, din, quality, mode);

                cv::InputArray idctIn(din);
                cv::OutputArray idctOut(dout);

                cv::idct(idctIn, idctOut);

                CopyBlockToMat(x, y, mat, dout);

                lastDC = block[0];
            }
        }
    }

    static void Decode(const Result& result, cv::Mat& m , int quality, MODE mode)
    {
        int xBlocks = result.xBlocks();
        int yBlocks = result.yBlocks();

        cv::Mat din(8, 8, CV_64FC1);
        cv::Mat dout(8, 8, CV_64FC1);

        for(int y = 0; y < yBlocks; y++)
        {
            for(int x = 0; x < xBlocks; x++)
            {                
                Result::Block block = result.get(x, y);

                block.Decode();

                DeQuantizeBlock(block, din, quality, mode);

                cv::InputArray idctIn(din);
                cv::OutputArray idctOut(dout);

                cv::idct(idctIn, idctOut);

                CopyBlockToMat(x, y, m, dout);

                block.Encode();
            }
        }
    }

    inline static void WriteM(uint64_t m, BitStream& bitstream)
    {
        for(int b = 64-5; b < 64; b++)
        {
            bitstream.WriteBit((m >> (63 - b)) & 1);
        }
    }

    inline static uint64_t ReadM(BitStream& bitstream)
    {
        uint64_t m = 0;
        
        for(int b = 64-5; b < 64; b++)
        {
            bool bit = false;
            bitstream.ReadBit(bit);

            m = m | (uint64_t)bit << (63-b);
        }

        return m;
    }

    inline static void CopyBlockToArray(int x, int y, const cv::Mat& m, cv::Mat& block)
    {
        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                block.at<double>(i, j) = (double)(m.at<uchar>(y * 8 + i, x * 8 + j)) - 127.0;
            }
        }
    }

    inline static void CopyBlockToMat(int x, int y, cv::Mat& m, const cv::Mat& block)
    {
        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                m.at<uchar>(y * 8 + i, x * 8 + j) = block.at<double>(i, j) + 127.0;
            }
        }
    }

    inline static void QuantizeBlock(Result::Block& iblock, const cv::Mat& block, double quality, MODE mode)
    {
        const constexpr int32_t lumaQuantMatrix[8][8] =   {
                                                        {16, 11, 10, 16, 24, 40, 51, 61},
                                                        {12, 12, 14, 19, 26, 58, 60, 55},
                                                        {14, 13, 16, 24, 40, 57, 69, 56},
                                                        {14, 17, 22, 29, 51, 87, 80, 62},
                                                        {18, 22, 37, 56, 68, 109, 103, 77},
                                                        {24, 35, 55, 64, 81, 104, 113, 92},
                                                        {49, 64, 78, 87, 103, 121, 129, 101},
                                                        {72, 92, 95, 98, 112, 100, 103, 99}
                                                        };

        const constexpr int32_t chromaQuantMatrix[8][8] =   {
                                                        {17, 18, 24, 47, 99, 99, 99, 99},
                                                        {18, 21, 26, 66, 99, 99, 99, 99},
                                                        {24, 26, 56, 99, 99, 99, 99, 99},
                                                        {47, 66, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99}
                                                        };

        double S;
        if(quality < 50) S = 5000 / quality;
        else S = 200 - 2 * quality;

        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                double quant;
                if(mode == Luma) quant = lumaQuantMatrix[i][j];
                else quant = chromaQuantMatrix[i][j];

                quant = std::floor((S * quant + 50) / 100);
                if(quant < 1.0) quant = 1.0;

                iblock[i * 8 + j] = std::round(block.at<double>(i, j) / (quant));
            }
        }
    }

    inline static void DeQuantizeBlock(Result::Block& iblock, cv::Mat& block, double quality, MODE mode)
    {
        const constexpr int32_t lumaQuantMatrix[8][8] =   {
                                                        {16, 11, 10, 16, 24, 40, 51, 61},
                                                        {12, 12, 14, 19, 26, 58, 60, 55},
                                                        {14, 13, 16, 24, 40, 57, 69, 56},
                                                        {14, 17, 22, 29, 51, 87, 80, 62},
                                                        {18, 22, 37, 56, 68, 109, 103, 77},
                                                        {24, 35, 55, 64, 81, 104, 113, 92},
                                                        {49, 64, 78, 87, 103, 121, 129, 101},
                                                        {72, 92, 95, 98, 112, 100, 103, 99}
                                                        };

        const constexpr int32_t chromaQuantMatrix[8][8] =   {
                                                        {17, 18, 24, 47, 99, 99, 99, 99},
                                                        {18, 21, 26, 66, 99, 99, 99, 99},
                                                        {24, 26, 56, 99, 99, 99, 99, 99},
                                                        {47, 66, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99},
                                                        {99, 99, 99, 99, 99, 99, 99, 99}
                                                        };

        double S;
        if(quality < 50) S = 5000 / quality;
        else S = 200 - 2 * quality;

        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                double quant;
                if(mode == Luma) quant = lumaQuantMatrix[i][j];
                else quant = chromaQuantMatrix[i][j];

                quant = std::floor((S * quant + 50) / 100);
                if(quant < 1.0) quant = 1.0;

                block.at<double>(i, j) = iblock[i * 8 + j] * quant;
            }
        }
    }
};