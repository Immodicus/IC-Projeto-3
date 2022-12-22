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

class MotionCompensation
{
public:
    struct Result
    {
        std::vector<cv::Vec2b> motionVectors;
        std::vector<int16_t> residuals;
        uint64_t totalDiff;
        const uint64_t frameSize;

        Result(uint64_t blockCount, uint64_t frameSize) : totalDiff(0), frameSize(frameSize)
        {
            residuals.resize(frameSize);
            residuals.clear();

            motionVectors.resize(blockCount);
            motionVectors.clear();
        }

        uint64_t Estimate() const
        {
            uint64_t m = std::ceil(totalDiff / frameSize);
            if(m == 0) m = 1;

            return GolombCoder::ComputeRequiredBits(residuals, m, false) + sizeof(uchar) * motionVectors.size() * 8 * 2;
        }

        uint64_t M() const
        {
            uint64_t m = std::ceil(totalDiff / frameSize);
            if(m == 0) m = 1;

            return m;
        }
    };
    
    static Result Encode(const cv::Mat& prevFrame, const cv::Mat& currFrame, uint64_t blockSize, uchar searchArea)
    {
        assert(!prevFrame.empty());
        assert(!currFrame.empty());
        assert(prevFrame.cols == currFrame.cols && prevFrame.rows == currFrame.rows);
        assert(blockSize > 0);
        assert((int)blockSize <= currFrame.rows && (int)blockSize <= currFrame.cols);
        assert((int)blockSize <= prevFrame.rows && (int)blockSize <= prevFrame.cols);
        
        uint64_t yBlocks = prevFrame.rows / blockSize;
        uint64_t xBlocks = prevFrame.cols / blockSize;
        uint64_t frameSize = prevFrame.rows * prevFrame.cols;
        uint64_t blockCount = xBlocks * yBlocks;

        cv::Mat currBlock(blockSize, blockSize, CV_8UC1);
        cv::Mat prevBlock(blockSize, blockSize, CV_8UC1);
        cv::Mat bestBlock(blockSize, blockSize, CV_8UC1);

        Result result(blockCount, frameSize);

        for(uint64_t y = 0; y < yBlocks; y++)
        {
            for(uint64_t x = 0; x < xBlocks; x++)
            {
                GetBlock(x, y, 0, 0, currFrame, currBlock, blockSize);

                uint64_t min = UINT64_MAX;
                uchar minX = 0, minY = 0;

                for(uchar xOffset = 0; xOffset < searchArea; xOffset++)
                {
                    for(uchar yOffset = 0; yOffset < searchArea; yOffset++)
                    {
                        if(GetBlock(x, y, xOffset, yOffset, prevFrame, prevBlock, blockSize))
                        {
                            uint64_t diff = CmpBlocks(currBlock, prevBlock);

                            if(diff < min)
                            {
                                min = diff;
                                minX = xOffset;
                                minY = yOffset;
                                prevBlock.copyTo(bestBlock);
                            }
                        }
                    }
                }

                // std::cout << "Vec2b(" << (int32_t)minX << ", " << (int32_t)minY << "): " << min << "\n";
                ComputeResiduals(currBlock, bestBlock, result);
                result.motionVectors.push_back(cv::Vec2b(minX, minY));
            }
        }

        return result;
    }

    static void Write(const Result& result, BitStream& bitstream)
    {
        uint16_t m = result.M();
        bitstream.WriteAlign(false);
        bitstream.Write(m);

        for(const auto& motVec : result.motionVectors)
        {
            uchar x = motVec[0];
            uchar y = motVec[1];

            bitstream.Write(x);
            bitstream.Write(y);
        }

        for(uint64_t r = 0; r < result.frameSize; r++)
        {
            bitstream.WriteNBits(GolombCoder::Encode(result.residuals[r], m));
        }
    }

    static bool Decode(BitStream& bitstream, const cv::Mat& prevFrame, cv::Mat& currFrame, uint64_t blockSize)
    {
        assert(!prevFrame.empty());
        assert(!currFrame.empty());
        assert(prevFrame.cols == currFrame.cols && prevFrame.rows == currFrame.rows);
        assert(blockSize > 0);
        assert((int)blockSize <= currFrame.rows && (int)blockSize <= currFrame.cols);
        assert((int)blockSize <= prevFrame.rows && (int)blockSize <= prevFrame.cols);
        
        uint64_t yBlocks = prevFrame.rows / blockSize;
        uint64_t xBlocks = prevFrame.cols / blockSize;
        uint64_t frameSize = prevFrame.rows * prevFrame.cols;
        uint64_t blockCount = xBlocks * yBlocks;

        std::vector<cv::Vec2b> motionVectors(blockCount);
        motionVectors.clear();

        std::vector<int64_t> residuals = Read(bitstream, motionVectors, frameSize, blockCount);
        if(residuals.size() < frameSize) return false;

        cv::Mat prevBlock(blockSize, blockSize, CV_8UC1);

        uint64_t block = 0, ptr = 0;
        for(uint64_t y = 0; y < yBlocks; y++)
        {
            for(uint64_t x = 0; x < xBlocks; x++, block++)
            {
                cv::Vec2b motionVector = motionVectors[block];
                
                GetBlock(x, y, motionVector[0], motionVector[1], prevFrame, prevBlock, blockSize);

                ComputePixels(ptr, prevBlock, residuals, blockSize);

                CopyBlockToFrame(x, y, prevBlock, currFrame, blockSize);         
            }
        }

        return true;
    }

private:
    static std::vector<int64_t> Read(BitStream& bitstream, std::vector<cv::Vec2b>& motionVectors, uint64_t frameSize, uint64_t blockCount)
    {
        uint16_t m = 1;
        bitstream.ReadAlign();
        bitstream.Read(m);

        for(uint64_t v = 0; v < blockCount; v++)
        {
            uchar x = 0, y = 0;

            bitstream.Read(x);
            bitstream.Read(y);

            motionVectors.push_back(cv::Vec2b(x, y));
        }

        return GolombCoder::Decode(bitstream, m, frameSize);
    }
    
    static inline void CopyBlockToFrame(uint64_t x, uint64_t y, const cv::Mat& block, cv::Mat& frame, uint64_t blockSize)
    {
        for (uint64_t row = 0; row < blockSize; row++)
        {
            for (uint64_t col = 0; col < blockSize; col++)
            {
                frame.at<uchar>(y * blockSize + row, x * blockSize + col) = block.at<uchar>(row, col);
            }
        }
    }

    static inline void ComputePixels(uint64_t& ptr, cv::Mat& block, const std::vector<int64_t>& residuals, uint64_t blockSize)
    {
        for (uint64_t row = 0; row < blockSize; row++)
        {
            for (uint64_t col = 0; col < blockSize; col++)
            {
                block.at<uchar>(row, col) -= residuals[ptr++];
            }
        }
    }

    // returns true on success
    // false if block goes outside of frame boundries
    static inline bool GetBlock(uint64_t x, uint64_t y, uchar xOffset, uchar yOffset, const cv::Mat& frame, cv::Mat& block, uint64_t blockSize)
    {
        const int64_t maxHeight = frame.rows;
        const int64_t maxWidth = frame.cols;
        
        for (uint64_t row = 0; row < blockSize; row++)
        {
            for (uint64_t col = 0; col < blockSize; col++)
            {
                int64_t pY = y * blockSize + row + yOffset;
                int64_t pX = x * blockSize + col + xOffset;

                if(pX >= maxWidth || pY >= maxHeight) return false;
                if(pX < 0 || pY < 0) return false;
                
                block.at<uchar>(row, col) = frame.at<uchar>(pY, pX);
            }
        }

        return true;
    }

    static inline uint64_t CmpBlocks(const cv::Mat& block1, const cv::Mat& block2)
    {
        uint64_t diff = 0;

        for (uint64_t row = 0; row < (uint64_t)block1.rows; row++)
        {
            for (uint64_t col = 0; col < (uint64_t)block1.cols; col++)
            {
                int16_t p1 = block1.at<uchar>(row, col);
                int16_t p2 = block2.at<uchar>(row, col);
                
                diff += std::abs(p2 - p1);
            }
        }

        return diff;
    }

    static inline void ComputeResiduals(const cv::Mat& block1, const cv::Mat& block2, Result& result)
    {
        uint64_t diff = 0;

        for (uint64_t row = 0; row < (uint64_t)block1.rows; row++)
        {
            for (uint64_t col = 0; col < (uint64_t)block1.cols; col++)
            {
                int16_t p1 = block1.at<uchar>(row, col);
                int16_t p2 = block2.at<uchar>(row, col);

                diff += std::abs(p2 - p1);
                
                result.residuals.push_back(p2 - p1);
            }
        }

        result.totalDiff += diff;
    }
};