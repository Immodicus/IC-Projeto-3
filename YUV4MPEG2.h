#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class YUV4MPEG2
{
public:
    struct YUV4MPEG2Description
    {
        int32_t width;
        int32_t height;
        char interlacing;
        int32_t f1;
        int32_t f2;
        int32_t a1;
        int32_t a2;

        std::string ToString() const
        {
            int size = std::snprintf(nullptr, 0, "YUV4MPEG2 W%d H%d F%d:%d I%c A%d:%d\n", width, height, f1, f2, interlacing, a1, a2);
            size++; // account for null terminator

            std::unique_ptr<char[]> str = std::unique_ptr<char[]>(new char[size]);
            std::snprintf(str.get(), size, "YUV4MPEG2 W%d H%d F%d:%d I%c A%d:%d\n", width, height, f1, f2, interlacing, a1, a2);

            return str.get();
        }

        friend std::ostream & operator<<(std::ostream& str, const YUV4MPEG2Description& desc) 
        { 
            return str << desc.ToString();
        }
    };

    enum Mode {Read, Write};

    YUV4MPEG2(const char* filename)
    {
        this->filename = filename;
        this->mode = Read;

        InitRead();
    }

    YUV4MPEG2(const char* filename, YUV4MPEG2Description& description)
    {
        this->filename = filename;
        this->description = description;
        this->mode = Write;

        InitWrite();
    }

    ~YUV4MPEG2()
    {
        if(filePtr) fclose(filePtr);
    }

    YUV4MPEG2Description Description()
    {
        return description;
    }

    int32_t ReadFrame(cv::Mat& Y, cv::Mat& Cb, cv::Mat& Cr)
    {
        assert(mode == Read);

        // Set Mats to correct size and type
        Y = cv::Mat(description.height, description.width, CV_8UC1);
        Cb = cv::Mat(description.height / 2, description.width / 2, CV_8UC1);
        Cr = cv::Mat(description.height / 2, description.width / 2, CV_8UC1);
        
        // Read Frame Header and ignore extra  parameters
        std::unique_ptr<char[]> frameHeader = std::unique_ptr<char[]>(new char[6]());

        fscanf(filePtr, "%5s", frameHeader.get());
        if(strcmp(frameHeader.get(), "FRAME")) return EOF;

        char c;
        while(c != 0x0A) fread(&c, sizeof(char), 1, filePtr);

        // Read the Frame
        for(uint64_t row = 0; row < (uint64_t)description.height; row++)
        {
            for(uint64_t col = 0; col < (uint64_t)description.width; col++)
            {
                assert(fread(&c, sizeof(char), 1, filePtr) == 1);
                Y.at<uchar>(row, col) = c;
            }
        }

        for(uint64_t row = 0; row < (uint64_t)description.height / 2; row++)
        {
            for(uint64_t col = 0; col < (uint64_t)description.width / 2; col++)
            {
                assert(fread(&c, sizeof(char), 1, filePtr) == 1);
                Cb.at<uchar>(row, col) = c;
            }
        }

        for(uint64_t row = 0; row < (uint64_t)description.height / 2; row++)
        {
            for(uint64_t col = 0; col < (uint64_t)description.width / 2; col++)
            {
                assert(fread(&c, sizeof(char), 1, filePtr) == 1);
                Cr.at<uchar>(row, col) = c;
            }
        }

        return 0;
    }

    void WriteFrame(const cv::Mat& Y, const cv::Mat& Cb, const cv::Mat& Cr)
    {
        assert(mode == Write);

        // Check Mats size and type
        assert(Y.type() == CV_8UC1 && Cb.type() == CV_8UC1 && Cr.type() == CV_8UC1);
        assert(Y.rows == description.height && Cb.rows == description.height / 2 && Cr.rows == description.height / 2);
        assert(Y.cols == description.width && Cb.cols == description.width / 2 && Cr.cols == description.width / 2);
        
        //Write Frame header
        fwrite("FRAME", sizeof(char), 5, filePtr);
        char c = 0x0A;
        fwrite(&c, sizeof(char), 1, filePtr);

        // Write the Frame
        uchar p;
        for(uint64_t row = 0; row < (uint64_t)description.height; row++)
        {
            for(uint64_t col = 0; col < (uint64_t)description.width; col++)
            {
                p = Y.at<uchar>(row, col);
                fwrite(&p, sizeof(uchar), 1, filePtr);
            }
        }

        for(uint64_t row = 0; row < (uint64_t)description.height / 2; row++)
        {
            for(uint64_t col = 0; col < (uint64_t)description.width / 2; col++)
            {
                p = Cb.at<uchar>(row, col);
                fwrite(&p, sizeof(uchar), 1, filePtr);
            }
        }

        for(uint64_t row = 0; row < (uint64_t)description.height / 2; row++)
        {
            for(uint64_t col = 0; col < (uint64_t)description.width / 2; col++)
            {
                p = Cr.at<uchar>(row, col);
                fwrite(&p, sizeof(uchar), 1, filePtr);
            }
        }
    }

private:
    Mode mode;
    const char* filename;
    FILE* filePtr;
    YUV4MPEG2Description description;

    void InitRead()
    {
        filePtr = fopen(filename, "r");
        if(!filePtr) throw std::runtime_error(std::string("Failed to open file: ") + filename);

        std::unique_ptr<char[]> header = std::unique_ptr<char[]>(new char[11]());

        fread(header.get(), sizeof(char), 10, filePtr);

        if(strcmp(header.get(), "YUV4MPEG2 ")) throw std::runtime_error(filename + std::string(" isnt a YUV4MPEG2 file"));

        assert(fscanf(filePtr, "W%d H%d F%d:%d I%c A%d:%d", &description.width, &description.height, 
                                                            &description.f1, &description.f2, &description.interlacing, 
                                                            &description.a1, &description.a2) == 7);

        std::unique_ptr<char[]> cBuff = std::unique_ptr<char[]>(new char[11]());

        int color = fscanf(filePtr, "C%7s", cBuff.get());

        if(!color) std::cout << "File doesn't appear to specify a color mode. Assuming YCbCr 4:2:0\n";
    }

    void InitWrite()
    {
        filePtr = fopen(filename, "w+");
        if(!filePtr) throw std::runtime_error(std::string("Failed to create file: ") + filename);

        std::string header = description.ToString();

        fwrite(header.c_str(), sizeof(char), header.length(), filePtr);
    }
};