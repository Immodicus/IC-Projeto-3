#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "YUV4MPEG2.h"
#include "IntraEncoding.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define VERBOSE(txt) if(verbose) std::cout << txt << '\n';

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "Usage: video_codec [-d] originalVideo compressedFile\n";
        return EXIT_FAILURE;
    }
    
    bool verbose = true;
    bool encode = true;

    for(int n = 1; n < argc; n++)
	{
        if(std::string(argv[n]) == "-d") 
        {
            encode = false;
            break;
		}
    }

    if(encode)
    {
        YUV4MPEG2 videoFile(argv[argc - 2]);

        cv::Mat Y;
        cv::Mat Cb;
        cv::Mat Cr;

        YUV4MPEG2::YUV4MPEG2Description desc = videoFile.Description();

        VERBOSE("Width: " << desc.width);
        VERBOSE("Height: " << desc.height);

        BitStream encoded(argv[argc - 1], "w+");

        assert(encoded.Write(desc));

        uint64_t frameCount = 0;
        while(videoFile.ReadFrame(Y, Cb, Cr) != EOF)
        {
            // cv::imwrite("Y.bmp", Y);
            // cv::imwrite("Cb.bmp", Cb);
            // cv::imwrite("Cr.bmp", Cr);

            IntraEncoding::LumaEncode(Y, encoded);
            IntraEncoding::ChromaEncode(Cb, encoded);
            IntraEncoding::ChromaEncode(Cr, encoded);

            frameCount++;
        }

        VERBOSE("Encoded " << frameCount << " frames");
    }
    else
    {
        BitStream encoded(argv[argc - 2], "r");

        YUV4MPEG2::YUV4MPEG2Description desc;
        assert(encoded.Read(desc));

        YUV4MPEG2 videoFile(argv[argc - 1], desc);

        VERBOSE("Width: " << desc.width);
        VERBOSE("Height: " << desc.height);

        uint64_t frameCount = 0;
        cv::Mat Y;
        while(!(Y = IntraEncoding::LumaDecode(encoded, desc)).empty())
        {         
            cv::Mat Cb = IntraEncoding::ChromaDecode(encoded, desc);
            cv::Mat Cr = IntraEncoding::ChromaDecode(encoded, desc);

            // cv::imwrite("Y2.bmp", Y);
            // cv::imwrite("Cb2.bmp", Cb);
            // cv::imwrite("Cr2.bmp", Cr);

            videoFile.WriteFrame(Y, Cb, Cr);

            frameCount++;
        }

        VERBOSE("Decoded " << frameCount << " frames");
    }

    return EXIT_SUCCESS;
}