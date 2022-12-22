#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "YUV4MPEG2.h"
#include "DCTIntraEncoding.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define VERBOSE(txt) if(verbose) std::cout << txt << '\n';

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "Usage: video_codec_dct [optional args] fileIn fileOut\n";
        std::cerr << "\nOptional arguments: \n";
        std::cerr << "     -v : Verbose Mode (default off) \n";
        std::cerr << "     -q : Encoding Quality (1 to 100) default 90 \n";
        std::cerr << "     -d : Decode (default encode) \n";
        return EXIT_FAILURE;
    }

    bool verbose = false;
    bool encode = true;

    int32_t quality = 90;
    uint64_t frameCount = 0;

    for(int n = 1; n < argc; n++)
	{
        if(std::string(argv[n]) == "-q") 
        {
            quality = atoi(argv[n+1]);
            if(quality < 1 || quality > 100) 
            {
                std::cerr << "Bad quality value. Must be in [1 100]\n";
                return EXIT_FAILURE;
            }
		}

        if(std::string(argv[n]) == "-d") 
        {
            encode = false;
		}

        if(std::string(argv[n]) == "-v") 
        {
            verbose = true;
		}
    }

    if(encode)
    {
        YUV4MPEG2 videoFile(argv[argc - 2]);
        YUV4MPEG2::YUV4MPEG2Description desc = videoFile.Description();

        BitStream encoded(argv[argc - 1], "w+");

        if(desc.width % 16 != 0 || desc.height % 16 != 0)
        {
            std::cerr << "\nVideo width and height must be evenly divisible by 16. Exiting.\n";
            return EXIT_FAILURE;
        }

        cv::Mat Y(desc.height, desc.width, CV_8UC1);
        cv::Mat Cb(desc.height / 2, desc.width / 2, CV_8UC1);
        cv::Mat Cr(desc.height / 2, desc.width / 2, CV_8UC1);

        VERBOSE("Video is " << desc.width << " x " << desc.height);
        VERBOSE("Quality is " << quality);

        assert(encoded.Write(desc));
        assert(encoded.Write(quality));
        assert(encoded.Write(videoFile.GetFrameCount()));
        
        VERBOSE("Encoding...");

        while(videoFile.ReadFrame(Y, Cb, Cr) != EOF)
        {  
            auto rY = DCTIntraEncoding::Encode(Y, quality, DCTIntraEncoding::Luma);
            auto rCb = DCTIntraEncoding::Encode(Cb, quality, DCTIntraEncoding::Chroma);
            auto rCr = DCTIntraEncoding::Encode(Cr, quality, DCTIntraEncoding::Chroma);

            DCTIntraEncoding::Write(rY.get(), encoded);
            DCTIntraEncoding::Write(rCb.get(), encoded);
            DCTIntraEncoding::Write(rCr.get(), encoded);

            frameCount++;
        }        

        VERBOSE("Encoded " << frameCount << " frames");
    }
    else
    {
        BitStream encoded(argv[argc - 2], "r");

        YUV4MPEG2::YUV4MPEG2Description desc;
        assert(encoded.Read(desc));
        assert(encoded.Read(quality));
        assert(encoded.Read(frameCount));

        YUV4MPEG2 videoFile(argv[argc - 1], desc);

        cv::Mat Y(desc.height, desc.width, CV_8UC1);
        cv::Mat Cb(desc.height / 2, desc.width / 2, CV_8UC1);
        cv::Mat Cr(desc.height / 2, desc.width / 2, CV_8UC1);

        VERBOSE("Video is " << desc.width << " x " << desc.height);
        VERBOSE("Quality is " << quality);
        VERBOSE("Decoding...");

        for(uint64_t f = 0; f < frameCount; f++)
        {                     
            DCTIntraEncoding::Decode(Y, quality, DCTIntraEncoding::Luma, encoded);
            DCTIntraEncoding::Decode(Cb, quality, DCTIntraEncoding::Chroma, encoded);
            DCTIntraEncoding::Decode(Cr, quality, DCTIntraEncoding::Chroma, encoded);

            videoFile.WriteFrame(Y, Cb, Cr);
        }

        VERBOSE("Decoded " << frameCount << " frames");
    }
}