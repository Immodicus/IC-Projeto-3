#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "YUV4MPEG2.h"
#include "DCTIntraEncoding.h"
#include "MotionCompensation.h"

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
        std::cerr << "     -i : Only use intra-frame encoding \n";
        std::cerr << "     -kf : Key Frame Interval (in frames)\n";
        std::cerr << "     -bs : Block size in pixel (square matrix) to be used for motion compensation\n";
        std::cerr << "     -sa : Search area in pixels to be used for motion compensation\n";
        return EXIT_FAILURE;
    }

    bool verbose = false;
    bool encode = true;
    bool intra = false;

    uint64_t blockSize = 16;
    uchar searchArea = 8;
    int32_t keyFrameInterval = 16;
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

        if(std::string(argv[n]) == "-i") 
        {
            intra = true;
		}

        if(std::string(argv[n]) == "-kf") 
        {
            keyFrameInterval = atoi(argv[n+1]);
		}

        if(std::string(argv[n]) == "-bs") 
        {
            blockSize = atoi(argv[n+1]);
		}

        if(std::string(argv[n]) == "-sa") 
        {
            searchArea = atoi(argv[n+1]);
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

        if(!intra && (desc.width % blockSize != 0 || desc.height % blockSize != 0 || 
            (desc.width / 2) % blockSize != 0 || (desc.height / 2) % blockSize))
        {
            std::cerr << "\nVideo width, width/2, height and height/2 must be evenly divisible by blockSize. Exiting.\n";
            return EXIT_FAILURE;
        }

        cv::Mat Y(desc.height, desc.width, CV_8UC1);
        cv::Mat Cb(desc.height / 2, desc.width / 2, CV_8UC1);
        cv::Mat Cr(desc.height / 2, desc.width / 2, CV_8UC1);

        cv::Mat prevY;
        cv::Mat prevCb;
        cv::Mat prevCr;

        VERBOSE("Video is " << desc.width << " x " << desc.height);
        VERBOSE("Quality is " << quality);

        if(intra)
        {
            VERBOSE("Using exclusively intra-encoding");
        }
        else
        {
            VERBOSE("Using intra-frame and inter-frame encoding");
            VERBOSE("Motion Compensation blockSize is " << blockSize << " x " << blockSize);
            VERBOSE("Motion Compensation search area is " << (int)searchArea);
        }

        assert(encoded.Write(desc));
        assert(encoded.Write(quality));
        assert(encoded.Write(videoFile.GetFrameCount()));
        assert(encoded.Write(intra));
        assert(encoded.Write(blockSize));
        assert(encoded.Write(searchArea));
        assert(encoded.Write(keyFrameInterval));
        
        VERBOSE("Encoding...");

        while(videoFile.ReadFrame(Y, Cb, Cr) != EOF)
        {  
            auto rY = DCTIntraEncoding::Encode(Y, quality, DCTIntraEncoding::Luma);
            auto rCb = DCTIntraEncoding::Encode(Cb, quality, DCTIntraEncoding::Chroma);
            auto rCr = DCTIntraEncoding::Encode(Cr, quality, DCTIntraEncoding::Chroma);

            if(!intra)
            {
                DCTIntraEncoding::Decode(*rY.get(), Y, quality, DCTIntraEncoding::Luma);
                DCTIntraEncoding::Decode(*rCb.get(), Cb, quality, DCTIntraEncoding::Chroma);
                DCTIntraEncoding::Decode(*rCr.get(), Cr, quality, DCTIntraEncoding::Chroma);

                if(frameCount > 0 && frameCount % keyFrameInterval == 0)
                {
                    MotionCompensation::Result pr1 = MotionCompensation::Encode(prevY, Y, blockSize, searchArea);
                    MotionCompensation::Result pr2 = MotionCompensation::Encode(prevCb, Cb, blockSize, searchArea);
                    MotionCompensation::Result pr3 = MotionCompensation::Encode(prevCr, Cr, blockSize, searchArea);

                    MotionCompensation::Write(pr1, encoded);
                    MotionCompensation::Write(pr2, encoded);
                    MotionCompensation::Write(pr3, encoded);
                }
                else
                {
                    DCTIntraEncoding::Write(rY.get(), encoded);
                    DCTIntraEncoding::Write(rCb.get(), encoded);
                    DCTIntraEncoding::Write(rCr.get(), encoded);
                }
            }
            else
            {
                DCTIntraEncoding::Write(rY.get(), encoded);
                DCTIntraEncoding::Write(rCb.get(), encoded);
                DCTIntraEncoding::Write(rCr.get(), encoded);
            }

            Y.copyTo(prevY);
            Cb.copyTo(prevCb);
            Cr.copyTo(prevCr);

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
        assert(encoded.Read(intra));
        assert(encoded.Read(blockSize));
        assert(encoded.Read(searchArea));
        assert(encoded.Read(keyFrameInterval));

        YUV4MPEG2 videoFile(argv[argc - 1], desc);

        cv::Mat Y(desc.height, desc.width, CV_8UC1);
        cv::Mat Cb(desc.height / 2, desc.width / 2, CV_8UC1);
        cv::Mat Cr(desc.height / 2, desc.width / 2, CV_8UC1);

        cv::Mat prevY;
        cv::Mat prevCb;
        cv::Mat prevCr;

        VERBOSE("Video is " << desc.width << " x " << desc.height);
        VERBOSE("Quality is " << quality);
        VERBOSE("Decoding...");

        for(uint64_t f = 0; f < frameCount; f++)
        {                     
            if(f > 0 && f % keyFrameInterval == 0 && !intra)
            {
                MotionCompensation::Decode(encoded, prevY, Y, blockSize);
                MotionCompensation::Decode(encoded, prevCb, Cb, blockSize);
                MotionCompensation::Decode(encoded, prevCr, Cr, blockSize);
            }
            else
            {
                DCTIntraEncoding::Decode(Y, quality, DCTIntraEncoding::Luma, encoded);
                DCTIntraEncoding::Decode(Cb, quality, DCTIntraEncoding::Chroma, encoded);
                DCTIntraEncoding::Decode(Cr, quality, DCTIntraEncoding::Chroma, encoded);
            }

            Y.copyTo(prevY);
            Cb.copyTo(prevCb);
            Cr.copyTo(prevCr);

            videoFile.WriteFrame(Y, Cb, Cr);
        }

        VERBOSE("Decoded " << frameCount << " frames");
    }
}