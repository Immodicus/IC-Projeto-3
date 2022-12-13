#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "YUV4MPEG2.h"
#include "IntraEncoding.h"
#include "MotionCompensation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define VERBOSE(txt) if(verbose) std::cout << txt << '\n';

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "Usage: video_codec [optional args] fileIn fileOut\n";
        std::cerr << "\nOptional arguments: \n";
        std::cerr << "     -v : Verbose Mode (default off) \n";
        std::cerr << "     -d : Decode (default encode) \n";
        std::cerr << "     -i : Only use intra-frame coding \n"; // not implemented
        std::cerr << "     -bs : Block size in pixel (square matrix) to be used for motion compensation\n";
        std::cerr << "     -sa : Search area in pixels to be used for motion compensation\n";
        std::cerr << "     -kf : Key Frame Interval (in frames)\n";
        return EXIT_FAILURE;
    }
    
    bool verbose = true;
    bool encode = true;
    bool intra = false;

    uint64_t blockSize = 16;
    uchar searchArea = 8;
    uint64_t keyFrameInterval = 16;

    for(int n = 1; n < argc; n++)
	{
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

        if(std::string(argv[n]) == "-bs") 
        {
            blockSize = atoi(argv[n+1]);
		}

        if(std::string(argv[n]) == "-sa") 
        {
            searchArea = atoi(argv[n+1]);
		}

        if(std::string(argv[n]) == "-kf") 
        {
            keyFrameInterval = atoi(argv[n+1]);
		}
    }

    if(encode)
    {
        YUV4MPEG2 videoFile(argv[argc - 2]);

        cv::Mat Y;
        cv::Mat Cb;
        cv::Mat Cr;

        cv::Mat prevY;
        cv::Mat prevCb;
        cv::Mat prevCr;

        YUV4MPEG2::YUV4MPEG2Description desc = videoFile.Description();

        if( desc.width % blockSize != 0 || desc.height % blockSize != 0 || 
            (desc.width / 2) % blockSize != 0 || (desc.height / 2) % blockSize)
        {
            std::cerr << "\nVideo width, width/2, height and height/2 must be evenly divisible by blockSize. Exiting.\n";
            return EXIT_FAILURE;
        }

        VERBOSE("Width: " << desc.width);
        VERBOSE("Height: " << desc.height);

        BitStream encoded(argv[argc - 1], "w+");

        assert(encoded.Write(desc));
        assert(encoded.Write(blockSize));
        assert(encoded.Write(searchArea));

        videoFile.ReadFrame(Y, Cb, Cr);
        IntraEncoding::Result r1 = IntraEncoding::LumaEncode(Y);
        IntraEncoding::Result r2 = IntraEncoding::ChromaEncode(Cb);
        IntraEncoding::Result r3 = IntraEncoding::ChromaEncode(Cr);
        IntraEncoding::Write(encoded, r1);
        IntraEncoding::Write(encoded, r2);
        IntraEncoding::Write(encoded, r3);
        Y.copyTo(prevY);
        Cb.copyTo(prevCb);
        Cr.copyTo(prevCr);

        uint64_t frameCount = 1;
        while(videoFile.ReadFrame(Y, Cb, Cr) != EOF)
        {           
            IntraEncoding::Result r1 = IntraEncoding::LumaEncode(Y);
            IntraEncoding::Result r2 = IntraEncoding::ChromaEncode(Cb);
            IntraEncoding::Result r3 = IntraEncoding::ChromaEncode(Cr);

            MotionCompensation::Result pr1 = MotionCompensation::Encode(prevY, Y, blockSize, searchArea);
            MotionCompensation::Result pr2 = MotionCompensation::Encode(prevCb, Cb, blockSize, searchArea);
            MotionCompensation::Result pr3 = MotionCompensation::Encode(prevCr, Cr, blockSize, searchArea);

            int64_t interBits = INT64_MIN;
            int64_t intraBits = INT64_MIN;
            
            interBits = pr1.Estimate() + pr2.Estimate() + pr3.Estimate();
            intraBits = r1.Estimate() + r2.Estimate() + r3.Estimate();
            
            if(frameCount % keyFrameInterval == 0 || intraBits < interBits)
            {
                //std::cout << "Intra frame\n";
                encoded.WriteBit(true);
                IntraEncoding::Write(encoded, r1);
                IntraEncoding::Write(encoded, r2);
                IntraEncoding::Write(encoded, r3);
            }
            else
            {
                //std::cout << "Inter frame\n";
                encoded.WriteBit(false);
                MotionCompensation::Write(pr1, encoded);
                MotionCompensation::Write(pr2, encoded);
                MotionCompensation::Write(pr3, encoded);
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
        assert(encoded.Read(blockSize));
        assert(encoded.Read(searchArea));

        YUV4MPEG2 videoFile(argv[argc - 1], desc);

        VERBOSE("Width: " << desc.width);
        VERBOSE("Height: " << desc.height);

        uint64_t frameCount = 0;
        cv::Mat Y;
        cv::Mat Cb;
        cv::Mat Cr;

        cv::Mat prevY;
        cv::Mat prevCb;
        cv::Mat prevCr;

        Y = IntraEncoding::LumaDecode(encoded, desc);
        Cb = IntraEncoding::ChromaDecode(encoded, desc);
        Cr = IntraEncoding::ChromaDecode(encoded, desc);
        videoFile.WriteFrame(Y, Cb, Cr);
        Y.copyTo(prevY);
        Cb.copyTo(prevCb);
        Cr.copyTo(prevCr);

        bool intra = true;
        while(encoded.ReadBit(intra))
        {         
            if(intra)
            {
                Y = IntraEncoding::LumaDecode(encoded, desc);
                if(Y.empty()) break;
                Cb = IntraEncoding::ChromaDecode(encoded, desc);
                Cr = IntraEncoding::ChromaDecode(encoded, desc);
            }
            else
            {
                if(!MotionCompensation::Decode(encoded, prevY, Y, blockSize)) break; //because of byte padding ReadBit may succeed after last frame. this won't
                MotionCompensation::Decode(encoded, prevCb, Cb, blockSize);
                MotionCompensation::Decode(encoded, prevCr, Cr, blockSize);
            }

            // cv::imwrite("Y2.bmp", Y);
            // cv::imwrite("Cb2.bmp", Cb);
            // cv::imwrite("Cr2.bmp", Cr);

            videoFile.WriteFrame(Y, Cb, Cr);

            Y.copyTo(prevY);
            Cb.copyTo(prevCb);
            Cr.copyTo(prevCr);

            frameCount++;
        }

        VERBOSE("Decoded " << frameCount << " frames");
    }

    return EXIT_SUCCESS;
}