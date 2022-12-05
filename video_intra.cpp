#include "YUV4MPEG2.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char ** argv)
{
    if(argc < 2) return EXIT_FAILURE;
    
    std::string filename = argv[argc -1];
    VideoCapture capture(filename);
    Mat frame;

    std::cout << filename << std::endl;

    if( !capture.isOpened() )
    {
        throw std::runtime_error("Error when reading " + filename + "\n");
    }

    return EXIT_SUCCESS;
}