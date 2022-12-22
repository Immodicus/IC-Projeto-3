cmake_minimum_required(VERSION 3.2)
project( IC-Project-3 )

#SET (CMAKE_BUILD_TYPE "Release")
SET (CMAKE_BUILD_TYPE "Debug")

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17 -march=native")
SET (CMAKE_CXX_FLAGS_RELEASE "-Ofast")
SET (CMAKE_CXX_FLAGS_DEBUG "-g3 -fsanitize=address")
#SET (CMAKE_CXX_FLAGS_DEBUG "-g3 -pg")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( video_intra video_intra.cpp )
target_link_libraries( video_intra ${OpenCV_LIBS} )

add_executable( video_cmp video_cmp.cpp )
target_link_libraries( video_cmp ${OpenCV_LIBS} )

add_executable( video_codec video_codec.cpp )
target_link_libraries( video_codec ${OpenCV_LIBS})

add_executable( video_codec_dct video_codec_dct.cpp )
target_link_libraries( video_codec_dct ${OpenCV_LIBS})