cmake_minimum_required(VERSION 2.8)

project(demo)

include_directories(${OpenCV_INCLUDE} ../SparseStereo)
link_directories(${OpenCV_LIB})

add_executable(demo main.cpp)
target_link_libraries(demo SparseStereo opencv_core243 opencv_highgui243 opencv_features2d243)
