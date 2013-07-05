cmake_minimum_required(VERSION 2.8)

project(demo)

include_directories(${OpenCV_INCLUDE} ../SparseStereo)
link_directories(${OpenCV_LIB})

add_executable(demo main.cpp)
target_link_libraries(demo optimized SparseStereo optimized opencv_core243 optimized opencv_highgui243 optimized opencv_features2d243)

target_link_libraries(demo debug SparseStereo debug opencv_core243d debug opencv_highgui243d debug opencv_features2d243d)
