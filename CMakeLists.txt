cmake_minimum_required(VERSION 2.8)

project(SparseStereo)

set(OpenCV_INCLUDE c:/opencv/build/include)
set(OpenCV_LIB c:/opencv/build/x86/vc10/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(SparseStereo)
add_subdirectory(demo)
