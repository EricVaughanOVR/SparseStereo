include_directories(${OpenCV_INCLUDE})
link_directories(${OpenCV_LIB})

add_library(SparseStereo Mat_32.hpp SparseStereo.cpp SparseStereo.hpp)

target_link_libraries(SparseStereo optimized opencv_core243)

target_link_libraries(SparseStereo debug opencv_core243d)
