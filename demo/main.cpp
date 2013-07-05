#include <iostream>
#include <string>
#include <vector>
#include "SparseStereo.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;

static void help( char** argv )
{
  std::cout<<"\nUsage: "<<argv[0]<<"[path/to/image1] [path/to/image2] [Max Hamming Dist] [Max Disparity] [Epipolar Range]\n"<< std::endl;
}

int main( int argc, char** argv ) 
{
  if( argc != 6 ) {
    help(argv);
    return -1;
  }

  // Load images
  Mat imgL = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  if( !imgL.data ) {
    std::cout<< " --(!) Error reading image " << argv[1] << std::endl;
    return -1;
  }

  Mat imgR = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  if( !imgR.data ) {
    std::cout << " --(!) Error reading image " << argv[2] << std::endl;
    return -1;
  }

  int maxDist = atoi(argv[3]);
  int maxDisparity = atoi(argv[4]);
  int epiRange = atoi(argv[5]);

  std::vector<cv::KeyPoint> keypointsL, keypointsR;
  std::vector<cv::DMatch> matches;
  SparseStereo census(imgL.step, maxDist, maxDisparity, epiRange);

  double t = (double)getTickCount();
  cv::FAST(imgL, keypointsL, 15, true);
  cv::FAST( imgR, keypointsR, 15, false);
  t = ((double)getTickCount() - t)/getTickFrequency();
  std::cout << "detection time [s]: " << t/1.0 << std::endl;

  // match
  t = (double)getTickCount();
  SparseStereo::TransformData transfmData1(imgL.rows, imgL.cols);
  SparseStereo::TransformData transfmData2(imgL.rows, imgL.cols);
  //Value of 0x80000000 indicates that a descriptor has not yet been calculated for the given pixel
  transfmData1.transfmImg = Mat_32(imgL.rows, imgL.cols, 0x80000000);
  transfmData2.transfmImg = Mat_32(imgL.rows, imgL.cols, 0x80000000);

  census.extractSparse(imgL, keypointsL, transfmData1);
  census.extractSparse(imgR, keypointsR, transfmData2);
  census.match(transfmData1, transfmData2, matches);
  t = ((double)getTickCount() - t)/getTickFrequency();
  std::cout << "matching time [s]: " << t << std::endl;

  std::cout << "Number of matches: "<<matches.size()<<std::endl;

  // Draw matches
  Mat imgMatch;
  std::vector<char> mask;
  drawMatches(imgL, keypointsL, imgR, keypointsR, matches, imgMatch, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, 2);

  namedWindow("matches", CV_WINDOW_KEEPRATIO);
  imshow("matches", imgMatch);
  waitKey(0);
}
