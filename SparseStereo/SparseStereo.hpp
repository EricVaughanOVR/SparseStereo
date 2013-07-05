#ifndef SPARSE_STEREO_HPP
#define SPARSE_STEREO_HPP

#include <vector>
#include <stdint.h>
#include <opencv2/features2d/features2d.hpp>
#include "Mat_32.hpp"

class SparseStereo
{
public:
  SparseStereo(const cv::Mat::MStep& step, const int filterDist = 20,
    const int validDisparity = 30, const int epipolarRange = 1);

  ~SparseStereo();

  struct KpLite
  {
    int x, y, idx, kpIdx;
    KpLite(int _x, int _y, int _idx, int _kpIdx)
      : x(_x),
        y(_y),
        idx(_idx),
        kpIdx(_kpIdx)//Index of the original list of keypoints, for using with drawMatches
    {};
  };

  struct SubVector
  {
    bool isEmpty;
    std::vector<KpLite>::iterator start;
    std::vector<KpLite>::iterator stop;
  };

  struct TransformData
  {
    std::vector<SubVector> rowBuckets;
    Mat_32 transfmImg;
    std::vector<KpLite> kps;

    //Value of 0x80000000 indicates that a descriptor has not yet been calculated for the given pixel
    TransformData(int rows, int cols)
      : transfmImg(rows, cols, 0x80000000)
    {};
  };

  /** Extracts 2-byte descriptor for each pixel in a sparse neighborhood about each keypoint
    * Adds the descriptors to resultImg, so that the extraction won't need to be repeated
    * for those locations
    * Only looks in the region of img contained by resultImg
    * resultImg must be the same size as img
    */
  void extractSparse(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, 
    TransformData& result);

  /** Finds best match in img2 for each feature in img1.
    */
  void match(TransformData& tfmData1, TransformData& tfmData2, 
    std::vector<cv::DMatch>& matches);

private:
  /* The neighborhood must be guaranteed to exist around the given 
   * pixel location, or these functions will crash
   */
  inline uint16_t transform9x9(const cv::Mat& img, const cv::Point2i pixelLoc)
  {
    uchar* pixelOffset = img.data + pixelLoc.y * img.step[0] + pixelLoc.x * img.step[1];
    return transform9x9(img, pixelOffset);
  }

  /* Computes the sparse 16-neighbor Census Transform about the given pixel.  Equivalent 
   * to a 9x9 dense transform about the given pixel.
   */
  uint16_t transform9x9(const cv::Mat& img, uchar* pixelLoc);
  
  /** Only looks in the region of img contained by resultImg
    * descriptor locations are always in terms of the parent img
    */
  void loadDescriptors(const std::vector<SubVector>& descriptors, 
    Mat_32& resultImg, std::vector<SubVector>& resultVector);

  /** Finds the Hamming Distance between two uint32_t
    */
  inline uint32_t calcHammingDist(const uint32_t _1, const uint32_t _2)
  {
    //TODO add a SIMD implementation, with memory reorganization

    //XOR the desc
    uint32_t newBitStr = _1 ^ _2;

    //From Bit Twiddling Hacks/  TODO SSSE3 impl
    uint32_t result; // store the total here
    static const int S[] = {1, 2, 4, 8, 16}; // Magic Binary Numbers
    static const int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};

    result = newBitStr - ((newBitStr >> 1) & B[0]);
    result = ((result >> S[1]) & B[1]) + (result & B[1]);
    result = ((result >> S[2]) + result) & B[2];
    result = ((result >> S[3]) + result) & B[3];
    result = ((result >> S[4]) + result) & B[4];

    return result;
  };

  /* Computes the sum of Hamming Distances between two sparse correlation windows
   * Assumes that the input Mat_32s were transformed with a compatible sparse-pattern.
   * i.e. - won't check validity of the pixels
   */
  uint32_t computeSHD(Mat_32& transformedL, Mat_32& transformedR, 
    const KpLite descL, const KpLite descR);

  /** Finds the best match for toMatch in the list of potentialMatches
    */
  cv::DMatch matchSparse(Mat_32& imgToMatch, Mat_32& imgPotMatches,
    const KpLite& toMatch, const std::vector<SubVector>& potMatches);

  int m_filterDist;

  int m_maxDisparity;
  int m_epipolarRange;

  std::vector<int> m_sampleOffsets_9x9;
};

#endif SPARSE_STEREO_HPP