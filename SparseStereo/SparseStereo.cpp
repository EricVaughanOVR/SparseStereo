#include "SparseStereo.hpp"
#include <iostream>

SparseStereo::SparseStereo(const cv::Mat::MStep& step, const int filterDist,
    const int validDisparity, const int epipolarRange)
  : m_filterDist(filterDist),
    m_maxDisparity(validDisparity),
    m_epipolarRange(epipolarRange)
{
  m_sampleOffsets_9x9.resize(16);
  m_sampleOffsets_9x9[0] = -4 * step[0];
  m_sampleOffsets_9x9[1] = -3 * step[0] + -2 * step[1];
  m_sampleOffsets_9x9[2] = -3 * step[0] + 2 * step[1];
  m_sampleOffsets_9x9[3] = -2 * step[0] + -3 * step[1];
  m_sampleOffsets_9x9[4] = -2 * step[0] + -2 * step[1];
  m_sampleOffsets_9x9[5] = -2 * step[0] + 2 * step[1];
  m_sampleOffsets_9x9[6] = -2 * step[0] + 4 * step[1];
  m_sampleOffsets_9x9[7] = -4 * step[1];
  m_sampleOffsets_9x9[8] = 3 * step[1];
  m_sampleOffsets_9x9[9] = 2 * step[0] + -3 * step[1];
  m_sampleOffsets_9x9[10] = 2 * step[0] + 4 * step[1];
  m_sampleOffsets_9x9[11] = 3 * step[0] + -3 * step[1];
  m_sampleOffsets_9x9[12] = 3 * step[0];
  m_sampleOffsets_9x9[13] = 3 * step[0] + 3 * step[1];
  m_sampleOffsets_9x9[14] = 4 * step[0] + -2 * step[1];
  m_sampleOffsets_9x9[15] = 4 * step[0] + 2 * step[1];
}

SparseStereo::~SparseStereo()
{
}

void SparseStereo::extractSparse(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, 
  TransformData& result)
{
  //copy to local var, to improve memory efficiency
  std::vector<int> sampleOffsets(m_sampleOffsets_9x9);
  
  if(result.transfmImg.cols != img.cols || result.transfmImg.rows != img.rows)
    return;

  std::vector<KpLite> _kps;

  SubVector initVal;
  initVal.isEmpty = true;
  result.rowBuckets.resize(img.rows, initVal);
  _kps.reserve(keypoints.size());

  int newIdx = 0;
  //Reject keypoints that are too close to the edge of the image
  for(size_t i = 0; i < keypoints.size(); ++i)
  {
    cv::Point2i pixelLoc(static_cast<int>(keypoints[i].pt.x), static_cast<int>(keypoints[i].pt.y));
    
    if(pixelLoc.x < 8 || pixelLoc.x > (img.cols - 8)//TODO make dynamic w/ window size, not hard-coded
      || pixelLoc.y < 8 || pixelLoc.y > (img.rows - 8))//Account for outer limit of correlation window
    {
      continue;
    }
        
    uint32_t* pixelResultLoc = &result.transfmImg.unsafeAt(pixelLoc.x, pixelLoc.y);
    uchar* pixelSrcLoc = img.data + pixelLoc.y * img.step[0] + pixelLoc.x * img.step[1];

    for(size_t k = 0; k < sampleOffsets.size(); ++k)
    {
      //Check if the value at this location has already been calculated
      if(!(*(pixelResultLoc + sampleOffsets[k]) & 0x80000000))
        continue;

      *(pixelResultLoc + sampleOffsets[k]) = transform9x9(img, pixelSrcLoc + sampleOffsets[k]);
    }
    
    _kps.push_back(KpLite(pixelLoc.x, pixelLoc.y, newIdx, i));
    _kps.back().idx = newIdx;//use class_id to store the keypoint's idx
    size_t newKpY = static_cast<size_t>(_kps.back().y);

    if(result.rowBuckets[newKpY].isEmpty)//First keypoint in its row
    {
      result.rowBuckets[newKpY].isEmpty = false;
      result.rowBuckets[newKpY].start = _kps.begin() + newIdx;
      result.rowBuckets[newKpY].stop = _kps.begin() + newIdx;
    }
    else if(result.rowBuckets[newKpY].stop->x < (_kps.begin() + newIdx)->x)//Last point(so far) in its row
      result.rowBuckets[newKpY].stop = _kps.begin() + newIdx;

    ++newIdx;
  }
  result.kps.swap(_kps);
}

void SparseStereo::match(TransformData& tfmData1,
    TransformData& tfmData2, std::vector<cv::DMatch>& matches)
{
  //check that left and right images have the same dimensions
  if(tfmData1.transfmImg.rows != tfmData2.transfmImg.rows || 
    tfmData1.transfmImg.cols != tfmData2.transfmImg.cols)
  {
    //TODO throw an exception here.
    matches.clear();
    return;
  }

  matches.reserve(tfmData1.kps.size());

  //Find best match in img2 for each feature in img1, by Hamming Distance
  for(size_t i = 0; i < tfmData1.kps.size(); ++i)//For each left feature
  {
    //Create submat to give to findBestMatch.  Epipolar + Disparity constraints.
    int x, y, width, height;

    int yMax;//Bottom of epipolarRegion

    x = tfmData1.kps[i].x - m_maxDisparity;
    y = tfmData1.kps[i].y - m_epipolarRange / 2;
    yMax = y + m_epipolarRange / 2 + 1;

    if(x < 8)//TODO make dynamic with (correlation window size / 2 + transform window size / 2)
      x = 8;
    if(y < 8)
      y = 8;
    if(yMax > tfmData1.transfmImg.rows - 8)
      yMax = tfmData1.transfmImg.rows - 8;

    width = tfmData1.kps[i].x - x;
    height = yMax - y;

    Mat_32 searchRegion(tfmData2.transfmImg, x, y, width, height);
    
    std::vector<SubVector> potentialMatches;
    //loads potential matches with the keypoints from the area of interest
    SparseStereo::loadDescriptors(tfmData2.rowBuckets, searchRegion, potentialMatches);

    //Find best match in the submat
    if(potentialMatches.empty())
      continue;
    cv::DMatch match = SparseStereo::matchSparse(tfmData1.transfmImg, tfmData2.transfmImg, 
      tfmData1.kps[i], potentialMatches);

    if(match.distance >= 0 && match.distance < m_filterDist)
      matches.push_back(match);
  }//end for loop
}

uint16_t SparseStereo::transform9x9(const cv::Mat& img, uchar* pixelLoc)
{
  uint16_t result = 0;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[0]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[1]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[2]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[3]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[4]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[5]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[6]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[7]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[8]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[9]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[10]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[11]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[12]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[13]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[14]) > *pixelLoc)) << 1;
  result = (result + (*(pixelLoc + m_sampleOffsets_9x9[15]) > *pixelLoc));

  return result;
}

void SparseStereo::loadDescriptors(const std::vector<SubVector>& descriptors, 
    Mat_32& resultImg, std::vector<SubVector>& resultVector)
{
  /* NOTE: If no subpixel interpolation has been done to the keypoints/descriptors, then they should remain 
     sorted, as they come from FAST.  Otherwise this will fail at boundaries
  */
  resultVector.clear();

  int descRefsIdx = 0;
  int lastRow = resultImg.rows + resultImg.offset.y - 1;

  for(int i = resultImg.offset.y; i <= lastRow; ++i)//For each row(element) of descriptors
  {
    if((descriptors.begin() + i)->isEmpty)//make sure that this row has descriptors in it
      continue;
    std::vector<KpLite>::iterator k = (descriptors.begin() + i)->start;

    do
    {
      if(resultImg.isWithin(k->x, k->y))
      {
        if(resultVector.size() < (static_cast<size_t>(descRefsIdx + 1)))
        {
          SubVector tmp;
          tmp.isEmpty = false;
          tmp.start = k;
          tmp.stop = k;
          resultVector.push_back(tmp);
        }
        else if(resultVector.back().stop->x < k->x)
          resultVector.back().stop = k;
      }
    }while(k++ != (descriptors.begin() + i)->stop);//TRICKY post-script ++ is necessary

    ++descRefsIdx;
  }//end for loop (rows)
}

uint32_t SparseStereo::computeSHD(Mat_32& transformedL, Mat_32& transformedR, 
    const KpLite descL, const KpLite descR)
{
  /* SSE VERSION (TODO - with memory reorganization)
    descriptors in contiguous memory

    Load 8x LDesc into a register  mm_load_128si(*LDesc)
    Load 8x RDesc into a register  mm_load_128si(*RDesc)

    XOR the registers
    PSHUFB-lookup the 4-bit table
    Add results
    return value
    */

  uint32_t result = 0;

  uint32_t* pixelOffsetL = &transformedL.unsafeAt(descL.x, descL.y);
  uint32_t* pixelOffsetR = &transformedR.unsafeAt(descR.x, descR.y);
  for(size_t i = 0; i < m_sampleOffsets_9x9.size(); ++i)
  {
    result += SparseStereo::calcHammingDist(*(pixelOffsetL + m_sampleOffsets_9x9[i]),
      *(pixelOffsetR + m_sampleOffsets_9x9[i]));
  }

  return result;
}

cv::DMatch SparseStereo::matchSparse(Mat_32& imgToMatch, Mat_32& imgPotMatches,
    const KpLite& toMatch, const std::vector<SubVector>& potMatches)
{
  //TODO ensure only unique matches
  //queryIdx is the right img, train idx is the left img
  int trainIdx = toMatch.kpIdx;
  int queryIdx = potMatches[0].start->kpIdx;//Should always be overwritten
  int bestDist = 2049;//greater than any possible distance, so it will be reset to a real distance immediately
  int dist = bestDist;

  std::for_each(potMatches.begin(), potMatches.end(), 
    [&](SubVector row)
    {
      for(std::vector<KpLite>::iterator iter = row.start; iter != row.stop; ++iter)
      {
        //Then calculate the Hamming Distance between it and the toMatch
        dist = SparseStereo::computeSHD(imgToMatch, imgPotMatches, toMatch, *iter);
  
        if(dist < bestDist)
        {
          bestDist = static_cast<int>(dist);
          queryIdx = iter->kpIdx;
          if(bestDist == 0)
            break;//cannot improve beyond a "perfect" match, and there is no uniqueness check, skip the rest
        }
      }//end for, iter
    });//end for_each, row

  cv::DMatch match(trainIdx, queryIdx, static_cast<float>(bestDist));
  return match;
}
