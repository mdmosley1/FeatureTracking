#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

class KPDetector
{
public:
    virtual std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false) = 0;
    virtual ~KPDetector() {}
};

class DetectorShiTomasi : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorShiTomasi() {}
};

class DetectorHarris : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorHarris() {}
private:
    std::vector<cv::KeyPoint> GetKeypoints(const cv::Mat dst_norm) const;
    // compute detector parameters based on image size
    int blockSize_ = 2;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize_ = 3;
    double k_ = 0.04;
    int minResponse_ = 100; // minimum value for a corner in the 8bit scaled response matrix
    double maxOverlap_ = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
};

class DetectorFast : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorFast() {}
};

class DetectorBrisk : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorBrisk() {}
};

class DetectorOrb : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorOrb() {}
};

class DetectorAkaze : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorAkaze() {}
};

class DetectorSift : public KPDetector
{
public:
    std::vector<cv::KeyPoint> DetectKeypoints(const cv::Mat&, bool bVis = false);
    ~DetectorSift() {}
};

std::vector<cv::KeyPoint> detKeypointsShiTomasi(const cv::Mat &img, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis=false);

cv::Mat descKeypoints(std::vector<cv::KeyPoint> &keypoints,
                      const cv::Mat &img,
                      const cv::Ptr<cv::DescriptorExtractor>& _descriptor,
                      const Params& params);


#endif /* matching2D_hpp */
