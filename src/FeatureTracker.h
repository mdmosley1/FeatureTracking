#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <vector>

#include "dataStructures.h" // DataFrame, Params

class FeatureTracker
{
public:
    FeatureTracker(const Params& params);
    
    std::vector<cv::DMatch> TrackFeatures(const DataFrame& newFrame);

private:
    void AddToRingBuffer(const DataFrame& frame);
    void VisualizeMatches(std::vector<cv::DMatch> matches);
    std::vector<cv::DMatch> matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                                             std::vector<cv::KeyPoint> &kPtsRef,
                                             cv::Mat &descSource,
                                             cv::Mat &descRef);
    
    int dataBufferSize_m = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::vector<DataFrame> dataBuffer_m; // list of data frames which are held in memory at the same time

    Params params_m;
};



#endif /* FEATURETRACKER_H */
