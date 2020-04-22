#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance

    DataFrame(cv::Mat img, std::vector<cv::KeyPoint> keypts, cv::Mat des) : cameraImg(img),
                                                                            keypoints(keypts),
                                                                            descriptors(des)
        {
        }

    cv::Mat cameraImg; // camera image
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};
struct Params
{
    std::string detectorType;
    std::string descriptorType;
    std::string matcherType;
    std::string selectorType;
    bool bFocusOnVehicle = true;
    int normType;
};


#endif /* dataStructures_h */
