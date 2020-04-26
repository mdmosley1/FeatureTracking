#include <memory> // unique_ptr
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "dataStructures.h" // DataFrame
#include "matching2D.hpp" // KPDetector

void AddToRingBuffer(const DataFrame& frame);
void VisualizeMatches(std::vector<cv::DMatch> matches);
void LimitKeyPoints(std::vector<cv::KeyPoint>& keypoints, const Params& p);

void LimitKeyPointsRect(std::vector<cv::KeyPoint>& keypoints);
void DetectAndTrackFeatures(const cv::Mat& imgGray,
                            const std::unique_ptr<KPDetector>& _detector,
                            const cv::Ptr<cv::DescriptorExtractor>& _descriptor,
                            const Params& params);
std::unique_ptr<KPDetector> CreateDetector(std::string _detectorType);
cv::Ptr<cv::DescriptorExtractor> CreateDescriptor(std::string _descriptorType);
Params LoadParamsFromFile(std::string fname);

