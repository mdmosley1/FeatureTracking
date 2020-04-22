#include <numeric>
#include "matching2D.hpp"

using namespace std;


void FilterMatches(std::vector<std::vector<cv::DMatch>>& knnMatches)
{
    const double minThresh = 0.8;
    // for each set of matches, compare best match with second best match
    cout << "Filtering Matches..." << "\n";

    int numMatches = knnMatches.size();
    int matchesFiltered = 0;
    //for (auto& kMatches : knnMatches)
    for (auto it = knnMatches.begin(); it != knnMatches.end();)
    {
        // delete this match if match is ambiguous (top two distances are too close)
        auto& matches = *it;
        double ratio = matches[0].distance / matches[1].distance;
        if (ratio > minThresh)
        {
            matchesFiltered++;
            it = knnMatches.erase(it);
        }
        else
            it++;
    }
    cout << "Filtered out " << matchesFiltered << " ambiguous matches out of " << numMatches << " total." << endl;
}

std::vector<cv::DMatch> ConvertMatches(std::vector<std::vector<cv::DMatch>>& knnMatches)
{
    std::vector<cv::DMatch> matches;
    for (auto& kMatches : knnMatches)
        matches.push_back(kMatches[0]);

    return matches;
}

// Find best matches for keypoints in two camera images based on several matching methods
std::vector<cv::DMatch> matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      const Params& params)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (params.matcherType.compare("MAT_BF") == 0)
        matcher = cv::BFMatcher::create(params.normType, crossCheck);
    else if (params.matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // convert descriptors to correct datatype if using flann
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }

    std::vector<cv::DMatch> matches;
    // perform matching task
    if (params.selectorType.compare("SEL_NN") == 0) // nearest neighbor (best match)
    {
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (params.selectorType.compare("SEL_KNN") == 0)
    {
        double t = (double)cv::getTickCount();
        int k = 2;
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, k);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        // filter matches using descriptor distance ratio test
        FilterMatches(knnMatches);
        matches = ConvertMatches(knnMatches);
    }
    return matches;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
cv::Mat descKeypoints(vector<cv::KeyPoint> &keypoints,
                      const cv::Mat &img,
                      const cv::Ptr<cv::DescriptorExtractor>& _descriptor,
                      const Params& params)
{
    cv::Mat descriptors;
    // perform feature description
    double t = (double)cv::getTickCount();
    _descriptor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << params.descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return descriptors;
}

// perform non-maximum supporesion to get only the good keypoints
std::vector<cv::KeyPoint> DetectorHarris::GetKeypoints(const cv::Mat dst_norm) const
{
    std::vector<cv::KeyPoint> keypoints;
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse_)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize_;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap_)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
    return keypoints;
}

std::vector<cv::KeyPoint> DetectorOrb::DetectKeypoints(const cv::Mat& img, bool bVis)
{
    vector<cv::KeyPoint> keypoints;

    double t = (double)cv::getTickCount();
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    return keypoints;
}

std::vector<cv::KeyPoint> DetectorSift::DetectKeypoints(const cv::Mat& img, bool bVis)
{
    vector<cv::KeyPoint> keypoints;

    double t = (double)cv::getTickCount();
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    sift->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    return keypoints;
}

std::vector<cv::KeyPoint> DetectorBrisk::DetectKeypoints(const cv::Mat& img, bool bVis)
{
    vector<cv::KeyPoint> keypoints;

    double t = (double)cv::getTickCount();
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    brisk->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    return keypoints;
}

std::vector<cv::KeyPoint> DetectorFast::DetectKeypoints(const cv::Mat& img, bool bVis)
{
    vector<cv::KeyPoint> keypoints;
    int threshold = 10;
    bool useNonMaxSuppression = true;

    int type = cv::FastFeatureDetector::TYPE_9_16;
    //int type = FastFeatureDetector::TYPE_7_12;
    //int type = FastFeatureDetector::TYPE_5_8;

    double t = (double)cv::getTickCount();
    cv::FAST(img, keypoints, threshold, useNonMaxSuppression, type);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    return keypoints;
}

std::vector<cv::KeyPoint> DetectorHarris::DetectKeypoints(const cv::Mat& img, bool bVis)
{
    // Apply corner detection
    double t = (double)cv::getTickCount();
    //vector<cv::Point2f> corners;
    cv::Mat dst_norm;
    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1 );
    cv::cornerHarris(img, dst, blockSize_, apertureSize_, k_);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    vector<cv::KeyPoint> keypoints = GetKeypoints(dst_norm);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    return keypoints;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
std::vector<cv::KeyPoint> DetectorShiTomasi::DetectKeypoints(const cv::Mat& img, bool bVis)
{
    vector<cv::KeyPoint> keypoints;
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return keypoints;
}