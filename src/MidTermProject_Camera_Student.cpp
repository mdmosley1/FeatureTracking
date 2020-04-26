/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <memory> // unique_ptr
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
vector<DataFrame> dataBuffer_g; // list of data frames which are held in memory at the same time

// Settings



const bool bLimitKpts = true;


bool bVis = true;            // visualize results

void AddToRingBuffer(const DataFrame& frame)
{
    dataBuffer_g.push_back(frame);
    while (dataBuffer_g.size() > dataBufferSize)
        dataBuffer_g.erase(dataBuffer_g.begin());
}

void VisualizeMatches(vector<cv::DMatch> matches)
{
    cv::Mat matchImg = ((dataBuffer_g.end() - 1)->cameraImg).clone();
    cv::drawMatches((dataBuffer_g.end() - 2)->cameraImg, (dataBuffer_g.end() - 2)->keypoints,
                    (dataBuffer_g.end() - 1)->cameraImg, (dataBuffer_g.end() - 1)->keypoints,
                    matches, matchImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    string windowName = "Matching keypoints between two camera images";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    cout << "Press key to continue to next image" << endl;
    cv::waitKey(0); // wait for key to be pressed
}

void LimitKeyPoints(vector<cv::KeyPoint>& keypoints, const Params& p)
{
    int maxKeypoints = 50;

    if (p.detectorType.compare("SHITOMASI") == 0)
    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
    }
    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
    cout << " NOTE: Keypoints have been limited!" << endl;
}

void LimitKeyPointsRect(vector<cv::KeyPoint>& keypoints)
{
    cv::Rect vehicleRect(535, 180, 180, 150);
    for (auto it = keypoints.begin(); it != keypoints.end(); )
    {
        // delete keypoint if not inside rectangle
        if (vehicleRect.contains(it->pt))
            it++;
        else
            it = keypoints.erase(it);
    }
}


void DetectAndTrackFeatures(const cv::Mat& imgGray,
                            const std::unique_ptr<KPDetector>& _detector,
                            const cv::Ptr<cv::DescriptorExtractor>& _descriptor,
                            const Params& params)
{
    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image        
    keypoints = _detector->DetectKeypoints(imgGray, false);

    //// TASK MP.3 -> only keep keypoints on the preceding vehicle
    if (params.bFocusOnVehicle) LimitKeyPointsRect(keypoints);
    // optional : limit number of keypoints (helpful for debugging and learning)
    if (bLimitKpts) LimitKeyPoints(keypoints, params);
    cout << "#2 : DETECT KEYPOINTS done" << endl;
    cv::Mat descriptors = descKeypoints(keypoints, imgGray, _descriptor, params);
    // push descriptors for current frame to end of data buffer
    DataFrame newFrame(imgGray, keypoints, descriptors);
    cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
    AddToRingBuffer(newFrame);

    if (dataBuffer_g.size() > 1) // wait until at least two images have been processed
    {
        //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
        //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
        auto currentFrame = dataBuffer_g.end() - 1;
        auto lastFrame = dataBuffer_g.end() - 2;
        vector<cv::DMatch> matches = matchDescriptors(lastFrame->keypoints, currentFrame->keypoints,
                                                      lastFrame->descriptors, currentFrame->descriptors,
                                                      params);
 
        // store matches in current data frame
        currentFrame->kptMatches = matches;
        cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
        // visualize matches between current and previous image
        if (bVis) VisualizeMatches(matches);
    }
}

std::unique_ptr<KPDetector> CreateDetector(std::string _detectorType)
{
    cout << "Creating detector with type: " << _detectorType << "\n";
    std::unique_ptr<KPDetector> detector;
    if (_detectorType.compare("SHITOMASI") == 0)
        detector = std::make_unique<DetectorShiTomasi>();
    else if (_detectorType.compare("HARRIS") == 0)
        detector = std::make_unique<DetectorHarris>();
    else if (_detectorType.compare("FAST") == 0)
        detector = std::make_unique<DetectorFast>();
    else if (_detectorType.compare("BRISK") == 0)
        detector = std::make_unique<DetectorBrisk>();
    else if (_detectorType.compare("ORB") == 0)
        detector = std::make_unique<DetectorOrb>();
    else if (_detectorType.compare("AKAZE") == 0)
        detector = std::make_unique<DetectorAkaze>();
    else if (_detectorType.compare("SIFT") == 0)
        detector = std::make_unique<DetectorSift>();
    else
    {
        cout << _detectorType  << " is not a valid detector type!"<< "\n";
        return nullptr;
    }
    return detector;
}

cv::Ptr<cv::DescriptorExtractor> CreateDescriptor(std::string _descriptorType)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (_descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (_descriptorType.compare("BRIEF") == 0)
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    else if (_descriptorType.compare("ORB") == 0)
        extractor = cv::ORB::create();
    else if (_descriptorType.compare("FREAK") == 0)
        extractor = cv::xfeatures2d::FREAK::create();
    else if (_descriptorType.compare("AKAZE") == 0)
        extractor = cv::AKAZE::create();
    else if (_descriptorType.compare("SIFT") == 0)
        extractor = cv::xfeatures2d::SIFT::create();
    else
    {
        cout << _descriptorType  << " is not a valid descriptor type!"<< "\n";
    }

    return extractor;
}

Params LoadParamsFromFile(std::string fname)
{
    Params p;
    std::ifstream file(fname);
    std::stringstream buffer;
    buffer << file.rdbuf();

    std::map<std::string, std::string> paramsMap;

    std::string line;
    while( std::getline(buffer, line) )
    {
        // skip line if it starts with #
        if (line.find('#') == 0)
            continue;

        std::istringstream is_line(line);
        std::string key;
        if( std::getline(is_line, key, '=') )
        {
            std::string value;
            if( std::getline(is_line, value) )
                paramsMap.emplace(key,value);
        }
    }

    // print params
    cout << "######### LOADED PARAMS ########" << "\n";
    for (auto it = paramsMap.begin(); it != paramsMap.end(); ++it)
        cout << it->first << " : " << it->second << "\n";
    cout << "################################" << "\n\n";

    p.detectorType = paramsMap["detectorType"];
    p.descriptorType = paramsMap["descriptorType"];
    p.matcherType = paramsMap["matcherType"];
    p.selectorType = paramsMap["selectorType"];
    p.bFocusOnVehicle = std::stoi(paramsMap["bFocusOnVehicle"]);
    p.normType = std::stoi(paramsMap["normType"]);
    return p;
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    auto params = LoadParamsFromFile("../src/settings.txt");

    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    string dataPath = "../";
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    auto detector = CreateDetector(params.detectorType);
    auto descriptor = CreateDescriptor(params.descriptorType);

    if (detector == nullptr)
    {
        cout << "Failed to create detector!" << "\n";
        return -1;
    }

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        DetectAndTrackFeatures(imgGray, detector, descriptor, params);
    }
    return 0;
}
