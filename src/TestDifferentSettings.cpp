#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "FeatureTracker.h"

#include "util.h"

using namespace std;

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

bool ValidCombination(const std::string detector, const std::string descriptor)
{
    if (detector == "AKAZE" && descriptor != "AKAZE")
        return false;
    if (descriptor == "AKAZE" && detector != "AKAZE")
        return false;

    return true;
}

void ProcessDatasetWithSettings(const std::unique_ptr<KPDetector>& detector, const cv::Ptr<cv::DescriptorExtractor>& descriptor, const Params& params)
{
    FeatureTracker featureTracker(params);

    // initialize feature tracker
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

        // detect and describe features
        DataFrame frame = DetectAndDescribeFeatures(imgGray, detector, descriptor, params);

        // trackFeatures
        featureTracker.TrackFeatures(frame);
    }
}

int main(int argc, const char *argv[])
{
    Params params = LoadParamsFromFile("../src/settings.txt");
    params.cvWaitTime = 10;

    // make list of strings of possible detectors and descriptors
    std::set<std::string> availableDetectors = {"HARRIS", "FAST", "SHITOMASI", "BRISK", "ORB", "AKAZE", "SIFT"};
    std::set<std::string> availableDescriptors = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    // make pairs of all possible combinations. Don't use invalid pairs
    std::set<std::pair<std::string, std::string>> combinations;
    for (auto detector : availableDetectors)
    {
        for (auto descriptor : availableDescriptors)
        {
            auto combo = std::make_pair(detector, descriptor);
            if (ValidCombination(combo.first, combo.second))
                combinations.insert

                    (combo);
            else
                cout << detector << " and " << descriptor << " are not valid combination." << "\n";
        }
    }


    // process dataset with each combination
    for (auto& combo : combinations)
    {
        cout << "Running with " << combo.first << " and " << combo.second << "\n";
        auto detector = CreateDetector(combo.first);
        auto descriptor = CreateDescriptor(combo.second);
    
        if (detector == nullptr)
        {
            cout << "Failed to create detector!" << "\n";
            return -1;
        }

        ProcessDatasetWithSettings(detector, descriptor, params);
    }

    // Write below to a results file
    // print out total number of keypoints detected on vehicle (all 10 images)

    // print out total matches

    // print out total time taken
    
    return 0;
}
