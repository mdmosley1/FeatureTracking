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
#include <fstream>

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

    // for some reason, I get the following error using SIFT and ORB together:
    // OpenCV Error: Insufficient memory (Failed to allocate 65763706112 bytes) in OutOfMemoryError
    if (detector == "SIFT" && descriptor == "ORB")
        return false;

    return true;
}

struct Results
{
    double time = 0.0;
    int numKeypoints = 0;
    int numMatches = 0;
};

Results ProcessDatasetWithSettings(const std::unique_ptr<KPDetector>& detector, const cv::Ptr<cv::DescriptorExtractor>& descriptor, const Params& params)
{
    FeatureTracker featureTracker(params);
    double totalTimeForDetectionAndDescription = 0;
    int totalKeypoints = 0;
    int totalMatches = 0;
    
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
        double t = (double)cv::getTickCount();
        DataFrame frame = DetectAndDescribeFeatures(imgGray, detector, descriptor, params);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTimeForDetectionAndDescription += t;
        totalKeypoints += frame.keypoints.size();
        
        vector<cv::DMatch> matches = featureTracker.TrackFeatures(frame);
        totalMatches += matches.size();
    }
    Results results;
    results.time = totalTimeForDetectionAndDescription;
    results.numKeypoints = totalKeypoints;
    results.numMatches = totalMatches;
    
    return results;
}

std::set<std::pair<std::string, std::string>> FormCombinations(std::set<std::string> availableDetectors, std::set<std::string> availableDescriptors)
{
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
    return combinations;
}

void WriteResultsToDisk(std::string detector, std::string descriptor, double processingTime, int numberKeypoints, int totalMatches)
{
    std::string filename = "/tmp/results.txt";
    std::ofstream file(filename, ios::out | ios::app);
    if (file.is_open())
    {
    // Write below to a results file
        file << "\n\nDetector/Descriptor: " << detector << ", " << descriptor << endl;
    // print out total number of keypoints detected on vehicle (all 10 images)
        file << "Total keypoints: " << numberKeypoints << endl;
    // print out total matches
        file << "Total matches: " << totalMatches << endl;
    // print out total time taken
        file << "Total time for detection and description: " << processingTime << endl;
        file.close();
    }
    else {
        cout << "unable to open " << filename << "\n";
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
    auto combinations = FormCombinations(availableDetectors, availableDescriptors);

    
    std::string filename = "/tmp/results.txt";
    std::ofstream file(filename, ios::out);
    file << "RESULTS OF FEATURE TRACKING\n\n";
    file.close();

    // process dataset with each combination
    for (auto& combo : combinations)
    {
        cout << "\n\n\n";
        cout << "################################################## \n";
        cout << "### Running with " << combo.first << " and " << combo.second << "\n";
        cout << "################################################## \n\n ";
        auto detector = CreateDetector(combo.first);
        auto descriptor = CreateDescriptor(combo.second);
    
        if (detector == nullptr)
        {
            cout << "Failed to create detector!" << "\n";
            return -1;
        }

        params.detectorType=combo.first;
        params.descriptorType=combo.second;
        if (combo.second == "SIFT") params.normType=4; // use L2 norm instead of hamming for gradient descriptors 

        Results res = ProcessDatasetWithSettings(detector, descriptor, params);

        WriteResultsToDisk(combo.first, combo.second, res.time, res.numKeypoints, res.numMatches);
    }        
    return 0;
}
