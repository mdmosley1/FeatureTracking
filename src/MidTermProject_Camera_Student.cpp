/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <sstream>
#include <iomanip>
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

#include "FeatureTracker.h"

#include "util.h"

using namespace std;


// Settings


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

    FeatureTracker featureTracker(params);

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

    // refactor above so I can run with every possible combination (30 total)
    
    // Write below to a results file
    // print out total number of keypoints detected on vehicle (all 10 images)

    // print out total matches

    // print out total time taken
    
    return 0;
}
