# MidTerm Report:

This is the midterm report for the camera based 2d feature tracking
project. Below I list each rubric point and detail how addressed each
one.

## Data Buffer
### Data buffer optimization
To keep the data buffer from growing too large, I made a while loop
that erases the first element from the databuffer until the size of
the buffer is less than or equal to 2.

## Keypoints
### Keypoint detection
I added a settings parser that reads a settings file and sets the
detector string appropriately. Available options for the detector
setting include Harris, Fast, Brisk, Orb, Akaze and Sift.
### Keypoint removal
I used Opencv's cv::Rect and cv::Rect::contains() to remove all
keypoints outside a predefined rectangle.

## Descriptors
### Keypoint Descriptors
As with the keypoint detectors, I made which keypoint descriptor to
use a setting in the settings file. Available options for this setting
are Brief, Orb, Freak, Akaze and Sift.
### Descriptor Matching
I implemented Flann matching and k-nearest neighbor selection.
### Descriptor Distance Ratio
I implemented the distance ratio that filters out matches if the
second best match was almost as good as the first.

## Performance
### Performance Eval 1
I build a separate executable that iterates over all possible
detectors and writes the total number of detected keypoints for each
detector to a file called results.txt
### Performance Eval 2 
I build a separate executable that iterates over all possible
detectors and descriptors and writes the total number of matched keypoints for each
detector/descriptor combination to a file called results.txt
### Performance Eval 3 
The separate executable I build also logs the time taken for each
combination of detector/descriptor.  The fastest three combinations
are summarized in the following table.

| Detector        | Descriptor  | Time (ms)|
| ------------- |:-------------:| -----:   |
| ORB           | BRISK         | 80       |
| ORB           | FREAK         | 121      |
| ORB           | ORB           | 126      |
