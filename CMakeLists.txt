cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++14)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

set("OpenCV_DIR" "/usr/local/Cellar/opencv@3/3.4.9_1/share/OpenCV/")
find_package(OpenCV 3.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (2D_feature_tracking src/matching2D_Student.cpp src/MidTermProject_Camera_Student.cpp src/util.cpp src/FeatureTracker.cpp)
target_link_libraries (2D_feature_tracking ${OpenCV_LIBRARIES})


add_executable (TestDifferentSettings src/TestDifferentSettings.cpp src/matching2D_Student.cpp src/util.cpp src/FeatureTracker.cpp)
target_link_libraries (TestDifferentSettings ${OpenCV_LIBRARIES})
