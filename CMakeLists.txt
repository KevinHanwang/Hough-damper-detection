cmake_minimum_required(VERSION 2.8.3)
project(imageBasics)

SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
# set(OpenCV_DIR /usr/local/share/OpenCV)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Hough_Circle Improved_Hough_Circle_Detection.cpp)
target_link_libraries(Hough_Circle ${OpenCV_LIBRARIES})

add_executable(wh1 wh1_houghCircle.cpp)
target_link_libraries(wh1 ${OpenCV_LIBRARIES})

add_executable(wh2 wh2_houghCircle.cpp)
target_link_libraries(wh2 ${OpenCV_LIBRARIES})

add_executable(wh3 wh3_houghCircle.cpp)
target_link_libraries(wh3 ${OpenCV_LIBRARIES})

add_executable(line LineDetector2.cpp)
target_link_libraries(line ${OpenCV_LIBRARIES})

add_executable(line_realtime LineDetector.cpp)
target_link_libraries(line_realtime ${OpenCV_LIBRARIES})

add_executable(erodeDilate erodeDilate.cpp)
target_link_libraries(erodeDilate ${OpenCV_LIBRARIES})

add_executable(preProcessParaChange preProcessParaChange.cpp)
target_link_libraries(preProcessParaChange ${OpenCV_LIBRARIES})

add_executable(erodeTuning erodeTuning.cpp)
target_link_libraries(erodeTuning ${OpenCV_LIBRARIES})

add_executable(FZCDetector FZCDetector.cpp)
target_link_libraries(FZCDetector ${OpenCV_LIBRARIES})

add_executable(FZCDetector_video FZCDetector_video.cpp)
target_link_libraries(FZCDetector_video ${OpenCV_LIBRARIES})

# add_executable(cvTest cvTest.cpp)
# target_link_libraries(cvTest ${OpenCV_LIBRARIES})