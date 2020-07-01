cmake_minimum_required(VERSION 3.16)
project(TreeFinder)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( include src ${OpenCV_INCLUDE_DIRS} )

add_executable( ${PROJECT_NAME} include/main.cpp src/bag_of_leaves.h src/svm_binary_classifier.h src/tree_finder.h)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )