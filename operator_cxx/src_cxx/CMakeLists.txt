cmake_minimum_required(VERSION 2.8.13)
project(lidar_detection_lib)

# Flag to determine if build python lib only
####################
## Python support ##
###################
## C++ -> Python: pybind11, Python -> C++: Built-in <Python.h>
## If something goes wrong (e.g. <Python.h> not found), try:
##      pip install python-dev

set(PYBIND11_PYTHON_VERSION 2.7 CACHE STRING "")
if (NOT TARGET pybind11)
    find_package(pybind11 REQUIRED)
endif ()
set(PYBIND11_CPP_STANDARD -std=c++11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -Wno-unused-result -Wno-logical-op-parentheses -fext-numeric-literals")

#set(SOURCE_FILES preprocessing.cpp preprocessing.h overlap.cpp overlap.h polygon_overlap.cpp polygon_overlap.h refine.cpp refine.h)
set(SOURCE_FILES overlap.cpp overlap.h assigner.h)

set(PYTHON_EXECUTABLE /usr/bin/python)

find_package(Eigen3 REQUIRED)
include_directories(include ${EIGEN3_INCLUDE_DIRS})

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/..)

#message(ERROR ${PROJECT_SOURCE_DIR}/..)

pybind11_add_module(processing_cxx SHARED THIN_LTO pybinding.cpp ${SOURCE_FILES})

