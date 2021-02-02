cmake_minimum_required(VERSION 3.0.2)
project(Particle_filter)

#Debug mode
# set(CMAKE_VERBOSE_MAKEFILE yes)
# set(CMAKE_BUILD_TYPE Debug)

#Release mode
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O0 -DDEBUG -std=c++11")

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
 roscpp
)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES Particle_filter
  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
 ${PROJECT_SOURCE_DIR}/include/
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/Particle_filter.cpp
# )

add_executable(${PROJECT_NAME}_node src/main.cpp)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

