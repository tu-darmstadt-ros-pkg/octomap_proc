cmake_minimum_required(VERSION 2.8.3)
project(octomap_proc_common)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  octomap_msgs
  octomap_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
  cv_image_proc
)

link_libraries(${OCTOMAP_LIBRARIES})

add_definitions(-DOCTOMAP_NODEBUGOUT)

catkin_package(
   INCLUDE_DIRS include
#   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS cv_bridge geometry_msgs octomap_msgs octomap_ros roscpp sensor_msgs std_msgs tf visualization_msgs cv_image_proc
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

#common commands for building c++ executables and libraries
add_executable(test_octo_cv_interface src/test_octo_cv_interface.cpp)
target_link_libraries(test_octo_cv_interface ${catkin_LIBRARIES})
#add_dependencies(worldmodel_server_node ${catkin_EXPORTED_TARGETS})

#install(TARGETS worldmodel_server_node
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

#install the launch files.
#install(DIRECTORY launch
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#  FILES_MATCHING PATTERN "*.launch"
#)
