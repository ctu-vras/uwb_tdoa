cmake_minimum_required(VERSION 3.0.2)
project(uwb_tdoa)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  ros_numpy
  rospy
  std_msgs
  tf2
  tf2_py
  tf2_ros
  message_generation
  dwm1001_ros
)

add_message_files(
 FILES
 TDoAAnchor.msg  TDoAMeas.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
 DEPENDENCIES
 std_msgs
)

catkin_package(
)

catkin_install_python(PROGRAMS
    scripts/driver.py
    scripts/tdoa_locator.py
    scripts/plot_intersect.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
