cmake_minimum_required(VERSION 3.0.2)
project(franka_test)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

# 设置 libfranka 的路径
set(Franka_DIR "/home/heinrich/Fuzzy-Adaptive-Compliance-Control-main/libfranka-0.8.0/cmake")
find_package(Franka REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS Franka
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Franka_INCLUDE_DIRS}
)

link_directories(
  /home/heinrich/libfranka/lib
)

add_executable(franka_node src/franka_node.cpp)

target_link_libraries(franka_node
  ${catkin_LIBRARIES}
  ${Franka_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/pid_executer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)