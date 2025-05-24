# Hardware Setup

## Notice

### Requirenment
you need to install kortex_driver first to drive kinova gen3: https://github.com/Kinovarobotics/ros_kortex
you need to install libfranka  first to drive franka panda: https://github.com/frankaemika/libfranka
or you other hardware plattform but you need to adjust the communication of pid_executer in /config dynamical_parameters.yaml of speed plan pkg

### Interface
For more flexible control, ROS Hardware Interface is not used in this work, but things are simplified: control msg can via this work to control the robot directly.

#### By Kinova Gen3
The moveit config of kinova gen3 is remodified, allowing the move goup receiving the joint state without hardwareinterface mechanism.

#### By Franka Panda
manually coded a pid system based on libfranka cpp without utilizing the franka_ros package.