
// COPYRIGHT (c) 2025, Heinrich HE
// mail: 2130238@tongji.edu.cn.
// Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License. | diese Project soll nicht ohne die Lizenz verwendet werden.
// You may obtain a copy of the License at                          | Die kopie der Lizenz kann auf http://www.apache.org/licenses/LICENSE-2.0  
// http://www.apache.org/licenses/LICENSE-2.0                       | oder in der Original-Datei erhalten werden.


#include <ros/ros.h>
#include <franka/robot.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <std_msgs/Float32.h>

#include <franka/robot_state.h>
#include <franka/exception.h>
#include <std_msgs/Float32MultiArray.h>
#include <iostream>
#include <array>
#include <atomic>
#include <thread>
#include <mutex>
#include <sensor_msgs/JointState.h>

// This is a ros node to excute impedance control of franka panda robot arm.
// the input signals are 
// "/desired_velocity" 
// and 
// "/desired_gripper_width"
// this node will also publish joint state with frequency of 1000 hz and topic 
// "base_feedback/joint_state"
// while using this code, you can replace the catkin_pkg franka_ros

// Dies ist eine ROS Node, die fuer Impetanz-Steuerung von Franka-Panda Roboter durchzufuehren ist.
// Die Eingaebe Signalen sind"
// "/desired_velocity" 
// und
// "/desired_gripper_width"
// Diese Node kann JointState mit dem Frequenz um 1000 Hz und Topic
// "base_feedback/joint_state"
// verbreiten
// Wenn diese Node verwendet ist, kann das Catkin_Pkg Franka_Ros ersetzt werden

// core-code
void record_loop(franka::Robot& robot, ros::Publisher& joint_pub, ros::Publisher& eef_pub) {
  
    
    sensor_msgs::JointState joint_msg;
    joint_msg.name = {  "panda_joint1", "panda_joint2", "panda_joint3",
                    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7" };

    while (ros::ok)
    {
        joint_msg.header.stamp = ros::Time::now();
        franka::RobotState state = robot.readOnce();

        joint_msg.position.assign(state.q.begin(), state.q.end());
        joint_msg.velocity.assign(state.dq.begin(), state.dq.end());
        joint_msg.effort.assign(state.tau_J.begin(), state.tau_J.end());

        // eef_pos.data = {T_EE[12], T_EE[13], T_EE[14]};

        joint_pub.publish(joint_msg);
    }
}



int main(int argc, char** argv) {
    ros::init(argc, argv, "franka_velocity_control");
    // ros::Subscriber()
    ros::NodeHandle nh;
    ros::Publisher js_pub = nh.advertise<sensor_msgs::JointState>("/base_feedback/joint_state", 10);
    ros::Publisher ee_pub = nh.advertise<std_msgs::Float32MultiArray>("/eef_position/", 10);
    franka::Robot robot("172.16.0.2");
    franka::Gripper gripper("172.16.0.2");

    // kernel priority
    robot.setCollisionBehavior(
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}
    );

    // robot control
    std::thread control_thread(record_loop, std::ref(robot), std::ref(js_pub), std::ref(ee_pub));
    // gripper control

    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();

    control_thread.join();
    return 0;
}
