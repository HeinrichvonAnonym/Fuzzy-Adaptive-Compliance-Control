// copyright (c) 2025-present Heinrich 2130238@tongji.edu.cn.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include<ros/ros.h>
#include<sensor_msgs/JointState.h>
#include<std_msgs/Float32.h>
#include<std_msgs/Float32MultiArray.h>
#include<std_msgs/Int16.h>
#include<math.h>
#include<yaml-cpp/yaml.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include<Eigen/Dense>
#include <array>
#include<franka/robot.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot_state.h>
#include <franka/exception.h>

class Franka_Interface { 
    public:
        Franka_Interface(ros::NodeHandle nh);
        ~Franka_Interface();
        ros::Publisher js_pub;
        ros::Publisher ee_pub;
        const std::array<double, 7> joint_min = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
        const std::array<double, 7> joint_max = { 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973};
        std::array<double, 7> k_stiffness;
        std::array<double, 7> k_damping;
        double force_threshold_N ;  // Complaince threshold
        double hysteresis_N;        // Hysteresissrc/franka_test/src/franka_node.cpp
        double lp_alpha;         // Filter 
        double comp_vel_gain;  // m/(N*s) Kraft --> Bewegung
        double pinv_damping;     // Inverse Damping
        double dq_step_limit1;    // max dt
        double max_qoffset_abs;
        double max_tau;
        franka::RobotState state;
        franka::RobotState init_state;
        std::array<double, 7> q_initial;
        std::array<double, 7> q_cmd;
        std::array<double, 7> q_offset;

        std::array<double, 7> potential;
        bool got_command = false;
        
        std::array<double, 7> angle_normalize(std::array<double, 7> vec);
        void potential_command_sub(const std_msgs::Float32MultiArray::ConstPtr& msg);
        void set_params();
        void run();
        void control_loop(franka::Robot& robot);
        void gripper_control_loop(franka::Gripper& gripper); 
    private:
        ros::Subscriber potential_sub;
};
