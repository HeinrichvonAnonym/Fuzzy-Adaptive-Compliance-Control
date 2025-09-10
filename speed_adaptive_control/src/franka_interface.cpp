// copyright (c) 2025-present Heinrich 2130238@tongji.edu.cn.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include<speed_adaptive_control/franka_interface.h>

Franka_Interface::Franka_Interface(ros::NodeHandle nh)
{
    potential_sub = nh.subscribe("facc/command", 10, &Franka_Interface::potential_command_sub, this);
    ee_pub = nh.advertise<sensor_msgs::JointState>("/base_feedback/ee_state", 10);
    js_pub = nh.advertise<sensor_msgs::JointState>("/base_feedback/joint_state", 10);
    set_params();

    run();
}

Franka_Interface::~Franka_Interface()
{ 
}

void Franka_Interface::set_params()
{
    k_stiffness = {50., 50., 50., 50., 50., 50., 50 };
    for (size_t i = 0; i < 7; i++){
        k_damping[i] = 2.0 * std::sqrt(k_stiffness[i]);
    }
    force_threshold_N = 12.0;
    hysteresis_N = 4;
    lp_alpha = 0.15;
    comp_vel_gain = 0.0008;
    pinv_damping = 0.10;
    dq_step_limit1 = 0.01;
    max_qoffset_abs = 0.5;
    max_tau = 50;
}

void Franka_Interface::potential_command_sub(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    for (size_t i = 0; i < 7; i++){
        potential[i] = msg->data[i];
    } 
    got_command = true;
}

void Franka_Interface::control_loop(franka::Robot& robot)
{
    ros::Rate r = ros::Rate(10);
    sensor_msgs::JointState joint_msg;
    joint_msg.name = {  "panda_joint1", "panda_joint2", "panda_joint3",
                    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7" };
    franka::Model model = robot.loadModel();
    Eigen::Vector3d f_lp = Eigen::Vector3d::Zero();
    while (ros::ok()){
        try{
            init_state = robot.readOnce();
            q_initial = init_state.q;
            q_cmd = init_state.q;
            q_offset = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
            std::cout << "Robot state read OK. Ready for control.\n";
            robot.control(
                [&](const franka::RobotState& robot_state, franka::Duration period) -> franka::Torques{
                    const double dt =  period.toSec();
                    if (!ros::ok()){
                        throw franka::Exception("ROS is shutting down, exiting control loop");
                    }
                    joint_msg.header.stamp = ros::Time::now();
                    joint_msg.position.assign(robot_state.q.begin(), robot_state.q.end());
                    joint_msg.velocity.assign(robot_state.dq.begin(), robot_state.dq.end());
                    joint_msg.effort.assign(robot_state.tau_J.begin(), robot_state.tau_J.end());
                    js_pub.publish(joint_msg);

                     // Kraft
                    Eigen::Vector3d f_now(  robot_state.O_F_ext_hat_K[0],
                                            robot_state.O_F_ext_hat_K[1],
                                            robot_state.O_F_ext_hat_K[2]);
                    f_lp = (1.0 - lp_alpha) * f_lp + lp_alpha * f_now;
                    const double F = f_lp.norm();

                    // env F calc

                   
                    // Rechnung des Kraftmoment
                    std::array<double, 7> tau_d;
                    if(got_command){
                        for (int i = 0; i < 7; i++){
                            double T_ex = potential[i] * 5;
                            q_cmd[i] += dt * (2 * robot_state.dq[i] + dt * T_ex) / 2;
                            q_cmd[i] = std::min(std::max(q_cmd[i], joint_min[i] + 0.05), joint_max[i] - 0.05);
                        }
                    }

                    for (size_t i = 0; i < 7; i++) {
                        // double kd_scales = k_damping[i] * stiffness_scale;
                        // double k_stiffness_scales = k_stiffness[i] * stiffness_scale;
                        double q_err = robot_state.q[i] - q_cmd[i];
                        tau_d[i] = - k_stiffness[i] * q_err - k_damping[i] * state.dq[i];
                    }
            

                    return franka::Torques(tau_d);
                }
            );
        } catch (const franka::Exception& e) {
            std::cout << "Network Exception: " << e.what() << std::endl;
        }
        auto robot_state = robot.readOnce();
        joint_msg.header.stamp = ros::Time::now();
        joint_msg.position.assign(robot_state.q.begin(), robot_state.q.end());
        joint_msg.velocity.assign(robot_state.dq.begin(), robot_state.dq.end());
        joint_msg.effort.assign(robot_state.tau_J.begin(), robot_state.tau_J.end());
        js_pub.publish(joint_msg);
        r.sleep();
    } 
}

void Franka_Interface::gripper_control_loop(franka::Gripper& gripper)
{

}

void Franka_Interface::run()
{
    franka::Robot robot("172.16.0.2");
    franka::Gripper gripper("172.16.0.2");
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

    ROS_INFO("Starting controller...");
    std::thread control_thread(&Franka_Interface::control_loop, this, 
                                std::ref(robot));
    std::thread gripper_thread(&Franka_Interface::gripper_control_loop, this,
                                std::ref(gripper));
    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();
    control_thread.join();
    gripper_thread.join();
    spinner.stop();
 
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "franka_interface");
    ros::NodeHandle nh;
    Franka_Interface fi(nh); 
}