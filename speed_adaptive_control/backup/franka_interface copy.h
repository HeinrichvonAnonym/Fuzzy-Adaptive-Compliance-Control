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
#include <mutex>

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
        double dq_step_limit; 
        franka::RobotState state;
        franka::RobotState init_state;
        std::array<double, 7> q_initial;
        std::array<double, 7> q_cmd;
        std::array<double, 7> q_offset;

        std::array<double, 7> cur_q;
        std::array<double, 7> cur_vel;;

        std::array<double, 7> desired_velocity = {0,0,0,0,0,0,0};
        
        void set_velo(std::array<double, 7>& desired_velo);
        void set_params();
        void start();
        void join();
        void control_loop(franka::Robot& robot);
        void gripper_control_loop(franka::Gripper& gripper); 

        std::thread control_thread;
        std::thread gripper_thread;
    private:
        franka::Robot robot;
        franka::Gripper gripper;

        Eigen::Vector3d dx;
        Eigen::Matrix3d JJt;
        Eigen::Vector3d f_lp;
        Eigen::Matrix<double, 7, 3> J_pinv;
        Eigen::Matrix<double, 7, 1> dq_c;
        std::array<double, 42> jacobian_array;
        Eigen::Matrix<double, 3, 7> Jv;
        Eigen::Matrix3d I3d = Eigen::Matrix3d::Identity();

        sensor_msgs::JointState joint_msg;
        std::mutex velocity_mutex;
};
