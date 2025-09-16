#include<ros/ros.h>
#include<Eigen/Dense>
#include<vector>
#include<cmath>
#include<iostream>
#include<geometry_msgs/Pose.h>
#include<geometry_msgs/PoseStamped.h>
#include<geometry_msgs/PoseArray.h>
#include<geometry_msgs/Quaternion.h>
#include<sensor_msgs/JointState.h>
#include<std_msgs/Float32MultiArray.h>

class JacobianCalculator
{ 
    public:
        JacobianCalculator(ros::NodeHandle nh);
        ~JacobianCalculator();
        int dynamic;
        std::vector<Eigen::Vector4d> dh_parmas = {
            {0.,            0.333,      0.,      0},
            {0.,            0.,         0.,    -M_PI/2},
            {0,             0.316,       0.,   M_PI/2},
            {0.,       0,        0.0825,    M_PI/2},
            {0.,    0.384,   -0.0825,   -M_PI/2},
            {0,     0,         0,     M_PI/2},
            {0,     0.207,   0.088,    M_PI/2},
            {0,     0.18,     0,       -M_PI}
        };

        std::array<bool, 7> joint_orient = {true, true, true, true, true, true, true};

        Eigen::Matrix<double, 6, 7> jacobian;
        Eigen::Matrix<double, 6, 8> jacobian_eef;
        Eigen::Matrix<double, 6, 7> jacobian_7_dof;
        Eigen::Matrix<double, 6, 6> jacobian_6_dof;
        Eigen::Matrix<double, 6, 5> jacobian_5_dof;
        Eigen::Matrix<double, 6, 4> jacobian_4_dof;
        Eigen::Vector3d O_end;
        Eigen::Matrix3d R_i;
        Eigen::Vector3d Z_i;
        Eigen::Vector3d O_i;

        const Eigen::Matrix3d identity3d = Eigen::Matrix3d::Identity();
        const Eigen::Matrix4d identity4d = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d T_dh;

        Eigen::Matrix4d T_dof;
        std::array<Eigen::Matrix4d, 7> T_dofs;
        Eigen::Matrix4d T_end;
        Eigen::Vector3d Z_end;

        geometry_msgs::PoseArray pose_array;
        
        Eigen::Quaterniond quat;
        Eigen::Matrix3d shouldI;
        Eigen::Matrix3d U;
        Eigen::Matrix3d V;

        Eigen::VectorXd q = Eigen::VectorXd::Zero(7);



        Eigen::Matrix4d dh_2_T(double theta, double theta_offset,  
        double d, double a, double alpha, bool orient);
        Eigen::Quaterniond quat_from_T(Eigen::Matrix3d T);
        void forwardKinematics(Eigen::VectorXd q);
        void spaceJacobian(std::array<double, 7>& cur_q);
};