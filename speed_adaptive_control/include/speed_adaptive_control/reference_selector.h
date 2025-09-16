#include<ros/ros.h>
#include<sensor_msgs/JointState.h>
#include<std_msgs/Float32.h>
#include<std_msgs/Float32MultiArray.h>
#include<std_msgs/Int16.h>
#include<Eigen/Dense>
#include<math.h>
#include <iostream>
#include<Eigen/Dense>
#include <array>
#include<moveit_msgs/RobotTrajectory.h>


class ReferenceSelector
{
    private:
        std::array<double, 7> qr_rl = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<double, 7> qr_ompl = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        std::array<double, 7> qv;
        std::vector<std::array<double, 7>> RL_buffer;
        int RL_index = 0;
        bool got_RL = false;
        std::vector<std::array<double, 7>> OMPL_buffer;
        int OMPL_index = 0;
        bool got_OMPL = false;

        std::array<double, 7> vec_buffer;
        std::array<double, 7> vec_cur;

        double euclidean_threshold;

        double weight_o;
        double weight_r;
        
        ros::Subscriber cur_pos_sub;
        ros::Subscriber RL_sub;
        ros::Subscriber OMPL_sub;

        ros::Publisher fb_pub;
        ros::Publisher qr_pub;

        std_msgs::Float32MultiArray qr_msg;

        const double rate = 100;
        const double dt = 1 / rate;
        int rl_feedback_count = int(rate * 0.01); //10
        int run_mpc_count = 3 * rl_feedback_count;
        int reset_count = std::max(rl_feedback_count, run_mpc_count);
        std::array<std::array<double, 7>, 30> MPC_buffer;

    public:
        std::array<double, 7> cur_pos;
        std::array<double, 7> cur_velo;
        bool got_cur_pos = true;
        std::array<double, 7> qr;

        ReferenceSelector(ros::NodeHandle nh);
        ~ReferenceSelector();

        void set_params();

        void cur_pos_callback(const sensor_msgs::JointState::ConstPtr& msg);
        void RL_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        void OMPL_callback(const moveit_msgs::RobotTrajectory::ConstPtr& msg);

        int calc_cur_idx(std::vector<std::array<double, 7>>& buffer, std::array<double, 7>& query_q);
        void get_qr(std::array<double, 7>& cur_q);
        void get_qv();
        void publish_qr();
        std::array<double, 2> get_weight();

        void run();

        double min_dis;

        std::array<double, 7> normalize_angle(std::array<double, 7> vec);
};


