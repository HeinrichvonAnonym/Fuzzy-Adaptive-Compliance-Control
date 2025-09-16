#include<ros/ros.h>
#include<sensor_msgs/JointState.h>
#include<std_msgs/Float32.h>
#include<std_msgs/Float32MultiArray.h>
#include<std_msgs/Int16.h>
#include<visualization_msgs/Marker.h>
#include<visualization_msgs/MarkerArray.h>
#include<geometry_msgs/PoseArray.h>
#include<geometry_msgs/Point.h>
#include<math.h>
#include<yaml-cpp/yaml.h>
#include <iostream>
#include<Eigen/Dense>
#include <array>

class APF_Controller{ 
    public:
        APF_Controller(ros::NodeHandle nh);
        ~APF_Controller();
        // basical callbacks
        // void target_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        // void joint_state_callback(const sensor_msgs::JointState::ConstPtr& msg);
        void human_callback(const visualization_msgs::MarkerArray::ConstPtr& msg);
        // void jacobian_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        // void jacobian_6_dof_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        // void jacobian_4_dof_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        // void robot_pose_callback(const geometry_msgs::PoseArray::ConstPtr& msg);
        void att_sacle_callback(const std_msgs::Float32::ConstPtr& msg);
        // optional callbacks

        ////////////////////////////////

        //basical functions
        void run();
        std::array<double, 7> cal_att_potential();
        std::array<double, 7> cal_rep_potential();
        std::array<double, 7> cartesian_2_joint(std::array<double, 3> vec, int i);
        void update_marker(std::array<double, 3> rep_vec,
                        std::array<double, 3> inf_pos,
                        double inner_dis,
                        double influence_margin);
        // optional functions
        std::array<double, 7> cal_command(std::array<double, 7> potential);
        std::array<double, 7> angle_normalize(std::array<double, 7> vec);
        std::array<double, 7> get_vel(std::array<double, 7>& qr);
        void publish_vel(std::array<double, 7>& qv);
        void set_params();
        ////////////////////////////////

        // basical params
        // static
        double smooth_att;
        double att_base;

        double human_influence_margin;
        double human_safe_margin;
        double human_k_rep;
        double k_lamda;
        double pinv_damping;

        double inner_dis;

        std::string base_link;
        std::string end_effector;
        std::string wrist;
        std::string forearm;

        double dt = 0.01;
        
        ros::Publisher command_pub;
        ros::Publisher marker_pub;
        int num_human_link;

        std::array<double, 7> innertia;
        std_msgs::Float32MultiArray command_msg;


        // dynamic////////////////////////////////////
        std::array<double, 7> cur_pos;
        std::array<double, 7> cur_vel;
        bool got_cur_pos = true; 
        std::array<double, 7> target_pos;
        bool got_target_pos = false;
        std::array<std::array<double, 3>, 33> human_poses;
        bool got_human_poses = false;

        Eigen::MatrixXd Jv_cached;
        Eigen::MatrixXd JJt_cached;
        Eigen::MatrixXd J_pinv_cached;
        Eigen::VectorXd dx_full_cached;
        Eigen::VectorXd dq_c_cached;

        double rep_norm;
        double att_norm;
        std::array<double, 7> prev_dq;
        std::array<double, 7> K_P;
        std::array<double, 7> K_D;


        Eigen::Matrix<double, 6, 7>* jacobian;
        Eigen::Matrix<double, 6, 6>* jacobian_6_dof;
        Eigen::Matrix<double, 6, 5>* jacobian_5_dof;
        Eigen::Matrix<double, 6, 4>* jacobian_4_dof;
        geometry_msgs::PoseArray* robot_poses;

        std::array<double, 3> res_rep_vec;
        std::array<double, 3> inf_pos;
        double vec_distance;
        double max_rep_norm = 0.;

        double att_scale = 1.0;

        visualization_msgs::Marker vis_rep;
        visualization_msgs::Marker vis_thr;
        visualization_msgs::MarkerArray ma;

        std::array<double,7> att_potential{}, rep_potential{}, potential{}, command{};

        // optional params

        ///////////////////////////////

    private:
        // ros::Subscriber target_sub;
        // ros::Subscriber joint_state_sub;
        ros::Subscriber human_sub;
        // ros::Subscriber jacobian_sub;
        // ros::Subscriber jacobian_4_dof_sub;
        // ros::Subscriber jacobian_6_dof_sub;
        ros::Subscriber att_scale_sub;
        // ros::Subscriber robot_pose_sub;

        ros::AsyncSpinner spinner(uint32_t thread_num = 4);
        
};