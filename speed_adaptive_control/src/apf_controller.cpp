// copyright (c) 2025-present Heinrich 2130238@tongji.edu.cn.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include<speed_adaptive_control/apf_controller.h>
#include<Eigen/Dense>
#include <array>

APF_Controller::APF_Controller(ros::NodeHandle nh)
{ 
    set_params();
    target_sub = nh.subscribe("desired_q", 10, &APF_Controller::target_callback, this);
    joint_state_sub = nh.subscribe("/base_feedback/joint_state", 10, &APF_Controller::joint_state_callback, this);
    human_sub = nh.subscribe("/mrk/human_skeleton", 10, &APF_Controller::human_callback, this);
    jacobian_sub = nh.subscribe("facc/jacobian", 10, &APF_Controller::jacobian_callback, this);
    jacobian_4_dof_sub = nh.subscribe("facc/jacobian_4_dof", 10, &APF_Controller::jacobian_4_dof_callback, this);
    jacobian_6_dof_sub = nh.subscribe("facc/jacobian_6_dof", 10, &APF_Controller::jacobian_6_dof_callback, this);
    att_scale_sub = nh.subscribe("facc/att_scale", 10, &APF_Controller::att_sacle_callback, this);
    robot_pose_sub = nh.subscribe("/facc/robot_pose", 10, &APF_Controller::robot_pose_callback, this);
    // command_pub = nh.advertise<std_msgs::Float32MultiArray>("/facc/pid_command", 10);
    command_pub = nh.advertise<std_msgs::Float32MultiArray>("/desired_velocity", 10);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/facc/rep_vec_arr", 10);
}

APF_Controller::~APF_Controller()
{
}

void APF_Controller::set_params()
{
    smooth_att = 0.15;
    att_base = 18000; 

    human_safe_margin = 0.05;
    human_influence_margin = 0.4;
    human_k_rep = 12000;
    k_lamda = 800;
    inner_dis = 0.1;
    pinv_damping = 0.05;

    innertia = {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33,};
    prev_dq = {0., 0., 0., 0., 0., 0., 0.};
    K_P = {0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2};
    K_D = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00};


    base_link = "panda_link0";
    end_effector = "panda_grip_site";
    wrist = "panda_link6";
    forearm = "panda_link4";

    vis_rep.header.frame_id = base_link;
    vis_rep.ns = "potential_rep";
    vis_rep.id = 0;
    vis_rep.type = vis_rep.ARROW;
    vis_rep.action = vis_rep.ADD;
    vis_rep.scale.x = 0.01;
    vis_rep.scale.y = 0.01;
    vis_rep.scale.z = 0.1;
    vis_rep.color.r = 1.0;
    vis_rep.color.g = 0.0;
    vis_rep.color.b = 0.0;
    vis_rep.color.a = 1.0;
    vis_rep.pose.orientation.x = 0;
    vis_rep.pose.orientation.y = 0;
    vis_rep.pose.orientation.z = 0;
    vis_rep.pose.orientation.w = 1;


    vis_thr.header.frame_id = base_link;
    vis_thr.ns = "threshold";
    vis_thr.id = 1;
    vis_thr.type = vis_thr.SPHERE;
    vis_thr.action = vis_thr.ADD;
    vis_thr.color.r = 0.3;
    vis_thr.color.g = 0.3;
    vis_thr.color.b = 0.3;
    vis_thr.color.a = 0.3;
    vis_thr.pose.orientation.x = 0;
    vis_thr.pose.orientation.y = 0;
    vis_thr.pose.orientation.z = 0;
    vis_thr.pose.orientation.w = 1;
    
}

void APF_Controller::target_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    for (int i = 0; i < 7; i++)
    {
        target_pos[i] = msg->data[i];
    }
    if(!got_target_pos){
        got_target_pos = true;
    }
}

void APF_Controller::joint_state_callback(const sensor_msgs::JointState::ConstPtr& msg)
{
    // printf(">>");
    for (int i = 0; i < 7; i++)
    {
        cur_pos[i] = msg->position[i];
        cur_vel[i] = msg->velocity[i];
    } 
    if (!got_cur_pos){
        got_cur_pos = true;
    }
}

void APF_Controller::human_callback(const visualization_msgs::MarkerArray::ConstPtr& msg)
{
    size_t n = sizeof(msg->markers);
    num_human_link = static_cast<int>(n);
    for (int i = 0; i < num_human_link; i++)
    {
        human_poses[i][0] = msg->markers[i].pose.position.x;
        human_poses[i][1] = msg->markers[i].pose.position.y;
        human_poses[i][2] = msg->markers[i].pose.position.z;
    } 

    got_human_poses = true;
}

void APF_Controller::jacobian_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    Eigen::Map<const Eigen::Matrix<float,6,7,Eigen::RowMajor>> map(msg->data.data());
    jacobian = map.cast<double>();
    got_jacobian = true;
}

void APF_Controller::jacobian_6_dof_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    // printf("6");
    Eigen::Map<const Eigen::Matrix<float,6,6,Eigen::RowMajor>> map(msg->data.data());
    jacobian_6_dof = map.cast<double>();
    got_jacobian_6_dof = true;
}

void APF_Controller::jacobian_4_dof_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    Eigen::Map<const Eigen::Matrix<float,6,4,Eigen::RowMajor>> map(msg->data.data());
    jacobian_4_dof = map.cast<double>();
    got_jacobian_4_dof = true;
}

void APF_Controller::robot_pose_callback(const geometry_msgs::PoseArray::ConstPtr& msg)
{
    for (int i = 0; i < 3; i++)
    {
        robot_poses[i][0] = msg->poses[i].position.x;
        robot_poses[i][1] = msg->poses[i].position.y;
        robot_poses[i][2] = msg->poses[i].position.z;
    }
    got_robot_poses = true;
}

void APF_Controller::att_sacle_callback(const std_msgs::Float32::ConstPtr& msg)
{
    att_scale = msg->data;
}

std::array<double, 7> APF_Controller::angle_normalize(std::array<double, 7> vec)
{
    for (int i = 0; i < 7; i++)
    {
        vec[i] = fmod(vec[i], 2 * M_PI);
        if (vec[i] > M_PI)
        {
            vec[i] -= 2 * M_PI;
        }
        else if (vec[i] < -M_PI)
        {
            vec[i] += 2 * M_PI;
        }
    }
    return vec;
}

std::array<double, 7> APF_Controller::cal_att_potential()
{ 
    std::array<double, 7> cur_pos_norm = angle_normalize(cur_pos);
    std::array<double, 7> target_pos_norm = angle_normalize(target_pos);
    std::array<double, 7> pos_dis;
    for (int i = 0; i < 7; i++)
    {
        pos_dis[i] = target_pos_norm[i] - cur_pos_norm[i];
    }
    pos_dis = angle_normalize(pos_dis);

    double euclidean_dis = sqrt(pow(pos_dis[0], 2) + 
                        pow(pos_dis[1], 2) + 
                        pow(pos_dis[2], 2) + 
                        pow(pos_dis[3], 2) + 
                        pow(pos_dis[4], 2) + 
                        pow(pos_dis[5], 2) + 
                        pow(pos_dis[6], 2));
    double scale = (att_base * att_scale) / std::max(euclidean_dis, smooth_att);
    att_norm = scale * euclidean_dis;
    std::array<double, 7> att_potential;
    for (int i = 0; i < 7; i++)
    {
        att_potential[i] = scale * pos_dis[i];
    }
    return att_potential;
}

std::array<double, 7> APF_Controller::cartesian_2_joint(std::array<double, 3> vec, int link_category)
{
    std::array<double, 7> joint_vec {0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    switch(link_category){
        case 0: Jv_cached = jacobian.cast<double>(); break;       // 6x7
        case 1: Jv_cached = jacobian_6_dof.cast<double>(); break; // 6x6
        case 2: Jv_cached = jacobian_4_dof.cast<double>(); break; // 6x4
        default: ROS_WARN("Invalid link category"); return joint_vec;
    }

    int n_rows = Jv_cached.rows();
    int n_joints = Jv_cached.cols();

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_rows, n_rows);

    dx_full_cached.setZero(n_rows);
    dx_full_cached.head(3) << vec[0], vec[1], vec[2];

    JJt_cached.noalias() = Jv_cached * Jv_cached.transpose();
    JJt_cached += I * pinv_damping * pinv_damping;

    J_pinv_cached.noalias() = Jv_cached.transpose() * JJt_cached.ldlt().solve(I);

    dq_c_cached.noalias() = J_pinv_cached * dx_full_cached;

    for(int i=0;i<n_joints;i++){
        joint_vec[i] = dq_c_cached[i];
    }

    return joint_vec;
}

void APF_Controller::update_marker(std::array<double, 3> rep_vec,
                        std::array<double, 3> inf_pos,
                        double inner_dis,
                        double influence_margin)
{ 
    geometry_msgs::Point start, end;
    start.x = inf_pos[0];
    start.y = inf_pos[1];
    start.z = inf_pos[2];

    end.x = inf_pos[0] + rep_vec[0] / human_k_rep;
    end.y = inf_pos[1] + rep_vec[1] / human_k_rep;
    end.z = inf_pos[2] + rep_vec[2] / human_k_rep;
    vis_rep.points.clear();
    vis_rep.points.push_back(start); vis_rep.points.push_back(end);
    vis_thr.pose.position = start;

    vis_thr.scale.x = inner_dis + influence_margin;
    vis_thr.scale.y = inner_dis + influence_margin;
    vis_thr.scale.z = inner_dis + influence_margin;

    ros::Time now = ros::Time::now();
    vis_rep.header.stamp = now;
    vis_thr.header.stamp = now;
    ma.markers.clear();
    ma.markers.push_back(vis_rep);
    ma.markers.push_back(vis_thr);

    marker_pub.publish(ma);
}

std::array<double, 7> APF_Controller::cal_rep_potential()
{
    std::array<double, 3> res_rep_vec;
    std::array<double, 3> inf_pos;
    double max_rep_norm = 0.0;
    double vec_distance;
    int selected_robot_link;
    int selected_human_link;
    int selected_obj_link;
    // query
    for (int i = 0; i < 3; i++)
    {
        std::array<double, 3> robot_pos = robot_poses[i];

        // human
        for (int j = 0; j < num_human_link; j++)
        {
            std::array<double, 3> human_pos = human_poses[j];
            std::array<double, 3> rep_vec;
            for (int k = 0; k < 3; k++)
            {
                rep_vec[k] = robot_pos[k] - human_pos[k];
            }
            double dis = sqrt(pow(rep_vec[0], 2) + pow(rep_vec[1], 2) + pow(rep_vec[2], 2));
            dis = std::max(dis - inner_dis, inner_dis + human_safe_margin);
            double rep_norm = std::max(human_influence_margin - dis, 0.0) / std::max((human_influence_margin - human_safe_margin), 0.01);
            if (rep_norm > max_rep_norm)
            {
                max_rep_norm = rep_norm;
                res_rep_vec = rep_vec;
                vec_distance = dis;
                inf_pos = robot_pos;
                selected_robot_link = i;
                selected_human_link = j;
            }
        }
    }
    if (max_rep_norm > 0)
    {
        rep_norm = max_rep_norm * human_k_rep / std::max(vec_distance, 0.01);
        for (int i = 0; i < 3; i++)
        {
            res_rep_vec[i] = rep_norm * res_rep_vec[i];
        }

        //visualize the repulsion
        update_marker(res_rep_vec, inf_pos, inner_dis, human_influence_margin);
 
        std::array<double, 7> rep_potential = cartesian_2_joint(res_rep_vec, selected_robot_link);
        return rep_potential;
    }else{
        return std::array<double, 7> {0., 0., 0., 0., 0., 0., 0.};
    }
}

std::array<double, 7> APF_Controller::cal_command(std::array<double, 7> potential)
{
    std::array<double, 7> command;
    std::array<double, 7> dq;
    std::array<double, 7> ddq;
    for (int i = 0; i < 7; i++)
    {
        // potential[i] = potential[i] / innertia[i];
        double damping = k_lamda * cur_vel[i];
        potential[i] -= damping;
        // command[i] = cur_pos[i] + dt * (cur_vel[i] + potential[i] * dt) / 2; 
        dq[i] = dt * (cur_vel[i] + potential[i] * dt) / 2;    
    }
    dq = angle_normalize(dq);
    for (int i = 0; i < 7; i++){
        ddq[i] = dq[i] - prev_dq[i];
        prev_dq[i] = dq[i];
        command[i] = K_P[i] * dq[i] + K_D[i] * ddq[i];
    }
    return command;
}

void APF_Controller::run()
{
    ros::AsyncSpinner spinner(9);  // 异步处理回调
    spinner.start();
    ros::Rate loop_rate(100);

    std::array<double,7> att_potential{}, rep_potential{}, potential{}, command{};

    int print_cnt = 0;

    while(ros::ok())
    {
        if(got_cur_pos && got_target_pos){
            att_potential = cal_att_potential();
        } else {
            att_potential.fill(0.0);
            att_norm = 0.;
        }

        if(got_human_poses && got_robot_poses){
            rep_potential = cal_rep_potential();
        } else {
            rep_potential.fill(0.0);
            rep_norm = 0.;
        }
        for(int i=0;i<7;i++){
            potential[i] = att_potential[i] + rep_potential[i];
        }

        if(got_cur_pos){
            command = cal_command(potential);
        } else {
            command.fill(0.0);
        }

        command_msg.data.clear();
        command_msg.data.assign(command.begin(), command.end());
        command_pub.publish(command_msg);

        loop_rate.sleep();

        if (++print_cnt % 20 == 0){
            printf("> att: %f, rep: %f \n", att_norm, rep_norm);
            print_cnt= 0;
        }
    }

    spinner.stop();
}

int main(int argc, char **argv)
{ 
    ros::init(argc, argv, "apf_controller");
    ros::NodeHandle nh;
    APF_Controller apf_controller = APF_Controller(nh);
    ROS_INFO("Starting APF controller node...\n");
    apf_controller.run();
}