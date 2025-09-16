#include<speed_adaptive_control/reference_selector.h>

ReferenceSelector::ReferenceSelector(ros::NodeHandle nh)
{ 
    set_params();
    cur_pos_sub = nh.subscribe("base_feedback/joint_state", 1, &ReferenceSelector::cur_pos_callback, this);
    RL_sub = nh.subscribe("/RL_controller/qr", 1, &ReferenceSelector::RL_callback, this);
    OMPL_sub = nh.subscribe("/facc/cartesian_trajectory", 1, &ReferenceSelector::OMPL_callback, this);
    qr_pub = nh.advertise<std_msgs::Float32MultiArray>("desired_q", 1);
}   

ReferenceSelector::~ReferenceSelector()
{ 

}

void ReferenceSelector::set_params()
{ 
    euclidean_threshold = 0.35;
    weight_o = 1.;
    weight_r = 0.;
}

void ReferenceSelector::cur_pos_callback(const sensor_msgs::JointState::ConstPtr& msg)
{ 
    // printf("got cur pos...\n");
    cur_pos = {msg->position[0], msg->position[1], msg->position[2], 
    msg->position[3], msg->position[4], msg->position[5], msg->position[6]};
    cur_velo = {msg->velocity[0], msg->velocity[1], msg->velocity[2], 
    msg->velocity[3], msg->velocity[4], msg->velocity[5], msg->velocity[6]};
    got_cur_pos = true;
}

void ReferenceSelector::RL_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{ 
    std::array<double, 7> q_rl;
    for(int i = 0; i < 7; i++){
        q_rl[i] = msg->data[i];
    }
    RL_buffer.push_back(qr_rl);
    bool got_RL = true;
}

void ReferenceSelector::OMPL_callback(const moveit_msgs::RobotTrajectory::ConstPtr& msg)
{ 
    std::array<double, 7> q_ompl;
    int length = msg->joint_trajectory.points.size();
    OMPL_buffer.clear();
    for (int i = 0; i < length; i++){
        for (int j = 0; j < 7; j++){
            q_ompl[j] = msg->joint_trajectory.points[i].positions[j];
        }
        OMPL_buffer.push_back(q_ompl);
    }
    got_OMPL = true;
}

std::array<double, 7> ReferenceSelector::normalize_angle(std::array<double, 7> vec)
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

int ReferenceSelector::calc_cur_idx(std::vector<std::array<double, 7>>& buffer,
    std::array<double, 7>& query_q)
{ 
    int idx = 0;
    min_dis = 1000;
    vec_cur = normalize_angle(query_q);
    double dis;
    for (int i = 0; i < buffer.size(); i++){
        dis = 0;
        // Die Abstand zwischen den aktuellen Position und der Referenzposition
        vec_buffer = normalize_angle(buffer[i]);
        for (int j = 0; j < 7; j++){
            dis += pow(vec_buffer[j] - vec_cur[j], 2);
        }
        dis = sqrt(dis);
        // Wenn der Abstand kleiner als der Min Abstand ist, wird der Index gespeichert
        if (dis < min_dis){
            min_dis = dis;
            idx = i;
        }   
        // Wenn der Abstand kleiner als der Schwellwert ist, wird der Index auf den nächsten Wert gesetzt
        while (dis < euclidean_threshold){
            if (idx >= buffer.size()-1) break;
            idx = idx+1;
            vec_buffer = normalize_angle(buffer[idx]);
            dis = 0;
            for (int j = 0; j < 7; j++){
                dis += pow(vec_buffer[j] - vec_cur[j], 2);
            }
            dis = sqrt(dis);
            min_dis = dis;
            break;
        }
    }
    // printf("min_dis: %f, idx: %d \n", min_dis, idx);
    return idx;
}

void ReferenceSelector::get_qr(std::array<double, 7>& query_q)
{ 
    // printf("getting qr... got_RL: %d, got_OMPL: %d\n", got_RL, got_OMPL);
    weight_o = 1;
    weight_r = 0;
    if (!got_RL && !got_OMPL){
        for (int i = 0; i < 7; i++){
            qr[i] = query_q[i];
        }
        return;
    }
    if (got_RL){
        RL_index = calc_cur_idx(RL_buffer, query_q); 
        qr_rl = RL_buffer[RL_index];
    }else{
        weight_r = 0.;
    }
    if (got_OMPL){
        OMPL_index = calc_cur_idx(OMPL_buffer, query_q);
        qr_ompl = OMPL_buffer[OMPL_index];
    }else{
        weight_o = 0.;
    }
    for (int i = 0; i < 7; i++){
        qr[i] = weight_r * qr_rl[i] + weight_o * qr_ompl[i];
    }
    qr = normalize_angle(qr);
}

void ReferenceSelector::publish_qr()
{
    qr_msg.data.clear();
    for (int i = 0; i < 7; i++){
        qr_msg.data.push_back(qr[i]);
    } 
    qr_pub.publish(qr_msg);
}

void ReferenceSelector::run()
{ 

}


// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "reference_selector");
//     ros::NodeHandle nh;
//     ReferenceSelector rs(nh);
//     rs.run();
//     return 0;
// }