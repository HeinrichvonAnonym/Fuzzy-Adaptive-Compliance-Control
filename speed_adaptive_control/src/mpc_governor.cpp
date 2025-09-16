#include<speed_adaptive_control/mpc_governor.h>

MPC_Governor::MPC_Governor(ros::NodeHandle nodehandle,
                APF_Controller& apf, 
                ReferenceSelector& ref,
                JacobianCalculator& jac)
{
    nh = nh;
    ac = &apf;
    rf = &ref;
    jc = &jac;
    //fi = &finterface;
}
MPC_Governor::~MPC_Governor()
{
}

void MPC_Governor::update_kinematic()
{ 
    ac->jacobian = &jc->jacobian_7_dof;
    ac->jacobian_4_dof = &jc->jacobian_4_dof;
    ac->jacobian_5_dof = &jc->jacobian_5_dof;
    ac->jacobian_6_dof = &jc->jacobian_6_dof;
    ac->robot_poses = &jc->pose_array;
}


void MPC_Governor::forward(std::array<double, 7>& query_q)
{
    if (rf->got_cur_pos)
    {
        
        rf->get_qr(query_q);
        qr = &(rf->qr); 

        jc->spaceJacobian(*cur_pos);
        update_kinematic();

        ac->cur_pos = *cur_pos;
        ac->cur_vel = *cur_vel;
        vel_cmd = ac->get_vel(*qr);  
    }else{
        vel_cmd = {0,0,0,0,0,0,0};
    }
}

void MPC_Governor::roll_predict()
{ 
    

}

void MPC_Governor::run()
{
    // ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseArray>("/facc/robot_pose", 10);
    ros::Rate loop_rate(rate);
    ros::AsyncSpinner spinner(9);
    spinner.start();

    int print_cnt = 0;
    int roll_cnt = 0;

    virtual_q = rf->qr;

    cur_pos = &(rf->cur_pos);
    cur_vel = &(rf->cur_velo);

    while (ros::ok())
    {
        // Haupt Loop
        

        forward(*cur_pos);

        // Rolling Prediction
        // 10hz
        roll_predict();
        roll_cnt++;
        if (roll_cnt >= 10){
            roll_cnt = 0;
            virtual_q = rf->cur_pos;
        }

        ac->update_marker(ac->res_rep_vec, ac->inf_pos, ac->inner_dis, 
        ac->human_influence_margin);  
        ac->publish_vel(vel_cmd);

        if (++print_cnt % 40 == 0)
        {
            printf("rf min ids: %f, apf att: %f, apf rep: %f, num_humanlink: %d \n",
             rf->min_dis, ac->att_norm, ac->rep_norm, ac->num_human_link);
            print_cnt = 0;
        }
        loop_rate.sleep();
    }
    // fi->join();
    spinner.stop();
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "mpc_governor");
    ros::NodeHandle nh;
    APF_Controller apf_controller(nh);
    ReferenceSelector reference_selector(nh);
    JacobianCalculator jacobian_calculator(nh);
    // Franka_Interface franka_interface(nh);
    MPC_Governor mpc_governor(nh, apf_controller, 
    reference_selector, jacobian_calculator);
    mpc_governor.run();
    return 0;
}