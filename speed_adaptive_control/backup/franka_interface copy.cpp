#include<speed_adaptive_control/franka_interface.h>
#include<franka/robot.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot_state.h>
#include <franka/exception.h>


Franka_Interface::Franka_Interface(ros::NodeHandle nh)
:robot("172.16.0.2"), gripper("172.16.0.2")
{
    ee_pub = nh.advertise<sensor_msgs::JointState>("/base_feedback/ee_state", 10);
    js_pub = nh.advertise<sensor_msgs::JointState>("/base_feedback/joint_state", 10);
    set_params();
}

Franka_Interface::~Franka_Interface()
{ 
}

void Franka_Interface::set_params()
{
    k_stiffness = {50, 50, 50, 40, 40, 40, 40};
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
    dq_step_limit = 0.01;

    joint_msg.name = {  "panda_joint1", "panda_joint2", "panda_joint3",
                    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7" };
}

void Franka_Interface::set_velo(std::array<double, 7>& desired_velo)
{
    std::lock_guard<std::mutex> lock(velocity_mutex);
    for (size_t i = 0; i < 7; i++){
        desired_velocity[i] = desired_velocity[i]*0.95 + desired_velo[i]*0.05;
        desired_velocity[i] = std::max(std::min(desired_velocity[i], 0.2), -0.2);
    } 
}

void Franka_Interface::control_loop(franka::Robot& robot)
{
    ros::Rate r = ros::Rate(10);
    
    // std::array<double, 7> q_d = initial_state.q;
    franka::Model model = robot.loadModel();


    f_lp = Eigen::Vector3d::Zero();

    while(ros::ok())
    {
        try {   
            franka::RobotState init_state = robot.readOnce();
            std::array<double, 7> q_cmd = init_state.q; 
            std::array<double, 7> q_offset = {{0., 0., 0., 0., 0., 0., 0.}};
            // franka::RobotState prev_state = robot.readOnce();
            // std::array<double, 7> prev_q = init_state.q; 
            double stiffness_scale = 1.0;
            std::cout << "Robot state read OK. Ready for control.\n";
            robot.control(
                [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques {

                    // const auto& T_EE = state.O_T_EE;
                    
                    // std_msgs::Float32MultiArray eef_pos;
                    if (!ros::ok()){
                        throw franka::Exception("ROS is shutting down, exiting control loop");
                    }

                    const double dt = period.toSec();

                    joint_msg.header.stamp = ros::Time::now();
                    
                    joint_msg.position.assign(state.q.begin(), state.q.end());
                    joint_msg.velocity.assign(state.dq.begin(), state.dq.end());
                    joint_msg.effort.assign(state.tau_J.begin(), state.tau_J.end());
                    js_pub.publish(joint_msg);

                    for (int i = 0; i < 7; i++)
                    {
                        cur_q[i] = state.q[i];
                        cur_vel[i] = state.dq[i];
                    }
                    
                    // Kraft
                    Eigen::Vector3d f_now(  state.O_F_ext_hat_K[0],
                                            state.O_F_ext_hat_K[1],
                                            state.O_F_ext_hat_K[2]);
                    // printf("force: %f, %f, %f, \n", state.O_F_ext_hat_K[0],state.O_F_ext_hat_K[1],state.O_F_ext_hat_K[2]);
                    f_lp = (1.0 - lp_alpha) * f_lp + lp_alpha * f_now;
                    const double F = f_lp.norm();
                    //Jacobian
                    jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
                    Eigen::Map<const Eigen::Matrix<double, 6, 7>> J_full(jacobian_array.data());
                    Jv = J_full.topRows<3>(); // linear velo

                 
                    if(F > force_threshold_N){
                        // linear movement
                        dx = comp_vel_gain * f_lp * dt;

                        // DLS pinv
                        JJt = Jv * Jv.transpose();
                        JJt += (pinv_damping * pinv_damping) * I3d;
                        J_pinv = Jv.transpose() * JJt.ldlt().solve(I3d);

                        //Compliance control
                        dq_c = J_pinv * dx;   

                        for (size_t i = 0; i < 7; i++) {
                            double step = std::max(std::min(dq_c[i], dq_step_limit), - dq_step_limit);
                            // q_cmd = state.q;
                    
                            if (q_offset[i] > max_qoffset_abs) q_offset[i] = max_qoffset_abs;
                            if (q_offset[i] < -max_qoffset_abs) q_offset[i] = -max_qoffset_abs;

                            q_cmd[i] = state.q[i];
                            q_cmd[i] += q_offset[i];
                            q_cmd[i] = std::min(std::max(q_cmd[i], joint_min[i] + 0.05), joint_max[i] - 0.05);
                        }
                                 
                    }else{
                        q_offset = {{0., 0., 0., 0., 0., 0., 0.}};
                        // stiffness_scale = 1;
                        std::lock_guard<std::mutex> lock(velocity_mutex);
                        for (size_t i = 0; i < 7; i++) {
                            // q_cmd[i] = state.q[i];
                            q_cmd[i] += desired_velocity[i] * dt; // Differenz 
                        }
                    }  

                    for (size_t i = 0; i < 7; i++) {                
                            q_cmd[i] = std::min(std::max(q_cmd[i], joint_min[i] + 0.05), joint_max[i] - 0.05);
                        }             
                      
                    // Rechnung des Kraftmoment
                    std::array<double, 7> tau_d;
                    for (size_t i = 0; i < 7; i++) {
                        // double kd_scales = k_damping[i] * stiffness_scale;
                        // double k_stiffness_scales = k_stiffness[i] * stiffness_scale;
                        double q_err = state.q[i] - q_cmd[i];
                        tau_d[i] = - k_stiffness[i] * q_err - k_damping[i] * state.dq[i];
                    }

                    // Ausgleichung des Gewichtes                    
                    // std::array<double, 7> gravity = model.gravity(state);
                    // for (size_t i = 0; i < 7; i++) {
                    //     tau_d[i] += gravity[i];
                    // }

                    return franka::Torques(tau_d);
                }
            );
        } catch (const franka::Exception& e) {
            std::cerr << "Franka Exception: " << e.what() << std::endl;
        }
        desired_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
        auto state = robot.readOnce();
        joint_msg.position.assign(state.q.begin(), state.q.end());
        joint_msg.velocity.assign(state.dq.begin(), state.dq.end());
        joint_msg.effort.assign(state.tau_J.begin(), state.tau_J.end());
        // eef_pos.data = {T_EE[12], T_EE[13], T_EE[14]};
        js_pub.publish(joint_msg);
     
        r.sleep();
    }
}

void Franka_Interface::gripper_control_loop(franka::Gripper& gripper)
{

}


void Franka_Interface::start()
{
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
    control_thread = std::thread(&Franka_Interface::control_loop, this, 
                                std::ref(robot));
    gripper_thread = std::thread(&Franka_Interface::gripper_control_loop, this,
                                std::ref(gripper));
}

void Franka_Interface::join()
{
    control_thread.join();
    gripper_thread.join();
}

// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "franka_interface");
//     ros::NodeHandle nh;
//     Franka_Interface fi(nh); 
// }