#include <ros/ros.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <franka/exception.h>
#include <std_msgs/Float32MultiArray.h>
#include <iostream>
#include <array>
#include <atomic>
#include <thread>
#include <mutex>
#include <sensor_msgs/JointState.h>

std::array<double, 7> desired_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
std::mutex velocity_mutex;

void velocityCallback(const std_msgs::Float32MultiArray::ConstPtr& msg){
    if (msg->data.size() < 7){
        ROS_WARN("Reiceived array with less than 7 elements");
        return;
    }

    std::lock_guard<std::mutex> lock(velocity_mutex);
    for (size_t i=0; i<7; i++){   
        desired_velocity[i] = static_cast<double>(msg->data[i]);
    }
}

void control_loop(franka::Robot& robot, ros::Publisher& joint_pub) {
    try {   
        auto state = robot.readOnce();
        std::cout << "Robot state read OK. Ready for control.\n";


        robot.control(
            [&](const franka::RobotState& state,
                                franka::Duration period) -> franka::JointVelocities {

                std::lock_guard<std::mutex> lock(velocity_mutex);
                const auto& T_EE = state.O_T_EE;
                std::cout << "EE Position: " << T_EE[12] << ", " << T_EE[13] << ", " << T_EE[14] << std::endl;
                
                sensor_msgs::JointState joint_msg;
                joint_msg.header.stamp = ros::Time::now();
                joint_msg.name = {  "panda_joint1", 
                                    "panda_joint2",
                                    "panda_joint3",
                                    "panda_joint4",
                                    "panda_joint5",
                                    "panda_joint6",
                                    "panda_joint7"};
                joint_msg.position.assign(state.q.begin(), state.q.end());
                joint_msg.velocity.assign(state.dq.begin(), state.dq.end());
                joint_msg.effort.assign(state.tau_J.begin(), state.tau_J.end());

                joint_pub.publish(joint_msg);

                return franka::JointVelocities(desired_velocity);
            }
        );
    } catch (const franka::Exception& e) {
        std::cerr << "Franka Exception: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "franka_velocity_control");
    // ros::Subscriber()
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/desired_velocity", 10, velocityCallback);
    ros::Publisher js_pub = nh.advertise<sensor_msgs::JointState>("base_feedback/joint_state", 10);
    franka::Robot robot("192.168.1.11");
    // 推荐设置较低的实时线程优先级，以兼容ROS和其他任务
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
    std::thread control_thread(control_loop, std::ref(robot), std::ref(js_pub));
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();
    control_thread.join();

    return 0;

}
