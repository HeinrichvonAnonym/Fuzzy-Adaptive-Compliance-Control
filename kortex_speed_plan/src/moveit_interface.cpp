#include <ros/ros.h>
// MoveIt!
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_msgs/AllowedCollisionMatrix.h>
#include <moveit_msgs/PlanningScene.h>

#include <Eigen/Core>


class JacobianMonitor{
    private:
        
    
    public:
        ros::NodeHandle nh_;
        moveit::planning_interface::MoveGroupInterface move_group_;
        const moveit::core::RobotModelConstPtr& robot_model_;
        const moveit::core::JointModelGroup* joint_model_group_;
        JacobianMonitor() : nh_("~"), 
                            move_group_("arm"), 
                            robot_model_(move_group_.getRobotModel()), 
                            joint_model_group_(robot_model_->getJointModelGroup("arm")) {

        }

        ~JacobianMonitor() {
        }

        void timerCallback() {
            ROS_INFO_STREAM("Timer callback");
            moveit::core::RobotStatePtr current_state_ = move_group_.getCurrentState();  
            Eigen::MatrixXd jacobian;

            const std::string& link_name = "link_6";
            const moveit::core::LinkModel* link_model = robot_model_->getLinkModel(link_name);
            const Eigen::Affine3d& tool_pose = current_state_->getGlobalLinkTransform(link_name);
            Eigen::Vector3d reference_point_position = tool_pose.translation();

            
            bool success = current_state_->getJacobian(joint_model_group_, link_model, reference_point_position, jacobian);
            if (success)
            {
                std::cout << "Jacobian: " << std::endl << jacobian << std::endl;
            }
            else
            {
                ROS_ERROR("Could not compute Jacobian");
            }
            }
       
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "moveit_interface");
    ros::NodeHandle nh;
    ros::Rate loop_rate(1);
    moveit::planning_interface::MoveGroupInterface move_group_("arm");
    const moveit::core::RobotModelConstPtr& robot_model_ = move_group_.getRobotModel();
    const moveit::core::JointModelGroup* joint_model_group_ = robot_model_->getJointModelGroup("arm");
    const std::string& link_name = "bracelet_link";
    const moveit::core::LinkModel* link_model = robot_model_->getLinkModel(link_name);

    while(ros::ok())
    {
        loop_rate.sleep();
        moveit::core::RobotStatePtr current_state_ = move_group_.getCurrentState(10);  
        Eigen::MatrixXd jacobian;
        const Eigen::Affine3d& tool_pose = current_state_->getGlobalLinkTransform(link_name);
        Eigen::Vector3d reference_point_position = tool_pose.translation();

        
        bool success = current_state_->getJacobian(joint_model_group_, link_model, reference_point_position, jacobian);
        if (success)
        {
            std::cout << "Jacobian: " << std::endl << jacobian << std::endl;
        }
        else
        {
            ROS_ERROR("Could not compute Jacobian");
        }
        }

        
    }

