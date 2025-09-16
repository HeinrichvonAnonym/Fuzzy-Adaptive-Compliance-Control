#include<speed_adaptive_control/jacobian_calculator.h>


JacobianCalculator::JacobianCalculator(ros::NodeHandle nh)
{ 
}

JacobianCalculator::~JacobianCalculator()
{

}

Eigen::Matrix4d JacobianCalculator::dh_2_T(double theta, double theta_offset, 
double d, double a, double alpha, bool orient)
{
    theta = theta + theta_offset;
    if (!orient) theta = -theta;
    double ct = cos(theta);
    double st = sin(theta);
    double ca = cos(alpha);
    double sa = sin(alpha);
    T_dh <<
        ct,       -st,       0,      a,
        st*ca,    ct*ca,    -sa,    -d*sa,
        st*sa,    ct*sa,     ca,     d*ca,
        0,        0,         0,      1;
    return T_dh;
}

Eigen::Quaterniond JacobianCalculator::quat_from_T(Eigen::Matrix3d T)
{ 
    // shouldI = T.transpose() * T;
    // double err = (shouldI - identity3d).norm();
    // if(err>1e-6){
    //     Eigen::JacobiSVD<Eigen::Matrix3d> svd(T, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //     U = svd.matrixU();
    //     V = svd.matrixV();
    //     T = U * V.transpose();
    //     if(T.determinant() < 0){
    //         U.col(2) *= -1;
    //         T = U * V.transpose();
    //     }
    // }

    Eigen::Quaterniond quat(T);
    quat.normalize();
    return quat;
}

void JacobianCalculator::forwardKinematics(Eigen::VectorXd q)
{
    pose_array.poses.clear();
    pose_array.header.frame_id = "panda_link0";
    pose_array.header.stamp = ros::Time::now();
    T_dof = identity4d;
    

    for(int i = 0; i < 7; i++)
    {
        geometry_msgs::Pose pose;
        //printf("forwarding %d, \n", i);
        T_dof = T_dof * dh_2_T(q(i), dh_parmas[i][0], dh_parmas[i][1], dh_parmas[i][2], dh_parmas[i][3], joint_orient[i]);
        T_dofs[i] = T_dof;
        
        quat = quat_from_T(T_dof.block<3, 3>(0, 0));
        Z_end = T_dof.block<3, 1>(0, 3);
        pose.position.x = Z_end(0);
        pose.position.y = Z_end(1);
        pose.position.z = Z_end(2);
        pose.orientation.w = quat.w();
        pose.orientation.x = quat.x();
        pose.orientation.y = quat.y();
        pose.orientation.z = quat.z();

        pose_array.poses.push_back(pose);
    }
}

void JacobianCalculator::spaceJacobian(std::array<double, 7>& cur_q)
{ 
    
    for (int i = 0; i < 7; i++)
    {
        q(i) = cur_q[i];
    }
    forwardKinematics(q);
    int dof, i;
    jacobian.setZero();
    for (dof = 0; dof < 7; dof++)
    {
        // printf("jacobian %d, \n", dof);
        T_end = T_dofs[dof];
        O_end = T_end.block<3, 1>(0, 3);
        jacobian.setZero();
        for(i = 0; i < dof; i++){
            //printf("sub jacobian %d, \n", i);
            R_i = T_dofs[i].block<3, 3>(0, 0);
            Z_i = R_i.col(2);
            O_i = T_dofs[i].block<3, 1>(0, 3);
            jacobian.block<3, 1>(0, i) = Z_i.cross(O_end - O_i);
            jacobian.block<3, 1>(3, i) = R_i.transpose() * (O_end - O_i);
        }
    }
    //printf("get jac 4 \n");
    jacobian_4_dof = jacobian.block<6, 4>(0, 0);

    jacobian_5_dof = jacobian.block<6, 5>(0, 0);
    ///printf("get jac 6 \n");
    jacobian_6_dof = jacobian.block<6, 6>(0, 0);
    //printf("get jac 7 \n");
    jacobian_7_dof = jacobian.block<6, 7>(0, 0);
}

Eigen::VectorXd q;
bool get_q = false;

// void js_callback(const sensor_msgs::JointState::ConstPtr& msg)
// {
//     q.resize(7);
//     for(int i = 0; i < 7; i++)
//     {
//         q(i) = msg->position[i];
//     }
//     get_q = true;
// }

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "jacobian_calculator");
//     ros::NodeHandle nh;
//     JacobianCalculator jc(nh);
//     geometry_msgs::PoseArray pose_array;
//     ros::Subscriber sub = nh.subscribe("joint_states", 1, &js_callback);
//     ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("/facc/robot_pose", 10);
//     ros::Rate rate(100);
//     while(ros::ok()){
//         rate.sleep();
//         ros::spinOnce();
//         if(! get_q) continue;
//         jc.spaceJacobian(q);
//         pose_array.poses.clear();
//         // pose_array.poses.push_back(jc.pose_array.poses[0]);
//         // pose_array.poses.push_back(jc.pose_array.poses[1]);
//         // pose_array.poses.push_back(jc.pose_array.poses[2]);
//         // pose_array.poses.push_back(jc.pose_array.poses[3]);
//         // pose_array.poses.push_back(jc.pose_array.poses[4]);
//         // pose_array.poses.push_back(jc.pose_array.poses[5]);
//         // pose_array.poses.push_back(jc.pose_array.poses[6]);
//         pose_array.header.stamp = ros::Time::now();
//         pose_array.header.frame_id = "panda_link0";
//         pub.publish(jc.pose_array);
//     }
    
//     return 0;
// }