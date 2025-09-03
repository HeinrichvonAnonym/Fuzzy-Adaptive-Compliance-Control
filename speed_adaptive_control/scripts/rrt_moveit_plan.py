import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander, RobotState
from moveit_msgs.msg import CollisionObject, RobotTrajectory
import sys
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import scipy.spatial.transform as transform
import tf
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import math
# import vector 3d
from geometry_msgs.msg import Vector3
import yaml

config_path = "/home/heinrich/franka_ws/src/speed_adaptive_control/config/dynamical_parameters_franka.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

algo_name = config["algo_name"]

link_names = config["link_names"]


critical_link_names = [link_names[4], link_names[-1]]

def quat_mul(q1, q2):
    q1_w, q1_x, q1_y, q1_z = q1
    q2_W, q2_x, q2_y, q2_z = q2
    q_w = q1_w * q2_W - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
    q_x = q1_w * q2_x + q1_x * q2_W + q1_y * q2_z - q1_z * q2_y
    q_y = q1_w * q2_y + q1_y * q2_W + q1_z * q2_x - q1_x * q2_z
    q_z = q1_w * q2_z + q1_z * q2_W + q1_x * q2_y - q1_y * q2_x
    return np.array([q_w, q_x, q_y, q_z])


class TrajectoryPlanner:
    def __init__(self):
        rospy.init_node('trajectory_planner', anonymous=True)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        # connect the moveit
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = MoveGroupCommander("robot")
        self.robot = moveit_commander.RobotCommander()
        rospy.Subscriber("cube_pose", Marker, self.target_callback)
        self.trajectory_pub = rospy.Publisher(f"/{algo_name}/cartesian_trajectory", RobotTrajectory, queue_size=10)
        self.tar_pose_pub = rospy.Publisher(f"/{algo_name}/target_pose", PoseStamped, queue_size=10)
        self.planning_scene_interface = PlanningSceneInterface("")
        self.add_ground()
        
        self.primitive_arr = None
        self.pose_arr = None
        self.dt = 0.01
        self.prev_time = rospy.Time.now()
        self.joint_state = None
        self.jacobian_msg = None
        self.arm.set_planning_time = 1.0
        
        # 设置目标点位置
        self.target_position = None
        self.target_orientation = None
        self.has_reached_target = False
        self.prev_position = None
        self.prev_orientation = None
        self.failed = False

    def add_ground(self):
        # add ground to planning scene
        ground_pose = Pose()
        ground_pose.position.z = -0.02
        ground_pose.position.x = 0.3
        ground_pose.position.y = 0.0
        ground_pose.orientation.w = 1.0
        ground_pose.orientation.x = 0.0
        ground_pose.orientation.y = 0.0
        ground_pose.orientation.z = 0.0

        ground_pose_stamped = PoseStamped()
        ground_pose_stamped.pose = ground_pose
        ground_pose_stamped.header.frame_id = "panda_link0"
        ground_pose_stamped.header.stamp = rospy.Time.now()
        self.planning_scene_interface.add_box("Schienen", ground_pose_stamped, (1.3, 0.3, 0.02))

        ground_pose_stamped.pose.position.z -= 0.05
        ground_pose_stamped.pose.position.x = 0.3
        self.planning_scene_interface.add_box("Tisch", ground_pose_stamped, (1.3, 1.5, 0.05))
    def target_callback(self, msg:Marker):
        
        self.target_position = msg.pose.position
        # w x y z
        q_rot = np.zeros(4)
        q_rot[1] = 1
        q_cube = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        q_tar = quat_mul(q_rot, q_cube)

        self.target_orientation = Quaternion()
        self.target_orientation.w = q_tar[0]
        self.target_orientation.x = q_tar[1]
        self.target_orientation.y = q_tar[2]
        self.target_orientation.z = q_tar[3]

        pose = PoseStamped()
        pose.pose.position = self.target_position
        pose.pose.orientation = self.target_orientation
        pose.header.frame_id = "panda_link0"
        pose.header.stamp = rospy.Time.now()
        self.tar_pose_pub.publish(pose)

    def joint_state_callback(self, msg):
        self.joint_state = msg

    def plan_to_target(self):
        if self.target_position is None or self.target_orientation is None:
            rospy.loginfo("No target position set")
            return False
        
        target_pose = Pose()
        target_pose.position = self.target_position
        target_pose.orientation = self.arm.get_current_pose("panda_grip_site").pose.orientation

        self.arm.set_pose_target(pose=target_pose, end_effector_link="panda_grip_site")     
        success, plan, planning_time, error_code_2 = self.arm.plan()
        
       
        if success :
            rospy.loginfo("Planned successfully")
            self.trajectory_pub.publish(plan)
            return True
        else:
            rospy.logwarn("Failed to plan to target position")
            return False   

    def check_target_changed(self):
        if self.prev_position is None or self.target_orientation is None:
            if self.target_position is not None and self.target_orientation is not None:
                self.prev_position = self.target_position
                self.prev_orientation = self.target_orientation
                return True
            else:
                return False
        position_changed =( self.prev_position.x != self.target_position.x or
                            self.prev_position.y != self.target_position.y or
                            self.prev_position.z != self.target_position.z )
        orientation_changed = (self.prev_orientation.x != self.target_orientation.x or
                               self.prev_orientation.y != self.target_orientation.y or
                               self.prev_orientation.z != self.target_orientation.z or
                               self.prev_orientation.w != self.target_orientation.w)
        self.prev_position = self.target_position
        self.prev_orientation = self.target_orientation
        return position_changed or orientation_changed
    
    def check_target_reached(self):
        if self.target_position is None or self.target_orientation is None:
            return False
        
        cur_pose = self.arm.get_current_pose("panda_grip_site").pose
        position_diff = np.linalg.norm(np.array([self.target_position.x, 
                                                 self.target_position.y, 
                                                 self.target_position.z]) 
                                                 - 
                                        np.array([cur_pose.position.x, 
                                                cur_pose.position.y, 
                                                cur_pose.position.z]))
        if position_diff > 0.01:
            return True
        else:
            return False
    
    def run(self):
        while not rospy.is_shutdown():
            cur_time = rospy.Time.now()
            if (cur_time - self.prev_time).to_sec() > 1 or self.check_target_changed():     
                if not self.check_target_reached():
                    rospy.loginfo("Target not reached")
                    # if self.check_target_changed():
                self.plan_to_target()
                self.prev_time = cur_time
            
            rospy.sleep(self.dt)

if __name__ == '__main__':
    trajectory_planner = TrajectoryPlanner()
    trajectory_planner.run()
    rospy.spin()