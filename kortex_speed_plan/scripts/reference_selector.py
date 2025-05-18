import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.msg import CollisionObject, RobotTrajectory
from geometry_msgs.msg import Pose, PoseArray
import sys
import numpy as np
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from std_msgs.msg import Float32, Int16, Float32MultiArray
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose
import scipy.spatial.transform as transform
import tf
from tf.transformations import quaternion_from_matrix
import yaml

config_path = "/home/heinrich/kinova/src/kortex_speed_plan/config/dynamical_parameters.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

algo_name = config["algo_name"]

USE_FUZZY = False
INDEX_OFFSET = 2

# theta d a alpha
# dh_params = np.array([
#     [0,           -(0.1564 + 0.1284),       0,          np.pi],
#     [0,             0,                      0,          np.pi / 2], 
#     [0,           -(0.2104 + 0.2104),       0,         -np.pi / 2],
#     [0,             0,                      0,          np.pi / 2],
#     [0,           -(0.2084 + 0.1059),       0,         -np.pi / 2],
#     [0,             0,                      0,          np.pi / 2],
#     [0,           -(0.1059 + 0.18),         0,         -np.pi / 2],
#     [0,            -0.18,                     0.,         np.pi],     # TOOL
# ])
dh_params = config["dh_params"]

def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct,       -st,       0,      a],
                     [st*ca,    ct*ca,    -sa,    -d*sa],
                     [st*sa,    ct*sa,     ca,     d*ca],
                     [0,        0,         0,      1]])

def compute_forward_kinematics(joint_angles):
    T = np.identity(4)
    for i in range(7):
        # print(i)
        theta_offset, d, a, alpha = dh_params[i]
        theta = joint_angles[i] + theta_offset
        A_i = dh_matrix(theta, d, a, alpha)
        T = np.dot(T, A_i)
        # print(T)
    theta_offset, d, a, alpha = dh_params[-1]
    theta = 0
    A_i = dh_matrix(theta, d, a, alpha)
    T = np.dot(T, A_i)

    return T

def cac_pose(angle):
    # print(angle)
    joint_angles = angle[:7]
    T = compute_forward_kinematics(joint_angles)

    pos = T[:3, 3]
    quat = quaternion_from_matrix(T)
    return pos, quat



def ref_fuzzy_slot(ri_weight, target_pos:list, pot_rep:list):
    if ri_weight is None:
        ri_weight = 0
    pot_rep = np.array(pot_rep)
    target_pos = np.array(target_pos)
    pot_rep     *= 0.00001
    ri_weight   =  ( ri_weight + 1 ) / 2
    target_pos  += ri_weight * pot_rep
    return list(target_pos)


class ReferenceSelector:
    def __init__(self):
        rospy.init_node("reference_selector", anonymous=True)
        rospy.Subscriber(f"/{algo_name}/cartesian_trajectory", RobotTrajectory, self.trajectory_callback)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        rospy.Subscriber(f"/{algo_name}/pot_rep", Float32MultiArray, self.pot_rep_callback)
        rospy.Subscriber(f"/{algo_name}/ri_weight", Float32, self.ri_callback)
        rospy.Subscriber(f"/{algo_name}/robot_pose", PoseArray, self.pose_callback)
        rospy.Subscriber(f"/{algo_name}/work_damping", Float32, self.work_damping_callback)
        self.target_pos_command = rospy.Publisher(f"/{algo_name}/target_pos", JointState, queue_size=10)
        # self.prev_target_pos_command = rospy.Publisher(f"{algo_name}/prev_target_pos", JointState, queue_size=10)
        self.refresh_pub = rospy.Publisher(f"{algo_name}/refresh", Int16, queue_size=10)
        
        # 添加参考选择器位置发布器
        self.reference_selector_pub = rospy.Publisher(f"/{algo_name}/reference_selector_pos", Float32MultiArray, queue_size=10)
        self.prev_reference_selector_pub = rospy.Publisher(f"/{algo_name}/prev_reference_selector_pos", Float32MultiArray, queue_size=10)
        self.arrived_pub = rospy.Publisher(f"/{algo_name}/arrived", String, queue_size=10)
        
        self.joint_state = None
        self.trajectory = None
        self.exe_index = 0
        self.threshold = 0.18
        self.working_threshold = 0.06
        self.last_threshold = 0.005
        self.working_last_threshold = 0.001
        self.refresh_trajectory = False
        self.joint_trajectory = None
        self.eef_pose = None
        
        # 添加当前选择的参考点
        self.current_reference_point = None
        self.pot_rep = None
        self.ri_weight = None
        self.work_damping = 0.

    def work_damping_callback(self, msg:Float32):
        self.work_damping = msg.data
    
    def pot_rep_callback(self, msg:Float32MultiArray):
        self.pot_rep = msg.data
    
    def ri_callback(self, msg:Float32):
        self.ri_weight = msg.data
    
    def pose_callback(self, msg:PoseArray):
        self.eef_pose = msg.poses[0]
    
    # def compute_new_trajectory_index(self, trajectory:RobotTrajectory):
    #     cur_pos = self.joint_state.position[:7]
    #     min_dis = 1e20
    #     idx = 1
    #     for i, point in enumerate(trajectory.joint_trajectory.points):
    #         pos_dis = np.linalg.norm(np.array(point.positions) - np.array(cur_pos))
    #         if i < len(trajectory.joint_trajectory.points) - 1:
    #             pos_dis_next = np.linalg.norm(np.array(trajectory.joint_trajectory.points[i+1].positions) - np.array(cur_pos))
    #             if pos_dis_next < min_dis:
    #                 min_dis = pos_dis_next
    #                 idx = i
    #     return idx     

    def compute_new_trajectory_index(self, trajectory:RobotTrajectory):
        idx = 0
        min_dis = 1e10
        for i, point in enumerate(trajectory.joint_trajectory.points):
            waypoint = point.positions
            pos, quat = cac_pose(waypoint)
            # print(pose)
            position = np.array(pos)
            # print(self.eef_pose.position)
            eef_position = np.array([self.eef_pose.position.x,
                                     self.eef_pose.position.y,
                                     self.eef_pose.position.z])
            position_dis = np.linalg.norm(eef_position - position)
            if position_dis < min_dis:
                min_dis = position_dis
                idx = i
        
        if self.work_damping < 0.5:
            if idx > 0 and idx < 0.5 * INDEX_OFFSET:
                idx += idx
            if idx > 0.5 * INDEX_OFFSET:
                idx += min(int(2 * idx), 8)
            idx = min(max(len(trajectory.joint_trajectory.points) - 3, 0), idx) 
        return idx       
    
    def check_collision(self, target_pos):
        return False
    
    def send_target_command(self, target_pos, prev_target_pos = None):
        tar_pos = JointState()
        tar_pos.name = self.trajectory.joint_trajectory.joint_names
        for i in range(len(target_pos)):
            tar_pos.position.append(target_pos[i]) 
            tar_pos.velocity.append(0)
            tar_pos.effort.append(0)
        tar_pos.header.stamp = rospy.Time.now()
        self.target_pos_command.publish(tar_pos)
        
        # 更新并发布当前参考点
        self.current_reference_point = target_pos
        reference_msg = Float32MultiArray()
        reference_msg.data = list(target_pos)
        self.reference_selector_pub.publish(reference_msg)

        if prev_target_pos is None:
            return
        reference_msg = Float32MultiArray()
        reference_msg.data = list(prev_target_pos)
        self.prev_reference_selector_pub.publish(reference_msg)

    def send_prev_target_command(self, target_pos):
        tar_pos = JointState()
        tar_pos.name = self.trajectory.joint_trajectory.joint_names
        for i in range(len(target_pos)):
            tar_pos.position.append(target_pos[i]) 
            tar_pos.velocity.append(0)
            tar_pos.effort.append(0)
        tar_pos.header.stamp = rospy.Time.now()
        self.target_pos_command.publish(tar_pos)
        
        # 更新并发布当前参考点
        self.current_reference_point = target_pos
        reference_msg = Float32MultiArray()
        reference_msg.data = list(target_pos)
        self.reference_selector_pub.publish(reference_msg)

    
    def trajectory_callback(self, msg: RobotTrajectory):
        print("Received trajectory")
        
        if self.trajectory is None:
            self.trajectory = msg
        else:
            self.refresh_trajectory = True
            self.trajectory = msg
            self.exe_index = self.compute_new_trajectory_index(msg)
            print(f"selected_start_idx: {self.exe_index}")
        
    
    def joint_state_callback(self, msg):
        self.joint_state = msg

    def distance_normalize(self, pos_dis):
        # 角度归一化
        for i in range(len(pos_dis)):
            pos_dis[i] = pos_dis[i] % (2 * np.pi)
            if pos_dis[i] > np.pi:
                pos_dis[i] = pos_dis[i] % np.pi - np.pi
            elif pos_dis[i] < -1 * np.pi:
                pos_dis[i] = -pos_dis[i] % np.pi + np.pi
        return pos_dis
    
    def command(self):
        target_pos = list(self.trajectory.joint_trajectory.points[self.exe_index].positions)
        cur_pos = list(self.joint_state.position[:7])
        if self.work_damping > 0.5:
            target_pos[-1] = cur_pos[-1]
        # self.send_target_command(target_pos)

        distance_vec = np.array(target_pos) - np.array(cur_pos)
        distance_vec = self.distance_normalize(distance_vec)
        distance = np.linalg.norm(distance_vec)

        if self.work_damping > 0.1:
            threshold = self.working_threshold - (self.working_threshold - self.working_last_threshold) * (self.exe_index / len(self.trajectory.joint_trajectory.points)) 
            # threshold = self.threshold
        #jubuqr

        else:
            threshold = self.working_threshold - (self.working_threshold - self.working_last_threshold) * (self.exe_index / len(self.trajectory.joint_trajectory.points)) ** 2
            threshold = self.threshold
        if self.exe_index == len(self.trajectory.joint_trajectory.points) - 1:
            threshold = self.last_threshold
            if self.work_damping > 0.5:
                threshold = self.working_last_threshold
            

    
        print(f"cur_idx: {self.exe_index}; threshold: {threshold}")
        while(distance > threshold and not rospy.is_shutdown()):
            if self.refresh_trajectory:
                self.refresh_trajectory = False
                return
            coli = self.check_collision(target_pos)
            cur_pos = self.joint_state.position[:7]

            if coli:
                self.exe_index -= 1
                target_pos = self.trajectory.joint_trajectory.points[self.exe_index].positions
                if USE_FUZZY:
                    target_pos_local = ref_fuzzy_slot(ri_weight=self.ri_weight, target_pos=target_pos, pot_rep=self.pot_rep)
                else:
                    target_pos_local = target_pos
                
                # if self.work_damping > 0.5:
                #     target_pos_local[-1] = cur_pos[-1]
                self.send_target_command(target_pos_local, target_pos)
                return
            else:
                if USE_FUZZY:
                    target_pos_local = ref_fuzzy_slot(ri_weight=self.ri_weight, target_pos=target_pos, pot_rep=self.pot_rep)
                else:
                    target_pos_local = target_pos
                # if self.work_damping > 0.5:
                #     target_pos_local[-1] = cur_pos[-1]
                self.send_target_command(target_pos_local, target_pos)

            
            distance_vec = np.array(target_pos_local) - np.array(cur_pos)
            distance = self.distance_normalize(distance_vec)
            # if self.work_damping > 0.5:
            #     distance[-1] = 0
            distance = np.linalg.norm(distance_vec)
            #print("distance: ", distance)

            rospy.sleep(0.02)

            # print(distance)

        self.exe_index += 1
        if not self.exe_index < len(self.trajectory.joint_trajectory.points):
            print("finished_trajectory")
            self.arrived_pub.publish("finished_trajectory")
        

    def hold_joint_state(self):
        target_pos = self.joint_state.position[:7]
        self.send_target_command(target_pos)
        rospy.sleep(0.02)
    
    def refresh_command(self):
        msg = Int16()
        msg.data = 0
        self.refresh_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            if self.joint_state is None:
                continue
            if self.trajectory is not None:
                if self.exe_index < len(self.trajectory.joint_trajectory.points):
                    self.command()
                    
                # release the robot 
                else:
                    pass
                    # self.arrived_pub.publish("finished_trajectory")
            
    
if __name__ == "__main__":
    selector = ReferenceSelector()
    selector.run()
    rospy.spin()