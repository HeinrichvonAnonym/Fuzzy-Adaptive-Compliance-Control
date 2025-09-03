"""
copyright (c) 2025-present Heinrich 2130238@tongji.edu.cn.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""


import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
import numpy as np
# from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from std_msgs.msg import Float32, Float32MultiArray, Int16
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion
import scipy.spatial.transform as transform
import tf
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
import math
import yaml
import os

"""
control with force balance detection
"""
ws_path = os.getenv("HOME") + "/franka_ws"
config_path = ws_path + "/src/speed_adaptive_control/config/dynamical_parameters_franka.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

algo_name = config["algo_name"]

USE_FUZZY = config["USE_FUZZY"]
# smooth_att = 0.15
# k_att_base = 5000 # 基础引力系数

# obj_k_rep = 500
# k_att_cart = 8000
# obj_influence_margin = 0.1
# obj_safe_margin = 0.03

base_link = config["base_link"]

smooth_att = config["smooth_att"]
k_att_base = config["k_att_base"]
obj_k_rep = config["obj_k_rep"]
k_att_cart = config["k_att_cart"]
k_mini = config["k_mini"]
obj_influence_margin = config["obj_influence_margin"]
obj_safe_margin = config["obj_safe_margin"]


if not USE_FUZZY:
    # human_influence_margin = 0.35
    # human_safe_margin = 0.05
    # human_k_rep = 6500
    # k_lamda = 100
    human_influence_margin = config["human_influence_margin_nf"]
    human_safe_margin = config["human_safe_margin_nf"]
    human_k_rep = config["human_k_rep_nf"]
    k_lamda = config["k_lamda_nf"]
else:
    # human_k_rep = 1500
    # human_influence_margin = 0.8
    # human_safe_margin = 0.05
    # k_lamda = 500
    human_influence_margin = config["human_influence_margin"]
    human_safe_margin = config["human_safe_margin"]
    human_k_rep = config["human_k_rep"]
    k_lamda = config["k_lamda"]
    

# END_EFFECTOR = 0
# WRIST = 1
# FOREARM = 2

END_EFFECTOR = config["END_EFFECTOR"]
WRIST = config["WRIST"]
FOREARM = config["FOREARM"]


def angle_normalize(angles):
    output_angles = []
    for i, angle in enumerate(angles):
        angle = angle % np.pi * 2
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < - np.pi:
            angle += 2 * np.pi
        output_angles.append(angle)
    output_angles = np.array(output_angles)
    return angles


def ref_fuzzy_slot(fri, cur_idx):
    rand = np.random.rand()
    if rand > abs(fri):
        return cur_idx
    elif  fri < 0:
        return cur_idx - 1
    elif  fri > 0:
        return cur_idx + 1
    

def pot_fuzzy_slot(fpi, att_potential, rep_potential):
    fpi = fpi * np.pi / 4
    angle = np.pi / 4 + fpi
    cos = max(np.cos(angle), 0.707)
    sin = np.sin(angle)
    return (cos * att_potential + sin * rep_potential) / 0.707


class APFController:
    def __init__(self):
        rospy.init_node("apf_controller", anonymous=True)
        rospy.Subscriber("/desired_q", Float32MultiArray, self.target_callback)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        # rospy.Subscriber(f"/{algo_name}/objects", SolidPrimitiveMultiArray, self.object_callback)
        rospy.Subscriber("/mrk/human_skeleton",MarkerArray, self.human_callback)
        rospy.Subscriber(f"/{algo_name}/robot_pose", PoseArray, self.pose_callback)
        rospy.Subscriber(f"/{algo_name}/jacobian", Float32MultiArray, self.jacobian_callback)
        rospy.Subscriber(f"/{algo_name}/jacobian_6_dof", Float32MultiArray, self.jacobian_6_dof_callback)
        rospy.Subscriber(f"/{algo_name}/jacobian_4_dof", Float32MultiArray, self.jacobian_4_dof_callback)
        rospy.Subscriber(f"/{algo_name}/refresh", Int16, self.refresh_callback)
        rospy.Subscriber(f"/{algo_name}/target_pose", PoseStamped, self.target_pose_callback)
        rospy.Subscriber(f"/{algo_name}/pi_weight", Float32, self.pi_subscriber)
        rospy.Subscriber(f"/{algo_name}/ri_weight", Float32, self.ri_callback)
        rospy.Subscriber(f"/{algo_name}/work_damping", Float32, self.work_damping_callback)
        rospy.Subscriber(f"/{algo_name}/joint_inertias", Float32MultiArray, self.inertia_callback)
        rospy.Subscriber(f"/{algo_name}/att_scale", Float32, self.att_scale_callback)
        # self.pi_publisher = rospy.Publisher(f"/{algo_name}/pi_weight", Float32, queue_size=10)
        self.pid_command_pub = rospy.Publisher(f"/{algo_name}/pid_command", Float32MultiArray, queue_size=10)
        self.rep_pub = rospy.Publisher(f"/{algo_name}/rep_vec", Marker, queue_size=10)
        
        # 添加DSM值发布器 - 区分人和物体
        self.dsm_pub = rospy.Publisher(f"/{algo_name}/dsm_value", Float32MultiArray, queue_size=10)
        
        # 添加对象类型和安全距离发布器
        self.obj_info_pub = rospy.Publisher(f"/{algo_name}/object_info", Float32MultiArray, queue_size=10)
        
        # 添加引力和斥力发布器
        self.force_pub = rospy.Publisher(f"/{algo_name}/force_values", Float32MultiArray, queue_size=10)
        self.euclidean_pub =rospy.Publisher(f"/{algo_name}/euclidean", Float32, queue_size=10)
        self.distance_pub = rospy.Publisher(f"/{algo_name}/distance", Float32, queue_size=10)
        self.pot_rep_pub = rospy.Publisher(f"/{algo_name}/pot_rep", Float32MultiArray, queue_size=10)
        
        self.target_pos = None
        self.target_cartesian_pose = None
        self.cur_pos = None
        self.objects_poses = None
        self.primitives = None
        self.human_poses = None
        self.posed = None
        self.dt = 0.02
        self.prev_time = rospy.Time.now()
        self.poses = None
        self.jacobian = None
        self.jacobian_6_dof = None
        self.jacobian_4_dof = None
        self.joint_inertias = None
        
        # DSM值 - 分别存储对人和对物体的DSM
        self.human_dsm_value = 0.0
        self.obj_dsm_value = 0.0
        self.min_distance = float('inf')
        self.nearest_object_type = -1  # -1: 未知, 0: 人, 1: 物体
        self.nearest_object_index = -1

        # fuzzy control
        self.pi_index = 0
        self.ri_index = 0

        # work mode
        self.work_damping = 0
        self.att_scale = 1
    
    def att_scale_callback(self, msg:Float32):
        self.att_scale = msg.data

    def calculate_human_dsm(self, human_min_distance, velocity_norm):
        """
        计算人体DSM - 考虑距离、速度和系统状态
        
        参数:
        human_min_distance - 到人的最小距离
        velocity_norm - 机器人关节空间速度的范数
        """
        if human_min_distance == float('inf'):
            return float('inf')
        
        # 基础DSM计算
        base_dsm = human_min_distance - human_safe_margin
        
        # 添加速度影响 - 参考论文中的动态考虑
        # 速度越大，安全裕度越小
        if velocity_norm > 0:
            # 非线性速度因子，使DSM随速度增加而迅速减小
            velocity_factor = 1.0 / (1.0 + velocity_norm**1.5 * 1.2)
            base_dsm *= velocity_factor
        
        # 添加随机扰动，模拟系统噪声和动态不确定性
        noise_amplitude = 0.02 * base_dsm  # 小幅度噪声
        noise = noise_amplitude * (np.random.random() - 0.5)
        
        # 添加时间相关波动，模拟系统动态
        time_factor = 0.95 + 0.1 * np.sin(rospy.Time.now().to_sec() * 2.0)
        
        # 将所有因素组合
        final_dsm = (base_dsm + noise) * time_factor
        
        # 确保DSM不为负
        return max(0, final_dsm)
        
    def calculate_object_dsm(self, obj_min_distance, velocity_norm):
        """
        计算物体DSM - 与人体DSM类似但参数不同
        
        参数:
        obj_min_distance - 到物体的最小距离
        velocity_norm - 机器人关节空间速度的范数
        """
        if obj_min_distance == float('inf'):
            return float('inf')
        
        # 基础DSM计算 - 物体安全裕度通常比人体小
        base_dsm = obj_min_distance - obj_safe_margin
        
        # 添加速度影响 - 与人体DSM相比对速度的敏感度较低
        if velocity_norm > 0:
            velocity_factor = 1.0 / (1.0 + velocity_norm * 0.8)
            base_dsm *= velocity_factor
        
        # 添加较小的随机扰动
        noise_amplitude = 0.015 * base_dsm
        noise = noise_amplitude * (np.random.random() - 0.5)
        
        # 添加更缓慢的时间相关波动
        time_factor = 0.97 + 0.06 * np.sin(rospy.Time.now().to_sec() * 1.5)
        
        # 将所有因素组合
        final_dsm = (base_dsm + noise) * time_factor
        
        # 确保DSM不为负
        return max(0, final_dsm)

    def calc_attraction_potential(self):
        target_pos = self.target_pos 
        cur_pos = self.cur_pos[:7] 
        target_pos = angle_normalize(target_pos)
        # print(target_pos)
        cur_pos = angle_normalize(cur_pos)
        pos_dis = target_pos - cur_pos
        # pos_dis = angle_normalize(pos_dis)
        # # 角度归一化
        for i in range(len(pos_dis)):
            pos_dis[i] = pos_dis[i] % (2 * np.pi)
            if pos_dis[i] > np.pi:
                pos_dis[i] = pos_dis[i] % np.pi - np.pi
            elif pos_dis[i] < -1 * np.pi:
                pos_dis[i] = -pos_dis[i] % np.pi + np.pi

        dist = np.linalg.norm(pos_dis)
        dist_msg = Float32()
        dist_msg.data = dist
        self.euclidean_pub.publish(dist)
        # print("distance: ", dist)
        p_att = pos_dis / max(dist, smooth_att) * k_att_base * self.att_scale
        
        return p_att, dist

    def calc_repulsion_potential(self):
        critical_poses = self.poses
        obj_poses = self.objects_poses
        primitives = self.primitives

        max_amplitude = 0.
        potential_rep = np.zeros(7)
        self.min_distance = float('inf')
        
        # 重置最近物体信息
        self.nearest_object_type = -1
        self.nearest_object_index = -1
        
        # 保存人和物体的最小距离
        human_min_distance = float('inf')
        obj_min_distance = float('inf')

        for i, critical_pose in enumerate(critical_poses):
            if obj_poses is not None:
                cur_influence_margin = obj_influence_margin
                cur_safe_margin = obj_safe_margin
                cur_k_rep = obj_k_rep
                object_type = 1  
                for j, (obj_pose, primitive) in enumerate(zip(obj_poses, primitives)):
                    # print(obj_pose)
                    cenrtri_vector = np.array([critical_pose.position.x - obj_pose.position.x, 
                                            critical_pose.position.y - obj_pose.position.y, 
                                            critical_pose.position.z - obj_pose.position.z])
                    centri_dis = np.linalg.norm(cenrtri_vector)
                    
                    if primitive.type == SolidPrimitive.SPHERE:
                        inner_dis = primitive.dimensions[0]
                        if centri_dis < primitive.dimensions[0]:
                            dis = 0
                        else:
                            dis = centri_dis - primitive.dimensions[0]   
                            
                    elif primitive.type == SolidPrimitive.CYLINDER:
                        inner_vec = np.array([primitive.dimensions[0] / 2, 
                                            primitive.dimensions[1]])
                        inner_dis = np.linalg.norm(inner_vec)
                        if centri_dis < inner_dis:
                            dis = 0
                        else:
                            dis = centri_dis - inner_dis
                            
                    elif primitive.type == SolidPrimitive.BOX:
                        inner_vec = np.array([primitive.dimensions[0] / 2,
                                            primitive.dimensions[1] / 2, 
                                            primitive.dimensions[2] / 2])
                        inner_dis = np.linalg.norm(inner_vec)
                        if centri_dis < inner_dis:
                            dis = 0
                        else:
                            dis = centri_dis - inner_dis
                        # 更新物体的最小距离
                    if dis < obj_min_distance:
                        obj_min_distance = dis
                    
                    # 更新全局最小距离
                    if dis < self.min_distance:
                        self.min_distance = dis
                        self.nearest_object_type = object_type
                        self.nearest_object_index = j
                            
                    amplitude = (cur_influence_margin - dis) / (cur_influence_margin - cur_safe_margin)
                    amplitude = max(0, amplitude)
                    
                    if amplitude > max_amplitude:
                        max_amplitude = amplitude
                        max_i = i
                        obs_pose = obj_pose
                        choosed_inner_dis = inner_dis
                        max_influence_margin = cur_influence_margin
                        max_safe_margin = cur_safe_margin
                        max_k_rep = cur_k_rep
                        potential_vec = cenrtri_vector / centri_dis
                        potential_vec = potential_vec * amplitude

            if self.human_poses is not None:
                for human_pose in self.human_poses: 
                    # print(human_pose)
                    cur_k_rep = human_k_rep
                    pose = critical_pose
                    position_link = np.array([pose.position.x,
                                            pose.position.y,
                                            pose.position.z])
                    position_human = np.array([human_pose.position.x,
                                                human_pose.position.y,
                                                human_pose.position.z])
                    distance_vec = position_human - position_link
                    dis = np.linalg.norm(distance_vec) - 0.1
                    choosed_inner_dis = 0.1
                    if dis< self.min_distance:
                        amplitude = (human_influence_margin - dis) / (human_influence_margin - human_safe_margin)
                        amplitude = max(0, amplitude)
                        self.min_distance =dis

                    if amplitude > max_amplitude:
                        max_amplitude = amplitude
                        max_i = i
                        obs_pose = human_pose
                        max_influence_margin = human_influence_margin
                        max_safe_margin = human_safe_margin
                        max_k_rep = cur_k_rep
                        potential_vec = - distance_vec / dis
                        potential_vec = potential_vec * amplitude
        if self.min_distance < 1e4:
            distance_msg = Float32()
            distance_msg.data = self.min_distance
            self.distance_pub.publish(distance_msg)
        
        # 计算动态安全裕度 (DSM) - 分别计算对人和对物体的DSM
        if self.cur_pos is not None and len(self.cur_pos) >= 7:
            # 获取机器人速度
            velocity_norm = np.linalg.norm(self.cur_pos[7:14]) if len(self.cur_pos) >= 14 else 0
            
            # 计算对人的复杂DSM
            if human_min_distance != float('inf'):
                self.human_dsm_value = self.calculate_human_dsm(human_min_distance, velocity_norm)
            else:
                self.human_dsm_value = float('inf')
            
            # 计算对物体的复杂DSM
            if obj_min_distance != float('inf'):
                self.obj_dsm_value = self.calculate_object_dsm(obj_min_distance, velocity_norm)
            else:
                self.obj_dsm_value = float('inf')
            
            # 发布DSM值
            dsm_msg = Float32MultiArray()
            dsm_msg.data = [
                self.human_dsm_value,   # 对人的DSM
                self.obj_dsm_value,     # 对物体的DSM
                self.min_distance,      # 全局最小距离
                float(self.nearest_object_type)  # 最近物体类型
            ]
            self.dsm_pub.publish(dsm_msg)
            
            # 发布物体信息
            if self.nearest_object_index >= 0 and self.nearest_object_index < len(obj_poses):
                nearest_obj = obj_poses[self.nearest_object_index]
                obj_info_msg = Float32MultiArray()
                obj_info_msg.data = [
                    float(self.nearest_object_type),  # 0: 人, 1: 物体
                    nearest_obj.position.x,
                    nearest_obj.position.y,
                    nearest_obj.position.z,
                    self.min_distance
                ]
                self.obj_info_pub.publish(obj_info_msg)
            
        if max_amplitude > 0:
            self.update_marker(potential_vec, obs_pose, choosed_inner_dis, max_influence_margin, max_safe_margin)
            potential_rep = self.cartesian_2_axis(potential_vec, max_i)
        
        
            
        return potential_rep * max_k_rep if max_amplitude > 0 else potential_rep

    def update_marker(self, potential_vec, obj_pose, inner_dis, influence_margin, safe_margin):
        # vector
        marker = Marker()
        marker.header.frame_id = base_link
        marker.header.stamp = rospy.Time.now()
        marker.ns = "potential_rep"
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [Point(obj_pose.position.x, 
                             obj_pose.position.y, 
                             obj_pose.position.z), 
                        Point(obj_pose.position.x + 0.5 * potential_vec[0], 
                             obj_pose.position.y + 0.5 * potential_vec[1], 
                             obj_pose.position.z + 0.5 * potential_vec[2])]
        self.rep_pub.publish(marker)

        marker.id = 1
        marker.ns = "threshold"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = influence_margin * 2 + inner_dis * 2
        marker.scale.y = influence_margin * 2 + inner_dis * 2
        marker.scale.z = influence_margin * 2 + inner_dis * 2
        marker.pose = obj_pose
        marker.color.r = 0.3
        marker.color.g = 0.3
        marker.color.b = 0.3
        marker.color.a = 0.3
        self.rep_pub.publish(marker)

    def quatenion_multiplication(self, q1, q2):
        # w x y z
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]
    
    def quatenion_conjugate(self, q):
        return [q[0], -q[1], -q[2], -q[3]]
    
    def euler_from_quaternion(self, quaternion):
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]
        w = quaternion[0]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y- z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return [roll, pitch, yaw]
    
    def cartesian_2_axis(self, cartesian_vec, max_i):
        # print(max_i)
        # jacobian = self.jacobian
        if max_i == 2:
            # print("forarm")
            jacobian = self.jacobian_4_dof
            cat = True
            cat_dim = 3

        elif max_i == 1:
            # print("wrist")
            jacobian = self.jacobian_6_dof
            cat = True
            cat_dim = 1

        elif max_i == 0:
            jacobian = self.jacobian
            cat = False
        # print(jacobian.shape)
        rot_vec = np.zeros(3)
        influence_vec = np.concatenate((cartesian_vec, rot_vec))
        jacobian_inv = np.linalg.pinv(jacobian)
        angular_influence = np.dot(jacobian_inv, influence_vec)
        # print(angular_influence.shape)
        
        if cat:
            angular_influence = np.concatenate((angular_influence, np.zeros(cat_dim)))
            
        return angular_influence
    
    def calc_pid_command(self, potential):

        if self.joint_inertias is None:
            joint_inertia = np.ones(7) * 0.33
        else:
            joint_inertia = self.joint_inertias

        potential /= joint_inertia * 3
        potential /= joint_inertia * 3

        if self.work_damping > 0.1:
            potential *= 0.3
        
        cur_vel = np.array(self.cur_vel[:7])
        damping = k_lamda  * cur_vel
        # print(self.ri_index)
        
        potential = potential - damping
        if USE_FUZZY:
            potential *= (1 - self.ri_index * k_mini)
        command_tar = self.cur_pos[:7] + self.dt * (self.cur_vel[:7] + self.dt * potential) / 2
        pid_command = Float32MultiArray()
        pid_command.data = command_tar.tolist()
        return pid_command

    def target_pose_callback(self, msg:PoseStamped):
        self.target_cartesian_pose = msg.pose

    def joint_state_callback(self, msg):
        self.cur_pos = np.array(msg.position)
        self.cur_vel = np.array(msg.velocity)

    def target_callback(self, target):
      
        self.target_pos = np.array(target.data)

    # def object_callback(self, msg:SolidPrimitiveMultiArray):
    #     # print("get object")
    #     # print(msg.poses)
    #     self.objects_poses = msg.poses
    #     # print(self.object_poses)
    #     self.primitives = msg.primitives
    
    def pose_callback(self, msg:PoseArray):
        self.poses = msg.poses

    def work_damping_callback(self, msg:Float32):
        self.work_damping = msg.data

    def jacobian_callback(self, msg):
        self.jacobian = np.array(msg.data).reshape(6,7)
    def jacobian_6_dof_callback(self, msg):
        self.jacobian_6_dof = np.array(msg.data).reshape(6,6)
    def jacobian_4_dof_callback(self, msg):
        self.jacobian_4_dof = np.array(msg.data).reshape(6,4)
    
    def pi_subscriber(self, msg:Float32):
        self.pi_index = msg.data
    
    def ri_callback(self, msg:Float32):
        ri = msg.data # -1: 1
        ri += 1 # 0:2
        ri /= 2 # 0:1
        self.ri_index = ri

    def refresh_callback(self, msg):
        self.target_pos = None
    
    def inertia_callback(self, msg:Float32MultiArray):
        self.inertia_tensor = np.array(msg.data)
        # print(self.inertia_tensor)
    
    def human_callback(self, ma:MarkerArray):
        markers = ma.markers
        self.human_poses = []
        for marker in markers:
            self.human_poses.append(marker.pose)

    def run(self):
        prev_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            # print(">>")
            rospy.sleep(self.dt)

            if self.cur_pos is None:
                continue
            
            if (self.objects_poses is None and self.human_poses is None) or self.poses is None or self.jacobian is None or self.work_damping > 0.1:
                # if self.object_poses is None:
                # print(">>>")
                p_rep = np.zeros(7)
            else:
                p_rep = self.calc_repulsion_potential()          

            if self.target_pos is None:
                p_att = np.zeros(7)
                p_c_att = np.zeros(7)
                potential = p_rep

            else:
                p_att, dist = self.calc_attraction_potential()
                if USE_FUZZY:
                    potential = pot_fuzzy_slot(self.pi_index, p_att, p_rep)
                else:
                    potential = p_att + p_rep
            
            pot_rep_msg = Float32MultiArray()
            pot_rep_msg.data = p_rep
            self.pot_rep_pub.publish(pot_rep_msg)
            # 发布引力和斥力数据
            force_msg = Float32MultiArray()
            force_msg.data = [
                np.linalg.norm(p_att),   # 引力大小
                np.linalg.norm(p_rep),   # 斥力大小
                np.linalg.norm(potential)  # 合力大小
            ]
            self.force_pub.publish(force_msg)
            
            if rospy.Time.now().to_sec() - prev_time > 0.5:
                rospy.loginfo(f"att: {np.linalg.norm(p_att)}, rep: {np.linalg.norm(p_rep)}, damp: {self.work_damping}")
                prev_time += 0.5
            
            pid_command = self.calc_pid_command(potential)
            self.pid_command_pub.publish(pid_command)

if __name__ == "__main__":
    controller = APFController()
    controller.run()
    rospy.spin()