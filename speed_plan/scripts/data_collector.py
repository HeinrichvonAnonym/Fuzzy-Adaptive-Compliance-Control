#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float32MultiArray, Int16, Float32
from visualization_msgs.msg import MarkerArray, Marker
import csv
import os
from datetime import datetime

import yaml
config_path = "/home/heinrich/kinova/src/speed_plan/config/dynamical_parameters_kortex.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

algo_name = config["algo_name"]

class SimplifiedRobotDataCollector:
    def __init__(self):
        # 初始化节点
        rospy.init_node('simplified_robot_data_collector', anonymous=True)
        
        # 创建数据存储目录
        self.data_dir = os.path.expanduser('~/robot_data_logs')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 创建时间戳用于文件名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化数据文件
        self.joint_error_file = self.init_csv_file('joint_error')
        self.dsm_data_file = self.init_csv_file('dsm_data')
        self.safety_data_file = self.init_csv_file('safety_data')
        self.trajectory_data_file = self.init_csv_file('trajectory_data')
        self.force_data_file = self.init_csv_file('force_data')
        self.target_points_file = self.init_csv_file('target_points')
        self.final_target_file = self.init_csv_file('final_target')  # 新增qrl记录文件
        self.fuzzy_index_file = self.init_csv_file("fuzzy_index")
        
        # 人体骨架位置文件
        self.human_skeleton_file = self.init_csv_file('human_skeleton')
        
        # 添加状态标志
        self.is_recording = False
        self.trajectory_started = False
        self.trajectory_completed = False
        self.target_points_history = []
        
        # 数据存储变量
        self.actual_joint_positions = None  # q
        self.actual_joint_velocities = None
        self.actual_joint_efforts = None
        self.target_joint_positions = None  # qr = qrs
        self.last_target_positions = None   # 每段轨迹的最后一个点 qrl
        self.prev_target_joint_positions = None
        self.end_effector_pose = None
        self.prev_end_effector_pose = None
        self.end_effector_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 线速度和角速度
        self.ee_velocity_time = rospy.Time.now()
        self.target_pose = None
        
        
        # 简化的DSM值 - 只保留人类和物体的DSM
        self.human_dsm = float('inf')
        self.object_dsm = float('inf')
        
        # 关节误差数据
        self.joint_errors = {
            'q_minus_qr_norm': float('nan'),
            'q_minus_qrl_norm': float('nan'),  # 与最终目标点的误差
            'joint_errors': [float('nan')] * 7
        }
        
        # 力数据
        self.attractive_force = 0.0
        self.repulsive_force = 0.0
        self.total_force = 0.0
        
        # 安全数据
        self.min_obstacle_distance = float('inf')
        self.nearest_object_type = -1
        self.nearest_object_position = [0.0, 0.0, 0.0]
        
        # 轨迹数据
        self.trajectory_progress = 0.0
        self.current_target_index = 0
        self.total_target_count = 0

        self.pi = 0
        self.raw_pi = 0
        self.ri = 0

        self.euclidean = 0
        self.velocity = 0 
        self.distance = 0
        
        # 轨迹终点标记
        self.is_final_target = False
        
        # 人体骨架数据 - 存储左右手四连杆关节位置
        self.human_joints = {
            # 左手四连杆 (11, 13, 15, 19)
            11: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 左肩
            13: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 左肘
            15: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 左腕
            19: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 左手
            
            # 右手四连杆 (12, 14, 16, 20)
            12: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 右肩
            14: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 右肘
            16: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 右腕
            20: {'x': 0.0, 'y': 0.0, 'z': 0.0},  # 右手
        }
        
        # 订阅相关话题
        rospy.Subscriber('/base_feedback/joint_state', JointState, self.joint_state_callback)
        rospy.Subscriber(f'/{algo_name}/target_pos', JointState, self.target_pos_callback)
        rospy.Subscriber(f'/{algo_name}/cartesian_trajectory', RobotTrajectory, self.trajectory_callback)
        rospy.Subscriber(f'/{algo_name}/robot_pose', PoseArray, self.robot_pose_callback)
        rospy.Subscriber(f'/{algo_name}/target_pose', PoseStamped, self.target_pose_callback)
        rospy.Subscriber(f'/{algo_name}/dsm_value', Float32MultiArray, self.dsm_callback)
        rospy.Subscriber(f'/{algo_name}/force_values', Float32MultiArray, self.force_values_callback)
        rospy.Subscriber(f'/{algo_name}/trajectory_progress', Float32MultiArray, self.trajectory_progress_callback)
        rospy.Subscriber(f'/{algo_name}/object_info', Float32MultiArray, self.object_info_callback)
        rospy.Subscriber('/mrk/human_skeleton', MarkerArray, self.human_skeleton_callback)
        rospy.Subscriber(f"/{algo_name}/prev_reference_selector_pos", Float32MultiArray, self.prev_target_callback)
        rospy.Subscriber(f"/{algo_name}/pi_weight", Float32, self.pi_callback)
        rospy.Subscriber(f"/{algo_name}/raw_pi_weight", Float32, self.raw_pi_callback)
        rospy.Subscriber(f"/{algo_name}/ri_weight", Float32, self.ri_callback)
        rospy.Subscriber(f"/{algo_name}/fuzzy/d_euclidean", Float32, self.euclidean_callback)
        rospy.Subscriber(f"/{algo_name}/fuzzy/velocity", Float32, self.velocity_callback)
        rospy.Subscriber(f"/{algo_name}/fuzzy/distance", Float32, self.min_distance_callback)
        
        # 设置数据记录频率
        self.record_rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("简化版机器人数据收集器已初始化，等待轨迹开始...")
    
    def euclidean_callback(self, msg:Float32):
        self.euclidean = msg.data

    def velocity_callback(self, msg:Float32):
        self.velocity = msg.data
    
    def min_distance_callback(self, msg:Float32):
        self.distance = msg.data
    
    def raw_pi_callback(self, msg:Float32):
        self.raw_pi = msg.data

    def pi_callback(self, msg:Float32):
        self.pi = msg.data
    
    def ri_callback(self, msg:Float32):
        self.ri = msg.data

    def prev_target_callback(self, msg:Float32MultiArray):
        # 获取目标位置并归一化
        normalized_target = self.normalize_joint_angles(msg.data)
        self.prev_target_joint_positions = normalized_target
    
    def init_csv_file(self, name):
        """初始化CSV文件并写入表头"""
        file_path = os.path.join(self.data_dir, f"{name}_{self.timestamp}.csv")
        f = open(file_path, 'w')
        writer = csv.writer(f)
        
        if name == 'joint_error':
            writer.writerow(['timestamp', 'q_minus_qr_norm', 'q_minus_qrl_norm', 'q_minus_prev_qr_norm',
                            'is_final_target', 'joint1_error', 'joint2_error', 
                            'joint3_error', 'joint4_error', 'joint5_error', 
                            'joint6_error', 'joint7_error',
                            'qr_joint1', 'qr_joint2', 'qr_joint3', 'qr_joint4', 
                            'qr_joint5', 'qr_joint6', 'qr_joint7'])
        elif name == 'dsm_data':
            writer.writerow(['timestamp', 'human_dsm', 'object_dsm'])
        elif name == 'safety_data':
            writer.writerow(['timestamp', 'min_obstacle_distance', 'nearest_object_type',
                            'object_pos_x', 'object_pos_y', 'object_pos_z'])
        elif name == 'trajectory_data':
            writer.writerow(['timestamp', 
                            'joint1_pos', 'joint2_pos', 'joint3_pos', 'joint4_pos', 
                            'joint5_pos', 'joint6_pos', 'joint7_pos',
                            'joint1_vel', 'joint2_vel', 'joint3_vel', 'joint4_vel', 
                            'joint5_vel', 'joint6_vel', 'joint7_vel',
                            'joint1_eff', 'joint2_eff', 'joint3_eff', 'joint4_eff', 
                            'joint5_eff', 'joint6_eff', 'joint7_eff',
                            'ee_pos_x', 'ee_pos_y', 'ee_pos_z', 
                            'ee_quat_x', 'ee_quat_y', 'ee_quat_z', 'ee_quat_w',
                            'ee_lin_vel_x', 'ee_lin_vel_y', 'ee_lin_vel_z',
                            'ee_ang_vel_x', 'ee_ang_vel_y', 'ee_ang_vel_z',
                            'target_pos_x', 'target_pos_y', 'target_pos_z',
                            'target_quat_x', 'target_quat_y', 'target_quat_z', 'target_quat_w',
                            'trajectory_progress', 'current_target_index', 'total_target_count'])
        elif name == 'force_data':
            writer.writerow(['timestamp', 'attractive_force', 'repulsive_force', 'total_force'])
        elif name == 'target_points':
            writer.writerow(['timestamp', 'target_pos_x', 'target_pos_y', 'target_pos_z',
                            'target_quat_x', 'target_quat_y', 'target_quat_z', 'target_quat_w',
                            'target_index'])
        elif name == 'final_target':
            # 新增qrl记录表头
            writer.writerow(['timestamp', 'trajectory_id', 
                            'qrl_joint1', 'qrl_joint2', 'qrl_joint3', 'qrl_joint4', 
                            'qrl_joint5', 'qrl_joint6', 'qrl_joint7'])
        elif name == 'human_skeleton':
            writer.writerow(['timestamp', 
                            'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                            'left_elbow_x', 'left_elbow_y', 'left_elbow_z',
                            'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
                            'left_hand_x', 'left_hand_y', 'left_hand_z',
                            'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                            'right_elbow_x', 'right_elbow_y', 'right_elbow_z',
                            'right_wrist_x', 'right_wrist_y', 'right_wrist_z',
                            'right_hand_x', 'right_hand_y', 'right_hand_z'])
        elif name == "fuzzy index":
            writer.writerow(['raw_fpi', 'fpi', 'fri', 'euclidean', 'velocity', 'distance'])
        
        return f
    
    def normalize_joint_angles(self, angles):
        """
        将关节角度归一化到[-π, π]区间，处理角度回绕问题
        
        Args:
            angles: 关节角度列表或数组
            
        Returns:
            normalized: 归一化后的角度数组，范围为[-π, π]
        """
        normalized = np.array(angles, dtype=float)
        
        # 使用取模运算将角度归一化到[-π, π]范围
        normalized = ((normalized + np.pi) % (2 * np.pi)) - np.pi
        
        return normalized
    
    def normalize_joint_error(self, error):
        """
        归一化关节角度误差，确保误差值在[-π, π]范围内
        
        Args:
            error: 关节角度误差
            
        Returns:
            normalized_error: 归一化后的误差，范围为[-π, π]
        """
        # 转换为numpy数组以便处理
        normalized_error = np.array(error, dtype=float)
        
        # 将每个误差分量限制在[-π, π]范围内
        # 如果误差大于π，则取反向的较短路径
        for i in range(len(normalized_error)):
            if normalized_error[i] > np.pi:
                normalized_error[i] -= 2 * np.pi
            elif normalized_error[i] < -np.pi:
                normalized_error[i] += 2 * np.pi
        
        return normalized_error
    
    def are_angles_equal(self, angles1, angles2, threshold=0.01):
        """
        检查两组关节角度是否相等，考虑角度回绕
        
        Args:
            angles1: 第一组角度
            angles2: 第二组角度
            threshold: 判定相等的阈值，默认为0.01弧度
            
        Returns:
            equal: 如果两组角度相等，返回True；否则返回False
        """
        if angles1 is None or angles2 is None:
            return False
            
        # 确保输入是numpy数组
        a1 = np.array(angles1)
        a2 = np.array(angles2)
        
        # 检查长度是否相同
        if len(a1) != len(a2):
            return False
        
        # 计算差异并考虑角度回绕
        diff = np.abs(a1 - a2)
        # 对于差异大于π的，计算另一个方向的差异
        diff = np.minimum(diff, 2*np.pi - diff)
        
        # 如果所有差异都小于阈值，则认为相等
        return np.all(diff <= threshold)
    
    def joint_state_callback(self, msg):
        """处理机器人实际关节状态"""
        # 获取前7个关节位置并归一化
        if len(msg.position) >= 7:
            self.actual_joint_positions = self.normalize_joint_angles(msg.position[:7])
            self.actual_joint_velocities = np.array(msg.velocity[:7]) if len(msg.velocity) >= 7 else np.zeros(7)
            if hasattr(msg, 'effort') and len(msg.effort) >= 7:
                self.actual_joint_efforts = np.array(msg.effort[:7])
            else:
                self.actual_joint_efforts = np.zeros(7)
            
            # 计算关节误差数据
            self.calculate_joint_errors()
    
    def target_pos_callback(self, msg):
        """处理目标关节位置，正确处理角度回绕"""
        if len(msg.position) == 0:
            return
            
        # 获取目标位置并归一化
        normalized_target = self.normalize_joint_angles(msg.position)
        
        # 检查是否是新的目标点
        is_new_target = False
        if self.target_joint_positions is None:
            is_new_target = True
        else:
            # 使用角度相等性检查函数判断是否是新目标
            is_new_target = not self.are_angles_equal(normalized_target, self.target_joint_positions)
        
        # 更新目标位置
        self.target_joint_positions = normalized_target
        
        # 如果是新目标点，开始记录
        if is_new_target:
            if not self.trajectory_started:
                self.trajectory_started = True
                self.is_recording = True
                rospy.loginfo("检测到第一个目标点，开始记录数据...")
            
            # 记录轨迹索引
            self.current_target_index += 1
    
    def trajectory_callback(self, msg):
        """处理轨迹数据，提取最终目标点"""
        if msg.joint_trajectory.points:
            # 标记轨迹开始
            if not self.trajectory_started:
                self.trajectory_started = True
                self.is_recording = True
                rospy.loginfo("检测到轨迹，开始记录数据...")
            
            # 获取轨迹的最后一个点作为qrl
            last_point = msg.joint_trajectory.points[-1]
            new_last_target = self.normalize_joint_angles(last_point.positions)
            
            # 检查是否是新的最终目标点
            is_new_final_target = False
            if self.last_target_positions is None:
                is_new_final_target = True
            else:
                is_new_final_target = not self.are_angles_equal(new_last_target, self.last_target_positions)
            
            # 如果是新的最终目标，记录它
            if is_new_final_target:
                self.last_target_positions = new_last_target
                self.record_final_target_data()
            
            # 标记是否为轨迹的最终点
            if self.target_joint_positions is not None and self.last_target_positions is not None:
                # 使用角度相等性检查判断是否是最终点
                self.is_final_target = self.are_angles_equal(
                    self.target_joint_positions, 
                    self.last_target_positions
                )
    
    def robot_pose_callback(self, msg):
        """处理机器人位姿数据，并计算末端执行器速度"""
        if len(msg.poses) >= 1:
            current_time = rospy.Time.now()
            self.end_effector_pose = msg.poses[0]  # 使用第一个位姿作为末端执行器
            
            # 计算末端执行器速度（通过有限差分）
            if self.prev_end_effector_pose is not None:
                dt = (current_time - self.ee_velocity_time).to_sec()
                if dt > 0:
                    # 计算线速度
                    dx = (self.end_effector_pose.position.x - self.prev_end_effector_pose.position.x) / dt
                    dy = (self.end_effector_pose.position.y - self.prev_end_effector_pose.position.y) / dt
                    dz = (self.end_effector_pose.position.z - self.prev_end_effector_pose.position.z) / dt
                    
                    # 简化角速度计算
                    dqx = (self.end_effector_pose.orientation.x - self.prev_end_effector_pose.orientation.x) / dt
                    dqy = (self.end_effector_pose.orientation.y - self.prev_end_effector_pose.orientation.y) / dt
                    dqz = (self.end_effector_pose.orientation.z - self.prev_end_effector_pose.orientation.z) / dt
                    
                    # 更新末端执行器速度
                    self.end_effector_velocity = [dx, dy, dz, dqx*2, dqy*2, dqz*2]
            
            self.prev_end_effector_pose = self.end_effector_pose
            self.ee_velocity_time = current_time
    
    def target_pose_callback(self, msg):
        """处理目标位姿数据，并记录每个新目标点"""
        # 检查是否是新的目标点
        is_new_target = False
        if self.target_pose is None or (
            self.target_pose.position.x != msg.pose.position.x or
            self.target_pose.position.y != msg.pose.position.y or
            self.target_pose.position.z != msg.pose.position.z
        ):
            is_new_target = True
            
        # 更新目标位姿
        self.target_pose = msg.pose
        
        # 如果是新目标点，记录它
        if is_new_target:
            timestamp = rospy.Time.now().to_sec()
            target_point = {
                'timestamp': timestamp,
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
                'qx': msg.pose.orientation.x,
                'qy': msg.pose.orientation.y,
                'qz': msg.pose.orientation.z,
                'qw': msg.pose.orientation.w,
                'target_index': self.current_target_index
            }
            self.target_points_history.append(target_point)
            
            # 记录到CSV
            writer = csv.writer(self.target_points_file)
            writer.writerow([
                timestamp,
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
                self.current_target_index
            ])
            self.target_points_file.flush()
    
    def dsm_callback(self, msg):
        """处理动态安全裕度数据 - 简化为只关注人和物体"""
        if len(msg.data) >= 4:
            # 获取人和物体的DSM
            self.human_dsm = msg.data[0]
            self.object_dsm = msg.data[1]
            self.min_obstacle_distance = msg.data[2]
            self.nearest_object_type = int(msg.data[3])
    
    def force_values_callback(self, msg):
        """处理力数据"""
        if len(msg.data) >= 3:
            self.attractive_force = msg.data[0]
            self.repulsive_force = msg.data[1]
            self.total_force = msg.data[2]
    
    def trajectory_progress_callback(self, msg):
        """处理轨迹进度数据"""
        if len(msg.data) >= 3:
            self.trajectory_progress = msg.data[0]
            self.current_target_index = int(msg.data[1])
            self.total_target_count = int(msg.data[2])
    
    def object_info_callback(self, msg):
        """处理最近障碍物信息"""
        if len(msg.data) >= 5:
            self.nearest_object_type = int(msg.data[0])
            self.nearest_object_position = [msg.data[1], msg.data[2], msg.data[3]]
            self.min_obstacle_distance = msg.data[4]
    
    def human_skeleton_callback(self, msg):
        """处理人体骨架数据，提取左右手四连杆关节位置"""
        joint_ids = [11, 13, 15, 19, 12, 14, 16, 20]  # 左手四连杆和右手四连杆的关节ID
        
        for marker in msg.markers:
            # 检查该标记是否为我们需要的关节ID之一
            if marker.id in joint_ids:
                # 更新关节位置数据
                self.human_joints[marker.id] = {
                    'x': marker.pose.position.x,
                    'y': marker.pose.position.y,
                    'z': marker.pose.position.z
                }
    
    def calculate_joint_errors(self):
        """计算关节误差数据 - 改进版，处理角度回绕"""
        self.joint_errors = {
            'q_minus_qr_norm': float('nan'),
            'q_minus_qrl_norm': float('nan'),
            'q_minus_prev_qr_norm': float('nan'),
            'joint_errors': [float('nan')] * 7
        }
        
        # 只在有实际位置和目标位置时计算q-qr
        if self.actual_joint_positions is not None and self.target_joint_positions is not None:
            # 计算实际位置与目标位置之间的误差
            error = self.target_joint_positions - self.actual_joint_positions
            
            # 归一化误差，处理角度回绕问题
            error = self.normalize_joint_error(error)
            
            # 计算误差的欧几里德范数
            self.joint_errors['q_minus_qr_norm'] = np.linalg.norm(error)
            self.joint_errors['joint_errors'] = error.tolist()

        if self.actual_joint_positions is not None and self.prev_target_joint_positions is not None:
            error = self.prev_target_joint_positions - self.actual_joint_positions
            
            # 归一化误差，处理角度回绕问题
            error = self.normalize_joint_error(error)
            self.joint_errors['q_minus_prev_qr_norm'] = np.linalg.norm(error)
        
        # 只在有实际位置和最终目标位置时计算q-qrl
        if self.actual_joint_positions is not None and self.last_target_positions is not None:
            # 计算实际位置与最终目标位置之间的误差
            error = self.last_target_positions - self.actual_joint_positions
            
            # 归一化误差，处理角度回绕问题
            error = self.normalize_joint_error(error)
            
            # 计算误差的欧几里德范数
            self.joint_errors['q_minus_qrl_norm'] = np.linalg.norm(error)
    
    def record_joint_error_data(self):
        """记录关节误差数据"""
        timestamp = rospy.Time.now().to_sec()
        
        # 准备行数据
        row_data = [timestamp, self.joint_errors['q_minus_qr_norm'], self.joint_errors['q_minus_qrl_norm'], self.joint_errors['q_minus_prev_qr_norm'] ,int(self.is_final_target)]
        
        # 添加关节误差数据
        if isinstance(self.joint_errors['joint_errors'], list):
            row_data.extend(self.joint_errors['joint_errors'])
        else:
            # 如果不是列表，添加7个NaN值
            row_data.extend([float('nan')] * 7)
        
        # 添加目标关节位置数据
        if self.target_joint_positions is not None:
            # 确保是列表类型
            if isinstance(self.target_joint_positions, np.ndarray):
                row_data.extend(self.target_joint_positions.tolist())
            else:
                row_data.extend(self.target_joint_positions)
        else:
            # 如果没有目标位置，添加7个NaN值
            row_data.extend([float('nan')] * 7)
        
        # 写入CSV
        writer = csv.writer(self.joint_error_file)
        writer.writerow(row_data)
        self.joint_error_file.flush()
    
    def record_final_target_data(self):
        """记录最终目标关节位置(qrl)数据"""
        timestamp = rospy.Time.now().to_sec()
        
        if self.last_target_positions is not None:
            # 准备行数据
            row_data = [timestamp, self.current_target_index]
            
            # 添加最终目标关节位置
            if isinstance(self.last_target_positions, np.ndarray):
                row_data.extend(self.last_target_positions.tolist())
            else:
                row_data.extend(self.last_target_positions)
            
            # 写入CSV
            writer = csv.writer(self.final_target_file)
            writer.writerow(row_data)
            self.final_target_file.flush()
            
            rospy.loginfo(f"记录了新的最终目标点 (qrl) 到CSV，轨迹ID: {self.current_target_index}")
    
    def record_dsm_data(self):
        """记录DSM数据 - 简化版本"""
        timestamp = rospy.Time.now().to_sec()
        
        writer = csv.writer(self.dsm_data_file)
        writer.writerow([
            timestamp,
            self.human_dsm,
            self.object_dsm
        ])
        self.dsm_data_file.flush()
    
    def record_safety_data(self):
        """记录安全相关数据"""
        timestamp = rospy.Time.now().to_sec()
        
        writer = csv.writer(self.safety_data_file)
        writer.writerow([
            timestamp,
            self.min_obstacle_distance,
            self.nearest_object_type,
            *self.nearest_object_position
        ])
        self.safety_data_file.flush()
    
    def record_force_data(self):
        """记录力数据"""
        timestamp = rospy.Time.now().to_sec()
        
        writer = csv.writer(self.force_data_file)
        writer.writerow([
            timestamp,
            self.attractive_force,
            self.repulsive_force,
            self.total_force
        ])
        self.force_data_file.flush()
    
    def record_trajectory_data(self):
        """记录轨迹执行数据"""
        timestamp = rospy.Time.now().to_sec()
        
        # 默认值
        joint_positions = [float('nan')] * 7
        joint_velocities = [float('nan')] * 7
        joint_efforts = [float('nan')] * 7
        ee_pos = [float('nan')] * 3
        ee_quat = [float('nan')] * 4
        ee_velocity = [float('nan')] * 6
        target_pos = [float('nan')] * 3
        target_quat = [float('nan')] * 4
        
        # 更新有效值
        if self.actual_joint_positions is not None:
            joint_positions = self.actual_joint_positions
        
        if self.actual_joint_velocities is not None:
            joint_velocities = self.actual_joint_velocities
        
        if self.actual_joint_efforts is not None:
            joint_efforts = self.actual_joint_efforts
        
        if self.end_effector_pose:
            ee_pos = [
                self.end_effector_pose.position.x,
                self.end_effector_pose.position.y,
                self.end_effector_pose.position.z
            ]
            ee_quat = [
                self.end_effector_pose.orientation.x,
                self.end_effector_pose.orientation.y,
                self.end_effector_pose.orientation.z,
                self.end_effector_pose.orientation.w
            ]
        
        ee_velocity = self.end_effector_velocity
        
        if self.target_pose:
            target_pos = [
                self.target_pose.position.x,
                self.target_pose.position.y,
                self.target_pose.position.z
            ]
            target_quat = [
                self.target_pose.orientation.x,
                self.target_pose.orientation.y,
                self.target_pose.orientation.z,
                self.target_pose.orientation.w
            ]
        
        writer = csv.writer(self.trajectory_data_file)
        row_data = [timestamp]
        
        # 添加关节位置、速度和扭矩
        row_data.extend(joint_positions)
        row_data.extend(joint_velocities)
        row_data.extend(joint_efforts)
        
        # 添加末端执行器位姿和速度
        row_data.extend(ee_pos)
        row_data.extend(ee_quat)
        row_data.extend(ee_velocity)
        
        # 添加目标位姿
        row_data.extend(target_pos)
        row_data.extend(target_quat)
        
        # 添加轨迹进度数据
        row_data.extend([
            self.trajectory_progress,
            self.current_target_index,
            self.total_target_count
        ])
        
        writer.writerow(row_data)
        self.trajectory_data_file.flush()
    
    def record_human_skeleton_data(self):
        """记录人体骨架数据（左右手四连杆）"""
        timestamp = rospy.Time.now().to_sec()
        
        # 提取左手四连杆关节位置
        left_shoulder = self.human_joints[11]
        left_elbow = self.human_joints[13]
        left_wrist = self.human_joints[15]
        left_hand = self.human_joints[19]
        
        # 提取右手四连杆关节位置
        right_shoulder = self.human_joints[12]
        right_elbow = self.human_joints[14]
        right_wrist = self.human_joints[16]
        right_hand = self.human_joints[20]
        
        writer = csv.writer(self.human_skeleton_file)
        writer.writerow([
            timestamp,
            # 左肩
            left_shoulder['x'], left_shoulder['y'], left_shoulder['z'],
            # 左肘
            left_elbow['x'], left_elbow['y'], left_elbow['z'],
            # 左腕
            left_wrist['x'], left_wrist['y'], left_wrist['z'],
            # 左手
            left_hand['x'], left_hand['y'], left_hand['z'],
            # 右肩
            right_shoulder['x'], right_shoulder['y'], right_shoulder['z'],
            # 右肘
            right_elbow['x'], right_elbow['y'], right_elbow['z'],
            # 右腕
            right_wrist['x'], right_wrist['y'], right_wrist['z'],
            # 右手
            right_hand['x'], right_hand['y'], right_hand['z']
        ])
        self.human_skeleton_file.flush()
    def record_fuzzy_index_data(self):
        writer = csv.writer(self.fuzzy_index_file)
        row_data = [self.raw_pi, self.pi, self.ri, self.euclidean, self.velocity, self.distance]
        writer.writerow(row_data)
        self.fuzzy_index_file.flush()
    
    def run(self):
        """运行数据收集器"""
        rospy.loginfo("数据收集器已准备就绪，等待机器人开始运动...")
        
        while not rospy.is_shutdown():
            try:
                # 只有在记录状态时才记录数据
                if self.is_recording:
                    self.record_joint_error_data()
                    self.record_dsm_data()
                    self.record_safety_data()
                    self.record_trajectory_data()
                    self.record_force_data()
                    self.record_human_skeleton_data()
                    self.record_fuzzy_index_data()
                
            except Exception as e:
                rospy.logerr(f"数据记录错误: {e}")
            
            self.record_rate.sleep()
        
        # 关闭文件
        self.joint_error_file.close()
        self.dsm_data_file.close()
        self.safety_data_file.close()
        self.trajectory_data_file.close()
        self.force_data_file.close()
        self.target_points_file.close()
        self.final_target_file.close()  # 关闭新增的qrl记录文件
        self.human_skeleton_file.close()
        
        rospy.loginfo("数据收集完成")

if __name__ == '__main__':
    try:
        collector = SimplifiedRobotDataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass