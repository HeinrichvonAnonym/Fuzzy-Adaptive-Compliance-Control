#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from moveit_msgs.msg import RobotTrajectory
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float32MultiArray, Int16
from visualization_msgs.msg import MarkerArray, Marker
import csv
import os
from datetime import datetime
import cv2
import socket
import json

server_ip = "192.168.31.183"
server_port = 8080


class SocketClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
    def send_data(self, data):
        self.socket.sendall(data)

class SimplifiedRobotDataCollector:
    def __init__(self, episode=1):
        # 初始化节点
        rospy.init_node('simplified_robot_data_collector', anonymous=True)
        
        # 创建数据存储目录
        self.data_dir = os.path.expanduser('~/robot_data_logs')
        self.episode = episode
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 创建时间戳用于文件名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.act_file = self.init_csv_file('act')
        self.vid_file_1_path = self.init_video_file(1)
        self.vid_file_2_path = self.init_video_file(2)
        self.vid_file_3_path = self.init_video_file(3)
        
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

        rospy.Subscriber('rrt/start_record', Int16, self.trajectory_callback)
        rospy.Subscriber('rrt/robot_pose', PoseArray, self.robot_pose_callback)
 

        rospy.Subscriber('rrt/force_values', Float32MultiArray, self.force_values_callback)
        rospy.Subscriber('rrt/trajectory_progress', Float32MultiArray, self.trajectory_progress_callback)

        rospy.Subscriber('/mrk/human_skeleton', MarkerArray, self.human_skeleton_callback)
        rospy.Subscriber('/kinect2/hd/image_color', Image, self.cam_1_cb)
        rospy.Subscriber('/camera_1/color/image_raw', Image, self.cam_2_cb)
        rospy.Subscriber('/camera_2/color/image_raw', Image, self.cam_3_cb)

        

        self.bridge = CvBridge()
        self.image_1 = None
        self.image_2 = None
        self.image_3 = None
        self.codec = rospy.get_param('~codec', 'XVID')

        
        # 设置数据记录频率
        hz = 20
        self.record_rate = rospy.Rate(hz)  # 10Hz
        self.fps = hz
        self.vid_writer_1 = None
        self.vid_writer_2 = None
        self.vid_writer_3 = None
        self.socket_client = SocketClient(server_ip, server_port)
        
        rospy.loginfo("简化版机器人数据收集器已初始化，等待轨迹开始...")
    
    def init_video_file(self, num:int):
        
        return os.path.join(self.data_dir, f"episode_{self.episode}", f"cam_{num}.avi")

    
    def init_csv_file(self, name):
        """初始化CSV文件并写入表头"""
        if not os.path.exists(os.path.join(self.data_dir, f"episode_{self.episode}")):
            os.makedirs(os.path.join(self.data_dir, f"episode_{self.episode}"))

        file_path = os.path.join(self.data_dir, f"episode_{self.episode}", "action.csv")
        f = open(file_path, 'w')
        writer = csv.writer(f)
        
        if name == 'act':
            writer.writerow(['timestamp', 
                             
                            # robot
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

                            'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                            'left_elbow_x', 'left_elbow_y', 'left_elbow_z',
                            'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
                            'left_hand_x', 'left_hand_y', 'left_hand_z',
                            'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                            'right_elbow_x', 'right_elbow_y', 'right_elbow_z',
                            'right_wrist_x', 'right_wrist_y', 'right_wrist_z',
                            'right_hand_x', 'right_hand_y', 'right_hand_z',
                            ])
        
        return f
    
    def cam_1_cb(self, msg:Image):
        image = self.bridge.imgmsg_to_cv2(msg)
        # cv2.imshow("cam1", image)
        # cv2.waitKey(30)
        self.image_1 = image

    def cam_2_cb(self, msg:Image):
        image = self.bridge.imgmsg_to_cv2(msg)
        # cv2.imshow("cam1", image)
        # cv2.waitKey(30)
        self.image_2 = image

    def cam_3_cb(self, msg:Image):
        image = self.bridge.imgmsg_to_cv2(msg)
        # cv2.imshow("cam1", image)
        # cv2.waitKey(30)
        self.image_3 = image
    
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
    
    
            self.current_target_index += 1
    
    def trajectory_callback(self, msg):
        if msg.data == 1:
            self.is_recording = True
            data = "1"
            data = data.encode("utf-8")
            self.socket_client.send_data(data)
            print(">>")
        elif msg.data == 0:
            self.is_recording = False
            data = "0"
            data = data.encode("utf-8")
            self.socket_client.send_data(data)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.episode += 1
            self.act_file = self.init_csv_file('act')
            self.vid_file_1_path = self.init_video_file(1)
            self.vid_file_2_path = self.init_video_file(2)
            self.vid_file_3_path = self.init_video_file(3)
            self.vid_writer_1 = None
            self.vid_writer_2 = None
            self.vid_writer_3 = None
            print("---")
    
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
        row_data = [timestamp, self.joint_errors['q_minus_qr_norm'], self.joint_errors['q_minus_qrl_norm'], int(self.is_final_target)]
        
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
    


    def record_act_data(self):
        """ Robot """
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
        
        writer = csv.writer(self.act_file)
        row_data = [timestamp]
        row_data.extend(joint_positions)
        row_data.extend(joint_velocities)
        row_data.extend(joint_efforts)
        
        # 添加末端执行器位姿和速度
        row_data.extend(ee_pos)
        row_data.extend(ee_quat)
        row_data.extend(ee_velocity)

        row_data.extend([left_shoulder['x'], 
                         left_shoulder['y'], 
                         left_shoulder['z'],
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
            right_hand['x'], right_hand['y'], right_hand['z']])
        
        writer.writerow(
            row_data  
        )
        self.act_file.flush()
        if self.image_1 is not None:
            if self.vid_writer_1 is None:
                frame_size = (self.image_1.shape[1], self.image_1.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.vid_writer_1 = cv2.VideoWriter(self.vid_file_1_path, fourcc, self.fps, frame_size)
            self.vid_writer_1.write(self.image_1)
        
        if self.image_2 is not None:
            if self.vid_writer_2 is None:
                frame_size = (self.image_2.shape[1], self.image_2.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.vid_writer_2 = cv2.VideoWriter(self.vid_file_2_path, fourcc, self.fps, frame_size)
            self.vid_writer_2.write(self.image_2)
        
        if self.image_3 is not None:
            if self.vid_writer_3 is None:
                frame_size = (self.image_3.shape[1], self.image_3.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.vid_writer_3 = cv2.VideoWriter(self.vid_file_3_path, fourcc, self.fps, frame_size)
            self.vid_writer_3.write(self.image_3)
    
    def run(self):
        """运行数据收集器"""
        rospy.loginfo("数据收集器已准备就绪，等待机器人开始运动...")
        
        while not rospy.is_shutdown():
            try:
                # 只有在记录状态时才记录数据
                if self.is_recording:
                    self.record_act_data()
 
                
            except Exception as e:
                rospy.logerr(f"数据记录错误: {e}")
            
            self.record_rate.sleep()
        
        
        
        rospy.loginfo("数据收集完成")

if __name__ == '__main__':
    try:
        collector = SimplifiedRobotDataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass