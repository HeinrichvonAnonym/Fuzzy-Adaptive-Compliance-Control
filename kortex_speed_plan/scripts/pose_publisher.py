import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import sys
import numpy as np
from std_msgs.msg import Float32
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion
import tf
from tf.transformations import quaternion_matrix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import math
import time
import yaml

class GraspTrajectoryPublisher:
    def __init__(self):
        rospy.init_node("grasp_trajectory_publisher", anonymous=True)
        
        # 创建发布器，用于发布目标位姿
        self.pose_publisher = rospy.Publisher("/rrt/target_pose", PoseStamped, queue_size=10)
        
        # 添加目标轨迹发布器
        self.trajectory_publisher = rospy.Publisher("/rrt/target_trajectory", PoseArray, queue_size=10)
        
        # 添加位置误差发布器 - 用于数据收集
        self.position_error_pub = rospy.Publisher("/rrt/position_error", Float32MultiArray, queue_size=10)
        
        # 添加轨迹进度发布器 - 用于数据收集
        self.trajectory_progress_pub = rospy.Publisher("/rrt/trajectory_progress", Float32MultiArray, queue_size=10)
        
        # 订阅机器人位姿信息
        self.current_pose_subscriber = rospy.Subscriber("/rrt/robot_pose", PoseArray, self.current_pose_callback)
        
        # 夹爪控制
        self.gripper_position_pub = rospy.Publisher('gripper/set_position', Float32, queue_size=10)
        self.grasp_service = rospy.ServiceProxy('gripper/grasp', Trigger)
        self.release_service = rospy.ServiceProxy('gripper/release', Trigger)
        
        # 初始化TF监听器 - 用于获取目标物体位置
        self.tf_listener = tf.TransformListener()
        
        # 创建基本的姿态消息结构
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "base_link"
        self.pose = Pose()
        
        # 初始化当前位置和目标轨迹
        self.current_end_effector_pose = None
        self.start_pose = None           # 初始位置，夹爪竖直向下
        self.object_pose = None          # 默认抓取位置，夹爪竖直向下但绕z轴旋转90度
        self.place_pose = None           # 放置位置，夹爪朝向x轴正方向
        self.return_start_pose = None    # 返回初始位置，夹爪竖直向下
        
        # 当前目标："start", "object", "place", "return_start"
        self.current_target = "start"  
        self.position_tolerance = 0.02  # 位置容差，2厘米
        self.target_reached = False
        self.trajectory_completed = False
        
        # 抓取状态
        self.gripper_open = False
        self.object_grasped = False
        
        # 添加轨迹开始时间 - 用于数据收集
        self.trajectory_start_time = None
        # 添加当前轨迹点起始时间 - 用于数据收集
        self.current_point_start_time = None
        
        # 存储所有轨迹点，以便计算进度
        self.all_trajectory_poses = []

        self.current_loop = 0
        self.loop_num = 0
        self.start_poses = []
        self.object_poses = []
        self.place_poses = []
        self.return_start_poses = []

    def create_default_orientation(self):
        """创建默认的末端执行器朝向（竖直向下）"""
        # 创建一个单位四元数（无旋转）
        self.pose = Pose()
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = 0.0
        self.pose.orientation.w = 1.0
        
        # 对初始姿态进行x轴旋转π，使末端执行器竖直向下
        self.x_axis_rot(np.pi)
    
    def quternion_multiplication(self, q1, q2):
        w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
        x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
        y = q1.w * q2.y + q1.x * q2.z + q1.y * q2.w - q1.z * q2.x
        z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w 
        return w, x, y, z

    def x_axis_rot(self, step):
        q = tf.transformations.quaternion_from_euler(step, 0, 0)
        q = Quaternion(q[0], q[1], q[2], q[3])
        w, x, y, z = self.quternion_multiplication(self.pose.orientation, q)
        self.pose.orientation.x = x
        self.pose.orientation.y = y
        self.pose.orientation.z = z
        self.pose.orientation.w = w

    def y_axis_rot(self, step):
        q = tf.transformations.quaternion_from_euler(0, step, 0)
        q = Quaternion(q[0], q[1], q[2], q[3])
        w, x, y, z = self.quternion_multiplication(self.pose.orientation, q)
        self.pose.orientation.x = x
        self.pose.orientation.y = y
        self.pose.orientation.z = z
        self.pose.orientation.w = w

    def z_axis_rot(self, step):
        q = tf.transformations.quaternion_from_euler(0, 0, step)
        q = Quaternion(q[0], q[1], q[2], q[3])
        w, x, y, z = self.quternion_multiplication(self.pose.orientation, q)
        self.pose.orientation.x = x
        self.pose.orientation.y = y
        self.pose.orientation.z = z
        self.pose.orientation.w = w
    
    def wait_for_services(self):
        """等待夹爪服务可用"""
        rospy.loginfo("等待夹爪服务...")
        try:
            rospy.wait_for_service('gripper/grasp', timeout=5.0)
            rospy.wait_for_service('gripper/release', timeout=5.0)
            rospy.loginfo("夹爪服务已就绪")
            return True
        except rospy.ROSException:
            rospy.logwarn("夹爪服务不可用，但程序将继续")
            return False

    def open_gripper(self):
        """打开夹爪"""
        try:
            rospy.loginfo("正在打开夹爪...")
            response = self.release_service()
            rospy.loginfo(f"夹爪释放服务响应: {response}")
            self.gripper_open = True
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"释放服务调用失败: {e}")
            return False

    def close_gripper(self):
        """闭合夹爪"""
        try:
            rospy.loginfo("正在闭合夹爪...")
            response = self.grasp_service()
            rospy.loginfo(f"夹爪抓取服务响应: {response}")
            self.gripper_open = False
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"抓取服务调用失败: {e}")
            return False

    def setup_poses(self, config="/home/heinrich/kinova/src/kortex_speed_plan/scripts/task_poeses.yaml"):
        """设置起点、默认抓取位置、放置位置和返回起点位置"""
        # 创建默认的末端执行器朝向（竖直向下）
        self.start_poses = []
        self.object_poses = []
        self.place_poses = []
        self.return_start_poses = []

        with open(config, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)['wayposes']
            print(data)
        self.loop_num = len(data)
        for loop in data:
            # print(loop)
            self.create_default_orientation()
            
            # 起点: 初始位置，末端执行器竖直向下
            self.start_pose = Pose()
            self.start_pose.position.x = loop['start_pose']['position']['x']
            self.start_pose.position.y = loop['start_pose']['position']['y']
            self.start_pose.position.z = loop['start_pose']['position']['z']
            self.start_pose.orientation = self.pose.orientation  # 已经设置为竖直向下
            self.start_poses.append(self.start_pose)
            
            # 保存起点姿态的拷贝，用于返回起点
            self.return_start_pose = Pose()
            self.return_start_pose.position.x = loop['return_start_pose']['position']['x']
            self.return_start_pose.position.y = loop['return_start_pose']['position']['y']
            self.return_start_pose.position.z = loop['return_start_pose']['position']['z']
            self.return_start_pose.orientation = self.start_pose.orientation
            self.return_start_poses.append(self.return_start_pose)
            
            # 放置位置: 位置(0.5, 0.4, 0.6)，绕y轴旋转-π/2，使夹爪朝向x轴正方向
            self.place_pose = Pose()
            self.place_pose.position.x = loop['place_pose']['position']['x']
            self.place_pose.position.y = loop['place_pose']['position']['y']
            self.place_pose.position.z = loop['place_pose']['position']['z']
            
            # 计算初始姿态的欧拉角（竖直向下，对应于绕x轴旋转π）
            initial_euler = tf.transformations.euler_from_quaternion([
                self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w
            ])
            
            # 计算放置位置的欧拉角（绕y轴旋转-π/2）
            place_euler = list(initial_euler)
            place_euler[1] += -(np.pi/2)  # y轴旋转-π/2
            
            # 放置位置姿态 - 从初始姿态（竖直向下）再绕y轴旋转-π/2
            q_place = tf.transformations.quaternion_from_euler(place_euler[0], place_euler[1], place_euler[2])
            self.place_pose.orientation.x = q_place[0]
            self.place_pose.orientation.y = q_place[1]
            self.place_pose.orientation.z = q_place[2]
            self.place_pose.orientation.w = q_place[3]
            
            self.place_poses.append(self.place_pose)
            # 设置默认抓取位置
            self.setup_default_object_pose(loop['object_pose']['position']['x'],
                                           loop['object_pose']['position']['y'],
                                           loop['object_pose']['position']['z'])
            self.object_poses.append(self.object_pose)
                
            # 保存所有轨迹点到一个列表中，以便计算进度
        self.all_trajectory_poses = [
            self.start_poses,
            self.object_poses,
            self.place_poses,
            self.return_start_poses
        ]
    
    def setup_default_object_pose(self, x ,y, z):
        """设置默认的抓取位置"""
        # 默认抓取位置: (0.35, -0.35, 0.42)，保持竖直向下的姿态并绕z轴旋转90度
        self.object_pose = Pose()
        self.object_pose.position.x = x
        self.object_pose.position.y = y
        self.object_pose.position.z = z
        
        # 复制起点姿态作为基础
        self.object_pose.orientation = self.pose.orientation
        z_rotated_pose = Pose()
        z_rotated_pose.orientation = self.pose.orientation
        z_rotated_pose.position = self.pose.position
        
        # 绕z轴旋转90度
        q = tf.transformations.quaternion_from_euler(0, 0, np.pi/2)
        q = Quaternion(q[0], q[1], q[2], q[3])
        w, x, y, z = self.quternion_multiplication(z_rotated_pose.orientation, q)
        self.object_pose.orientation.x = x
        self.object_pose.orientation.y = y
        self.object_pose.orientation.z = z
        self.object_pose.orientation.w = w
    
    
    def current_pose_callback(self, pose_array_msg):
        """接收当前机器人位置的回调函数"""
        if len(pose_array_msg.poses) >= 2:
            # 根据提供的代码，第二个位姿是末端执行器的位姿
            self.current_end_effector_pose = pose_array_msg.poses[0]
            
            # 检查是否已经到达当前目标点 not self.target_reached and
            if  not self.trajectory_completed:
                self.check_if_target_reached()
                
            # 发布位置误差数据
            if not self.trajectory_completed:
                self.publish_position_error()
        else:
            rospy.logwarn("收到的PoseArray消息中没有足够的位姿信息")
    
    def check_if_target_reached(self):
        """检查机器人是否到达当前目标点"""
        if self.current_end_effector_pose is None:
            return
            
        # 确定当前目标
        # print(self.current_target)
        target_pose = None
        if self.current_target == "start" and len(self.start_poses) > self.current_loop:
            target_pose = self.start_poses[self.current_loop]
            target_desc = "起点"
            target_index = 0
        elif self.current_target == "object" and len(self.object_poses) > self.current_loop:
            target_pose = self.object_poses[self.current_loop]
            target_desc = "抓取位置"
            target_index = 1
        elif self.current_target == "place" and len(self.place_poses) > self.current_loop:
            target_pose = self.place_poses[self.current_loop]
            target_desc = "放置位置"
            target_index = 2
        elif self.current_target == "return_start" and len(self.return_start_poses) > self.current_loop:
            target_pose = self.return_start_poses[self.current_loop]
            target_desc = "返回起点"
            target_index = 3
        
        # 如果目标位姿未设置，返回
        if target_pose is None:
            rospy.logwarn(f"当前目标 {self.current_target} 位姿未设置")
            return
        
      
        # 计算当前位置与目标位置之间的距离
        dx = self.current_end_effector_pose.position.x - target_pose.position.x
        dy = self.current_end_effector_pose.position.y - target_pose.position.y
        dz = self.current_end_effector_pose.position.z - target_pose.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 如果距离小于容差，认为已经到达目标点
        if distance < self.position_tolerance:
            rospy.loginfo(f"已到达{target_desc}，距离: {distance:.3f}m")
            self.target_reached = True
            
            # 计算当前点所花费的时间
            current_point_time = 0
            if self.current_point_start_time:
                current_point_time = time.time() - self.current_point_start_time
                rospy.loginfo(f"{target_desc}执行时间: {current_point_time:.2f}秒")
            
            
                
            # 执行到达当前目标点后的操作
            if self.current_target == "start":
                # 到达起点后打开夹爪
                rospy.loginfo("到达起点，打开夹爪")
                self.open_gripper()
                rospy.sleep(0.2)  # 等待夹爪动作完成
                
                # 切换到下一个目标点
                self.current_target = "object"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始前往抓取位置")
                
                
                # 发布轨迹进度
                self.publish_trajectory_progress(0.25, current_point_time)
            
            elif self.current_target == "object":
                # 到达物体位置后闭合夹爪
                rospy.loginfo("到达抓取位置，闭合夹爪")
                self.close_gripper()
                # self.close_gripper()
                self.object_grasped = True
                rospy.sleep(0.2)  # 等待夹爪动作完成
                
                # 切换到下一个目标点
                self.current_target = "place"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始前往放置位置")
                
                # 发布轨迹进度
                self.publish_trajectory_progress(0.5, current_point_time)
            
            elif self.current_target == "place":
                # 到达放置位置后打开夹爪
                rospy.loginfo("到达放置位置，打开夹爪")
                self.open_gripper()
                # self.open_gripper()
                self.object_grasped = False
                rospy.sleep(0.2)  # 等待夹爪动作完成
                
                # 切换到下一个目标点
                self.current_target = "return_start"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始返回起点")
                
                # 发布轨迹进度
                self.publish_trajectory_progress(0.75, current_point_time)
            
            elif self.current_target == 'return_start':
                self.current_target = "object"
                self.current_loop += 1
                rospy.loginfo("到达返回起点，next loop")
            
            if self.current_loop >= self.loop_num:  # "return_start"
                # 如果当前目标是返回起点，标记轨迹完成
                rospy.loginfo("到达返回起点，轨迹执行完成")
                self.trajectory_completed = True
                
                # 发布最终的轨迹进度
                self.publish_trajectory_progress(1.0, current_point_time)
                
                # 计算总体执行时间
                if self.trajectory_start_time is not None:
                    total_time = time.time() - self.trajectory_start_time
                    rospy.loginfo(f"轨迹总体执行时间: {total_time:.2f}秒")
    
    def publish_current_target(self):
        """发布当前目标点"""
        if not self.trajectory_completed:
            # 确定当前目标
            target_pose = None
            if self.current_target == "start" and len(self.start_poses)>self.current_loop:
                target_pose = self.start_poses[self.current_loop]
                target_desc = "起点"
            elif self.current_target == "object" and  len(self.start_poses)>=self.current_loop:
                target_pose = self.object_poses[self.current_loop]
                target_desc = "抓取位置"
            elif self.current_target == "place" and len(self.start_poses)>self.current_loop:
                target_pose = self.place_poses[self.current_loop]
                target_desc = "放置位置"
            elif self.current_target == "return_start" and len(self.start_poses)>self.current_loop:
                target_pose = self.return_start_poses[self.current_loop]
                target_desc = "返回起点"
            
            # 如果目标位姿未设置，返回
            if target_pose is None:
                rospy.logwarn(f"当前目标 {self.current_target} 位姿未设置，跳过发布")
                return
            
            # 更新消息
            # print(self.pose_msg)
            self.pose_msg.pose = target_pose
            self.pose_msg.header.stamp = rospy.Time.now()
            self.pose_publisher.publish(self.pose_msg)
            
            # 打印当前与目标之间的距离，方便调试
            if self.current_end_effector_pose and not self.target_reached:
                dx = self.current_end_effector_pose.position.x - target_pose.position.x
                dy = self.current_end_effector_pose.position.y - target_pose.position.y
                dz = self.current_end_effector_pose.position.z - target_pose.position.z
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                rospy.logdebug(f"当前距离{target_desc}: {distance:.3f}m")
                
    def publish_position_error(self):
        """发布当前位置误差数据"""
        if self.current_end_effector_pose is None:
            return
            
        # 确定当前目标
        target_pose = None
        if self.current_target == "start" and len(self.start_poses) > self.current_loop:
            target_pose = self.start_poses[self.current_loop]
            target_index = 0
        elif self.current_target == "object" and len(self.object_poses) > self.current_loop:
            target_pose = self.object_poses[self.current_loop]
            target_index = 1
        elif self.current_target == "place" and len(self.place_poses) > self.current_loop:
            target_pose = self.place_poses[self.current_loop]
            target_index = 2
        elif self.current_target == "return_start" and len(self.return_start_poses) > self.current_loop:
            target_pose = self.return_start_poses[self.current_loop]
            target_index = 3
            
        # 如果目标位姿未设置，返回
        if target_pose is None:
            return
            
        # 计算位置误差
        dx = self.current_end_effector_pose.position.x - target_pose.position.x
        dy = self.current_end_effector_pose.position.y - target_pose.position.y
        dz = self.current_end_effector_pose.position.z - target_pose.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 计算方向误差 (简化为欧拉角差异)
        current_quat = [
            self.current_end_effector_pose.orientation.x,
            self.current_end_effector_pose.orientation.y,
            self.current_end_effector_pose.orientation.z,
            self.current_end_effector_pose.orientation.w
        ]
        target_quat = [
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w
        ]
        
        current_euler = tf.transformations.euler_from_quaternion(current_quat)
        target_euler = tf.transformations.euler_from_quaternion(target_quat)
        
        orientation_error = [
            abs(current_euler[0] - target_euler[0]),
            abs(current_euler[1] - target_euler[1]),
            abs(current_euler[2] - target_euler[2])
        ]
        
        # 发布误差数据
        error_msg = Float32MultiArray()
        error_msg.data = [
            dx, dy, dz, distance,  # 位置误差
            orientation_error[0], orientation_error[1], orientation_error[2],  # 方向误差
            float(target_index)  # 当前目标点索引
        ]
        self.position_error_pub.publish(error_msg)
        
    def publish_trajectory_progress(self, progress=None, point_time=None):
        """发布轨迹执行进度数据"""
        if self.trajectory_start_time is None:
            return
            
        # 计算轨迹总时间
        trajectory_time = 0
        if self.trajectory_start_time:
            trajectory_time = time.time() - self.trajectory_start_time
        
        # 如果未提供point_time，计算当前点的执行时间
        if point_time is None and self.current_point_start_time:
            point_time = time.time() - self.current_point_start_time
        
        # 如果未提供进度，根据当前目标计算
        if progress is None:
            # 4个关键点，每个点占25%的进度
            if self.current_target == "start":
                progress = 0.0
            elif self.current_target == "object":
                progress = 0.25
            elif self.current_target == "place":
                progress = 0.5
            elif self.current_target == "return_start":
                progress = 0.75
            
            if self.trajectory_completed:
                progress = 1.0
        
        # 获取当前目标点索引
        target_index = self.get_target_index()
        
        # 创建并发布进度消息
        progress_msg = Float32MultiArray()
        progress_msg.data = [
            progress,                          # 轨迹进度百分比
            float(target_index),               # 当前目标点索引
            float(len(self.all_trajectory_poses[self.current_loop])),  # 轨迹点总数
            trajectory_time,                   # 轨迹执行总时间
            point_time if point_time is not None else 0.0  # 当前点执行时间
        ]
        self.trajectory_progress_pub.publish(progress_msg)
        
    def get_target_index(self):
        """获取当前目标点的索引"""
        if self.current_target == "start":
            return 0
        elif self.current_target == "object":
            return 1
        elif self.current_target == "place":
            return 2
        else:  # "return_start"
            return 3
            
    def get_all_poses(self):
        """获取所有路径点，用于发布完整轨迹"""
        poses = []
        if self.start_poses[self.current_loop]:
            poses.append(self.start_poses[self.current_loop])
        if self.object_poses[self.current_loop]:
            poses.append(self.object_poses[self.current_loop])
        if self.place_poses[self.current_loop]:
            poses.append(self.place_poses[self.current_loop])
        if self.return_start_poses[self.current_loop]:
            poses.append(self.return_start_poses[self.current_loop])
        return poses
        
    def publish_trajectory(self):
        """发布整个轨迹"""
        poses = self.get_all_poses()
        if poses:
            pose_array = PoseArray()
            pose_array.header.frame_id = "base_link"
            pose_array.header.stamp = rospy.Time.now()
            pose_array.poses = poses
            self.trajectory_publisher.publish(pose_array)
            rospy.loginfo(f"已发布完整轨迹，包含 {len(poses)} 个路径点")
    
    def run(self):
        """执行完整的抓取-放置任务流程"""
        # 等待夹爪服务
        self.wait_for_services()
        
        # 等待接收机器人位姿信息
        rospy.loginfo("等待机器人位姿信息...")
        timeout = 10.0  # 等待10秒
        start_time = rospy.Time.now().to_sec()
        while self.current_end_effector_pose is None and not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logwarn("等待机器人位姿信息超时，但程序将继续")
                break
            rospy.sleep(0.2)
    
        # 设置所有目标位姿
        self.setup_poses()
            
        rospy.loginfo("开始执行抓取-放置轨迹")
        rospy.loginfo("轨迹包含四个关键点：")
        rospy.loginfo("1. 起点: 位置(0.2, -0.2, 0.55)，末端执行器竖直向下")
        rospy.loginfo("2. 抓取位置: 默认位置(0.35, -0.35, 0.42)，末端执行器竖直向下并绕z轴旋转90度")
        rospy.loginfo("3. 放置位置: 位置(0.5, 0.4, 0.6)，末端执行器朝向x轴正方向")
        rospy.loginfo("4. 返回起点: 回到位置(0.2, -0.2, 0.55)，末端执行器竖直向下")
        
        # 发布整个轨迹，以便数据收集器记录
        self.publish_trajectory()
        
        # 初始化轨迹开始时间和当前点开始时间
        self.trajectory_start_time = time.time()
        self.current_point_start_time = time.time()
        
        # 发布初始轨迹进度
        self.publish_trajectory_progress(0.0, 0.0)
        
        rospy.loginfo("开始前往起点")
        
        rate = rospy.Rate(10)  # 10Hz，提高反馈频率
        
        while not rospy.is_shutdown():
            # 发布当前目标点
            self.publish_current_target()
            
            # 如果轨迹已经执行完毕，结束循环
            if self.trajectory_completed:
                rospy.loginfo("所有点都已到达，轨迹执行完毕")
                break
                
            rate.sleep()
        
        rospy.loginfo("轨迹发布器已结束")


if __name__ == "__main__":
    try:
        grasp_publisher = GraspTrajectoryPublisher()
        grasp_publisher.run()
    except rospy.ROSInterruptException:
        pass

    