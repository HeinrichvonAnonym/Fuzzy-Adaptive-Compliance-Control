import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.msg import CollisionObject, RobotTrajectory
import sys
import numpy as np
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from std_msgs.msg import Float32
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion
import scipy.spatial.transform as transform
import tf
from kortex_speed_plan.msg import SolidPrimitiveMultiArray
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import math
import time


class TrajectoryPublisher:
    def __init__(self):
        rospy.init_node("trajectory_publisher", anonymous=True)
        # 创建发布器，用于发布目标位姿
        self.pose_publisher = rospy.Publisher("/rrt/target_pose", PoseStamped, queue_size=10)
        
        # 添加目标轨迹发布器
        self.trajectory_publisher = rospy.Publisher("/rrt/target_trajectory", PoseArray, queue_size=10)
        
        # 添加位置误差发布器 - 用于数据收集
        self.position_error_pub = rospy.Publisher("/rrt/position_error", Float32MultiArray, queue_size=10)
        
        # 添加轨迹进度发布器 - 用于数据收集 (与pose_publisher相同接口)
        self.trajectory_progress_pub = rospy.Publisher("/rrt/trajectory_progress", Float32MultiArray, queue_size=10)
        
        # 订阅机器人位姿信息
        self.current_pose_subscriber = rospy.Subscriber("/rrt/robot_pose", PoseArray, self.current_pose_callback)
        
        # 初始化姿态
        self.init_pose()
        
        # 对初始姿态进行x轴旋转π，使末端执行器竖直向下
        self.x_axis_rot(np.pi)
        
        # 初始化当前位置和目标轨迹
        self.current_end_effector_pose = None
        self.start_pose = None
        self.middle_pose = None
        self.return_start_pose = None
        self.intermediate_pose = None
        self.final_pose = None
        # 当前目标："start", "middle", "return_start", "intermediate", "final"
        self.current_target = "start"  
        self.position_tolerance = 0.02  # 位置容差，2厘米
        self.target_reached = False
        self.trajectory_completed = False
        
        # 添加轨迹开始时间 - 用于数据收集
        self.trajectory_start_time = None
        # 添加当前轨迹点起始时间 - 用于数据收集
        self.current_point_start_time = None
        
        # 添加初始位置作为终点 (返回起点的逻辑)
        self.initial_pose = None
        
        # 存储所有轨迹点，以便计算进度
        self.all_trajectory_poses = []

        self.mid_pose_reached = False

    def init_pose(self):
        self.pose_msg = PoseStamped()
        self.pose = Pose()
        self.pose.position.x = 0.55
        self.pose.position.y = 0.0
        self.pose.position.z = 0.42
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = 0.0
        self.pose.orientation.w = 1.0
        self.pose_msg.pose = self.pose
        self.pose_msg.header.frame_id = "base_link"
        self.pose_msg.header.stamp = rospy.Time.now()
    
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

    def setup_poses(self):
        """设置起点、中间点、返回起点、中间终点和最终终点姿态"""
        # 起点: 初始位置，末端执行器竖直向下（通过绕x轴旋转π实现）
        self.start_pose = Pose()
        self.start_pose.position.x = self.pose.position.x
        self.start_pose.position.y = self.pose.position.y
        self.start_pose.position.z = self.pose.position.z
        self.start_pose.orientation = self.pose.orientation  # 已经通过x_axis_rot(np.pi)设置为竖直向下
        
        # 保存初始位置，用于最后返回
        self.initial_pose = Pose()
        self.initial_pose.position.x = self.pose.position.x
        self.initial_pose.position.y = self.pose.position.y
        self.initial_pose.position.z = self.pose.position.z
        self.initial_pose.orientation.x = self.pose.orientation.x
        self.initial_pose.orientation.y = self.pose.orientation.y
        self.initial_pose.orientation.z = self.pose.orientation.z
        self.initial_pose.orientation.w = self.pose.orientation.w
        
        # 保存起点姿态的拷贝，用于返回起点
        self.return_start_pose = Pose()
        self.return_start_pose.position.x = self.pose.position.x
        self.return_start_pose.position.y = self.pose.position.y
        self.return_start_pose.position.z = self.pose.position.z
        self.return_start_pose.orientation = self.pose.orientation
        
        # 中间点: 位置(0.5, 0.3, 0.4)，末端执行器竖直向下（保持与起点相同的姿态）
        self.middle_pose = Pose()
        self.middle_pose.position.x = 0.5
        self.middle_pose.position.y = 0.3
        self.middle_pose.position.z = 0.42
        self.middle_pose.orientation = self.pose.orientation  # 与起点相同，末端执行器竖直向下
        
        # 计算初始姿态的欧拉角（竖直向下，对应于绕x轴旋转π）
        initial_euler = tf.transformations.euler_from_quaternion([
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ])
        
        # 计算中间终点姿态的欧拉角（在初始姿态基础上再绕x轴旋转π/2，使末端执行器朝向x轴正方向）
        intermediate_euler = list(initial_euler)
        intermediate_euler[0] += -(np.pi/2)  # x轴再旋转π/2
        
        # 中间终点: 位置(0.5, 0.4, 0.525)，末端执行器朝向x轴正方向
        self.intermediate_pose = Pose()
        self.intermediate_pose.position.x = 0.5
        self.intermediate_pose.position.y = -0.4
        self.intermediate_pose.position.z = 0.6
        
        # 中间终点姿态 - 从初始姿态（竖直向下）再绕x轴旋转π/2
        q_intermediate = tf.transformations.quaternion_from_euler(intermediate_euler[0], intermediate_euler[1], intermediate_euler[2])
        self.intermediate_pose.orientation.x = q_intermediate[0]
        self.intermediate_pose.orientation.y = q_intermediate[1]
        self.intermediate_pose.orientation.z = q_intermediate[2]
        self.intermediate_pose.orientation.w = q_intermediate[3]
        
        # 计算最终终点姿态的欧拉角（绕y轴旋转π/2）
        final_euler = list(initial_euler)
        final_euler[1] += -(np.pi/2)  # y轴旋转π/2
        
        # 最终终点: 位置(0.5, 0.0, 0.6)，末端执行器绕y轴旋转90度
        self.final_pose = Pose()
        self.final_pose.position.x = 0.7
        self.final_pose.position.y = 0.0
        self.final_pose.position.z = 0.5
        
        # 最终终点姿态 - 从初始姿态（竖直向下）再绕y轴旋转π/2
        q_final = tf.transformations.quaternion_from_euler(final_euler[0], final_euler[1], final_euler[2])
        self.final_pose.orientation.x = q_final[0]
        self.final_pose.orientation.y = q_final[1]
        self.final_pose.orientation.z = q_final[2]
        self.final_pose.orientation.w = q_final[3]
        
        # 保存所有轨迹点到一个列表中，以便计算进度
        self.all_trajectory_poses = [
            self.start_pose,
            self.middle_pose,
            self.return_start_pose,
            self.intermediate_pose,
            self.final_pose
        ]
    
    def current_pose_callback(self, pose_array_msg):
        """接收当前机器人位置的回调函数"""
        if len(pose_array_msg.poses) >= 2:
            # 根据轨迹发布中等.py，第二个位姿是末端执行器的位姿
            self.current_end_effector_pose = pose_array_msg.poses[1]
            
            # 检查是否已经到达当前目标点
            if (self.start_pose or self.end_pose) and not self.target_reached and not self.trajectory_completed:
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
        if self.current_target == "start":
            target_pose = self.start_pose
            target_desc = "起点"
            target_index = 0
        elif self.current_target == "middle":
            target_pose = self.middle_pose
            target_desc = "中间点"
            target_index = 1
        elif self.current_target == "return_start":
            target_pose = self.return_start_pose
            target_desc = "返回起点"
            target_index = 2
        elif self.current_target == "intermediate":
            target_pose = self.intermediate_pose
            target_desc = "中间终点"
            target_index = 3
        else:  # "final"
            target_pose = self.final_pose
            target_desc = "最终终点"
            target_index = 4
        
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
            
            # 根据当前目标点切换到下一个目标点
            if self.current_target == "start":
                self.current_target = "middle"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始前往中间点")
            elif self.current_target == "middle":
                self.current_target = "return_start"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始返回起点")
            elif self.current_target == "return_start":
                self.current_target = "intermediate"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始前往中间终点")
            elif self.current_target == "intermediate":
                self.current_target = "final"
                self.target_reached = False
                self.current_point_start_time = time.time()
                rospy.loginfo("开始前往最终终点")
            else:  # "final"
                # 已完成所有路径点
                rospy.loginfo("轨迹执行完成，到达最终终点")
                self.trajectory_completed = True
                
                # 发布最终的轨迹进度
                self.publish_trajectory_progress(1.0, current_point_time)
                
                # 计算总体执行时间
                if self.trajectory_start_time is not None:
                    total_time = time.time() - self.trajectory_start_time
                    rospy.loginfo(f"轨迹总体执行时间: {total_time:.2f}秒")
            
            # 发布轨迹进度
            progress = float(target_index) / len(self.all_trajectory_poses)
            self.publish_trajectory_progress(progress, current_point_time)
    
    def publish_current_target(self):
        """发布当前目标点"""
        if not self.trajectory_completed:
            # 确定当前目标
            if self.current_target == "start":
                target_pose = self.start_pose
                target_desc = "起点"
            elif self.current_target == "middle":
                target_pose = self.middle_pose
                target_desc = "中间点"
            elif self.current_target == "return_start":
                target_pose = self.return_start_pose
                target_desc = "返回起点"
            elif self.current_target == "intermediate":
                target_pose = self.intermediate_pose
                target_desc = "中间终点"
            else:  # "final"
                target_pose = self.final_pose
                target_desc = "最终终点"
            
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
        if self.current_target == "start":
            target_pose = self.start_pose
            target_index = 0
        elif self.current_target == "middle":
            target_pose = self.middle_pose
            target_index = 1
        elif self.current_target == "return_start":
            target_pose = self.return_start_pose
            target_index = 2
        elif self.current_target == "intermediate":
            target_pose = self.intermediate_pose
            target_index = 3
        else:  # "final"
            target_pose = self.final_pose
            target_index = 4
            
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
    
    def publish_trajectory_progress(self, progress, point_time):
        """发布轨迹进度信息 (与pose_publisher一致的接口)"""
        # 计算轨迹总时间
        trajectory_time = 0
        if self.trajectory_start_time:
            trajectory_time = time.time() - self.trajectory_start_time
        
        # 获取当前目标点索引
        target_index = 0
        if self.current_target == "start":
            target_index = 0
        elif self.current_target == "middle":
            target_index = 1
        elif self.current_target == "return_start":
            target_index = 2
        elif self.current_target == "intermediate":
            target_index = 3
        else:  # "final"
            target_index = 4
        
        # 创建并发布进度消息
        progress_msg = Float32MultiArray()
        progress_msg.data = [
            progress,                          # 轨迹进度百分比
            float(target_index),               # 当前目标点索引
            float(len(self.all_trajectory_poses)),  # 轨迹点总数
            trajectory_time,                   # 轨迹执行总时间
            point_time                         # 当前点执行时间
        ]
        self.trajectory_progress_pub.publish(progress_msg)
        
    def get_all_poses(self):
        """获取所有路径点，用于发布完整轨迹"""
        return self.all_trajectory_poses
        
    def publish_trajectory(self):
        """发布整个轨迹"""
        poses = self.get_all_poses()
        if poses:
            pose_array = PoseArray()
            pose_array.header.frame_id = "base_link"
            pose_array.header.stamp = rospy.Time.now()
            pose_array.poses = poses
            self.trajectory_publisher.publish(pose_array)
        
    def run(self):
        # 设置起点、中间点、返回起点、中间终点、最终终点和初始位置
        self.setup_poses()
        
        # 等待接收机器人位姿信息
        rospy.loginfo("等待机器人位姿信息...")
        while self.current_end_effector_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.2)
        
        if self.current_end_effector_pose is None:
            rospy.logerr("无法获取机器人位姿信息，退出程序")
            return
            
        rospy.loginfo("获取到机器人位姿信息，开始执行轨迹")
        rospy.loginfo("轨迹包含五个关键点：")
        rospy.loginfo("1. 起点: 初始位置，末端执行器竖直向下")
        rospy.loginfo("2. 中间点: 位置(0.5, 0.3, 0.42)，末端执行器竖直向下")
        rospy.loginfo("3. 返回起点: 初始位置，末端执行器竖直向下")
        rospy.loginfo("4. 中间终点: 位置(0.5, -0.4, 0.6)，末端执行器朝向x轴正方向")
        rospy.loginfo("5. 最终终点: 位置(0.7, 0.0, 0.5)，末端执行器绕y轴旋转90度")
        
        # 发布整个轨迹，以便数据收集器记录
        self.publish_trajectory()
        
        # 初始化轨迹开始时间和当前点开始时间
        self.trajectory_start_time = time.time()
        self.current_point_start_time = time.time()
        
        # 发布初始轨迹进度
        self.publish_trajectory_progress(0.0, 0.0)
        
        rospy.loginfo("开始前往起点")
        
        rate = rospy.Rate(5)  # 5Hz，与robot_pose话题刷新率相匹配
        
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
        trajectory_publisher = TrajectoryPublisher()
        trajectory_publisher.run()
    except rospy.ROSInterruptException:
        pass