#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import moveit_commander
from moveit_commander import MoveGroupCommander
import tf.transformations as tf
from std_msgs.msg import Int16

class TargetAdapter:
    def __init__(self):
        rospy.init_node('target_adapter', anonymous=True)
        
        # 初始化MoveIt接口
        moveit_commander.roscpp_initialize([])
        self.arm = MoveGroupCommander("arm")
        
        # 设置规划参数 - 平衡规划时间和频率
        self.arm.set_planning_time(10.0)  # 10秒规划时间
        self.arm.set_num_planning_attempts(1)  # 单次尝试
        self.arm.set_goal_tolerance(0.02)  # 2cm容差
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        
        # 订阅目标位置话题
        rospy.Subscriber('/target_pose', Pose, self.target_callback)
        
        # 发布完整轨迹到轨迹规划器
        self.trajectory_pub = rospy.Publisher("rrt/cartesian_trajectory", RobotTrajectory, queue_size=1)
        
        # 订阅刷新信号，用于知道何时规划新轨迹
        rospy.Subscriber("rrt/refresh", Int16, self.refresh_callback)
        
        self.target_pose = None
        self.planning_active = False
        self.rate = rospy.Rate(0.2)  # 降低到0.2Hz，每5秒尝试一次规划，与原始系统相似
        
        # 上次成功规划的目标
        self.last_planned_target = None
        
        # 上次发布的轨迹
        self.last_trajectory = None
        
        # 设置默认目标位置
        self.default_target = Pose()
        self.default_target.position.x = 0.4
        self.default_target.position.y = 0.0
        self.default_target.position.z = 0.45
        self.default_target.orientation.w = 1.0
        
        rospy.loginfo("Target adapter initialized with 10s planning time, 5s cycle")
    
    def target_callback(self, msg):
        """接收目标位置的回调函数"""
        self.target_pose = msg
    
    def refresh_callback(self, msg):
        """接收刷新信号的回调函数"""
        rospy.loginfo("Planning refresh signal received")
        self.planning_active = False
    
    def plan_trajectory(self, use_default=False):
        """规划到目标位置的完整轨迹"""
        target = self.default_target if use_default else self.target_pose
        
        if target is None:
            return False

        # 设置目标位姿
        self.arm.set_pose_target(target)
        
        # 获取当前位置，用于调试
        current_pose = self.arm.get_current_pose().pose
        
        # 生成完整轨迹
        rospy.loginfo("Planning trajectory from (%.3f, %.3f, %.3f) to %s (%.3f, %.3f, %.3f)", 
                     current_pose.position.x, current_pose.position.y, current_pose.position.z,
                     "DEFAULT" if use_default else "TARGET",
                     target.position.x, target.position.y, target.position.z)
        
        # 增加时间记录
        start_time = rospy.Time.now()
        rospy.loginfo("Starting planning with timeout of 10 seconds...")
        
        success, plan, planning_time, error_code = self.arm.plan()
        
        end_time = rospy.Time.now()
        actual_time = (end_time - start_time).to_sec()
        
        if success:
            rospy.loginfo("Trajectory planning succeeded in %.3f seconds (reported: %.3f)",
                         actual_time, planning_time)
            self.last_trajectory = plan
            self.last_planned_target = target
            return True
        else:
            rospy.logwarn("Trajectory planning failed with error code: %s after %.3f seconds", 
                         str(error_code), actual_time)
            return False
    
    def significant_target_change(self):
        """判断目标是否有显著变化，需要重新规划"""
        if self.last_planned_target is None or self.target_pose is None:
            return True
        
        dx = self.target_pose.position.x - self.last_planned_target.position.x
        dy = self.target_pose.position.y - self.last_planned_target.position.y
        dz = self.target_pose.position.z - self.last_planned_target.position.z
        
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        return distance > 0.02  # 2cm变化阈值
    
    def run(self):
        """主循环函数"""
        # 首先尝试规划到默认目标位置
        use_default = True
        default_attempts = 0
        max_default_attempts = 3
        
        while not rospy.is_shutdown():
            # 判断是否需要规划
            if not self.planning_active:
                success = False
                
                # 如果还在尝试默认位置
                if use_default and default_attempts < max_default_attempts:
                    success = self.plan_trajectory(use_default=True)
                    default_attempts += 1
                    
                    if success:
                        rospy.loginfo("Successfully planned to default position")
                        self.trajectory_pub.publish(self.last_trajectory)
                        self.planning_active = True
                    elif default_attempts >= max_default_attempts:
                        rospy.logwarn("Failed to plan to default position after %d attempts", 
                                     default_attempts)
                        use_default = False
                
                # 如果有自定义目标且需要规划
                elif self.target_pose is not None and self.significant_target_change():
                    success = self.plan_trajectory(use_default=False)
                    
                    if success:
                        rospy.loginfo("Publishing trajectory with %d waypoints", 
                                     len(self.last_trajectory.joint_trajectory.points))
                        self.trajectory_pub.publish(self.last_trajectory)
                        self.planning_active = True
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        adapter = TargetAdapter()
        adapter.run()
    except rospy.ROSInterruptException:
        pass