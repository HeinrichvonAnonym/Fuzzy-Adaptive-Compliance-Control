import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander, RobotState
from moveit_msgs.msg import CollisionObject, RobotTrajectory
import sys
import numpy as np
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point
import scipy.spatial.transform as transform
import tf
from kortex_speed_plan.msg import SolidPrimitiveMultiArray
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import math


critical_link_names = ["forearm_link", "tool_frame"]

class TrajectoryPlanner:
    def __init__(self):
        rospy.init_node('trajectory_planner', anonymous=True)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        # connect the moveit
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = MoveGroupCommander("arm")
        self.robot = moveit_commander.RobotCommander()
        rospy.Subscriber("rrt/target_pose", PoseStamped, self.target_callback)
        self.trajectory_pub = rospy.Publisher("rrt/cartesian_trajectory", RobotTrajectory, queue_size=10)
        self.planning_scene_interface = PlanningSceneInterface("")
        self.pose_publisher = rospy.Publisher("rrt/robot_pose", PoseArray, queue_size=10)
        self.jacobian_pub = rospy.Publisher("rrt/jacobian", Float32MultiArray, queue_size=10)
        self.object_pub = rospy.Publisher("rrt/objects", SolidPrimitiveMultiArray, queue_size=10)
        
        # 添加规划统计和安全距离发布器
        self.planning_stats_pub = rospy.Publisher("rrt/planning_stats", Float32MultiArray, queue_size=10)
        self.min_distance_pub = rospy.Publisher("rrt/min_distance", Float32MultiArray, queue_size=10)

        rospy.Subscriber("rrt/work_damping", Float32, self.work_damping_callback)
        
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
        
        # 添加规划统计数据
        self.last_planning_time = 0.0
        self.last_planning_iterations = 0
        self.min_obstacle_distance = float('inf')
    
        self.work_damping = 0.

    def work_damping_callback(self, msg:Float32):
        self.work_damping = msg.data

    def target_callback(self, msg: PoseStamped):
        self.target_position = msg.pose.position
        self.target_orientation = msg.pose.orientation

    def joint_state_callback(self, msg):
        self.joint_state = msg
    
    def get_mid_pose(self, target_pose):
        current_position = self.pose_arr.poses[-1].position
        target_position = target_pose.position
        mid_pose = Pose()
        mid_pose.position.x = (current_position.x + target_position.x) / 2
        mid_pose.position.y = (current_position.y + target_position.y) / 2
        mid_pose.position.z = (current_position.z + target_position.z) / 2
        mid_pose.orientation = target_pose.orientation
        object_poses = self.primitive_arr.poses
        for pose in object_poses:
            position = pose.position
            dis = np.array([position.x - mid_pose.position.x, 
                            position.y - mid_pose.position.y, 
                            position.z - mid_pose.position.z])
            if np.linalg.norm(dis) <= 0.2:
                mid_pose.position.z = mid_pose.position.z + 0.2
        return mid_pose

    def plan_cartesian_to_target(self, step = 0.00005):
        if self.target_position is None or self.target_orientation is None:
            rospy.loginfo("No target position set")
            return
        start_pose = self.arm.get_current_pose("end_effector_link").pose
        wayposes = []
        # wayposes.append(start_pose)

        target_pose = Pose()
        target_pose.position = self.target_position
        target_pose.orientation = self.target_orientation

        # distance_vec = np.array([target_pose.position.x - start_pose.position.x,
        #                      target_pose.position.y - start_pose.position.y,
        #                      target_pose.position.z - start_pose.position.z])
        # distance = np.linalg.norm(distance_vec)
        # num = math.floor(distance / 0.001) - 1

        # distance_vec = 0.001 * distance_vec / (distance + 1e-8)

        # for i in range(num):
        #     pose = Pose()
        #     pose.position.x = start_pose.position.x + (i+1) * distance_vec[0]
        #     pose.position.y = start_pose.position.y + (i+1) * distance_vec[1]
        #     pose.position.z = start_pose.position.z + (i+1) * distance_vec[2]
        #     pose.orientation = self.target_orientation
        #     wayposes.append(pose)

        wayposes.append(target_pose)

        plan, fraction = self.arm.compute_cartesian_path(waypoints=wayposes, eef_step=step)
        if fraction <= 0.99:
            rospy.logwarn("failed to plan cartesian trajectory")
            return
        else:
            rospy.loginfo("Planned cartesian successfully")
            self.trajectory_pub.publish(plan)



    def plan_to_target(self):
        if self.failed:
            rospy.loginfo("current target is unreachable")
            return False
        
        if self.target_position is None or self.target_orientation is None:
            rospy.loginfo("No target position set")
            return False
        
        target_pose = Pose()
        target_pose.position = self.target_position
        target_pose.orientation = self.target_orientation

        mid_pose = self.get_mid_pose(target_pose)
        # print(f"mid_poe: {mid_pose}")

        self.arm.set_pose_target(target_pose)     
        success, plan, planning_time, error_code_2 = self.arm.plan()
        
        # 记录规划统计数据
        self.last_planning_time = planning_time 
        # 假设每次规划只有一次迭代，这是个简化
        self.last_planning_iterations = 1 if success else 0
        
        # 发布规划统计数据
        stats_msg = Float32MultiArray()
        stats_msg.data = [self.last_planning_time, self.last_planning_iterations]
        self.planning_stats_pub.publish(stats_msg)
       
        if success :
            rospy.loginfo("Planned successfully")
            self.trajectory_pub.publish(plan)
            return True
        else:
            rospy.logwarn("Failed to plan to target position")
            self.plan_to_target()
            self.failed = True
            return False

    def get_collision_object(self):
        objects = self.planning_scene_interface.get_objects()
        primitive_arr = SolidPrimitiveMultiArray()
        for key in objects.keys():
            object = objects[key]
            primitive = object.primitives[0]
            primitive_arr.primitives.append(primitive)
            primitive_arr.poses.append(object.pose)
        self.primitive_arr = primitive_arr

    def get_critical_link_position(self):
        pose_arr = PoseArray()
        for link in critical_link_names:
            pose = self.arm.get_current_pose(link)
            pose_arr.poses.append(pose.pose)
        self.pose_arr = pose_arr
    
    # def get_jacobian(self):
    #     if self.joint_state is None:
    #         return None 
    #     cur_joint_values = list(self.joint_state.position[:7])
    #     jacobian_matrix = self.arm.get_jacobian_matrix(cur_joint_values, reference_point=[0., 0., 0.15])

    #     jacobian_msg = Float32MultiArray()
    #     jacobian_msg.layout.dim.append(MultiArrayDimension())
    #     jacobian_msg.layout.dim[0].label = "row"
    #     jacobian_msg.layout.dim[0].size = jacobian_matrix.shape[0]
    #     jacobian_msg.layout.dim[0].stride = jacobian_matrix.shape[0] * jacobian_matrix.shape[1]
    #     jacobian_msg.layout.dim.append(MultiArrayDimension())
    #     jacobian_msg.layout.dim[1].label = "col"
    #     jacobian_msg.layout.dim[1].size = jacobian_matrix.shape[1]
    #     jacobian_msg.layout.dim[1].stride = jacobian_matrix.shape[1]

    #     for i in range(jacobian_matrix.shape[0]):
    #         for j in range(jacobian_matrix.shape[1]):
    #             jacobian_msg.data.append(jacobian_matrix[i][j])

    #     self.jacobian_msg = jacobian_msg

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
        position_diff = np.linalg.norm(np.array([self.target_position.x, 
                                                 self.target_position.y, 
                                                 self.target_position.z]) 
                                                 - 
                                        np.array([self.arm.get_current_pose().pose.position.x, 
                                                self.arm.get_current_pose().pose.position.y, 
                                                self.arm.get_current_pose().pose.position.z]))
        if position_diff < 0.01:
            return True
        else:
            return False
    
    def calculate_min_obstacle_distance(self):
        """计算机器人与障碍物的最小距离"""
        # 这里是一个简化实现，实际应该使用MoveIt的碰撞检测API
        min_distance = float('inf')
        
        # 通过planning_scene_interface获取障碍物，并计算与关键链接的距离
        if self.primitive_arr and self.pose_arr:
            for pose in self.pose_arr.poses:
                for i, obj_pose in enumerate(self.primitive_arr.poses):
                    primitive = self.primitive_arr.primitives[i]
                    
                    # 计算到障碍物中心的距离
                    dx = pose.position.x - obj_pose.position.x
                    dy = pose.position.y - obj_pose.position.y
                    dz = pose.position.z - obj_pose.position.z
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # 根据障碍物类型减去半径/尺寸
                    if primitive.type == SolidPrimitive.SPHERE:
                        distance -= primitive.dimensions[0]  # 减去半径
                    elif primitive.type == SolidPrimitive.CYLINDER:
                        # 简化：使用圆柱体的半径
                        distance -= primitive.dimensions[0] / 2
                    elif primitive.type == SolidPrimitive.BOX:
                        # 简化：使用盒子的对角线长度的一半
                        half_diagonal = np.sqrt(
                            (primitive.dimensions[0]/2)**2 + 
                            (primitive.dimensions[1]/2)**2 + 
                            (primitive.dimensions[2]/2)**2
                        )
                        distance -= half_diagonal
                    
                    # 更新最小距离
                    if distance > 0 and distance < min_distance:
                        min_distance = distance
        
        self.min_obstacle_distance = min_distance
        
        # 发布最小距离数据
        distance_msg = Float32MultiArray()
        distance_msg.data = [min_distance]
        self.min_distance_pub.publish(distance_msg)
        
        return min_distance
        
    def clear_trajectory(self):
        self.trajectory_pub.publish(RobotTrajectory())
        self.has_reached_target = False

    def run(self):
        while not rospy.is_shutdown():
            cur_time = rospy.Time.now()
            if (cur_time - self.prev_time).to_sec() > 1 or self.check_target_changed():     
                if not self.check_target_reached():
                    rospy.loginfo("Target not reached")
                # if self.check_target_changed():
                #     rospy.loginfo("Target changed")
                #     self.failed = False
                #     self.clear_trajectory()  
                #     # else:
                if self.work_damping > 0.5:
                    self.plan_cartesian_to_target()
                if self.work_damping <= 0.5:
                    self.plan_to_target()
                self.prev_time = cur_time
            self.get_collision_object()
            self.get_critical_link_position()  
            # print(self.primitive_arr)
            self.object_pub.publish(self.primitive_arr)
            # self.pose_publisher.publish(self.pose_arr)

            # if self.jacobian_msg is not None:
            #     self.jacobian_pub.publish(self.jacobian_msg)
            
            # self.get_jacobian()
            
            # 计算并发布最小障碍物距离
            self.calculate_min_obstacle_distance()
            
            rospy.sleep(self.dt)

if __name__ == '__main__':
    trajectory_planner = TrajectoryPlanner()
    trajectory_planner.run()
    rospy.spin()