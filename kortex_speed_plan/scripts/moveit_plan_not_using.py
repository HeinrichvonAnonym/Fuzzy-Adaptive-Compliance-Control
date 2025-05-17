import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.msg import CollisionObject
import sys
import numpy as np
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from std_msgs.msg import Float32
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose
import scipy.spatial.transform as transform
import tf

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, target_position, current_position, dt):
        error = target_position - current_position
        print(error)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt>0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output


search_target_positions = [-0.56, -0.30, 2.90, -1.49, -2.75, 1.30, 2.00]
fake_grasp_position = [-0.41, 0.16, -2.89, -2.24, -2.77, -0.70, 2.43]
search_human_positions = [-0.60, 0.29, -1.08, -1.46, -2.49, 0.32, 2.00]

class KortexPlanner:
    def __init__(self):
        rospy.init_node('kortex_planner', anonymous=True)
        self.joint_velocity_pub = rospy.Publisher("in/joint_velocity", Base_JointSpeeds, queue_size=10)
        # connect the moveit
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = MoveGroupCommander("arm")

        self.step_dis = 0.15
        self.step_angle = 25 * np.pi / 180

        self.pid_controllers = []
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1, 0.0, 0.05))

        self.gripper_grasp_client = rospy.ServiceProxy("/gripper/grasp_command", Trigger, persistent=True)
        self.gripper_release_client = rospy.ServiceProxy("/gripper/release_command", Trigger, persistent=True)

    
    def get_planning_scene_object(self):
        scene = PlanningSceneInterface("")
        cos = scene.get_objects()
        return cos
    
    def go_home(self):
        self.arm.set_named_target("retract")
        self.arm.go(wait=True)
        print("Go home")
    
    def get_jacobian_matrix(self):
        cur_joint_values = self.arm.get_current_joint_values()
        jacobian_matrix = self.arm.get_jacobian_matrix(cur_joint_values, reference_point=[0., 0., 0.17])
        return jacobian_matrix
    
    def write_vel(self, speed):
        joint_speeds_msg = Base_JointSpeeds()
        # print(self.target_vel)
        for i in range(7):        
            joint_speed = JointSpeed()

            joint_speed.value = speed[i]

            joint_speed.joint_identifier = i
            joint_speed.duration = 1

            joint_speeds_msg.joint_speeds.append(joint_speed)
        # print(joint_speeds_msg)
        self.joint_velocity_pub.publish(joint_speeds_msg)
    
    def cartesian_move(self, orientation, distance):
        prev_time = rospy.Time.now()
        step_dis = self.step_dis
        step_angle = self.step_angle
        steps = 0
        end_effector_velo = orientation[:3]
        end_effector_angular_velo = orientation[3:]
        start_scale = 0.0
        end_scale = 5
        while not rospy.is_shutdown():
            
            
            jacobian_matrix = planner.get_jacobian_matrix()
            cur_time = rospy.Time.now()
            dt = (cur_time - prev_time).to_sec()
            end_effector_velo = end_effector_velo / np.linalg.norm(end_effector_velo)
            end_effector_angular_velo = end_effector_angular_velo / (np.linalg.norm(end_effector_angular_velo) + 1e-4)
            end_effector_velo *= step_dis/dt
            end_effector_angular_velo *= step_angle/dt
            ori = np.concatenate((end_effector_velo, end_effector_angular_velo))

            prev_time = cur_time

            
            jacobian_matrix_psuedo_inverse = np.linalg.pinv(jacobian_matrix)
            joint_velocities = np.matmul(jacobian_matrix_psuedo_inverse, ori)

            steps += 1

            if steps >= 0.8 * (distance / step_dis):
                joint_velocities = joint_velocities / end_scale
                end_scale += 3
                if steps >= 1.3 * (distance / step_dis):
                    joint_velocities = np.zeros(7)
                    planner.write_vel(joint_velocities)
                    return
                planner.write_vel(joint_velocities)
            
            if steps <= 0.5 / dt:
                joint_velocities = joint_velocities * start_scale**2
                start_scale += 2 * dt
                
            print(dt, joint_velocities)
            planner.write_vel(joint_velocities)
        
    def cartesian_vel(self, orientation, jacobian_matrix, dt, step_dis, step_angle):
        if step_dis is None:
            step_dis = self.step_dis
        if step_angle is None:
            step_angle = self.step_angle

        end_effector_velo = orientation[:3]
        end_effector_angular_velo = orientation[3:]

        end_effector_velo = end_effector_velo / (np.linalg.norm(end_effector_velo) + 1e-4)
        end_effector_angular_velo = end_effector_angular_velo / (np.linalg.norm(end_effector_angular_velo) + 1e-4)

        end_effector_velo *= step_dis/dt
        end_effector_angular_velo *= step_angle/dt
        ori = np.concatenate((end_effector_velo, end_effector_angular_velo))


        jacobian_matrix_psuedo_inverse = np.linalg.pinv(jacobian_matrix)
        joint_velocities = np.matmul(jacobian_matrix_psuedo_inverse, ori)
        planner.write_vel(joint_velocities)

    def move_to_pose(self, pose:Pose, threshold_pos = 0.02, threshold_ori = 0.02, step_dis = None, step_angle = None):
        executed = False
        prev_time = rospy.Time.now()
        while(not executed and not rospy.is_shutdown()):
            cur_tcp_pose = self.arm.get_current_pose().pose
            position_dis = np.array([cur_tcp_pose.position.x - pose.position.x, 
                                    cur_tcp_pose.position.y - pose.position.y, 
                                    cur_tcp_pose.position.z - pose.position.z])
            position_dis *= -1
            cur_q = np.array([cur_tcp_pose.orientation.x, 
                              cur_tcp_pose.orientation.y, 
                              cur_tcp_pose.orientation.z, 
                              cur_tcp_pose.orientation.w])
            cur_q = transform.Rotation.from_quat(cur_q)
            target_q = np.array([pose.orientation.x, 
                                  pose.orientation.y, 
                                  pose.orientation.z, 
                                  pose.orientation.w])
            target_q = transform.Rotation.from_quat(target_q)
            error_q = target_q * cur_q.inv()
            error_q = error_q.as_rotvec()
            euler_dis = error_q
            # euler_dis = np.zeros(3)
    


            orientation = np.concatenate((position_dis, euler_dis))

            pos_err = np.linalg.norm(orientation[:3])
            ori_err = np.linalg.norm(orientation[3:])
            print(f"pos_err {pos_err}, ori_err {ori_err}")
            if pos_err < threshold_pos and ori_err < threshold_ori:
                joint_velocities = np.zeros(7)
                self.write_vel(joint_velocities)
                executed = True
                break

            jacobian_matrix = planner.get_jacobian_matrix()
            cur_time = rospy.Time.now()
            dt = (cur_time - prev_time).to_sec()
            
            self.cartesian_vel(orientation, jacobian_matrix, dt, step_dis, step_angle)
    
    def get_collision_object(self, id):
        plannning_scene_interface = PlanningSceneInterface("")
        cos = planner.get_planning_scene_object(str(id))
        return cos

from copy import deepcopy


if __name__ == '__main__':
    planner = KortexPlanner()
    velo = np.array([0, 0, ])
    trigger = TriggerRequest()
    ret = planner.gripper_release_client.call(trigger)
    rospy.sleep(0.1)
    ret = planner.gripper_grasp_client.call(trigger)
    rospy.sleep(0.1)
    ret = planner.gripper_release_client.call(trigger)
    rospy.sleep(0.1)
    ret = planner.gripper_release_client.call(trigger)
    rospy.sleep(0.1)


    cur_pose = planner.arm.get_current_pose().pose
    pre_pose = deepcopy(cur_pose)
    cur_pose.position.x += 0.3
    planner.move_to_pose(cur_pose)

    ret = planner.gripper_grasp_client.call(trigger)
    rospy.sleep(0.1)
    ret = planner.gripper_grasp_client.call(trigger)
    rospy.sleep(0.5)
    

    
    cur_pose.position.z += 0.4
    cur_pose.position.x -= 0.3
    # planner.move_to_pose(cur_pose)

    
    # cur_pose.orientation.x = 0.5
    # cur_pose.orientation.y = 0.5
    # cur_pose.orientation.z = 0.5
    # cur_pose.orientation.w = 0.5
    # planner.move_to_pose(cur_pose, threshold_ori=0.2, step_angle=np.pi * 30 / 180)
    
    cur_pose.orientation.x = 0.732 / 4
    cur_pose.orientation.y = 2.732 / 4
    cur_pose.orientation.z = 2.732 / 4
    cur_pose.orientation.w = 0.732 / 4
    cur_pose.position.y + 0.7
    planner.move_to_pose(cur_pose, 
                         step_dis= 0.2,
                        threshold_pos=0.2, threshold_ori=0.65, step_angle=np.pi * 45 / 180)

    ret = planner.gripper_release_client.call(trigger)
    rospy.sleep(0.5)
    ret = planner.gripper_grasp_client.call(trigger)

    planner.move_to_pose(pre_pose, step_angle=np.pi * 45 / 180)

    # # rospy.spin()


        # print(jacobian_matrix)
    #     cos = planner.get_planning_scene_object()
    #     pos_human = cos['0'].pose.position
    #     human_x = pos_human.x
    #     human_y = pos_human.y
    #     planner.towards_human(human_x, human_y)
    #     rospy.sleep(0.5)
        
        