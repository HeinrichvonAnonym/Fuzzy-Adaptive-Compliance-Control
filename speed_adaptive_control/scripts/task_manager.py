import rospy
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32MultiArray, Float32
from std_srvs.srv import Empty, Trigger, TriggerRequest
import yaml
import numpy as np
from visualization_msgs.msg import Marker

import yaml

config_path = "/home/heinrich/franka_ws/src/speed_adaptive_control/config/dynamical_parameters_franka.yaml"
task_path = "/home/heinrich/franka_ws/src/speed_adaptive_control/config/task.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

algo_name = config["algo_name"]
base_link = config["base_link"]

IDLE = 0     # interrupt pd control
HIGH = 1     # start pd control
LOW = 2      # rise damping factor
SCREW = 3    # lock robot position and act screw
WAIT = 4

class AL:
    def __init__(self):
        self.al = IDLE
    
    def auto_update(self, al_msg):
        al_msg = str(al_msg)
        if al_msg == "targeting":
            # print(">>")
            self.al = LOW
            # print(self.al)
        elif al_msg == "arrived":
            self.al = HIGH
        elif al_msg == "screwing":
            self.al = SCREW
        elif al_msg == "idle":
            self.al = IDLE 
        elif al_msg == "start":
            self.al = HIGH
        elif al_msg == "wait":
            self.al = WAIT
        return self.al


class TaskManager:
    def __init__(self, config):
        rospy.init_node('task_manager', anonymous=True)
        rospy.Subscriber(f"/{algo_name}/arrived", String, self.task_arrived_callback)
        self.current_pose_subscriber = rospy.Subscriber(f"/{algo_name}/robot_pose", PoseArray, self.current_pose_callback)

        self.go_on_subscriber = rospy.Subscriber(f"/{algo_name}/robot_pose", PoseArray, self.current_pose_callback)
        rospy.Subscriber(f"/{algo_name}/go_forward", String, self.go_forward_callback)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        # self.current_pose_subscriber = rospy.Subscriber(f"/{algo_name}/robot_pose", PoseArray, self.current_pose_callback)

        self._init_marker()

        self.pose_publisher =  rospy.Publisher('/cube_pose', Marker, queue_size=10)
        self.damping_publisher = rospy.Publisher(f"/{algo_name}/work_damping", Float32, queue_size=10)
        self.idle_publisher = rospy.Publisher(f"/{algo_name}/idle", String, queue_size=5)
        # self.tool_silence_publisher = rospy.Publisher(f"/{algo_name}/tool_silence", String, queue_size=10)
        self.config = config
        self._load_config(self.config)
        self.tar_pose = None
        self.work_damping = Float32() # 0 - 80
        self.waiting_for_succ = False

        self.thread_wait = False
        self.cur_pose = None
        self.wait_for_human_task = 0
        self.human_task = 0

        self.tool_torque = 0
        self.tool_torque_threshold = 2


    def joint_state_callback(self, msg:JointState):
        self.tool_torque =  msg.effort[6]
    def go_forward_callback(self, msg:String):
        if msg.data == "over":
            print(f"another human task overed: idx{self.human_task}")
            self.human_task += 1
        if msg.data == "go on":
            self.al.auto_update("start")
        if msg.data == "stop":
            self.al.auto_update("idle")
        
        if msg.data == "jump":
            if self.task_index < self.task_list_len - 1:
                print(f"jump pose idx from {self.task_index} to {self.task_index+1}")
                self.task_arrived_callback("pass")
            else:
                print(f"jump pose idx from {self.task_index} to 0")
                self.task_index = 0
                self.task_arrived_callback("pass")

        
        if msg.data == "back":
            print(f"back pose idx from {self.task_index} to {self.task_index-1}")
            self.task_index -= 2
            self.task_index = max(self.task_index, 0)
            self.task_arrived_callback("pass")


        

    def check_actual_arrived(self):
        if self.cur_pose is None:
            return False
        cur_x, cur_y, cur_z = self.cur_pose.position.x, self.cur_pose.position.y, self.cur_pose.position.z
        tar_x, tar_y, tar_z = self.tar_pose.position.x, self.tar_pose.position.y, self.tar_pose.position.z

        cur = np.array([cur_x, cur_y, cur_z])
        tar = np.array([tar_x, tar_y, tar_z])

        dis = np.linalg.norm(tar - cur)

        if dis <= 0.01:
            return True
        else:
            return False

    def current_pose_callback(self, msg:PoseArray):
        self.cur_pose = msg.poses[0]

    def task_arrived_callback(self, msg):
        # print(msg)
        if not self.waiting_for_succ and msg != "pass":
            return
        
        if msg != "pass" and not self.check_actual_arrived():
            return
        
        self.thread_wait = True
        if self.task_index < len(self.task_list) - 1:
            self.task_index += 1
        
            # print(self.task_list[self.task_index]['al_msg'])
            self.al.auto_update(self.task_list[self.task_index]['al_msg'])
            self.task_status = self.al.al
            print(f"al status is shifting to {self.task_status}")
            self.damping_control()
        else:
            self.wait_for_human_task = 0
            self.human_task = 0
            print("execution finished")
        
        
    
    def _load_config(self, config):
        self.task_list = config
        self.task_poses = []
        for pose_dict in self.task_list:
            pose = Pose()
            pose.position.x = pose_dict['position']['x']
            pose.position.y = pose_dict['position']['y']
            pose.position.z = pose_dict['position']['z']
            pose.orientation.x = pose_dict['orientation']['x']
            pose.orientation.y = pose_dict['orientation']['y']
            pose.orientation.z = pose_dict['orientation']['z']
            pose.orientation.w = pose_dict['orientation']['w']
            self.task_poses.append(pose)
        self.task_list_len = len(self.task_list)
        self.task_index = -1
        self.al = AL()
        self.task_status = self.al.al

    def damping_control(self):
        self.idle_publisher.publish(" ")
        if self.task_status == LOW:
            self.work_damping.data = 0.501
            self.damping_publisher.publish(self.work_damping)
        else:
            self.work_damping.data = 0
            self.damping_publisher.publish(self.work_damping)

    def _init_marker(self):
        markerA = Marker()
        markerA.header.frame_id = "panda_link0"
        markerA.ns = "cube"
        markerA.id = 0
        markerA.type = Marker.CUBE
        markerA.action = Marker.ADD
        markerA.scale.x = 0.05
        markerA.scale.y = 0.05
        markerA.scale.z = 0.05
        markerA.color.a = 1.0
        markerA.color.r = 1.0
        markerA.color.g = 0.0
        markerA.color.b = 0.0
        markerA.pose.orientation.w = 1.0
        markerA.pose.orientation.x = 0.0
        markerA.pose.orientation.y = 0.0
        markerA.pose.orientation.z = 0.0

        markerB = Marker()
        markerB.header.frame_id = "panda_link0"
        markerB.ns = "cube"
        markerB.id = 1
        markerB.type = Marker.CUBE
        markerB.action = Marker.ADD
        markerB.scale.x = 0.075
        markerB.scale.y = 0.075
        markerB.scale.z = 0.075
        markerB.color.a = 1.0
        markerB.color.r = 1.0
        markerB.color.g = 1.0
        markerB.color.b = 0.8
        

        self.markerA = markerA
        self.markerB = markerB

    def run(self):
        self.task_arrived_callback("pass")
        prev = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            rospy.sleep(3)
            if self.thread_wait:
                # rospy.sleep(0.5)
                self.thread_wait = False
            
            pose = Pose()
            pose = self.task_poses[self.task_index]
            self.tar_pose = pose
            self.markerA.pose = pose
            
            print(self.al.al)
            print(self.task_index)

            if self.al.al is IDLE:
                self.idle_publisher.publish("idle")
            else:
                if self.al.al is WAIT:
                    if self.wait_for_human_task >= self.human_task:
                        pass
                    else:
                        self.wait_for_human_task += 1
                        self.task_status = self.al.auto_update("start")  

                elif self.al.al is SCREW:
                    ########################################################
                    self.idle_publisher.publish("screw")
                    print(self.tool_torque)
                    if abs(self.tool_torque) > self.tool_torque_threshold:
                        self.task_arrived_callback("pass")
                    ########################################################

                else:
                    self.waiting_for_succ = True
                    self.pose_publisher.publish(self.markerA)
                    now = rospy.Time.now().to_sec()
                    

                    if  now - prev > 1:
                        self.damping_control()
                        prev = now     
            
                
            
            
            rospy.sleep(0.1)
              


if __name__ == '__main__':
    with open(task_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
           
    task_manager = TaskManager(data)
    task_manager.run()
