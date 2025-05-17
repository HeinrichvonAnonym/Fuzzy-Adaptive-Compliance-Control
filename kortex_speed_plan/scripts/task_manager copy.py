import rospy
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32MultiArray
from std_srvs.srv import Empty, Trigger, TriggerRequest
import yaml
import numpy as np

IDLE = 0     # interrupt pd control
HIGH = 1     # start pd control
LOW = 2      # rise damping factor
SCREW = 3    # lock robot position and act screw

class AL:
    def __init__(self):
        self.al = IDLE
    
    def update(self, al_callback):
        if al_callback == "take over":
            self.al = IDLE
        if self.al == IDLE:
            if al_callback == "start" or al_callback == "go_forward":
                self.al = HIGH
        if self.al == HIGH:
            if al_callback == "arrived":
                self.al = HIGH
            elif al_callback == "targeting":
                self.al = LOW
        if self.al == LOW:
            if al_callback == "screw":
                self.al = SCREW    
        if self.al == SCREW:
            if al_callback == "leave":
                self.al = HIGH
        
        return self.al


class TaskManager:
    def __init__(self, config):
        rospy.init_node('task_manager', anonymous=True)
        rospy.Subscriber("rrt/arived", String, self.task_arrived_callback)
        rospy.Subscriber("rrt/go_forward", String, self.go_forward_callback)
        self.current_pose_subscriber = rospy.Subscriber("/rrt/robot_pose", PoseArray, self.current_pose_callback)

        self.pose_publisher = rospy.Publisher("/rrt/target_pose", PoseStamped, queue_size=10)
        self.config = config
        self._load_config(self.config)
        self.tar_pose = None

    def current_pose_callback(self, pose_array_msg):
       pass

    def task_arrived_callback(self, msg):
        pass
    def go_forward_callback(self, msg):
        pass
    
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
        self.task_index = 0
        self.al = AL()
        self.task_status = self.al.al

    

    def run(self):
        self.task_status = self.al.update("start")
        while not rospy.is_shutdown():
            pose = PoseStamped()
            pose.header.frame_id = "base_link"  
            pose.pose = self.task_poses[self.task_index]
            pose.header.stamp = rospy.Time.now()
            self.tar_pose = pose
            self.pose_publisher.publish(pose)
            
            if self.task_status == 'EXECUTING ROBOT':
                if self.task_finished == True:
                    self.task_finished = False
                    
                    print(f"Executing task {self.task_index} : {self.task_poses[self.task_index]}")
                    self.task_index += 1

            rospy.sleep(0.5)
              


if __name__ == '__main__':
    config_path = "/home/heinrich/kinova/src/kortex_speed_plan/scripts/task.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)['wayposes']
           
    task_manager = TaskManager(data)
    task_manager.run()
