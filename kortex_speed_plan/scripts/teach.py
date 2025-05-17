import rospy
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32MultiArray, Int16
import pynput
from pynput import keyboard
import yaml

PREPARE = 0
WORK = 1
IDLE = 2
WAIT = 3

class KortexTeach:
    def __init__(self, output):
        rospy.init_node("kortex_teach", anonymous=True)
        rospy.Subscriber("rrt/robot_pose", PoseArray, self.current_pose_callback)

        self.eef_pose = None
        self.teached_poses = []
        self.types = []
        self.point_num = 0
        self.pose_type = PREPARE

        self.idle = False

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.output = output
        self.idle_publisher = rospy.Publisher("rrt/idle", String, queue_size=5)
    
    def current_pose_callback(self, pose_array_msg):
        self.eef_pose = pose_array_msg.poses[0]
    
    def on_press(self, key):
        try:
            if key.char == "i":# idle
                if not self.idle:
                    self.idle_publisher.publish("idle")
                    self.idle = True
                else:
                    self.idle_publisher.publish(" ")
                    self.idle = False
                print(f"set idle: {self.idle}")
            if key.char == 'a':# add
                print(f"saving pose num {self.point_num}, type {self.pose_type}")
                self.teached_poses.append(self.eef_pose)
                self.types.append(self.pose_type)
                self.point_num += 1
            if key.char == 'p':# prepare
                print("set point type: PREPARE (0)")
                self.pose_type = PREPARE
            if key.char == 'w':# work
                print("set point typr: WORK (1)")
                self.pose_type = WORK
            if key.char == 'e':# idle
                print("set point typr: IDLE (2)")
                self.pose_type = IDLE
            if key.char == 't':# idle
                print("set point typr: WAIT (3)")
                self.pose_type = WAIT
            if key.char == 'd':# delete
                self.teached_poses.pop(-1)
                self.types.pop(-1)
                self.point_num -= 1
                print(f"deleting pose num {self.point_num}, type {self.pose_type}")
            if key.char == "l":# list
                print("teached poses:")
                for pose, P_type in zip(self.teached_poses, self.types):
                    print(pose)
                    print(f"type: {P_type}")
                    print(">>>")
            if key.char =='s':# save
                print("saving teached poses")
                self.save_config()
            
        except AttributeError:
            print(f"special key pressed: {key}")
    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False
    
    def save_config(self):
        # print(">>")
        config = []
        for pose, p_type in zip(self.teached_poses, self.types):
            # print(pose)
            dict = {}
            dict['position'] = {'x': pose.position.x,
                                'y': pose.position.y,
                                'z': pose.position.z,}
            dict['orientation'] = {'x': pose.orientation.x,
                                   'y': pose.orientation.y,
                                   'z': pose.orientation.z,
                                   'w': pose.orientation.w,}
            if p_type == PREPARE:
                dict['al_msg'] = 'arrived'
            elif p_type == WORK:
                dict['al_msg'] = 'targeting'
            elif p_type == IDLE:
                dict['al_msg'] = 'idle'
            elif p_type == WAIT:
                dict['al_msg'] = 'wait'
            config.append(dict)
        # print(">>")
        with open(self.output, "w", encoding="utf-8") as file:
            yaml.dump(config, file, allow_unicode=True)

        

    def run(self):
        self.listener.run()
        while(not rospy.is_shutdown()):
            # print(self.eef_pose)
            rospy.sleep(0.1)
        
        rospy.spin()
        self.listener.join()

if __name__ == '__main__':
    output = "/home/heinrich/kinova/src/kortex_speed_plan/scripts/config/task.yaml"
    teach = KortexTeach(output)
    teach.run()