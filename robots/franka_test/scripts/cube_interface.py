import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PointStamped
import numpy as np
from pynput.keyboard import Listener, Key


class cubeInterface:
    def __init__(self):
        rospy.init_node('cube_interface', anonymous=True)   
        self.markerA_pub = rospy.Publisher('/cube_pose', Marker, queue_size=10)
        self.markerB_pub = rospy.Publisher('/obs_pose', Marker, queue_size=10)
        self.target_x, self.target_y, self.target_z = 0.4, 0.0, 0.3
        self._init_marker()
        self._init_pose()

        self.keyboard_listener = Listener(on_press=self.keyboard_callback)

        # subscribers
        pass
    def keyboard_callback(self, key):

        if key == Key.esc:
            self.keyboard_listener.stop()
            rospy.signal_shutdown("Esc key pressed")
        
        else:
            try:
                if key.char == 'w':
                    print("w pressed")
                    self.target_x += 0.01
                elif key.char == 's':
                    self.target_x -= 0.01
                elif key.char == 'a':
                    print("A")
                    self.target_y += 0.01
                elif key.char == 'd':
                    self.target_y -= 0.01
                elif key.char == 'q':
                    self.target_z += 0.01
                elif key.char == 'e':
                    self.target_z -= 0.01
            except:
                pass
             

    def _init_pose(self):
        poseA = Pose()
        poseA.position.x = self.target_x
        poseA.position.y = self.target_y
        poseA.position.z = self.target_z

        poseA.orientation.x = 0.0
        poseA.orientation.y = 0.0
        poseA.orientation.z = 0.0
        poseA.orientation.w = 1.0

        poseB = Pose()
        poseB.position.x = 0.3 + np.random.uniform(-0.1,0.05)
        poseB.position.y = -0.1 + np.random.uniform(-0.1,0.1)
        poseB.position.z = 0.0 
        poseB.orientation.x = 0.0
        poseB.orientation.y = 0.0
        poseB.orientation.z = 0.0
        poseB.orientation.w = 1.0

        
        self.poseA, self.poseB = poseA, poseB

        self.dt = 0.01

        
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
        frame_count = 0
        self.keyboard_listener.start()
        while(not rospy.is_shutdown()):
            # print(">")
            self.markerA.header.stamp = rospy.Time.now()
            self.markerA.pose.position .x = self.target_x
            self.markerA.pose.position .y = self.target_y
            self.markerA.pose.position .z = self.target_z
            self.markerB.header.stamp = rospy.Time.now()
            self.markerB.pose = self.poseB

            self.markerA_pub.publish(self.markerA)
            self.markerB_pub.publish(self.markerB)

            rospy.sleep(self.dt)
       
    

if __name__ == '__main__':
    cube_interface = cubeInterface()
    cube_interface.run()