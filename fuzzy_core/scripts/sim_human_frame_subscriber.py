import rospy
from zess_msgs.msg import ObjectsStamped
from kompi_msgs.msg import HumanTrajectory
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import pandas as pd

data_path = "/home/heinrich/robot_data_logs/episode_86/action.csv"




            

class HumanTrajectoryPublisher:
    def __init__(self, path, robot_name="", freq = 8):
        print(">")
        
        print(">")
        self.robot_name = robot_name
        self.freq = freq
        source_data = pd.read_csv(path)
        source_data = source_data.iloc[1:, 34:].dropna().reset_index(drop=True).to_numpy()
        print(source_data)

        self.marker_array = MarkerArray()
        
        self.human_publisher = rospy.Publisher("/mrk/human_skeleton", MarkerArray, queue_size=5)
    
    
    
    def loop(self):
        pass
                




    def run(self):
        while(not rospy.is_shutdown()):
            rospy.sleep(0.1)
            self.loop()
        
if __name__ == '__main__':
    print(">")
    rospy.init_node('sim_human_trajectory_subscriber', anonymous=True)
    publisher = HumanTrajectoryPublisher(path=data_path)
    # publisher.run()
    # rospy.spin()