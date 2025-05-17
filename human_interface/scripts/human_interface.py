"""
edited by Heinrich 17,Jan 2025
"""

import rospy
# from zess_msgs.msg import ObjectsStamped, Object, Skeleton3D, Keypoint3D, Keypoint2Df
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
# from kompi_msgs.msg import HumanTrajectory
import mediapipe as mp
# import cv bridge
from cv_bridge import CvBridge
import numpy as np
import time
import cv2

## mp pose land marker
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HumanInterface:
    def __init__(self):
        self.bridge = CvBridge()
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8)
        self.publisher = rospy.Publisher('/mrk/human_frame', PoseArray, queue_size=10)
    
    def get_pose(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def get_pose_landmark(self, image, results):
        image_height, image_width, _ = image.shape
        pose_landmarks = results.pose_landmarks
        pa = PoseArray()
        pa.header.frame_id = "world"
        pa.header.stamp = rospy.Time.now()
        if pose_landmarks is not None:
            for landmark in pose_landmarks.landmark:
                x = landmark.x 
                y = landmark.y
                z = landmark.z 
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z

                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1

                pa.poses.append(pose)
                frame_x = min(int(x * image_width), image_width - 1)
                frame_y = min(int(y * image_height), image_height - 1)
                cv2.circle(image, (frame_x, frame_y), 5, (0, 255, 0), -1)
            return image, pa
        else:
            return image, None
    
    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        # print(image.shape)
        image, results = self.get_pose(image)
        
        image, pa = self.get_pose_landmark(image, results)
        if results.pose_landmarks is not None:
            self.publisher.publish(pa)
        cv2.imshow('image', image)
        cv2.waitKey(30)
                   
    def subsribe(self):
        rospy.Subscriber('/kinect2/hd/image_color', Image, self.callback)
        rospy.spin()

rospy.init_node('human_interface', anonymous=True)
HumanInterface().subsribe()
