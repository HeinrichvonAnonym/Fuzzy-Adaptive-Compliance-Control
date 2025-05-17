"""
edited by Heinrich 17,Jan 2025
"""

import rospy
from zess_msgs.msg import ObjectsStamped, Object, Skeleton3D, Keypoint3D, Keypoint2Df
from sensor_msgs.msg import Image
from kompi_msgs.msg import HumanTrajectory
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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.publisher = rospy.Publisher('/mrk/human_frame', ObjectsStamped, queue_size=10)
    
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
        xs = []
        ys = []
        zs = []
        if pose_landmarks is not None:
            for landmark in pose_landmarks.landmark:
                x = min(int(landmark.x * image_width), image_width - 1)
                y = min(int(landmark.y* image_height), image_height - 1)
                z = min(int(landmark.z * image_width), image_width - 1)
                xs.append(landmark.x)
                ys.append(landmark.y)
                zs.append(landmark.z)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            return image, xs, ys, zs
    
    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        # print(image.shape)
        image, results = self.get_pose(image)
        if results.pose_landmarks is None:
            return
        image, xs, ys, zs = self.get_pose_landmark(image, results)
        cv2.imshow('image', image)
        cv2.waitKey(1)
        obj = ObjectsStamped()
        obj.header.stamp = rospy.Time.now()

        obj.objects = []
        object = Object()
        skeleton_3d = Skeleton3D()
        for i in range(len(xs)):
            keypoint_3d = Keypoint3D()
            keypoint_3d.kp = [xs[i], ys[i], zs[i]]
            skeleton_3d.keypoints[i] = keypoint_3d

        object.skeleton_3d = skeleton_3d

        obj.objects.append(object)
        self.publisher.publish(obj)
            
        
    def subsribe(self):
        rospy.Subscriber('/kinect2/hd/image_color', Image, self.callback)
        rospy.spin()



rospy.init_node('human_interface', anonymous=True)
HumanInterface().subsribe()