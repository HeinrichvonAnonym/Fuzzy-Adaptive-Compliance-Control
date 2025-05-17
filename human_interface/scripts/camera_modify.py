import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraModify:
    def __init__(self):
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.resize(image, (640 * 3 , 360 * 3))
            
        except Exception as e:
            print(e)
        
        
        width, height = cv_image.shape[1], cv_image.shape[0]
        # print(width, height)
        center_x, center_y = width // 2, height // 2

        cv2.circle(cv_image, (center_x, center_y), 20, (225, 0, 255), 2)

        cv2.line(cv_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)
        cv2.line(cv_image, (0, center_y), (width, center_y), (0, 255, 0), 2)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)


        


    
    def subscribe(self):
        rospy.Subscriber("/kinect2/hd/image_color", Image, self.callback)
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("camera_modify")
    camera_modify = CameraModify()
    camera_modify.subscribe()