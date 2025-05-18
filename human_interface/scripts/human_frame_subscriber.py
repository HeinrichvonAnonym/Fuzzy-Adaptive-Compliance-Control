import rospy
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math

scale = 1.0
PI = np.pi

MEDIAPIPE = 33
human_format = MEDIAPIPE

ZED_FRAME_WIDTH                 = 640
ZED_FRAME_HEIGHT                = 360
KINECT_FRAME_WIDTH              = 1920
KINECT_FRAME_HEIGHT             = 1080

ZED_VERTICLE_WIDE_ANGLE         = 70   * PI / 180
ZED_HORIZONTAL_WIDE_ANGLE       = 110    * PI / 180
KINECT_VERTICLE_WIDE_ANGLE      = 53.8  * PI / 180 # vertical
KINECT_HORIZONTAL_WIDE_ANGLE    = 84.1  * PI / 180 # horizon

KINECT_POS                      = 1.77    # X AXIS
ZED_POS                         = 2.5    # -Y AXIS

z_angle = 10
R_A = np.array([[ 1,  0, 0],
               [  0,  1,  0],
               [  0,  0,  1]])
R_B = np.array([[ 0,  1,  0],
                [-1,  0,  0],
                [ 0,  0,  1]])
# R_B = np.array([[ 0.33552987,  0.87485346,  0.34925817],
#                 [-0.9181785,  0.38685917,   -0.08708161],
#                 [-0.21133606, -0.29149856, 0.9329339]])

d_P = np.array([-2.50, 2.27, 0.059])

def get_lambdas(th_x1, th_y1, th_x2, th_y2):
    lambda_as = np.zeros(3)
    lambda_bs = np.zeros(3)
    for i in range(3):
        lambda_as[i] = (    R_A[i, 0] * np.cos(th_y1) * np.sin(th_x1) +
                            R_A[i, 1] * np.cos(th_y1) * np.cos(th_x1) +
                            R_A[i, 2] * np.sin(th_y1))
        lambda_bs[i] = (    R_B[i, 0] * np.cos(th_y2) * np.sin(th_x2) +
                            R_B[i, 1] * np.cos(th_y2) * np.cos(th_x2) +
                            R_B[i, 2] * np.sin(th_y2))
    return lambda_as, lambda_bs

def get_ds(lambda_as, lambda_bs):
    M = np.array([  [lambda_as[0], - lambda_bs[0]],
                    [lambda_as[1], - lambda_bs[1]], 
                    [lambda_as[2], - lambda_bs[2]]])
    y = np.array([d_P[0], d_P[1], d_P[2]])
    M = np.linalg.pinv(M)
    return np.dot(M, y)

def get_position(ds, lambda_as, lambda_bs):
    x_a = ds[0] * lambda_as[0]
    x_b = ds[1] * lambda_bs[0] + d_P[0]

    y_a = ds[0] * lambda_as[1]
    y_b = ds[1] * lambda_bs[1] + d_P[1]

    z_a = ds[0] * lambda_as[2]
    z_b = ds[1] * lambda_bs[2] + d_P[2]

    x   = (x_a + x_b) / 2
    d_x = np.abs(x_a - x_b)
    y   = (y_a + y_b) / 2
    d_y = np.abs(y_a - y_b)
    z   = (z_a + z_b) / 2
    d_z = np.abs(z_a - z_b)

    return np.array([x, y, z]), np.array([d_x, d_y, d_z])

def reposition(th_x1, th_y1, th_x2, th_y2):
    lambda_as, lambda_bs = get_lambdas(th_x1, th_y1, th_x2, th_y2)
    ds = get_ds(lambda_as, lambda_bs)
    return get_position(ds, lambda_as, lambda_bs)


            

class HumanTrajectoryPublisher:
    def __init__(self, robot_name="", freq = 8):
        rospy.init_node('human_trajectory_subscriber')
        self.robot_name = robot_name
        self.prev_time = rospy.Time.now()
        self.prev_zed_time = rospy.Time.now()
        self.kinect_buffer = np.zeros([33, 3])
        self.zed_buffer = np.zeros([33, 3])
        self.freq = freq
        rospy.Subscriber("/mrk/human_frame", PoseArray, self.callback)

        rospy.Subscriber("/mrk/zed/human_frame", PoseArray, self.zed_callback)

        self.marker_array = MarkerArray()
        
        self.human_publisher = rospy.Publisher("/mrk/human_skeleton", MarkerArray, queue_size=5)
    
    def callback(self, data:PoseArray):
        poses = data.poses
        # print(len(objects))   # 1
        
        
        # print(len(object.skeleton_3d.keypoints)) # 70
        for p in range(human_format):
            pose = poses[p]
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            frame_x = x
            frame_y = 1 - z
            frame_z = 1 - y 

            self.kinect_buffer[p, 0] = frame_x
            self.kinect_buffer[p, 1] = frame_y 
            self.kinect_buffer[p, 2] = frame_z
    
    def loop(self):
        marker_array = self.marker_array
        marker_array.markers = []
        for p in range(human_format):
            kinect_frame_z = self.kinect_buffer[p, 2] - 0.5
            kinect_frame_x = self.kinect_buffer[p, 0] - 0.5
            zed_frame_z = self.zed_buffer[p, 2] -0.5
            zed_frame_x = self.zed_buffer[p, 0] -0.5

            tan_x1 = 2 * kinect_frame_x * np.tan(KINECT_HORIZONTAL_WIDE_ANGLE / 2)
            th_x1 = np.arctan(tan_x1)
            tan_y1 = 2 * kinect_frame_z * np.tan(KINECT_VERTICLE_WIDE_ANGLE / 2)
            th_y1 = np.arctan(tan_y1)

            tan_x2 = 2 * zed_frame_x * np.tan(ZED_HORIZONTAL_WIDE_ANGLE / 2)
            th_x2 = np.arctan(tan_x2)
            tan_y2 = 2 * zed_frame_z * np.tan(ZED_VERTICLE_WIDE_ANGLE / 2)
            th_y2 = np.arctan(tan_y2)

            positions, position_err = reposition(th_x1=th_x1, th_y1=th_y1, th_x2=th_x2, th_y2=th_y2)
            pos_x = positions[0]
            pos_y = positions[1]
            pos_z = positions[2]

            pose = Pose()
            y =  - pos_y + 1.8
            x =  - pos_x + 1
            z =  pos_z + 0.4

            # y_angle = 0
            # mod_arr = np.array([[np.cos(y_angle ), 0,  np.sin(y_angle )],
            #                     [0, 1, 0],
            #                     [-np.sin(y_angle ) , 0, np.cos(y_angle )]])

            # pos_vec = np.array([x, y, z])
            # pos_vec = np.dot(mod_arr, pos_vec)
            
            # z_angle = 0
            # mod_arr = np.array([[np.cos(z_angle ), - np.sin(z_angle ), 0],
            #                     [np.sin(z_angle ), np.cos(z_angle ), 0],
            #                     [0, 0, 1]])
            
           
            # pos_vec = np.dot(mod_arr, pos_vec)
            pose.position.x = x - 0.3
            pose.position.y = y + 0.15
            pose.position.z = z + 0.4
            
            
            # z_angle = np.arctan(0.25)


            # pose.position.x = frame_x
            # pose.position.y = - (0.5 + 
            #                      (np.tan(ZED_HORIZONTAL_WIDE_ANGLE) / np.tan(KINECT_HORIZONTAL_WIDE_ANGLE))
            #                     * (zed_frmae_x - 0.5))
            # pose.position.z = frame_z 

            # print(f"x: {x}, y: {y}, z: {z}")
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 1

            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.action = Marker.ADD
            marker.type = Marker.SPHERE
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 0.
            marker.color.g = 1.
            marker.color.b = 0
            marker.color.a = 1

            marker.pose = pose
            marker.id = p+1
            marker.header.stamp = rospy.Time.now()
            marker_array.markers.append(marker)

        self.human_publisher.publish(marker_array)
                

    def zed_callback(self, data:PoseArray):
        print(len(data.poses))
        for id, p in enumerate(data.poses):
            pose = Pose()
            pose.position.x = p.position.x
            pose.position.y = 1 - p.position.z
            pose.position.z = 1 - p.position.y

            self.zed_buffer[id, 0] = pose.position.x
            self.zed_buffer[id, 1] = pose.position.y 
            self.zed_buffer[id, 2] = pose.position.z

            # co = self.get_collission_object(pose, str(100 + id*(id+1)))
            # planning_scene_interface.add_object(co)


    def run(self):
        while(not rospy.is_shutdown()):
            rospy.sleep(0.1)
            self.loop()
        
if __name__ == '__main__':
    publisher = HumanTrajectoryPublisher()
    publisher.run()
    rospy.spin()