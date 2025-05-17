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
from std_msgs.msg import Float32, Float32MultiArray, Int16
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion
import scipy.spatial.transform as transform
import tf
from kortex_speed_plan.msg import SolidPrimitiveMultiArray
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
import math

"""
control with force balance detection
"""
Bc = 0.1
Sm = 0.1
Threshold = 0.1

USE_FUZZY = False
smooth_att = 0.15
k_att_base = 5000 # 基础引力系数
human_k_rep = 1500
obj_k_rep = 500
k_att_cart = 8000
k_lamda = 500

if not USE_FUZZY:
    human_influence_margin = 0.35
    human_safe_margin = 0.05
    human_k_rep = 6500
    k_lamda = 100
else:
    human_influence_margin = 0.8
    human_safe_margin = 0.05

obj_influence_margin = 0.1
obj_safe_margin = 0.03




END_EFFECTOR = 0
WRIST = 1
FOREARM = 2




def angle_normalize(angles):
    output_angles = []
    for i, angle in enumerate(angles):
        angle = angle % np.pi * 2
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < - np.pi:
            angle += 2 * np.pi
        output_angles.append(angle)
    output_angles = np.array(output_angles)
    return angles


def ref_fuzzy_slot(fri, cur_idx):
    rand = np.random.rand()
    if rand > abs(fri):
        return cur_idx
    elif  fri < 0:
        return cur_idx - 1
    elif  fri > 0:
        return cur_idx + 1
    

def pot_fuzzy_slot(fpi, att_potential, rep_potential):
    fpi = fpi * np.pi / 4
    angle = np.pi / 4 + fpi
    cos = max(np.cos(angle), 0.707)
    sin = np.sin(angle)
    return (cos * att_potential + sin * rep_potential) / 0.707


class APFController:
    def __init__(self):
        rospy.init_node("apf_controller", anonymous=True)
        rospy.Subscriber("rrt/target_pos", JointState, self.target_callback)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        rospy.Subscriber("rrt/objects", SolidPrimitiveMultiArray, self.object_callback)
        rospy.Subscriber("/mrk/human_skeleton",MarkerArray, self.human_callback)
        rospy.Subscriber("rrt/robot_pose", PoseArray, self.pose_callback)
        rospy.Subscriber("rrt/jacobian", Float32MultiArray, self.jacobian_callback)
        rospy.Subscriber("rrt/jacobian_6_dof", Float32MultiArray, self.jacobian_6_dof_callback)
        rospy.Subscriber("rrt/jacobian_4_dof", Float32MultiArray, self.jacobian_4_dof_callback)
        rospy.Subscriber("rrt/refresh", Int16, self.refresh_callback)
        rospy.Subscriber("rrt/target_pose", PoseStamped, self.target_pose_callback)
        rospy.Subscriber("/rrt/pi_weight", Float32, self.pi_subscriber)
        rospy.Subscriber("rrt/ri_weight", Float32, self.ri_callback)
        rospy.Subscriber("rrt/work_damping", Float32, self.work_damping_callback)
        rospy.Subscriber("/rrt/fuzzy/velocity", Float32, self.velocity_callback)
        # self.pi_publisher = rospy.Publisher("rrt/pi_weight", Float32, queue_size=10)
        self.pid_command_pub = rospy.Publisher("rrt/pid_command", Float32MultiArray, queue_size=10)
        self.rep_pub = rospy.Publisher("rrt/rep_vec", Marker, queue_size=10)
        
        # 添加DSM值发布器 - 区分人和物体
        self.dsm_pub = rospy.Publisher("rrt/dsm_value", Float32MultiArray, queue_size=10)
        
        # 添加对象类型和安全距离发布器
        self.obj_info_pub = rospy.Publisher("rrt/object_info", Float32MultiArray, queue_size=10)
        
        # 添加引力和斥力发布器
        self.force_pub = rospy.Publisher("rrt/force_values", Float32MultiArray, queue_size=10)
        self.euclidean_pub =rospy.Publisher("/rrt/euclidean", Float32, queue_size=10)
        self.distance_pub = rospy.Publisher("/rrt/distance", Float32, queue_size=10)
        self.pot_rep_pub = rospy.Publisher("/rrt/pot_rep", Float32MultiArray, queue_size=10)
        
        self.target_pos = None
        self.target_cartesian_pose = None
        self.cur_pos = None
        self.objects_poses = None
        self.primitives = None
        self.human_poses = None
        self.posed = None
        self.dt = 0.02
        self.prev_time = rospy.Time.now()
        self.poses = None
        self.jacobian = None
        self.jacobian_6_dof = None
        self.jacobian_4_dof = None
        self.rel_vel = 0
        self.max_i = 0
        self.distance_vec = np.array([10, 10, 10])
        
        # DSM值 - 分别存储对人和对物体的DSM
        self.human_dsm_value = 0.0
        self.obj_dsm_value = 0.0
        self.min_distance = float('inf')
        self.nearest_object_type = -1  # -1: 未知, 0: 人, 1: 物体
        self.nearest_object_index = -1

        # fuzzy control
        self.pi_index = 0
        self.ri_index = 0

        # work mode
        self.work_damping = 0

    def velocity_callback(self, msg):
        self.rel_vel =  msg.data

    def calc_attraction_potential(self):
        target_pos = self.target_pos 
        cur_pos = self.cur_pos[:7] 
        target_pos = angle_normalize(target_pos)
        # print(target_pos)
        cur_pos = angle_normalize(cur_pos)
        pos_dis = target_pos - cur_pos
        # pos_dis = angle_normalize(pos_dis)
        # # 角度归一化
        for i in range(len(pos_dis)):
            pos_dis[i] = pos_dis[i] % (2 * np.pi)
            if pos_dis[i] > np.pi:
                pos_dis[i] = pos_dis[i] % np.pi - np.pi
            elif pos_dis[i] < -1 * np.pi:
                pos_dis[i] = -pos_dis[i] % np.pi + np.pi

        dist = np.linalg.norm(pos_dis)
        dist_msg = Float32()
        dist_msg.data = dist
        self.euclidean_pub.publish(dist)
        # print("distance: ", dist)
        p_att = pos_dis / max(dist, smooth_att) * k_att_base
        
        return p_att, dist

    def calc_repulsion_potential(self):
        critical_poses = self.poses
        obj_poses = self.objects_poses
        primitives = self.primitives

        max_amplitude = 0.
        potential_rep = np.zeros(7)
        self.min_distance = float('inf')
        
        # 重置最近物体信息
        self.nearest_object_type = -1
        self.nearest_object_index = -1
        
        # 保存人和物体的最小距离
        human_min_distance = float('inf')
        obj_min_distance = float('inf')

        for i, critical_pose in enumerate(critical_poses):
            if obj_poses is not None:
                cur_influence_margin = obj_influence_margin
                cur_safe_margin = obj_safe_margin
                cur_k_rep = obj_k_rep
                object_type = 1  
                for j, (obj_pose, primitive) in enumerate(zip(obj_poses, primitives)):
                    # print(obj_pose)
                    cenrtri_vector = np.array([critical_pose.position.x - obj_pose.position.x, 
                                            critical_pose.position.y - obj_pose.position.y, 
                                            critical_pose.position.z - obj_pose.position.z])
                    centri_dis = np.linalg.norm(cenrtri_vector)
                    
                    if primitive.type == SolidPrimitive.SPHERE:
                        inner_dis = primitive.dimensions[0]
                        if centri_dis < primitive.dimensions[0]:
                            dis = 0
                        else:
                            dis = centri_dis - primitive.dimensions[0]   
                            
                    elif primitive.type == SolidPrimitive.CYLINDER:
                        inner_vec = np.array([primitive.dimensions[0] / 2, 
                                            primitive.dimensions[1]])
                        inner_dis = np.linalg.norm(inner_vec)
                        if centri_dis < inner_dis:
                            dis = 0
                        else:
                            dis = centri_dis - inner_dis
                            
                    elif primitive.type == SolidPrimitive.BOX:
                        inner_vec = np.array([primitive.dimensions[0] / 2,
                                            primitive.dimensions[1] / 2, 
                                            primitive.dimensions[2] / 2])
                        inner_dis = np.linalg.norm(inner_vec)
                        if centri_dis < inner_dis:
                            dis = 0
                        else:
                            dis = centri_dis - inner_dis
                        # 更新物体的最小距离
                    if dis < obj_min_distance:
                        obj_min_distance = dis
                    
                    # 更新全局最小距离
                    if dis < self.min_distance:
                        self.min_distance = dis
                        self.nearest_object_type = object_type
                        self.nearest_object_index = j
                            
                    amplitude = (cur_influence_margin - dis) / (cur_influence_margin - cur_safe_margin)
                    amplitude = max(0, amplitude)
                    
                    if amplitude > max_amplitude:
                        max_amplitude = amplitude
                        max_i = i
                        obs_pose = obj_pose
                        choosed_inner_dis = inner_dis
                        max_influence_margin = cur_influence_margin
                        max_safe_margin = cur_safe_margin
                        max_k_rep = cur_k_rep
                        potential_vec = cenrtri_vector / centri_dis
                        potential_vec = potential_vec * amplitude

            if self.human_poses is not None:
                for human_pose in self.human_poses: 
                    # print(human_pose)
                    cur_k_rep = human_k_rep
                    pose = critical_pose
                    position_link = np.array([pose.position.x,
                                            pose.position.y,
                                            pose.position.z])
                    position_human = np.array([human_pose.position.x,
                                                human_pose.position.y,
                                                human_pose.position.z])
                    distance_vec = position_human - position_link
                    dis = np.linalg.norm(distance_vec) - 0.1
                    choosed_inner_dis = 0.1
                    if dis< self.min_distance:
                        amplitude = (human_influence_margin - dis) / (human_influence_margin - human_safe_margin)
                        amplitude = max(0, amplitude)
                        self.min_distance =dis
                        self.distance_vec = distance_vec

                    if amplitude > max_amplitude:
                        max_amplitude = amplitude
                        max_i = i
                        obs_pose = human_pose
                        max_influence_margin = human_influence_margin
                        max_safe_margin = human_safe_margin
                        max_k_rep = cur_k_rep
                        potential_vec = - distance_vec / dis
                        potential_vec = potential_vec * amplitude
        if self.min_distance < 1e4:
            distance_msg = Float32()
            distance_msg.data = self.min_distance
            self.distance_pub.publish(distance_msg)
        

            
        if max_amplitude > 0:
            # self.update_marker(potential_vec, obs_pose, choosed_inner_dis, max_influence_margin, max_safe_margin)
            self.max_i = i
            potential_rep = self.cartesian_2_axis(potential_vec, max_i)
        
        
            
        return potential_rep * max_k_rep if max_amplitude > 0 else potential_rep

    def quatenion_multiplication(self, q1, q2):
        # w x y z
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]
    
    def quatenion_conjugate(self, q):
        return [q[0], -q[1], -q[2], -q[3]]
    
    def euler_from_quaternion(self, quaternion):
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]
        w = quaternion[0]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y- z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return [roll, pitch, yaw]
    
    def cartesian_2_axis(self, cartesian_vec, max_i):
        # print(max_i)
        # jacobian = self.jacobian
        if max_i == 2:
            # print("forarm")
            jacobian = self.jacobian_4_dof
            cat = True
            cat_dim = 3

        elif max_i == 1:
            # print("wrist")
            jacobian = self.jacobian_6_dof
            cat = True
            cat_dim = 1

        elif max_i == 0:
            jacobian = self.jacobian
            cat = False
        # print(jacobian.shape)
        rot_vec = np.zeros(3)
        influence_vec = np.concatenate((cartesian_vec, rot_vec))
        jacobian_inv = np.linalg.pinv(jacobian)
        angular_influence = np.dot(jacobian_inv, influence_vec)
        # print(angular_influence.shape)
        
        if cat:
            angular_influence = np.concatenate((angular_influence, np.zeros(cat_dim)))
            
        return angular_influence
    
    def calc_velo(self, potential):
   
        cur_vel = np.array(self.cur_vel[:7])
        damping = k_lamda  * cur_vel
        # print(self.ri_index)
        
        potential = potential - damping
 
        # command_tar = self.cur_pos[:7] + self.dt * (self.cur_vel[:7] + self.dt * potential) / 2
        # pid_command = Float32MultiArray()
        # pid_command.data = command_tar.tolist()
        velo = self.cur_vel[:7] + self.dt * potential
        return velo
    
    def send_pid_command(self, velo):
        command_tar = self.cur_pos[:7] + self.dt * (self.cur_vel[:7] + velo) / 2
        pid_command = Float32MultiArray()
        pid_command.data = command_tar.tolist()
        self.pid_command_pub.publish(pid_command)

    def target_pose_callback(self, msg:PoseStamped):
        self.target_cartesian_pose = msg.pose

    def joint_state_callback(self, msg):
        self.cur_pos = np.array(msg.position)
        self.cur_vel = np.array(msg.velocity)

    def target_callback(self, target):
        self.target_pos = np.array(target.position)

    def object_callback(self, msg:SolidPrimitiveMultiArray):
        # print("get object")
        # print(msg.poses)
        self.objects_poses = msg.poses
        # print(self.object_poses)
        self.primitives = msg.primitives
    
    def pose_callback(self, msg:PoseArray):
        self.poses = msg.poses

    def work_damping_callback(self, msg:Float32):
        self.work_damping = msg.data

    def jacobian_callback(self, msg):
        self.jacobian = np.array(msg.data).reshape(6,7)
    def jacobian_6_dof_callback(self, msg):
        self.jacobian_6_dof = np.array(msg.data).reshape(6,6)
    def jacobian_4_dof_callback(self, msg):
        self.jacobian_4_dof = np.array(msg.data).reshape(6,4)
    
    def pi_subscriber(self, msg:Float32):
        self.pi_index = msg.data
    
    def ri_callback(self, msg:Float32):
        ri = msg.data # -1: 1
        ri += 1 # 0:2
        ri /= 2 # 0:1
        self.ri_index = ri

    def refresh_callback(self, msg):
        self.target_pos = None
    
    def human_callback(self, ma:MarkerArray):
        markers = ma.markers
        self.human_poses = []
        for marker in markers:
            self.human_poses.append(marker.pose)

    def get_Sc(self, velo, cos_rc):
        return self.rel_vel * 2 * self.dt + np.linalg.norm(velo) * cos_rc * self.dt + Bc + Sm
    
    def calc_nova_velo(self, velo, cos_rc, Cc):
        rel_velo = self.rel_vel
        Sc = rel_velo * 2 * self.dt + np.linalg.norm(velo) * cos_rc * self.dt + Bc + Sm
        p_f_n = Sc - Cc

        norm_v = np.linalg.norm(velo)
        p_f_d_norm =  3 * self.dt * cos_rc 

        norm = norm_v - p_f_n / p_f_d_norm

        delta_norm = (norm_v - norm)

        # print(norm / norm_v)
        n_velo = velo * norm/norm_v
        
        norm_v = np.linalg.norm(velo)

        Sc = (rel_velo - delta_norm) * 2 * self.dt + np.linalg.norm(n_velo) * cos_rc * self.dt + Bc + Sm
    
        f_n = Sc - Cc
        # print(f"{p_f_n} - > {f_n}")
       
        return n_velo, norm>0

    def run(self):
        prev_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():

            rospy.sleep(self.dt)
            if (self.objects_poses is None and self.human_poses is None) or self.poses is None or self.jacobian is None:
                # if self.object_poses is None:
                # print(">>>")
                p_rep = np.zeros(7)
            else:
                p_rep = self.calc_repulsion_potential()     

            if self.cur_pos is None:
                continue
            
            if self.target_pos is None:
                p_att = np.zeros(7)
                p_c_att = np.zeros(7)
                potential = p_att

            else:
                potential, dist = self.calc_attraction_potential()

            velo_j = self.calc_velo(potential)
            if self.max_i == 0:
                jacobian = self.jacobian
                dim = 7
            elif self.max_i == 1:
                jacobian = self.jacobian_6_dof
                dim = 6
            elif self.max_i == 2:
                jacobian = self.jacobian_4_dof
                dim = 4
            velo_dot = velo_j[:dim]
            velo = np.dot(jacobian, velo_dot)[:3]

            cos_rc = np.dot(velo, self.distance_vec) / (np.linalg.norm(velo) * np.linalg.norm(self.distance_vec) + 1e-8)

            velo_rc = velo * cos_rc
            Sc = self.get_Sc(velo_rc, cos_rc)
            Cc = self.min_distance

            if rospy.Time.now().to_sec() - prev_time > 5:
                print(f"velo: {np.linalg.norm(velo)}, distance: {self.min_distance}, rel_val: {self.rel_vel} >> Sc: {Sc} Cc: {Cc}" )
                prev_time += 5
            
            if Cc > Sm:
                if cos_rc > 0 and self.rel_vel> 0:
                    if Sc > Cc:
                        velo_command, keep = self.calc_nova_velo(velo, cos_rc, Cc)
                        k = np.linalg.norm(velo_command) / np.linalg.norm(velo)
                        print(f">>>>>>>>>>>>>> Mid Situation K: {k}")
                        velo_command = velo_j * k
                        if not keep:
                            velo_command *= -1
                    else:
                        velo_command = velo_j
                else: 
                    velo_command = velo_j
              

            else:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>BREAK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                velo_command = [0., 0., 0., 0., 0., 0., 0.]
            

            
            self.send_pid_command(velo_command)



if __name__ == "__main__":
    controller = APFController()
    controller.run()
    rospy.spin()