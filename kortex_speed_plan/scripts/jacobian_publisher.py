import rospy
import tf
from sensor_msgs.msg import JointState
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, PoseStamped
import tf.transformations as tr
import yaml

config_path = "/home/heinrich/kinova/src/kortex_speed_plan/config/dynamical_parameters.yaml"
with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
algo_name = config["algo_name"]

# link_names = ["shoulder_link", "half_arm_1_link", "half_arm_2_link", "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link", "tool_frame"]
# jointstate_msg_topic = '/joint_states'
# base_link = "base_link"
# eef_link_names = ["tool_frame", "spherical_wrist_2_link", "forearm_link"]
# dofs = [7, 6, 4]

link_names = config['link_names']
jointstate_msg_topic = config['jointstate_msg_topic']
base_link = config['base_link']
eef_link_names = config['eef_link_names']
dofs = config['dofs']

# tool_mass = 0
# tool_com = [0, 0, 0]

tool_mass = config['tool_mass']
tool_com = np.array(config['tool_com'])

# link_params = {
    # # name: [mass, [com_x, com_y, com_z], [I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]]
    # 'shoulder_link': [1.4699, [-2.522E-05, -0.0075954, -0.088651], [0.0043269, 0.0044703, 0.0014532, 2.5E-07, 9.4E-07, 0.0001016]],
    # 'half_arm_1_link': [1.2357745, [-4.533E-05, -0.12951716, -0.01354356], [0.0115879, 0.00104574, 0.0116684, -1.05E-06, 5E-08, -0.00096902]],
    # 'half_arm_2_link': [1.2357745, [-4.533E-05, -0.00361448, -0.14407154], [0.01009873, 0.01017801, 0.00104697, 5.7E-07, 1.89E-06, 0.00013166]],
    # 'forearm_link': [0.89954802, [-0.00030188, -0.104938, -0.01559665], [0.00889854, 0.00060297, 0.00898975, 1.98E-05, -2.39E-06, -0.00074456]],
    # 'spherical_wrist_1_link': [0.70588351, [-0.00035363, -0.00659443, -0.07560343], [0.00145671, 0.00145189, 0.00039299, 3.35E-06, 7.62E-06, 0.00012055]],
    # 'spherical_wrist_2_link': [0.70583924, [-0.00035547, -0.06159424, -0.00850171], [0.00187208, 0.00041077, 0.0018494, 6.1E-06, -2.17E-06, -0.00033774]],
    # 'tool_frame': [0.31573861 + tool_mass, [ (tool_com[0] * tool_mass + 0.31573861 * -0.00010337) / (tool_mass + 0.31573861),
    #                                            (tool_com[1] * tool_mass + 0.31573861 * 0.00015804)/ (tool_mass + 0.31573861),
    #                                            (tool_com[2] * tool_mass + 0.31573861 * -0.02874642) / (tool_mass + 0.31573861)],
    #                                            [0.00018712, 0.00019576, 0.0002257, 6E-08, 7.7E-07, -1.62E-06]]}

link_params = config['link_params']

def get_tf_matrix(tf_listener, base_link_name, target_link_name):
    try:
        (trans, rot) = tf_listener.lookupTransform(base_link_name, target_link_name, rospy.Time(0))
        return tr.concatenate_matrices(tr.translation_matrix(trans), tr.quaternion_matrix(rot)), trans, rot
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return None


class JacobianPublisher():
    def __init__(self):
        rospy.init_node('jacobian_publisher', anonymous=True)
        rospy.Subscriber(jointstate_msg_topic, JointState, self.jointstate_cb)
        self.tf_listener = tf.TransformListener()
        self.jacobian_publishers = [rospy.Publisher(f'/{algo_name}/jacobian', Float32MultiArray, queue_size=10),
                                    rospy.Publisher(f'/{algo_name}/jacobian_6_dof', Float32MultiArray, queue_size=10),
                                    rospy.Publisher(f'/{algo_name}/jacobian_4_dof', Float32MultiArray, queue_size=10)]
        self.pose_publisher = rospy.Publisher(f"/{algo_name}/robot_pose", PoseArray, queue_size=10)
        self.joint_inertias_publisher = rospy.Publisher(f'/{algo_name}/joint_inertias', Float32MultiArray, queue_size=10)
        
        self.joint_states = None
        self.joint_inertias = None
        self.jacobian_mtrs = []
        self.poses = []
    
    def jointstate_cb(self, msg:JointState):
        self.joint_states = msg

    def calc_jacobian(self, eef_link_name, dof=len(link_names)):
        if self.joint_states is None:
            return None, None, None
        end_T, trans, rot = get_tf_matrix(self.tf_listener, base_link, eef_link_name)
        # print(trans, rot)
        
        jacobian_mtr = np.zeros((6, dof))

        for i, link in enumerate(link_names[:dof]):
            link_T, _, _ = get_tf_matrix(self.tf_listener, base_link, link)
            if link_T is None:
                continue
            joint_origin = link_T[:3, 3]
            eef_origin = end_T[:3, 3]
            z_axis = link_T[:3, 2]
            jacobian_mtr[:3, i] = np.cross(z_axis, eef_origin - joint_origin)
            
            jacobian_mtr[3:, i] = z_axis
        
        return jacobian_mtr, trans, rot

    def calc_jacobians(self):
        self.jacobian_mtrs = []
        self.poses = []
        for i, eef_link_name in enumerate(eef_link_names):
            jacobian_mtr, trans, rot = self.calc_jacobian(eef_link_name, dof = dofs[i])
            self.jacobian_mtrs.append(jacobian_mtr)
            self.poses.append(self.get_pose(trans, rot))

    def get_pose(self, trans, rot):
        if trans is None:
            return None
        pose = PoseStamped()
        pose.pose.position.x = trans[0]
        pose.pose.position.y = trans[1]
        pose.pose.position.z = trans[2]
        pose.pose.orientation.x = rot[0]
        pose.pose.orientation.y = rot[1]
        pose.pose.orientation.z = rot[2]
        pose.pose.orientation.w = rot[3]
        return pose
    
    def inertia_tensor_to_rotation_axis(self, inertia_tensor, rot_axis):
        rot_aixs = np.array(rot_axis)
        rot_aixs = rot_aixs / np.linalg.norm(rot_aixs)

        moment_of_inertia = np.dot(rot_aixs, np.dot(inertia_tensor, rot_aixs))

        return moment_of_inertia
    
    def calc_joint_inertias(self):
        if self.joint_states is None:
            return None
        
        num_joints = len(link_names)
        joint_inertias = np.zeros((num_joints))

        for i, link in enumerate(link_names):
            # print(link)
            if link_params[link] is None:
                continue
            mass, com, inertia = link_params[link]

            # print(mass, com, inertia)

            I_xx, I_yy, I_zz, I_xy, I_xz, I_yz = inertia
            inertia_tensor = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

            link_T, _, _ = get_tf_matrix(self.tf_listener, base_link, link)
            if link_T is None:
                continue

            rotation_matrix = link_T[:3, :3]

            inertia_world = np.dot(rotation_matrix, inertia_tensor)

            for j in range(i+1):

                joint_link = link_names[j]
                joint_T, _, _ = get_tf_matrix(self.tf_listener, base_link, joint_link)
                if joint_T is None:
                    continue

                joint_axis = joint_T[:3, 2]
                joint_pos = joint_T[:3, 3]

                com_pos = np.append(com, 1)
                # print(com_pos)
                com_world = np.dot(link_T, com_pos)[:3]
                
                # joint position -> mass center
                r_vec = com_world - joint_pos

                # projection vector
                proj_length = np.dot(r_vec, joint_axis)
                # perp vector
                perp_vec = r_vec - proj_length * joint_axis

                d_squared = np.dot(perp_vec, perp_vec)

                I_axis = self.inertia_tensor_to_rotation_axis(inertia_world, joint_axis)

                I_parallel_axis = mass * d_squared
                joint_inertias[j] += I_parallel_axis + I_axis

        self.joint_inertias = joint_inertias

        return joint_inertias
        
    def publish_jacobian(self):
        for i, jacobian_mtr in enumerate(self.jacobian_mtrs):
            if jacobian_mtr is None:
                return
            jacobian_msg = Float32MultiArray()
            jacobian_msg.data = jacobian_mtr.flatten().tolist()
            self.jacobian_publishers[i].publish(jacobian_msg)
    def publish_inertia(self):
        if self.joint_inertias is None:
            return
        inertia_msg = Float32MultiArray()
        inertia_msg.data = self.joint_inertias.flatten().tolist()
        self.joint_inertias_publisher.publish(inertia_msg)
    
    def publish_pose(self):
        pose_arr = PoseArray()
        if len(self.poses) > 0:
            for pose in self.poses:
                if pose is not None:
                    pose_arr.poses.append(pose.pose)
            pose_arr.header.frame_id = "base_link"
            self.pose_publisher.publish(pose_arr)


    def run(self):
        while(not rospy.is_shutdown()):
            rospy.sleep(0.02)
            self.calc_jacobians()
            self.publish_jacobian()
            self.publish_pose()
            self.calc_joint_inertias()
            if self.joint_inertias is not None:
                self.publish_inertia()
                # print(self.joint_inertias)
                # print(np.mean(self.joint_inertias))


if __name__ == '__main__':# quit
    jacobian_publisher = JacobianPublisher()
    jacobian_publisher.run()
    rospy.spin() 
