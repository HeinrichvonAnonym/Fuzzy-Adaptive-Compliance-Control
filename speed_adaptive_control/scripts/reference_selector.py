"""
copyright (c) 2025-present Heinrich 2130238@tongji.edu.cn.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""


import time
import threading
import numpy as np
from packaging import version
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
import rospy
from std_msgs.msg import Float32

class mpc_core:
    def __init__(self):
        pass


RL_MODE_SCALE = 1.0
OMPL_MODE_SCALE = 0.5


class ReferenceSelector():
    def __init__(self, dof, num_envs):
        # rospy.Subscriber()
        self.euclidean_threshold = 0.35
        self.dof = dof
        self.rl_buffer = []
        self.cur_pos = np.zeros(dof)

        self.num_envs = num_envs
        self.traj_len_threshold = 16

        for _ in range(num_envs):
            self.rl_buffer.append([])
        
        self.ompl_buffer = None

        self.distance_start_threshold = 0.15
        self.eval_threshold = 0.01
        self.prev_time = rospy.Time.now().to_sec()
        self.att_scale_pub = rospy.Publisher("facc/att_scale", Float32, queue_size=10)
        self.att_scale_msg = Float32()
    
    def normalize_angle(self, q):
        q = q % (2 * np.pi) 
        q = np.where(q > np.pi, q - 2 * np.pi, q)
        q = np.where(q < -np.pi, q + 2 * np.pi, q)
        return q       

    def _refresh_trajectory(self, action=None):        
        pass                                                                    
    
    def get_target_q(self, action, idx):
        target_q = action[idx]
        if len(self.rl_buffer[idx]) > 0:
            target_q = self.rl_buffer[idx][0]
     
        return self.normalize_angle(target_q)
  
    
    def pop_trajectory(self, idx):
        rl_buffer = self.rl_buffer
        for idx in range(len(rl_buffer)):
            if len(rl_buffer[idx]) > 1:
                rl_buffer[idx].pop(0)

    def get_action_idx_mask(self):
        idx = np.zeros(self.num_envs, dtype=np.int8)
        idx[-1] = 1
        return idx

    def reset_traj(self, reset_signal):
        reset_idx = np.where(reset_signal)[0].tolist()
        for idx in reset_idx:
            self.rl_buffer[idx] = []
    
    def query_rl_traget(self, action, cur_pos, action_idx):
        cur = self.normalize_angle(cur_pos)   
        tar = self.get_target_q(action, action_idx)
        err = tar - cur
        err = self.normalize_angle(err)
        euclidean_distance = np.linalg.norm(err)
        # print(err)

        while euclidean_distance < self.euclidean_threshold:
            # print("go on")
            if len(self.rl_buffer[action_idx]) > 1:   
                self.pop_trajectory(action_idx)   
                tar = self.get_target_q(action, action_idx)     
                err = tar - cur
                err = self.normalize_angle(err)
                euclidean_distance = np.linalg.norm(err)
            else:
                break
        
        return tar
    
    def query_ompl_target(self, cur_pos):
        min_distance = 1e3
        min_idx = 0
        for idx, point in enumerate(self.ompl_buffer):
            distance = np.linalg.norm(self.normalize_angle(self.normalize_angle(point)
                                                            - 
                                                        self.normalize_angle(cur_pos))) 
            if distance < min_distance:
                min_distance = distance
                min_idx = idx

        self.ompl_buffer = self.ompl_buffer[min_idx:]
        tar_pos = self.ompl_buffer[0]
        tar = self.normalize_angle(tar_pos)
        cur = self.normalize_angle(cur_pos)
        err = tar - cur
        euclidean_distance = np.linalg.norm(err)
        while euclidean_distance < self.euclidean_threshold:
            if len(self.ompl_buffer) > 1:
                self.ompl_buffer = self.ompl_buffer[1:]
                tar_pos = self.ompl_buffer[0]
                tar = self.normalize_angle(tar_pos)
                cur = self.normalize_angle(cur_pos)
                err = tar - cur
                euclidean_distance = np.linalg.norm(err)
            else:
                break
        return tar

    def choose_bufer(self, cur_pos):
        if self.ompl_buffer is None:
            return 0
        
        ompl_end_point = self.ompl_buffer[-1]
        distance = np.linalg.norm(self.normalize_angle(self.normalize_angle(cur_pos) - self.normalize_angle(ompl_end_point)))
        # distance = distance % np.pi
        # print(distance)
        if distance < 0.5:
            return 1
        else:
            return 1
    
    def step(self, action, cur_pos):
        reset_signal = np.zeros(self.num_envs)
        if action is not None: 
            for append_idx in range(len(self.rl_buffer)):
                if len(self.rl_buffer[append_idx]) >= self.traj_len_threshold:
                    reset_signal[append_idx] = 1
                self.rl_buffer[append_idx].append(action[append_idx])
            
            bufer_flag = self.choose_bufer(cur_pos)
        
            if bufer_flag == 0:
                self.att_scale_msg.data = RL_MODE_SCALE
                
                selected_idx = self.get_action_idx_mask()
                action_idx = np.where(selected_idx)[0][0]
                tar = self.query_rl_traget(action, cur_pos, action_idx)
            else:
                self.att_scale_msg.data = OMPL_MODE_SCALE
                tar = self.query_ompl_target(cur_pos)
            cur_time = rospy.Time.now().to_sec()
            distance_t = cur_time - self.prev_time
            if distance_t > 0.1:
                reset_signal[:] = 1
                self.prev_time = cur_time

            self.reset_traj(reset_signal)
        else:
            if self.ompl_buffer is None:
                return None, None
            self.att_scale_msg.data = OMPL_MODE_SCALE
            tar = self.query_ompl_target(cur_pos)

        self.att_scale_pub.publish(self.att_scale_msg)
        qr = tar
        return qr,  reset_signal


class ReachingFrankaRos():
    def __init__(self):
        rospy.init_node('reference_selector', anonymous=True)
        self.reference_selector = ReferenceSelector(7, 
                                                    16 # nur im RL Zustand genutzt wird
                                                    )
        rospy.Subscriber("base_feedback/joint_state", JointState, self.js_callback)
        rospy.Subscriber("/eef_position/", Float32MultiArray, self.eef_callback)
        rospy.Subscriber("/facc/cartesian_trajectory", RobotTrajectory, self.traj_callback)
        self.qr_publisher = rospy.Publisher("desired_q", Float32MultiArray, queue_size=10)
        self.dt = 0.02
        self.joint_positions = None
        self.joint_velocities = None
    
    def traj_callback(self, msg:RobotTrajectory):
        trajectory = msg.joint_trajectory.points
        ompl_buf = []
        for point in trajectory:
            joint_positions = np.array(point.positions)
            ompl_buf.append(joint_positions)
        self.reference_selector.ompl_buffer = ompl_buf
    
    def eef_callback(self, msg:Float32MultiArray):
        self.eef_pos = np.array(msg.data)
    
    def js_callback(self, msg:JointState):
        #     print(">")
        self.joint_positions = np.array(msg.position[:7])
        self.joint_velocities = np.array(msg.velocity[:7])

    def reset(self):
        self.reference_selector._refresh_trajectory()

    def step(self, action=None):
        if self.joint_positions is None:
            return None, None, None, None
        
        action, reset_signal = self.reference_selector.step(action, self.joint_positions)

        if action is None:
            return None, None, None, None
        # print(action)
        msg = Float32MultiArray()
        msg.data = action[:7].tolist()
        self.qr_publisher.publish(msg)

        cube_state = None
        return self.joint_positions, self.joint_velocities,  cube_state,  reset_signal,

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def run(self):
        while not rospy.is_shutdown():
            self.step()
            rospy.sleep(self.dt)


if __name__ == '__main__':
    robot = ReachingFrankaRos()
    robot.run()
   