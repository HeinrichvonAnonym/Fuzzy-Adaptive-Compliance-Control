#!/usr/bin/env python

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

import rospy
from sensor_msgs.msg import JointState
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import yaml


class PIDController:
    def __init__(self, kp, ki, kd, alpha = 0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0
        self.alpha = alpha
    
    def compute(self, target_position, current_position, dt):
        error = target_position - current_position
        # print(error)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt>0 else 0.0

        derivative = self.prev_derivative * self.alpha + derivative * ( 1 - self.alpha)
        self.prev_derivative = derivative

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output, error


class PIDExecuter:
    def __init__(self):
        rospy.init_node("pid_executer", anonymous=True)
        self.joint_velocity_pub = rospy.Publisher("/desired_velocity", Float32MultiArray, queue_size=10)


        self.joint_state = None
        self.target_position = None
        self.errors = [0.0] * 7  # 存储每个关节的误差

        self.pid_controllers = []

        self.pid_controllers.append(PIDController(1.2, 0.0, 0.05, 0.5))
        self.pid_controllers.append(PIDController(1.2, 0.0, 0.05, 0.7))
        self.pid_controllers.append(PIDController(1.2, 0.0, 0.05, 0.5))
        self.pid_controllers.append(PIDController(1.2, 0.000, 0.05, 0.7))
        self.pid_controllers.append(PIDController(1.2, 0.0, 0.05))
        self.pid_controllers.append(PIDController(1.2, 0.0, 0.05, 0.5))
        self.pid_controllers.append(PIDController(0.9, 0.0, 0.05))

        self.is_idle = False
        self.screw = False

        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        rospy.Subscriber("/facc/pid_command", Float32MultiArray, self.drive_callback)
        self.init_dt = 0.01
        self.dt = 0.01
        self.cur_time = rospy.Time.now()
        self.breaked = False
    
    def joint_state_callback(self, msg):
        self.joint_state = msg

    def send_command(self, velocities):
        joint_velocity_msg = Float32MultiArray()
        joint_velocity_msg.data = []
        for i in range(len(velocities)):
            joint_velocity_msg.data.append(velocities[i])
        self.joint_velocity_pub.publish(joint_velocity_msg)
    

        
    def drive_callback(self, msg):
        self.target_position = msg.data[:7]
        
    

    def update_dt(self, event):
        self.dt = (rospy.Time.now() - self.cur_time).to_sec()
        self.cur_time = rospy.Time.now()

        
    def run(self):
        prev = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if self.joint_state is None:     
                continue
            if self.target_position is None:
                continue
            
            velocities = []
            self.cur_position = self.joint_state.position[:7]
            

            for i in range(len(self.target_position)):
                vel, error = self.pid_controllers[i].compute(self.target_position[i], self.cur_position[i], self.dt)
                # error = self.target_position[i] - self.cur_position[i]
                # vel = 0.01 * error
                velocities.append(vel)
                self.errors[i] = error  # 存储误差
                # print(vel)

            now = rospy.Time.now().to_sec()
            if (now -prev) > 1:
                print(">>>>>>>>>>>>>")
                print("cur:")
                print(self.cur_position)
                print("tar:")
                print(self.target_position)
                print("err:")
                print(self.errors)
                prev = now
                
            self.send_command(velocities)
            self.breaked = False
            
            
            rospy.sleep(self.init_dt)
            self.update_dt(self)
        
        

if __name__ == "__main__":
    pid_executer = PIDExecuter()
    pid_executer.run()
    rospy.spin()