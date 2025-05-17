import rospy
from sensor_msgs.msg import JointState
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
import numpy as np
import matplotlib.pyplot as plt
import time
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

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
        self.joint_velocity_pub = rospy.Publisher("in/joint_velocity", Base_JointSpeeds, queue_size=10)
        
        # 添加误差发布器
        self.joint_error_pub = rospy.Publisher("rrt/joint_error", Float32MultiArray, queue_size=10)
        # 添加虚拟参考点发布器
        self.virtual_reference_pub = rospy.Publisher("rrt/virtual_reference", Float32MultiArray, queue_size=10)

        self.joint_state = None
        self.target_position = None
        self.errors = [0.0] * 7  # 存储每个关节的误差

        self.pid_controllers = []

        # self.pid_controllers.append(PIDController(0.5, 0.0, 0.05, 0.5))
        # self.pid_controllers.append(PIDController(0.4, 0.0, 0.04, 0.7))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.07, 0.5))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.07, 0.5))
        # self.pid_controllers.append(PIDController(0.4, 0.0, 0.05))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.07, 0.5))
        # self.pid_controllers.append(PIDController(0.7, 0.0, 0.1))

        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.05, 0.5))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.01, 0.7))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.001, 0.5))
        # self.pid_controllers.append(PIDController(0.4, 0.0, 0.001, 0.5))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.01))
        # self.pid_controllers.append(PIDController(0.3, 0.0, 0.01, 0.5))
        # self.pid_controllers.append(PIDController(0.3, 0.0, 0.01))

        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.005, 0.5))
        # self.pid_controllers.append(PIDController(0.7, 0.0, 0.001, 0.7))
        # self.pid_controllers.append(PIDController(0.7, 0.0, 0.001, 0.5))
        # self.pid_controllers.append(PIDController(0.5, 0.000, 0.001, 0.7))
        # self.pid_controllers.append(PIDController(0.7, 0.0, 0.001))
        # self.pid_controllers.append(PIDController(0.6, 0.0, 0.001, 0.5))
        # self.pid_controllers.append(PIDController(0.3, 0.0, 0.01))

        self.pid_controllers.append(PIDController(0.65, 0.0, 0.001, 0.5))
        self.pid_controllers.append(PIDController(0.8, 0.0, 0.001, 0.7))
        self.pid_controllers.append(PIDController(0.8, 0.0, 0.001, 0.5))
        self.pid_controllers.append(PIDController(0.55, 0.000, 0.002, 0.7))
        self.pid_controllers.append(PIDController(0.8, 0.0, 0.001))
        self.pid_controllers.append(PIDController(0.7, 0.0, 0.001, 0.5))
        self.pid_controllers.append(PIDController(0.5, 0.0, 0.01))

        self.is_idle = False
        self.screw = False

        rospy.Subscriber("rrt/pid_command", Float32MultiArray, self.drive_callback)
        rospy.Subscriber("/base_feedback/joint_state", JointState, self.joint_state_callback)
        rospy.Subscriber("rrt/idle", String, self.idle_callback)

        self.init_dt = 0.01
        self.dt = 0.01
        self.cur_time = rospy.Time.now()
        self.breaked = False
    
    def joint_state_callback(self, msg):
        self.joint_state = msg
    
    def idle_callback(self, msg:String):
        # print(msg)
        if msg.data == "idle":
            print("idle")
            self.is_idle = True
            self.screw = False
        elif msg.data == "screw":
            print("screw")
            self.is_idle = True
            self.screw = True
        else:
            print("go")
            self.is_idle = False
            self.screw = False
    def send_command(self, velocities):
        joint_velocity_msg = Base_JointSpeeds()
        for i in range(len(velocities)):
            joint_speed = JointSpeed()

            joint_speed.joint_identifier = i
            joint_speed.value = velocities[i]
            joint_speed.duration = 1
            joint_velocity_msg.joint_speeds.append(joint_speed)
        self.joint_velocity_pub.publish(joint_velocity_msg)
    
    def publish_joint_error(self):
        """发布关节误差数据"""
        error_msg = Float32MultiArray()
        error_msg.data = self.errors
        self.joint_error_pub.publish(error_msg)
        
    def drive_callback(self, msg):
        self.target_position = msg.data
        
        # 发布虚拟参考点数据
        virtual_ref_msg = Float32MultiArray()
        virtual_ref_msg.data = self.target_position
        self.virtual_reference_pub.publish(virtual_ref_msg)
    
    def update_dt(self, event):
        self.dt = (rospy.Time.now() - self.cur_time).to_sec()
        self.cur_time = rospy.Time.now()
        
    def run(self):
        while not rospy.is_shutdown():
            if self.joint_state is None:     
                continue
            if self.target_position is None:
                continue
            
            velocities = []
            self.cur_position = self.joint_state.position[:7]
            for i in range(len(self.target_position)):
                vel, error = self.pid_controllers[i].compute(self.target_position[i], self.cur_position[i], self.dt)
                velocities.append(vel)
                self.errors[i] = error  # 存储误差
                # print(vel)
            
            if self.is_idle:
                if not self.breaked:
                    self.send_command([0., 0., 0., 0., 0., 0., 0.])
                    self.breaked = True
            elif self.screw:
                self.send_command([0., 0., 0., 0., 0., 0., -0.1])
                self.breaked = False
            else:
                self.send_command(velocities)
                self.breaked = False
            
            
                
            
            # 发布误差数据
            self.publish_joint_error()
            
            rospy.sleep(self.init_dt)
            self.update_dt(self)
        
        

if __name__ == "__main__":
    pid_executer = PIDExecuter()
    pid_executer.run()
    rospy.spin()