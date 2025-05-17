import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float32, Float64
import numpy as np

class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add(self, value):
        if len(self.buffer) < self.size:
            self.buffer.append(value)
        else:
            self.buffer.pop(0)
            self.buffer.append(value)
    def get_mean(self):
        return np.mean(self.buffer)

    def get_len(self):
        return len(self.buffer)


class FuzzyMonitor:
    def __init__(self):
        rospy.init_node('fuzzy_monitor', anonymous=True)

        self.distance_pub = rospy.Publisher('/rrt/fuzzy/distance', Float32, queue_size=10)
        self.velocity_pub = rospy.Publisher('/rrt/fuzzy/velocity', Float32, queue_size=10)
        self.deuclidean_pub = rospy.Publisher('/rrt/fuzzy/d_euclidean', Float32, queue_size=10)

        rospy.Subscriber('/rrt/distance', Float32, self.distance_callback)
        rospy.Subscriber('/rrt/euclidean', Float32, self.euclidean_callback)

        self.distance_buffer = Buffer(10)
        self.distance_vel = Buffer(10)
        self.euclidean_buffer = Buffer(10)
        self.euclidean_vel = Buffer(3)

        self.dt = 0.05


    def distance_callback(self, data):
        if self.distance_buffer.get_len()>0:

            cur_time = rospy.Time.now().to_sec()
            dt = cur_time - self.prev_distance_time
            self.prev_distance_time = cur_time

            self.distance_vel.add((data.data - self.distance_buffer.get_mean())/dt)
            # print(dt)
        else:
            self.prev_distance_time = rospy.Time.now().to_sec()
        self.prev_distance = self.distance_buffer.get_mean()
        self.distance_buffer.add(data.data)

    def euclidean_callback(self, data):
        # print(">>")
        if self.euclidean_buffer.get_len() > 0:
            cur_time = rospy.Time.now().to_sec()
            dt = cur_time - self.prev_euclidean_time
            self.prev_euclidean_time = cur_time
            self.euclidean_vel.add((data.data - self.euclidean_buffer.get_mean()) / dt)
        else:
            self.prev_euclidean_time = rospy.Time.now().to_sec()
        self.euclidean_buffer.add(data.data)

    def run(self):
        max_vel = 0
        start_time = rospy.Time.now().to_sec()
        while(not rospy.is_shutdown()):
            distance_msg = Float32()
            velo = Float32()
            euclidean_msg = Float32()
            if len(self.distance_buffer.buffer) > 0:
                distance = self.distance_buffer.get_mean()
                distance_msg.data = distance
                self.distance_pub.publish(distance_msg)

            if len(self.distance_vel.buffer) > 0:
                vel = self.distance_vel.get_mean()
                velo = - vel
                self.velocity_pub.publish(velo)

            if len(self.euclidean_buffer.buffer) > 0:
                euclidean = self.euclidean_buffer.get_mean()

            if len(self.euclidean_buffer.buffer) > 0:
                euclidean = self.euclidean_buffer.get_mean()
                # print(euclidean_vel)
                euclidean_msg.data = euclidean
                self.deuclidean_pub.publish(euclidean_msg)

            rospy.sleep(self.dt)
        rospy.spin()

# # when fri is 1 100 % return cur_udx + 1
# # when fri is -1 100 % return cur_udx - 1
# def ref_fuzzy_slot(fri, cur_idx):
#     rand = np.random.rand()
#     if rand > abs(fri):
#         return cur_idx
#     elif  fri < 0:
#         return cur_idx - 1
#     elif  fri > 0:
#         return cur_idx + 1

# # -1 ~ att
# # 1 ~ rep
# def pot_fuzzy_slot(fpi, att_potential, rep_potential):
#     fpi = fpi * np.pi / 4
#     angle = np.pi / 4 + fpi
#     cos = np.cos(angle)
#     sin = np.sin(angle)
#     return cos * att_potential + sin * rep_potential

if __name__ == '__main__':
    monitor = FuzzyMonitor()
    monitor.run()