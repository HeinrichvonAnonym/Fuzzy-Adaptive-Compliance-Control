import rospy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
import cv2
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.msg import CollisionObject
import sys
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray, Int16
from std_srvs.srv import Empty, Trigger, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point
import scipy.spatial.transform as transform
import tf
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker
import  yaml
import curses

max_velo = 1.5
max_dis = 0.8
max_e = 1 # rad

EMERGENCY_COUNT = 10

class Fuzzyficator:
    def __init__(self, rules:dict):
        self.rules = rules
    
    def fuzzyficate(self, value):

        res_dict = {}
        for key in self.rules:
            mid = self.rules[key]["mid"]
            area = self.rules[key]["area"]
            p = (area - 2 * abs(value - mid)) / area
            p = max(min(p, 1), 0)
            res_dict[key] = p
        return res_dict

class Fuzzificators:
    def __init__(self, rules:dict):
        self.rules = rules
        self.embedding_rules = rules
        self.distance_fuzzificator = Fuzzyficator(rules["distance"])
        self.velocity_fuzzificator = Fuzzyficator(rules["velocity"])
        self.weight_fuzzificator = Fuzzyficator(rules["weight"])
        self.euclidean_fuzzificator = Fuzzyficator(rules["euclidean"])


class FuzzzyCore:
    def __init__(self, embedding_rules:dict, logic_rules:list, name:str, input_key_1:str, input_key_2:str, output_key:str):
        self.input_key_1 = input_key_1
        self.input_key_2 = input_key_2
        self.output_key = output_key
        self.embedding_rules = embedding_rules
        self.logic_rules = logic_rules
        self.fuzzificators = Fuzzificators(embedding_rules)
        self.init_params()
        self.load_logic_rules(logic_rules[name])
        
  
    def init_params(self):
        self.len_input_1_keys = len(self.embedding_rules[self.input_key_1].keys())
        self.len_input_2_keys = len(self.embedding_rules[self.input_key_2].keys())
        self.len_output_keys = len(self.embedding_rules[self.output_key].keys())
        self.fuzzificator_input_1 = eval(f"self.fuzzificators.{self.input_key_1}_fuzzificator")
        self.fuzzificator_input_2 = eval(f"self.fuzzificators.{self.input_key_2}_fuzzificator")
        self.fuzzificator_output = eval(f"self.fuzzificators.{self.output_key}_fuzzificator")

    def load_logic_rules(self, logic_rules):
        self.logic_rules = logic_rules
        self.rules = []
        for rule in self.logic_rules:
            
            key_input_1 = rule["condition"][self.input_key_1]
            key_input_2 = rule["condition"][self.input_key_2]
            key_output = rule["action"][self.output_key]

            # print(self.rules)
            input_1_embedding  = self.fuzzificator_input_1.fuzzyficate(self.embedding_rules[self.input_key_1][key_input_1]['mid'])
            input_2_embedding  = self.fuzzificator_input_2.fuzzyficate(self.embedding_rules[self.input_key_2][key_input_2]['mid'])
            output_embedding   = self.fuzzificator_output.fuzzyficate(self.embedding_rules[self.output_key][key_output]['mid'])

            if rule["type"] == "AND":
                v_condition = self.and_embedding(input_1_embedding, input_2_embedding)
            elif rule["type"] == "OR":
                v_condition = self.or_embedding(input_1_embedding, input_2_embedding)
            elif rospy.logerr("fuzzy_controller.py", "load_logic_rules", "Unknown rule type: {}".format(rule["type"])):
                continue

            v_action = np.zeros(self.len_output_keys)
            for i, key_weight in enumerate(output_embedding.keys()):
                v_action[i] = output_embedding[key_weight]
            
            r_matrix = np.zeros([self.len_input_1_keys * self.len_input_2_keys, self.len_output_keys])
            for i in range(self.len_input_1_keys * self.len_input_2_keys):
                for j in range(self.len_output_keys):
                    r_matrix[i, j] = min(v_action[j] , v_condition[i])
            # print(r_matrix)
            
            if rule["type"] == "AND":
                self.rules.append({"type": "AND",
                                "R": r_matrix})
            elif rule["type"] == "OR":
                self.rules.append({"type": "OR",
                                "R": r_matrix})

    def and_embedding(self, input_1_embedding, input_2_embedding):
        v_condition = np.zeros(self.len_input_1_keys * self.len_input_2_keys)
        for i, key_distance in enumerate(input_1_embedding.keys()):
            for j, key_velocity in enumerate(input_2_embedding.keys()):
                v_condition[i * self.len_input_2_keys + j] = min(input_1_embedding[key_distance], input_2_embedding[key_velocity])
        return v_condition
    
    def or_embedding(self, input_1_embedding, input_2_embedding):
        v_condition = np.zeros(self.len_input_1_keys * self.len_input_2_keys)
        for i, key_distance in enumerate(input_1_embedding.keys()):
            for j, key_velocity in enumerate(input_2_embedding.keys()):
                v_condition[i * self.len_input_2_keys + j] = max(input_1_embedding[key_distance], input_2_embedding[key_velocity])
        return v_condition
    
    def get_action(self, input_1, input_2):
        inpyt_1_embedding = self.fuzzificator_input_1.fuzzyficate(input_1)
        inpyt_2_embedding = self.fuzzificator_input_2.fuzzyficate(input_2)
        v_actions = []
        for rule in self.rules:
            if rule["type"] == "AND":
                v_condition = self.and_embedding(inpyt_1_embedding, inpyt_2_embedding)
                # print(v_condition)
            elif rule["type"] == "OR":
                v_condition = self.or_embedding(inpyt_1_embedding, inpyt_2_embedding) 
            elif rospy.logwarn("fuzzy_controller.py", "get_action", "Unknown rule type: {}".format(rule["type"])):
                continue
            R = rule["R"]
            v_action = np.zeros(self.len_output_keys)
            for i in range(self.len_input_1_keys * self.len_input_2_keys):
                for j in range(self.len_output_keys):
                    v_action[j] += v_condition[i] * R[i, j]
            # print(v_action)
            v_actions.append(v_action)
        v_actions = np.stack(v_actions, axis=0)
        # print(v_actions)
        v_actions = self.de_fuzzy(v_actions)
        return v_actions
    
    def de_fuzzy(self, v_actions):
        output = 0
        weight_summary = 0
        for v_action in v_actions:
            for i, key in enumerate(self.embedding_rules[self.output_key].keys()):
                output += v_action[i] * self.embedding_rules[self.output_key][key]["mid"]
                weight_summary += v_action[i]
        # output /= len(v_actions)
        return output / max(weight_summary, 0.0001)

class FuzzyController:
    def __init__(self, embedding_rules:dict, logic_rules:list, stdscr:curses.window):
        rospy.init_node("fuzzy_controller", anonymous=True)
        self.fpi_core = FuzzzyCore(embedding_rules, logic_rules, "FPI", "distance", "velocity", "weight")
        self.fri_core = FuzzzyCore(embedding_rules, logic_rules, "FRI", "distance", "euclidean", "weight")
        rospy.loginfo("Fuzzy core initialized")

        rospy.Subscriber("/rrt/fuzzy/d_euclidean", Float32, self.euclidean_callback)
        rospy.Subscriber("/rrt/fuzzy/velocity", Float32, self.velocity_callback)
        rospy.Subscriber("/rrt/fuzzy/distance", Float32, self.min_distance_callback)
        self.pi_publisher = rospy.Publisher("/rrt/pi_weight", Float32, queue_size=10)
        self.ri_publisher = rospy.Publisher("/rrt/ri_weight", Float32, queue_size=10)
        self.raw_pi_publisher = rospy.Publisher("/rrt/raw_pi_weight", Float32, queue_size=10)

        self.euclidean = None
        self.velocity = None
        self.min_distance = None

        self.weight_p = None
        self.weight_r = None
        self.raw_weight_p = None

        curses.curs_set(0)
        self.stdscr = stdscr
        self.stdscr.nodelay(1)
        self.stdscr.timeout(100)
        self.max_slider_value = 100
        self.min_slider_value = 0

        self.emergemcy = False
        self.emergemcy_count = 0

        self.dt = 0.05
    
    def get_fpi_weight(self, distance, velocity):
        return self.fpi_core.get_action(distance, velocity)
    
    def get_fri_weight(self, euclidean, distance):
        return self.fri_core.get_action(euclidean, distance)
    
    def euclidean_callback(self, msg:Float32):
        self.euclidean = msg.data

    def velocity_callback(self, msg:Float32):
        self.velocity = msg.data
    
    def min_distance_callback(self, msg:Float32):
        self.min_distance = msg.data
    
    def update_slider(self, distance, velo, euclid, weight_p, weight_r):
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        slider_width = width - 4

        min_distance = min(100, distance * self.max_slider_value / max_dis)
        velocity = min(100, velo * self.max_slider_value / max_velo)
        euclidean = min(100, euclid * 100 / max_e)

        p = (weight_p + 1) * 100 / 2
        r = (weight_r + 1) * 100 / 2
   
        # print(velocity)
        slider_pos_distance = int((min_distance- self.min_slider_value) / (self.max_slider_value - self.min_slider_value) * slider_width)
        slider_pos_velocity = int((velocity - self.min_slider_value) / (self.max_slider_value - self.min_slider_value) * slider_width) - 1
        slider_pos_velocity = max(0, slider_pos_velocity)
        euclidean = int((euclidean - self.min_slider_value) / (self.max_slider_value - self.min_slider_value) * slider_width)

        p = int((p - self.min_slider_value) / (self.max_slider_value - self.min_slider_value) * slider_width)
        r = int((r - self.min_slider_value) / (self.max_slider_value - self.min_slider_value) * slider_width)
        
        self.stdscr.addstr(height // 2 - 4, 2, "[" + "=" * slider_pos_distance + " " * (slider_width - slider_pos_velocity))
        self.stdscr.addstr(height // 2 - 3, int(width // 2 - len(f"Distance: {distance}") // 2), f"Distance: {distance}")
        self.stdscr.addstr(height // 2 - 2, 2, "[" + "=" * slider_pos_velocity + " " * (slider_width - slider_pos_velocity))
        self.stdscr.addstr(height // 2 - 1, int(width // 2 - len(f"Velo: {velo}") // 2), f"Velo: {velo}")
        self.stdscr.addstr(height // 2    , 2, "[" + "=" * euclidean + " " * (slider_width - slider_pos_velocity))
        self.stdscr.addstr(height // 2 + 1, int(width // 2 - len(f"Euclidean: {euclid}") // 2), f"Euclidean: {euclid}")

        self.stdscr.addstr(height // 2 + 3  , 2, "[" + "=" * p + " " * (slider_width - slider_pos_velocity))
        self.stdscr.addstr(height // 2 + 4, int(width // 2 - len(f"Weight_p: {weight_p}") // 2), f"Weight_p: {weight_p}")
        self.stdscr.addstr(height // 2 + 5  , 2, "[" + "=" * r + " " * (slider_width - slider_pos_velocity))
        self.stdscr.addstr(height // 2 + 6, int(width // 2 - len(f"Weight_r: {weight_r}") // 2), f"Weight_r: {weight_r}")
        
        self.stdscr.refresh()

    def update_weight_p(self, w_p, alpha=0.99):
        if self.weight_p is None:
            self.weight_p = w_p
            return w_p
        
        if self.weight_p < w_p:
            self.weight_p = w_p
            return w_p
        
        output = self.weight_p * alpha + w_p * (1-alpha)
        self.weight_p = w_p
        return output
    def run(self):
        while not rospy.is_shutdown():
            weight_p = 0.
            weight_r = 0.
            min_distance = 0.5
            velocity = 0
            euclidean = 0
            if self.velocity is not None and self.min_distance is not None:
                min_distance = self.min_distance
                velocity = self.velocity
                d = min(self.min_distance, max_dis)
                d = max(d, 0)
                v = min(self.velocity, max_velo)
                v = max(v, 0)
                weight_p = self.get_fpi_weight(d, v)
                raw_p = Float32()
                raw_p.data = weight_p
                self.raw_pi_publisher.publish(raw_p)
                if weight_p > 0.75:
                    self.emergemcy = True
                    self.emergemcy_count = EMERGENCY_COUNT
                
                if self.emergemcy:
                    weight_p = max(0.75, weight_p)
                    self.emergemcy_count -= 1
                    self.emergemcy = self.emergemcy_count > 0
       
                # print(weight)

                self.pi_publisher.publish(weight_p)
               
            #     if weight > 0.5:
            #         rospy.loginfo(f"disytance:{self.min_distance} velo:{self.velocity * 1e3} > weight:{weight}>>>>>>")
            if self.euclidean is not None and self.min_distance is not None:
                d = min(self.min_distance, max_dis)
                d = max(d, 0)
                e = min(self.euclidean, max_e)
                e = max(e, 0)
                weight_r = self.get_fri_weight(d, e)
                # weight_r = min(max(- weight_p, 0), weight_r)
                self.ri_publisher.publish(weight_r)
                min_distance = self.min_distance
                euclidean = self.euclidean


            self.update_slider(min_distance, velocity, euclidean, weight_p, weight_r)
            rospy.sleep(self.dt)
        rospy.spin()

def main(stdscr):
    with open("/home/heinrich/fuzzy_ws/src/fuzzy_core/scripts/fuzzy_rules.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        embedding_rules = data["embedding_rules"]
        logic_rules = data["logic_rules"]
    controller = FuzzyController(embedding_rules, logic_rules, stdscr)
    #print(controller.get_fpi_weight(0.08, 0.02))
    controller.run()

if __name__ == "__main__":
    curses.wrapper(main)
    
    # print(controller.distance_fuzzificator.fuzzyficate(0.08))

