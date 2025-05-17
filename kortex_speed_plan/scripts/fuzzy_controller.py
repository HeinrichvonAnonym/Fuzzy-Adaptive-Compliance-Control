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
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point
import scipy.spatial.transform as transform
import tf
from kortex_speed_plan.msg import SolidPrimitiveMultiArray
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker


class FuzzyController:

    """
    --------------------------------------
        if 
            human and robot is:  too close    
            to the human   
        and
            relative velo between
            robot and human is:  too fast

        the 
            robot should:        noly
            move away from human
    ---------------------------------------
        if 
            human and robot is:  close    
            to the human   
        and
            relative velo between
            robot and human is:  too fast
            
        the 
            robot should:        obviously
            move away from human
    ---------------------------------------
        if 
            human and robot is:  not close    
            to the human   
        and
            relative velo between
            robot and human is:  too fast
            
        the 
            robot should:        at the same time
            move away from human
    ---------------------------------------

    --------------------------------------
        if 
            human and robot is:  too close    
            to the human   
        and
            relative velo between
            robot and human is:  fast

        the 
            robot should:        obviously
            move away from human
    ---------------------------------------
        if 
            human and robot is:  close    
            to the human   
        and
            relative velo between
            robot and human is:  fast
            
        the 
            robot should:        at the same time
            move away from human
    ---------------------------------------
        if 
            human and robot is:  not close    
            to the human   
        and
            relative velo between
            robot and human is:  fast
            
        the 
            robot should:        by the way
            move away from human
    ---------------------------------------

    --------------------------------------
        if 
            human and robot is:  too close    
            to the human   
        and
            relative velo between
            robot and human is:  not fast

        the 
            robot should:        at the same time
            move away from human
    ---------------------------------------
        if 
            human and robot is:  close    
            to the human   
        and
            relative velo between
            robot and human is:  not fast
            
        the 
            robot should:        at the same time
            move away from human
    ---------------------------------------
        if 
            human and robot is:  not close    
            to the human   
        and
            relative velo between
            robot and human is:  not fast
            
        the 
            robot should:        by the way
            move away from human
    ---------------------------------------
    """
    
    def __init__(self, rules:dict, logic_rules:list):
        self.load_embedding_rules(rules)
        self.load_logic_rules(logic_rules)

    def load_embedding_rules(self, embedding_rules):
        self.fuzzy_embedings = embedding_rules
        self.distance_fuzzy_embeddings = self.fuzzy_embedings["distance"]
        self.velocity_fuzzy_embeddings = self.fuzzy_embedings["velocity"]
        self.weight_fuzzy_embeddings = self.fuzzy_embedings["weight"]
        self.len_distance_keys = len(self.distance_fuzzy_embeddings.keys())
        self.len_velocity_keys = len(self.velocity_fuzzy_embeddings.keys())
        self.len_weight_keys = len(self.weight_fuzzy_embeddings.keys())

    def load_logic_rules(self, logic_rules):
        self.logic_rules = logic_rules
        self.rules = []
        for rule in self.logic_rules:
            if rule["type"] == "AND":
                key_distance = rule["condition"]["distance"]
                key_velocity = rule["condition"]["velocity"]
                key_weight = rule["action"]["weight"]

                distance_embedding = self.human_robot_distance_embedding(self.distance_fuzzy_embeddings[key_distance]['mid'])
                velocity_embedding = self.human_robot_velo_embedding(self.velocity_fuzzy_embeddings[key_velocity]['mid'])
                weight_embedding = self.weight_fuzzy_embedding(self.weight_fuzzy_embeddings[key_weight]['mid'])

                v_condition = self.and_embedding(distance_embedding, velocity_embedding)

                v_action = np.zeros(self.len_weight_keys)
                for i, key_weight in enumerate(weight_embedding.keys()):
                    v_action[i] = weight_embedding[key_weight]
                
                r_matrix = np.zeros([self.len_distance_keys * self.len_velocity_keys, self.len_weight_keys])
                for i in range(self.len_distance_keys * self.len_velocity_keys):
                    for j in range(self.len_weight_keys):
                        r_matrix[i, j] = min(v_action[j] , v_condition[i])
                # print(r_matrix)
                
                
                self.rules.append({"type": "AND",
                                   "R": r_matrix})

    
    def and_embedding(self, distance_embedding, velocity_embedding):
        v_condition = np.zeros(self.len_distance_keys * self.len_velocity_keys)
        for i, key_distance in enumerate(distance_embedding.keys()):
            for j, key_velocity in enumerate(velocity_embedding.keys()):
                v_condition[i * self.len_velocity_keys + j] = min(distance_embedding[key_distance], velocity_embedding[key_velocity])
        return v_condition
        
    def or_embedding(self, distance_embedding, velocity_embedding):
        v_condition = np.zeros(self.len_distance_keys * self.len_velocity_keys)
        for i, key_distance in enumerate(distance_embedding.keys()):
            for j, key_velocity in enumerate(velocity_embedding.keys()):
                v_condition[i * self.len_velocity_keys + j] = max(distance_embedding[key_distance], velocity_embedding[key_velocity])
        return v_condition
          

    def human_robot_distance_embedding(self, distance):
        res_dict = {}
        for key in self.distance_fuzzy_embeddings:
            mid = self.distance_fuzzy_embeddings[key]["mid"]
            area = self.distance_fuzzy_embeddings[key]["area"]
            p = (area - 2 * abs(distance - mid)) / area
            p = max(min(p, 1), 0)
            res_dict[key] = p
        return res_dict
    
    def human_robot_velo_embedding(self, velo):
        res_dict = {}
        for key in self.velocity_fuzzy_embeddings:
            mid = self.velocity_fuzzy_embeddings[key]["mid"]
            area = self.velocity_fuzzy_embeddings[key]["area"]
            p = (area - 2 * abs(velo - mid)) / area
            p = max(min(p, 1), 0)
            res_dict[key] = p
        return res_dict
    
    def weight_fuzzy_embedding(self, weight):
        res_dict = {}
        for key in self.weight_fuzzy_embeddings:
            mid = self.weight_fuzzy_embeddings[key]["mid"]
            area = self.weight_fuzzy_embeddings[key]["area"]
            p = (area - 2 * abs(weight - mid)) / area
            p = max(min(p,1), 0)
            res_dict[key] = p
        return res_dict
    
    def get_action(self, human_robot_distance, human_robot_velo):
        distance_embedding = self.human_robot_distance_embedding(human_robot_distance)
        velocity_embedding = self.human_robot_velo_embedding(human_robot_velo)
        v_actions = []
        for rule in self.rules:
            if rule["type"] is "AND":
                v_condition = self.and_embedding(distance_embedding, velocity_embedding)
                # print(v_condition)
                R = rule["R"]
                v_action = np.zeros(self.len_weight_keys)
                for i in range(self.len_distance_keys * self.len_velocity_keys):
                    for j in range(self.len_weight_keys):
                        v_action[j] += v_condition[i] * R[i, j]
                # print(v_action)
                v_actions.append(v_action)
        v_actions = np.stack(v_actions, axis=0)
        print(v_actions)
        v_actions = self.de_fuzzy(v_actions)
        return v_actions

    def de_fuzzy(self, v_actions):
        output = 0
        for v_action in v_actions:
            for i, key in enumerate(self.weight_fuzzy_embeddings.keys()):
                output += v_action[i] * self.weight_fuzzy_embeddings[key]["mid"]
        # output /= len(v_actions)
        return output
    
if __name__ == "__main__":
    embedding_rules = {
        "distance": {
           "too close":{"mid": 0.0, "area": 0.25},
           "close": {"mid": 0.1, "area": 0.25},
           "not close": {"mid": 0.2, "area": 0.25},
           "far": {"mid": 0.3, "area": 0.25},
        },
        "velocity": {
            "too fast": {"mid": 0.4, "area": 0.35},
            "fast": {"mid": 0.25, "area": 0.3},
            "not fast": {"mid": 0.15, "area": 0.25},
            "mid": {"mid": 0.1, "area": 0.2},
            "slow": {"mid": 0.0, "area": 0.2},
        },
        "weight":{
            "light": {"mid": 0.1, "area": 0.5},
            "midium": {"mid": 1, "area": 2},
            "heavy": {"mid": 2, "area": 3},
            "very heavy": {"mid": 5, "area": 6},

        }
    }

    logic_rules = [
        {
            "type": "AND",
            "condition": {
                "distance": "too close",
                "velocity": "too fast"
            },
            "action": {
                "weight": "very heavy"
                }
        },

        {
            "type": "AND",
            "condition": {
                "distance": "too close",
                "velocity": "fast"
            },
            "action": {
                "weight": "heavy"
                }
        }
    ]
    controller = FuzzyController(embedding_rules, logic_rules)
    distance_key = controller.human_robot_distance_embedding(0.08)
    print(controller.get_action(0.02, 0.4))
    print(controller.get_action(0.02, 0.3))

