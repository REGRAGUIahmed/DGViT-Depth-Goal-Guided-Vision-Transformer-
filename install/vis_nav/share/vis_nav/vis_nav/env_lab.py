#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import path
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
import time
import math
import math
import random
import numpy as np
from numpy import inf
from collections import deque
from squaternion import Quaternion
import rclpy
import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseStamped

last_odom = None
last_image = None
last_dist = None
scan_data =None
goal_pose_rviz = None
velodyne_data = np.ones(20) * 10 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goalOK = False

    if -6.7 < x < -6.5 and -3.4 < y < 3.6:
        goalOK = True
    
    elif 3.5 < x < 5.2 and -3.4 < y < 3.6:
        goalOK = True
        
    elif -6.4 < x < 3.5 and -3.4 < y <-2.8:
        goalOK = True
    
    elif -6.4 < x < 3.5 and 3.1 < y < 3.6:
        goalOK = True
    
    elif -1.8 < x < -1.4 and -3.4 < y < 3.6:
        goalOK = True

    return goalOK


# Function to put the laser data in bins
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])

#launchfile, ROS_MASTER_URI, height, width, nchannels
class GazeboEnv(Node):
    """Superclass for all Gazebo environments.
    """

    def __init__(self):
        super().__init__('env') 
        self.entity_name = 'goal'
        self.entity_dir_path='/home/regmed/dregmed/vis_to_nav/src/vis_nav/description/sdf'
        self.entity_path = os.path.join(self.entity_dir_path, 'obstacle.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.odomX = 0.0
        self.odomY = 0.0

        self.goalX = 1.0
        self.goalY = 0.0
        self.angle = 0.0
        self.upper = 7.0 #10.0
        self.lower = -7.0 #-10.0
        # Changement de la position initial de robot mobile 
        self.collision = 0.0
        self.last_act = 0.0
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.x_pos_list = deque(maxlen=5)
        self.y_pos_list = deque(maxlen=5)
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(EntityState, "gazebo/set_entity_state", 10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        topic = 'goal_mark_array'
        self.publisher = self.create_publisher(MarkerArray, topic, 3)
        self.reset_proxy = self.create_client(Empty, "/reset_world")

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def spawn_entity(self):
        goal_pose1 = Pose()
        goal_pose1.position.x = self.goalX
        goal_pose1.position.y = self.goalY
        req_s = SpawnEntity.Request()
        req_s.name = self.entity_name
        req_s.xml = self.entity
        req_s.initial_pose = goal_pose1
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req_s)
    def delete_entity(self, entity_name):
        req = DeleteEntity.Request()
        req.name = entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.delete_entity_client.call_async(req)
    def calculate_observation(self, data):
        min_range = 0.5
        min_laser = 2.0
        done = False
        col = False
        for i, item in enumerate(data.ranges):
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                col = True
        return done, col, min_laser

    # Perform an action and read a new state
    def stop(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        #time.sleep(1.0)
    def step1(self, act, timestep):
        self.spawn_entity()
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        target = False
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...') # type: ignore

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(0.1)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e: # type: ignore
            print("/gazebo/pause_physics service call failed")

        data = scan_data
        dataOdom = last_odom
        data_obs = last_image
        v_state = []
        v_state[:] = velodyne_data[:]
        done, col, min_laser = self.calculate_observation(data)
        # Calculate robot heading from odometry data
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        self.x_pos_list.append(round(self.odomX,2))
        self.y_pos_list.append(round(self.odomY,2))
        
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        # Calculate the angle distance between the robots heading and heading toward the goal
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
                
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        '''Bunch of different ways to generate the reward'''
        
        r_heuristic = (self.distOld - Dist) * 20 #* math.cos(act[0]*act[1]/4)
        r_action = act[0]*2 - abs(act[1])
        r_smooth = - abs(act[1] - self.last_act)/4
              
        self.distOld = Dist

        r_target = 0.0
        r_collision = 0.0
        r_freeze = 0.0

        # Detect if the goal has been reached and give a large positive reward
        if Dist < 0.5:
            target = True
            #done = True 1/7
            self.delete_entity(self.entity_name)
            self.get_logger().info('Goal reached...')
            x =  self.goalX
            while goal_pose_rviz.pose.position.x == x :
                self.get_logger().info('Please give the goal location...')
                self.stop()
                time.sleep(2)
            self.goalX = goal_pose_rviz.pose.position.x
            self.goalY = goal_pose_rviz.pose.position.y    
            #self.change_goal()
            self.spawn_entity()            
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            r_target = 100

        # Detect if ta collision has happened and give a large negative reward
        if col:
            self.collision += 1
            r_collision = -100

        if timestep>10 and self.check_list(self.x_pos_list) and self.check_list(self.y_pos_list):
            r_freeze = -1

        reward = r_heuristic + r_action + r_collision + r_target + r_smooth #+ r_freeze
        Dist  = min(Dist/15, 1.0) #max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, act[0], act[1]])
        image = np.expand_dims(cv2.resize(data_obs, (160, 128)), axis=2)
        state = image / 255
        self.last_act = act[1]
        return state, r_heuristic, r_action, r_freeze, r_collision, r_target, reward, done, toGoal, target

    def step(self, act, timestep):
        self.spawn_entity()
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        target = False
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...') # type: ignore

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(0.1)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e: # type: ignore
            print("/gazebo/pause_physics service call failed")
        data = scan_data
        dataOdom = last_odom
        data_obs = last_image
        laser_state = np.array(data.ranges[:])
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        done, col, min_laser = self.calculate_observation(data)

        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        self.x_pos_list.append(round(self.odomX,2))
        self.y_pos_list.append(round(self.odomY,2))
        
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
                
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)


        '''Bunch of different ways to generate the reward'''
        
        r_heuristic = (self.distOld - Dist) * 20 #* math.cos(act[0]*act[1]/4)
        r_action = act[0]*2 - abs(act[1])
        r_smooth = - abs(act[1] - self.last_act)/4
              
        self.distOld = Dist

        r_target = 0.0
        r_collision = 0.0
        r_freeze = 0.0

        # Detect if the goal has been reached and give a large positive reward
        #if Dist < 0.5:
        if Dist < 0.15:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            r_target = 100

        # Detect if ta collision has happened and give a large negative reward
        if col:
            self.collision += 1
            r_collision = -100

        if timestep>10 and self.check_list(self.x_pos_list) and self.check_list(self.y_pos_list):
            r_freeze = -1

        reward = r_heuristic + r_action + r_collision + r_target + r_smooth #+ r_freeze
        Dist  = min(Dist/15, 1.0) #max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, act[0], act[1]])
        image = np.expand_dims(cv2.resize(data_obs, (160, 128)), axis=2)
        state = image / 255
        self.last_act = act[1]
        return state, r_heuristic, r_action, r_freeze, r_collision, r_target, reward, done, toGoal, target

    def check_list(self, buffer):
        it = iter(buffer)
        try:
            first = next(it)
        except StopIteration:
            return True
        return all((abs(first-x)<0.1) for x in buffer)
    
    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        '''rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")'''
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e: # type: ignore
            print("/gazebo/reset_simulation service call failed")
        #print(f'x= {self.odomX}  y = {self.odomY}')

        self.delete_entity(self.entity_name)
        self.change_goal()
        self.spawn_entity()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        data = scan_data
        data_obs = None
        data_obs_fish = last_image

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.2)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")
            
        '''while data is None:
            print("HHHHHHHHHHHHHHHHHHHHHHHHHHH")
            self.spawn_model()'''
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)

        '''while data_obs_fish is None:
            try:
                data_obs_fish = rospy.wait_for_message('/camera/fisheye/image_raw', Image, timeout=0.1)
            except:
                pass'''
        
        ##### FishEye Image ####
        camera_image = data_obs_fish
        
        #image = self.br.imgmsg_to_cv2(camera_image, "mono8")
        '''print("Saluuuut")
        print(type(camera_image))'''
        image = np.expand_dims(cv2.resize(camera_image, (160, 128)), axis=2)
        
        #image = np.expand_dims(cv2.resize(image[80:400, 140:500], (160, 128)), axis=2)
        state = image/255

        ######## Depth Image ##########
        # image = self.br.imgmsg_to_cv2(camera_image, "passthrough")
        # image = image.copy()
        # image[np.isnan(image)] = 10.0
        # image = np.expand_dims(cv2.resize(image, (128, 64)), axis=2)
        # state = image/10

        '''while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e: # type: ignore
            print("/gazebo/pause_physics service call failed")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")'''

        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY

        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - self.angle)

        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        Dist  = min(Dist/15, 1.0) # max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, 0.0, 0.0])
        return state, toGoal
    
    def reset1(self, cntr=0):
        # Resets the state of the environment and returns an initial observation.
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e: # type: ignore
            print("/gazebo/reset_simulation service call failed")
        if cntr ==1:
            self.delete_entity(self.entity_name)
            x =  self.goalX
            while goal_pose_rviz == None or goal_pose_rviz.pose.position.x == x:
                self.get_logger().info('Please give the goal location...')
                self.stop()
                time.sleep(2)
            self.goalX = goal_pose_rviz.pose.position.x
            self.goalY = goal_pose_rviz.pose.position.y    
            #self.change_goal()
            self.spawn_entity()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        data = scan_data
        data_obs_fish = last_image

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.2)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)
        
        ##### FishEye Image ####
        camera_image = data_obs_fish
        image = np.expand_dims(cv2.resize(camera_image, (160, 128)), axis=2)
        
        state = image/255

        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY

        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - self.angle)

        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        Dist  = min(Dist/15, 1.0) # max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, 0.0, 0.0])
        return state, toGoal

    # Place a new goal and check if its lov\cation is not on one of the obstacles


        
        
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.008
        if self.lower > -10:
            self.lower -= 0.008

        gOK = False
        #and self.goalX==self.modelX and self.goalY==self.goalY
        while not gOK :
            self.goalX = self.odomX + random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + random.uniform(self.upper, self.lower)

            euclidean_dist = math.sqrt((self.goalX - self.odomX)**2 + (self.goalY - self.odomY)**2)
            if self.upper > 4 and euclidean_dist < 3:
                gOK = False
                continue
            elif self.upper > 8 and euclidean_dist < 6:
                gOK = False
                continue

            gOK = check_pos(self.goalX, self.goalY)
    def change_pose(self):
        self.angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., self.angle)
        x = 0.0
        y = 0.0
        chk = False
        while not chk :
            x = np.random.uniform(-7.0, 7.0)
            y = np.random.uniform(-4.0, 4.0)
            chk = check_pos(x, y)
        self.modelX = x
        self.modelY = y
        self.orientationX = quaternion.x
        self.orientationY = quaternion.y
        self.orientationZ = quaternion.z
        self.orientationW = quaternion.w
        self.odomX = self.orientationX
        self.odomY = self.orientationY
        return x,y,quaternion.x,quaternion.y,quaternion.z,quaternion.w

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / 20]]
        for m in range(20 - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / 20]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(20) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

class LaserScan_subscriber(Node):

    def __init__(self):
        super().__init__('laserScan_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/front_laser/scan',
            self.laser_callback,
            1)
        self.subscription

    def laser_callback(self, od_data):
        global scan_data
        scan_data = od_data
         
class GoalPose_subscriber(Node):

    def __init__(self):
        super().__init__('GoalPose_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            1)
        self.subscription

    def goal_pose_callback(self, data):
        global goal_pose_rviz
        goal_pose_rviz = data
class DepthImage_subscriber(Node):
    def __init__(self):
        super().__init__('depth_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Replace with your depth image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            cv_image_normalized = cv_image_normalized.astype(np.uint8)
            
            global last_image
            last_image = cv_image_normalized[80:400, 140:500]  # Crop to (440, 640)
            # last_image = cv_image_normalized
            
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))
class Image_fish_subscriber(Node):
    def __init__(self):
        super().__init__('image_fish_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera2/image_raw',  # Replace with your depth image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, rgb_data):
        try:
            image = self.bridge.imgmsg_to_cv2(rgb_data, "mono8")
            #original_image = self.br.imgmsg_to_cv2(rgb_data, "rgb8")
            #last_image_fish = image[80:400, 140:500]
            last_image_fish = image[80:400, 118:523]
            #image_ = self.br.imgmsg_to_cv2(rgb_data, "rgb8") 
            global last_image
            last_image = last_image_fish
            
        except Exception as e:
            self.get_logger().error('Could not convert fish image: %s' % str(e))


class Image_subscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber') 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback, qos_profile_sensor_data)
        self.subscription
        self.bridge = CvBridge()
    def image_callback(self, im_data):
        # print(".....................HI  image_callback!!.........................")
        global last_image
        cv_image = self.bridge.imgmsg_to_cv2(im_data, "mono8")
        last_image = cv_image
