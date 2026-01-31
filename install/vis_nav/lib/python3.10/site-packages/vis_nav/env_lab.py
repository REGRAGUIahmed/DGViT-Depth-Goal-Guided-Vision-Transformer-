#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
import time
import math
import random
import numpy as np
from numpy import inf
from collections import deque
from squaternion import Quaternion
import rclpy
import cv2
from cv_bridge import CvBridge
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, LaserScan
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from utils import binning
last_odom = None
last_image = None
last_dist = None
scan_data =None
goal_pose_rviz = None
trajectorie=[]
from skimage.segmentation import slic
# from skimage.color import rgb2lab

def get_center_band(image):
    """Returns coordinates for a horizontal center band with height h/5"""
    h, w = image.shape[:2]
    band_height = h // 5
    y1 = h // 2 - band_height // 2
    y2 = y1 + band_height
    return y1, y2

def pixel_occlusion(image):
    """Occludes a horizontal center band with black pixels"""
    # import matplotlib.pyplot as plt
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_before_30_06.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    image = image.astype(np.float32)
    pertubo_image = image.copy()
    y1, y2 = get_center_band(image)
    pertubo_image[y1:y2, :] = 0
    # plt.imshow(pertubo_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_after_30_06.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    return pertubo_image

def superpixel_occlusion(image, segments=50):
    """Occludes superpixels overlapping the horizontal center band"""
    # import matplotlib.pyplot as plt
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_before_superpixel_occlusion.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    segments_slic = slic(image, n_segments=segments, compactness=4, start_label=1, channel_axis=None)
    pertubo_image = image.copy()
    y1, y2 = get_center_band(image)

    # Mask: find superpixels that overlap the horizontal band
    affected_segments = np.unique(segments_slic[y1:y2, :])
    for seg in affected_segments:
        pertubo_image[segments_slic == seg] = 0
    # plt.imshow(pertubo_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_after_superpixel_occlusion.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()    
    return pertubo_image

def greying_out(image):
    """Replaces the horizontal center band with gray color"""
    # import matplotlib.pyplot as plt
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_before_greying_out.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    pertubo_image = image.copy()
    y1, y2 = get_center_band(image)
    grey = 128
    pertubo_image[y1:y2, :] = grey
    # plt.imshow(pertubo_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_after_greying_out.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()    
    return pertubo_image

def blurring(image):
    """Blurs the horizontal center band"""
    # import matplotlib.pyplot as plt
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_before_blurring.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    pertubo_image = image.copy()
    y1, y2 = get_center_band(image)
    region = pertubo_image[y1:y2, :]
    blurred = cv2.GaussianBlur(region, (11, 11), 0)
    pertubo_image[y1:y2, :] = blurred
    # plt.imshow(pertubo_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_after_blurring.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()  
    return pertubo_image

def add_nose(image, noise_level=0.02):
    import matplotlib.pyplot as plt
    """
    Add Gaussian noise to the image.
    :param image: Input image.
    :param noise_level: Standard deviation of the Gaussian noise.
    :return: Noisy image.
    """
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_add_nose_{noise_level}.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    image = image.astype(np.float32)
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    blurred1 = cv2.GaussianBlur(noisy_image, (5, 5), 0)
    # plt.imshow(blurred1, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'/home/regmed/dregmed/vis_to_nav/metrics_data/image_noisy{noise_level}.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.close()
    return blurred1
class GazeboEnv(Node):
    """
    Superclass for all Gazebo environments.
    """

    def __init__(self):
        super().__init__('env') 
        self.entity_name = 'goal'
        self.entity_dir_path='/home/regmed/dregmed/vis_to_nav/src/vis_nav/description/sdf'
        self.entity_path = os.path.join(self.entity_dir_path, 'obstacle.sdf')
        self.entity = open(self.entity_path, 'r').read()
        # Load the positions from the .npz file
        # self.positions = np.load('/home/regmed/dregmed/vis_to_nav/src/vis_nav/resource/curriculum_positions.npz', allow_pickle=True)
        self.positions = np.load('/home/regmed/dregmed/vis_to_nav/src/vis_nav/resource/test_position_100_goals2.npz', allow_pickle=True)
        # self.positions = np.load('/home/regmed/dregmed/vis_to_nav/src/vis_nav/resource/test_positions.npz', allow_pickle=True)
        # Convert to a list of records
        self.records = [self.positions[key].item() for key in self.positions]
        # self.indice_position_test = np.load('/home/regmed/dregmed/vis_to_nav/metrics_data/goals.npz')
        self.indice_position = 0 #83
        self.flag = True #if self.indice_position>0 else True
        # self.indice_test = []
        self.odomX = 0.0
        self.odomY = 2.0
        self.entityX = 0.0
        self.entityY = 2.0
        self.quaterX = 0.0
        self.quaterY = 0.0
        self.quaterZ = 0.0
        self.quaterW = 1.0
        self.goalX = 2.0
        self.goalY = 2.0
        self.angle = 0.0
        self.upper = 5.0 #10.0
        self.lower = -5.0 #-10.0
        self.collision = 0.0
        self.last_act = [0,0]
        self.cntr_traj = 0
        self.set_entity_client = self.create_client(SetEntityState, 'gazebo/set_entity_state')
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.publisher = self.create_publisher(MarkerArray, 'goal_mark_array', 3)
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        # self.x_pos_list = deque(maxlen=5)
        # self.y_pos_list = deque(maxlen=5)
    def set_seed(self, seed):
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
    def set_entity(self, name, x ,y):
        request = SetEntityState.Request()
        request.state.name = name
        request.state.pose.position.x = x
        request.state.pose.position.y = y
        request.state.pose.position.z = 0.0 
        
        request.state.pose.orientation.x = self.quaterX
        request.state.pose.orientation.y = self.quaterY
        request.state.pose.orientation.z = self.quaterZ
        request.state.pose.orientation.w = self.quaterW       
        
        future = self.set_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Failed to change entity position.')
   
    def calculate_observation(self, data):
        min_range = 0.2 # For testing 0.2 For training 0.3
        min_laser = 2.0
        done = False
        col = False
        for i, item in enumerate(data.ranges):
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                # self.get_logger().info(f'Collision !!... {data.ranges[i]} ')
                col = True
        return done, col

    # Perform an action and read a new state
    def stop(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
    
    def step(self, act, timestep):
        # self.spawn_entity()
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)                                  
        target = False
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...') 
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
        except (rclpy.ServiceException) as e: 
            print("/gazebo/pause_physics service call failed")
        data = scan_data
        dataOdom = last_odom
        data_obs = last_image
        done, col = self.calculate_observation(data)
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        trajectorie.append(self.odomX)
        trajectorie.append(self.odomY)
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
        # r_action = act[0]*2 - abs(act[1])
        r_heuristic = (self.distOld - Dist) * 20
        # r_smooth = - abs(act[1] - self.last_act[1])/4
        self.distOld = Dist

        r_target = 0.0
        r_collision = 0.0
        r_arret = 0.0
        # r_freeze = 0.0
        # Detect if the goal has been reached and give a large positive reward
        if Dist < 0.5:
            self.get_logger().info(f'Goal reached... Goal number = {self.indice_position}')
            # np.save(os.path.join('trajectories', 'trajectorie_'+str(self.cntr_traj)), trajectorie, allow_pickle=True, fix_imports=True)
            self.cntr_traj += 1
            trajectorie.clear()
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            r_target = 200 
            r_arret = 50*(2-abs(act[1]))*(1-act[0])
            # if act[0]<0.1 and abs(act[1])<0.1:
            #     r_target +=100
            # self.indice_test.append(self.indice_position)
            # np.savez('/home/regmed/dregmed/vis_to_nav/src/vis_nav/resource/positions_clean.npz', *self.indice_test)
        # Detect if ta collision has happened and give a large negative reward
        # Goal number = {self.indice_position} in xG = {self.goalX} yG = {self.goalY}
        if col:
            # self.get_logger().info(f'Collision !!...')
            self.collision += 1
            r_collision = -100          
        # if timestep>10 and self.check_list(self.x_pos_list) and self.check_list(self.y_pos_list):
        #     r_freeze = -1
        reward =  r_collision + r_target   + r_heuristic #+ r_arret+ r_freeze  + r_smooth+r_arret+ r_action +
        #reward = r_collision + r_target #+ r_freeze r_heuristic + 
        image = np.expand_dims(cv2.resize(data_obs, (160, 128)), axis=2)
        # image = np.expand_dims(data_obs, axis=2)
        Dist  = min(Dist/15, 1.0) #max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, act[0], act[1]])
        state = image / 255
        # state = image
        # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/state.npz', state=state)
        
        self.last_act = act 
        reward =np.clip(reward,-200,500)
        return state,reward, done, toGoal, target
        # return state,r_heuristic, r_action, r_freeze, r_collision, r_target, reward, done,toGoal, target
    # def check_list(self, buffer):
    #     it = iter(buffer)
    #     try:
    #         first = next(it)
    #     except StopIteration:
    #         return True
    #     return all((abs(first-x)<0.1) for x in buffer)
    
    def reset(self):
        # while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('/gazebo/reset_world service not available, waiting...')
        # request = Empty.Request()
        # future = self.reset_proxy.call_async(request)
        # rclpy.spin_until_future_complete(self, future)
        # if future.result() is not None:
        #     self.get_logger().info('World reset successfully.')
        # else:
        #     self.get_logger().error('Failed to call /gazebo/reset_world service')   
        # Select a random record
        # random_record = random.choice(self.records)
        # indice = self.indice_position_test['goals'][self.indice_position]
        random_record = self.records[self.indice_position]
        # random_record = self.records[indice]
        if self.flag :
            if self.indice_position < len(self.records) - 1:
                self.indice_position += 1
            else :
                self.indice_position = 0 #len(self.records) -100
        
        # Retrieve the values
        xR = random_record['xR']#0.0
        yR = random_record['yR']#2.5
        xG = random_record['xG']
        yG = random_record['yG']
        # quaterX = random_record['quaterX']
        # quaterY = random_record['quaterY']
        # quaterZ = random_record['quaterZ']
        # quaterW = random_record['quaterW']
        # self.delete_entity(self.entity_name)
        # self.quaterX = 0.0 #quaterX
        # self.quaterY = 0.0 #quaterY
        # self.quaterZ = quaterZ
        # self.quaterW = quaterW
        self.odomX = xR
        self.odomY = yR
        self.goalX = xG
        self.goalY = yG
        self.set_entity('scout',xR, yR)
        # self.get_logger().info(f'The new Robot initial pose is xR = {self.odomX} yR = {self.odomY}')
        self.set_entity('target_cone',xG, yG)
        # self.get_logger().info(f'The new Goal pose is xG = {self.goalX} yG = {self.goalY} ')
        
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        data = scan_data
        camera_image = last_image
        # cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/metrics_data/fish_goal.png', camera_image)
        # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/odom{self.indice_position}.npz', xR=xR, yR=yR,quaterZ=quaterZ,quaterW=quaterW)
        while camera_image is None:
            self.get_logger().info('image not available, waiting again...')


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
        # camera_image = add_nose(camera_image)
        image = np.expand_dims(cv2.resize(camera_image, (160, 128)), axis=2)
        # image = np.expand_dims(camera_image, axis=2)
        # print(image)
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
        # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/goal_depth_h{self.indice_position}/goal{self.indice_position}.npz', goal=toGoal)
        # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/goal_depth_h{self.indice_position}/state{self.indice_position}.npz', state=state)
        # time.sleep(3)
        return state, xR,yR,toGoal
    
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
            # self.get_logger().error(f'the type of the image is {cv_image.dtype}!!!!!!')
            # cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            # cv_image_normalized = cv_image_normalized.astype(np.uint8)
            if cv_image.dtype == np.float32 or cv_image.dtype == np.float64:
                depth_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = depth_normalized.astype(np.uint8)
            elif cv_image.dtype == np.uint16:
                depth_normalized = (cv_image / cv_image.max() * 255).astype(np.uint8)
            else:
                depth_normalized = cv_image

            # cv_image = cv_image.astype(np.uint8)
            # cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/metrics_data/depth_image_before.png', depth_normalized)

            global last_image
            # last_image = cv_image_normalized[80:400, 140:500]  # Crop to (440, 640)
            # noise1 = np.random.normal(0.0, 0.04, depth_normalized.shape)
            # noisy1 = depth_normalized + noise1
            # noisy1 = np.clip(noisy1, 0, None)
            # blurred1 = cv2.GaussianBlur(noisy1, (5, 5), 0)
            # cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/metrics_data/depth_image_after_4.png', blurred1)
            # noise2 = np.random.normal(0.0, 0.08, depth_normalized.shape)
            # noisy2 = depth_normalized + noise2
            # noisy2 = np.clip(noisy2, 0, None)
            # blurred2 = cv2.GaussianBlur(noisy2, (5, 5), 0)
            # cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/metrics_data/depth_image_after_8.png', blurred2)
            # noise3 = np.random.normal(0.0, 0.10, depth_normalized.shape)
            # noisy3 = depth_normalized + noise3
            # noisy3 = np.clip(noisy3, 0, None)
            # blurred3 = cv2.GaussianBlur(noisy3, (5, 5), 0)
            # cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/metrics_data/depth_image_after_10.png', blurred3)
            # noise4 = np.random.normal(0.0, 0.20, depth_normalized.shape)
            # noisy4 =  + noise4
            # noisy4 = np.clip(noisy4, 0, None)
            # blurred4 = cv2.GaussianBlur(noisy4, (5, 5), 0)
            # cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/metrics_data/depth_image_after_20.png', blurred4)
            depth_normalized = add_nose(depth_normalized, noise_level=50)
            # self.get_logger().info(f'shape is = {depth_normalized.shape}')
            # depth_normalized = pixel_occlusion(depth_normalized)
            # depth_normalized = greying_out(depth_normalized)
            # depth_normalized = superpixel_occlusion(depth_normalized)
            depth_normalized = blurring(depth_normalized)
            # self.get_logger().info(f'shape is after = {depth_normalized.shape}') blurring
            last_image = depth_normalized  # Crop to (440, 640) 
            # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/last_image.npz', last_image=last_image)
            if np.all(last_image == 0):
                self.get_logger().error('Image nullll!!!!!!')
            # self.get_logger().info(f'cv_image is = {cv_image} is Zero = {np.all(cv_image == 0)}')
            # self.get_logger().info(f'cv_image_normalized is = {cv_image_normalized} is Zero = {np.all(cv_image_normalized == 0)}')
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))
            
class Image_fish_subscriber(Node):
    def __init__(self):
        super().__init__('image_fish_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera_fesh/image_raw',  # Replace with your fish image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
    def listener_callback(self, rgb_data):
        try:
            original_image = self.bridge.imgmsg_to_cv2(rgb_data, "mono8")
            last_image_fish_ori = original_image[80:400, 118:523]
            global last_image
            last_image_fish_ori = blurring(last_image_fish_ori)
            last_image =last_image_fish_ori
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
        global last_image
        original_image = self.bridge.imgmsg_to_cv2(im_data, "mono8")
        last_image =original_image