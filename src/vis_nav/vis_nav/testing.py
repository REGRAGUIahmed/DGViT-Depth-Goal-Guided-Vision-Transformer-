#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jan 7 12:46:17 2025

@author: ahmed
"""

#!/usr/bin/env python

import sys
sys.path.append('/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav')




import os
import time
import yaml
from collections import deque
import rclpy
import threading
from env_lab import GazeboEnv, Odom_subscriber, LaserScan_subscriber, DepthImage_subscriber, Image_subscriber, Image_fish_subscriber
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from natsort import natsorted
import copy
from got_sac_network import GoTPolicy as Actor

def main():
    rclpy.init(args=None)
    path = os.getcwd()
    yaml_path = os.path.join(path, 'src/vis_nav/vis_nav/config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    max_episodes = 100 #config['MAX_EPISODES']
    max_steps = config['MAX_STEPS']
    linear_cmd_scale = config['L_SCALE']
    angular_cmd_scale = config['A_SCALE']
    
    
    max_action = 1.0
    cntr2 = 0
    total_timestep = 0
    total_timestep_rel=max_episodes
    block=4 #@param
    head=4 #@param
    lfs =64
    seed = 3407
    action_dim = 2
    physical_state_dim = 2 
    policy_attention_fix = False 
    norm_type = 'RMS'
    ##### Entropy ###### self, nb_actions, nb_pstate, block, head,l_f_size, action_space=None):
    # ego =  Actor(action_dim, physical_state_dim, policy_attention_fix, seed, 
    #           l_f_size=lfs, block=block, head=head, parallel=False, norm_type=norm_type)
    ego =  Actor(action_dim, physical_state_dim, block, head, l_f_size=lfs)
    bh = block*10+head 
    # name = 'eval_96_183_reward_125_nbCol_2_seed_3407'
    # name = 'eval_96_84_reward_138_nbCol_0_seed_3407' #model depth image1
    # name = 'eval_95_205_reward_268_nbCol_0_seed_3407' #model depth image2
    # name = 'eval_95_202_reward_260_nbCol_0_seed_3407' #model depth image3 
    # name = 'eval_95_199_reward_237_nbCol_0_seed_3407' #model depth image4
    name = 'eval_90_91_reward_136_nbCol_3_seed_3407' #model fisheye image
    directory=f"/home/regmed/dregmed/vis_to_nav/final_models/data_{config['VIS_SENSOR']}/extanded_64"
    # directory=f"/home/regmed/dregmed/vis_to_nav/final_models/data_depth_image/extanded_64"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ego.load_actor(name,directory="/home/regmed/Downloads")
    goal_number = 92
    ego.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name)))
    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    if config['VIS_SENSOR'] == 'image':
        image_subscriber = Image_subscriber()
    if config['VIS_SENSOR'] == 'depth_image': 
        image_subscriber = DepthImage_subscriber()
    if config['VIS_SENSOR'] == 'fish_image': 
        image_subscriber = Image_fish_subscriber()
    laserScan_subscriber = LaserScan_subscriber()
    env.get_logger().warn(f"Testing with model {name} and sensor {config['VIS_SENSOR']}")   
    env.get_logger().warn(f"Testing with block {bh} and lfs {lfs}")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(image_subscriber)
    executor.add_node(laserScan_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = odom_subscriber.create_rate(2)
    time.sleep(5)
    done = False
    durations = []
    try:
        for ep in tqdm(range(0, max_episodes), ascii=True):
            s, x, y, goal = env.reset()
            state = s #np.squeeze(s, axis=2)
            # env.get_logger().warn("The episode number is : %d" % ep)
            start_time = time.time()
            for timestep in range(max_steps):
                # time.sleep(300)
                if timestep == 0:
                    action = ego.choose_action(np.array(state), np.array(goal[:2]), True)
                    action = action.clip(-max_action, max_action)
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                    last_goal = goal
                    s_,reward, done, goal, target = env.step(a_in, timestep)
                    state  = s_ #np.squeeze(s_, axis=2)
                    if done:
                        total_timestep_rel -= 1
                        env.get_logger().warn("Bad Initialization, skip this episode.")
                        start_time = time.time()
                        break
                    continue
                if done or timestep == max_steps-1:
                    done = False
                    total_timestep += timestep 
                    start_time = time.time()
                    break
                action = ego.choose_action(np.array(state), np.array(goal[:2]), True).clip(-max_action, max_action)
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                
                s_,reward, done, goal,  target = env.step(a_in, timestep)
                s_ = s_ #np.squeeze(s_, axis=2)
                state = s_
                if(target):
                    end_time = time.time()
                    cntr2=cntr2+1
                    durations.append((end_time-start_time))
                    env.get_logger().warn(f'Goal reached successfully : {cntr2} !!')
        s_r = cntr2/total_timestep_rel
        env.get_logger().warn(f'Number total of success {bh} {lfs} : {cntr2} with percentage : {s_r*100} % !!')
        '''mean_duration = np.mean(durations)
        standard_deviation_duration = np.std(durations)
        aver_duration = np.var(durations)
        np.save(os.path.join('trajectories', 'duration_'+str(goal_number)+'_'+str(name)), durations, allow_pickle=True, fix_imports=True)'''
    finally:
        with open(f'/home/regmed/dregmed/vis_to_nav/results/testing_data.txt', 'a') as f:
            f.write("\n----------------------------------------/*/*/*/*/*/*/----------------------------------------------\n")
            f.write(f"Model = {name} Sensor = {config['VIS_SENSOR']} \n")
            # f.write(f'Mean duration for goal {goal_number} = {mean_duration} with average = {aver_duration} and standard deviation = {standard_deviation_duration}\n')
            f.write(f'Number total of success : {cntr2} with percentage : {s_r*100} % for blurring \n')

        rclpy.shutdown()
        executor_thread.join()
        odom_subscriber.destroy_node()
        image_subscriber.destroy_node()
        laserScan_subscriber.destroy_node()
if __name__ == '__main__':
    main() #call the main function
