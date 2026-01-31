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
import numpy as np
from tqdm import tqdm
from collections import deque
import rclpy
import threading
from env_lab import GazeboEnv, Odom_subscriber, LaserScan_subscriber, DepthImage_subscriber, Image_subscriber

from got_sac_network import GoTPolicy 


import torch
from torch.optim import Adam
import torch.nn.functional as F
from DRL import SAC
import copy


class SAC_teacher(object):
    def __init__(self, block, head, l_f_size=32, action_dim=2, pstate_dim=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pstate_dim = pstate_dim
        self.action_dim = action_dim
        self.block = block
        self.head = head
        self.l_f_size = l_f_size

        ######### Initializing Transformer based Actor ##########
        self.policy = GoTPolicy(self.action_dim, self.pstate_dim,
                                                self.block, self.head, self.l_f_size).to(self.device)
    def choose_action(self, istate, pstate):
        #print(istate)
        if istate.ndim < 4:
            #print(f'istate.ndim = {istate.ndim}')
            istate = torch.FloatTensor(istate).float().permute(2,0,1).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)
        _, _, action = self.policy.sample([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()


    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
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
    valid_values = {1, 100, 200, 300, 301, 302, 303, 306}
    total_timestep_rel=max_episodes
    toGol_list = []
    # Model Config
    
    block=2 #@param
    head=1 #@param
    lfs =64
    seed = 3407
    action_dim = 2
    physical_state_dim = 2 
    policy_attention_fix = False 
    norm_type = 'RMS'
    ##### Entropy ######
    auto_tune = config['AUTO_TUNE'] #True
    alpha = config['ALPHA'] # 1.0
    lr_alpha = config['LR_ALPHA'] #0.0001
    ego =  SAC_teacher(block, head,l_f_size = lfs)

    bh = block*10+head 
    # name = f'gtrl{bh}_seed{seed}_{lfs}'
    name = 'eval_74_91_reward74.31384758332835_seed3407'
    # name = 'gtrl66_im_34_A_SCALE_1_583_reward152.6700250946039_seed2030'
    # ego.load_actor(name,directory="./final_models/data_depth_image/extanded_64")
    ego.load_actor(name,directory="/home/regmed/dregmed/vis_to_nav/final_models/data_depth_image/extanded_64")
    #Env Config
    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    image_subscriber = DepthImage_subscriber()
    laserScan_subscriber = LaserScan_subscriber()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(image_subscriber)
    executor.add_node(laserScan_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = odom_subscriber.create_rate(2)
    time.sleep(5)
    is_state_zero = True
    # data = np.load('/home/regmed/dregmed/vis_to_nav/metrics_data/data.npz')
    # state_o = data['state']
    # list_goal_for_test = []
    # list_goal_for_test = np.load('/home/regmed/dregmed/vis_to_nav/metrics_data/list_goal_for_test.npz')
    done = False
    try:
        for ep in tqdm(range(0, max_episodes), ascii=True):
            # if ep not in (list_goal_for_test['list_goal_for_test']):
            #     env.get_logger().warn("Je passe !!")
            #     break
            s_list = deque(maxlen=4)
            s, x, y, goal = env.reset()
            # for i in range(4):
            #         s_list.append(s)
            # state = np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)
            state = s
            # is_state_zero = np.all(state == 0)
            # if is_state_zero == False:
            #     np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/goal{ep}.npz', goal=goal)
            time.sleep(155)
            for timestep in range(max_steps):
                # env.get_logger().warn(f'timestep : {timestep} !!')
                if timestep == 0:
                    action = ego.choose_action(np.array(state), np.array(goal[:2]))
                    action = action.clip(-max_action, max_action)
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                    last_goal = goal
                    s_,reward, done, goal, target = env.step(a_in, timestep)
                    state  = s_
                    # state = np.concatenate((s_, s_, s_, s_), axis=-1)
                    # for i in range(4):
                    #     s_list.append(s_)
                    if done:
                        total_timestep_rel -= 1
                        env.get_logger().warn("Bad Initialization, skip this episode.")
                        break
                    continue
                if done or timestep == max_steps-1:
                    done = False
                    total_timestep += timestep 
                    break
                action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                # a_in = [action[0]*0.5, action[1]*1.0]
                # a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                last_goal = goal
                
                s_,reward, done, goal,  target = env.step(a_in, timestep)
                # next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)
                state = s_
                # s_list.append(s_)
                if(target):
                    # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/toGol_list.npz', toGol_list=toGol_list)
                    cntr2=cntr2+1
                    env.get_logger().warn(f'Goal reached successfully : {cntr2} !!')
                    # list_goal_for_test.append(ep)
                # if timestep in valid_values:
                #     toGol_list.append(last_goal)
                #     env.get_logger().info(f'La taille de toGol_list est = {len(toGol_list)}')
                #     time.sleep(10)
        s_r = cntr2/total_timestep_rel
        env.get_logger().warn(f'Number total of success {bh} {lfs} : {cntr2} with percentage : {s_r*100} % !!')
        # np.savez(f'/home/regmed/dregmed/vis_to_nav/metrics_data/list_goal_for_test.npz', list_goal_for_test=list_goal_for_test)
    finally:
        rclpy.shutdown()
        executor_thread.join()
        odom_subscriber.destroy_node()
        image_subscriber.destroy_node()
        laserScan_subscriber.destroy_node()
if __name__ == '__main__':
    main() #call the main function