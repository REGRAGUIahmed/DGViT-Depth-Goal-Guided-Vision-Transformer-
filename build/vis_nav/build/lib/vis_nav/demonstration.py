#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:33:39 2022

@author: oscar
"""

import sys
sys.path.append('/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav')


import os
import time
import statistics
import numpy as np
from tqdm import tqdm 
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from rclpy.node import Node
from env_lab import GazeboEnv
import rclpy
import threading
from geometry_msgs.msg import Twist
from env_lab import GazeboEnv, Odom_subscriber, LaserScan_subscriber, DepthImage_subscriber, Image_fish_subscriber, Image_subscriber


key_cmd = Twist()
class Telekey_subscriber(Node):
    def __init__(self):
        super().__init__('telekey_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            '/scout/cmd_vel',
            self.key_callback,
            1)
        self.subscription

    def key_callback(self,cmd):
        global key_cmd
        key_cmd.linear.x = cmd.linear.x
        key_cmd.angular.z = cmd.angular.z
def plot_animation_figure(env_name, ep,
                          reward_list, reward_mean_list):
    ep -= 200
    fig = plt.figure()
    #plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC"))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep), reward_list)
    plt.plot(np.arange(ep), reward_mean_list)

    plt.tight_layout()

    plt.savefig('/home/regmed/dregmed/vis_to_nav/results/plot_new_dem_'+ env_name +'.png')
    plt.close(fig)
def plot_animation_figure2(ep,env_name, reward_target_list, reward_collision_list, ep_real, pedal_list, steering_list, reward_list,reward_mean_list):

    ep -= 200
    plt.figure()
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC") +' Target Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep), reward_target_list)

    # plt.subplot(2, 2, 2)
    # plt.title('Collision Reward')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.plot(np.arange(ep), reward_collision_list)


    # plt.subplot(2, 2, 3)
    # plt.title('Pedal ' + str(ep_real))
    # plt.scatter(np.arange(len(pedal_list)), pedal_list, s=6, c='coral')
    
    # plt.subplot(2, 2, 4)
    # plt.title('Steering')
    # plt.scatter(np.arange(len(steering_list)), steering_list, s=6, c='coral')
    
    plt.tight_layout()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC"))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep), reward_list)
    plt.plot(np.arange(ep), reward_mean_list)

    # plt.subplot(2, 2, 2)
    # plt.title('Heuristic Reward')
    # plt.xlabel('Episode')
    # plt.ylabel('Heuristic Reward')
    # plt.plot(np.arange(ep), reward_heuristic_list)

    # plt.subplot(2, 2, 3)
    # plt.title('Action Reward')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.plot(np.arange(ep), reward_action_list)

    # plt.subplot(2, 2, 4)
    # plt.title('Freeze Reward')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.plot(np.arange(ep), reward_freeze_list)

    plt.tight_layout()

    plt.pause(0.001)  # pause a bit so that plots are updated


def main():
    rclpy.init(args=None)
    # Set the parameters for the implementation
    device = "cuda"
    env_name = "Bachelor"
    driver = "Regragui_depth_Image"
    robot = 'bot'
    seed = 3407 # Random seed number
    max_steps = 700
    max_episodes = int(600)  # Maximum number of steps to perform
    save_models = True  # Weather to save the model or not
    batch_size = 32  # Size of the mini-batch
    frame_stack = 4
    file_name = "SAC_bachelor_bot_depth_image_transformer"  # name of the file to store the policy
    plot_interval = int(1)

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists("./Data/" + str(env_name) + '/' + str(driver)):
        os.makedirs("./Data/" + str(env_name) + '/' + str(driver))

    #master_uri = '11311'
    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    #image_subscriber = Image_subscriber() 
    image_subscriber = DepthImage_subscriber()
    ##image_subscriber = Image_fish_subscriber()
    laserScan_subscriber = LaserScan_subscriber()
    telekey_subscriber = Telekey_subscriber()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(image_subscriber)
    executor.add_node(laserScan_subscriber)
    executor.add_node(telekey_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    odom_subscriber.create_rate(2)
    time.sleep(3)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    engage = 0
    


    env.set_seed(seed)
    
    state, _,_ ,_= env.reset()
    state_dim = state.shape
    action_dim = 2
    physical_state_dim = 2 # Polar coordinate
    max_action = 1
    
    # Create evaluation data store
    evaluations = []
    
    ep_real = 200
    done = False
    reward_list = []
    reward_heuristic_list = []
    reward_action_list = []
    reward_freeze_list = []
    reward_target_list = []
    reward_collision_list = []
    reward_mean_list = []
    
    pedal_list = []
    steering_list = []

    plt.ion()

    total_timestep = 0

    # Begin the training loop
    for ep in tqdm(range(0, max_episodes), ascii=True):
        episode_reward = 0
        # episode_heu_reward = 0.0
        # episode_act_reward = 0.0
        # episode_tar_reward = 0.0
        # episode_col_reward = 0.0
        # episode_fr_reward = 0.0
        s_list = deque(maxlen=frame_stack)
        s,_,_ ,goal = env.reset()
        
        # for i in range(4):
        #     s_list.append(s)

        state = s #np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)

        ######## Demonstration List #########
        obs_list = []
        act_list = []
        goal_list = []
        r_list = []
        next_obs_list = []
        next_goal_list = []
        done_list = []
        for timestep in range(max_steps):
            # On termination of episode
            if timestep < 3:
                a_in = [0.0, 0.0]
                last_goal = goal
                s_,reward, done,goal, target = env.step(a_in, timestep)
                state = s_ #np.concatenate((s_, s_, s_, s_), axis=-1)
                
                # for i in range(4):
                #     s_list.append(s_)           

                if done:
                    env.get_logger().info("Bad Initialization, skip this episode.")
                    break

                continue
            
            if done or timestep == max_steps-1:
                ep_real += 1
    
                done = False
                np.savez('Data/{}/{}/demo_test_{}_{}.npz'.format(env_name, driver, robot, ep_real),
                          obs=np.array(obs_list, dtype=np.float32), 
                          act=np.array(act_list, dtype=np.float32),
                          goal=np.array(goal_list, dtype=np.float32),
                          reward=np.array(r_list,dtype=np.float32),
                          next_obs=np.array(next_obs_list, dtype=np.float32),
                          next_goal=np.array(next_goal_list, dtype=np.float32),
                          done = np.array(done_list,dtype=bool)
                          )

                reward_list.append(episode_reward)
                reward_mean_list.append(np.mean(reward_list[-20:]))
                # reward_heuristic_list.append(episode_heu_reward)
                # reward_action_list.append(episode_act_reward)
                # reward_target_list.append(episode_tar_reward)
                # reward_collision_list.append(episode_col_reward)
                # reward_freeze_list.append(episode_fr_reward)
                
                pedal_list.clear()
                steering_list.clear()
                total_timestep += timestep 
                env.get_logger().info(f"\n Robot: Scout Episode: {ep_real} Step:{timestep} Total Steps: {total_timestep} R: {episode_reward} seed:{seed}  Env:{env_name} Filename: {file_name} \n")
                
                if ep_real % plot_interval == 0:
                    plot_animation_figure(env_name, ep_real,
                          reward_list, reward_mean_list)
                    # plot_animation_figure(ep,env_name, reward_target_list, reward_collision_list, ep_real, pedal_list, steering_list, reward_list,reward_mean_list)
                    # plot_animation_figure(ep_real)
                    plt.ioff()
                    plt.show()

                break

            action = [key_cmd.linear.x, key_cmd.angular.z]
            # if key_cmd.linear.x == 0.0 and key_cmd.angular.z==0.0:
            #     a_in = [0.0, 0.0]
            # else:
                # a_in = [(action[0] + 1) * 0.5, action[1]*np.pi*2]
                # a_in = [action[0], action[1]]
            last_goal = goal
            s_,reward, done,goal, target = env.step(action, timestep)
            
            episode_reward += reward
            # episode_heu_reward += r_h
            # episode_act_reward += r_a
            # episode_fr_reward += r_f
            # episode_col_reward += r_c
            # episode_tar_reward += r_t
            # pedal_list.append(round((action[0] + 1)/2,2))
            # steering_list.append(round(action[1],2))
            # env.get_logger().info(f'l action a enregistrer est = {action}')
            next_state = s_  # np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)
            if action == [0.0, 0.0]:
                # env.get_logger().info(f'En attendant une action non null')
                continue
            if state.ndim == 3:
                # env.get_logger().info(f'state Avant {state.shape}')
                state = np.squeeze(state, axis=2)
                # env.get_logger().info(f'state apres {state.shape}')
            if next_state.ndim == 3:
                # env.get_logger().info(f'next_state Avant {next_state.shape}')
                next_state = np.squeeze(next_state, axis=2)
                # env.get_logger().info(f'next_state apres {next_state.shape}')
            obs_list.append(state)
            act_list.append(action)
            goal_list.append(last_goal)
            r_list.append(reward)
            if next_state.shape == state.shape:
                next_obs_list.append(next_state)
            else:
                env.get_logger().error(f'Shape mismatch: state shape {state.shape}, next_state shape {next_state.shape}')
                continue
            next_goal_list.append(goal)
            done_list.append(done)          
            # env.get_logger().info(f"********************J enregistre********************\n")
            # env.get_logger().info(f'satate  = {len(obs_list)}\n')
            # env.get_logger().info(f'action = {len(act_list)}\n')
            # env.get_logger().info(f'last_goal = {len(goal_list)}\n')
            # env.get_logger().info(f'reward = {len(r_list)}\n')
            # env.get_logger().info(f'next_state = {len(next_obs_list)}\n')
            # env.get_logger().info(f'goal = {len(next_goal_list)}\n')
            # env.get_logger().info(f'done = {len(done_list)}')

            # Update the counters
            state = next_state
            # s_list.append(s_)
    rclpy.shutdown()
    executor_thread.join()
if __name__ == '__main__':
    main() #call the main function
