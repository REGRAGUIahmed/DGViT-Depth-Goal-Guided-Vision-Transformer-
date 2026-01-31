#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav')

import sys
import os
import time
import glob
import yaml
import statistics
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import torch
from DRL import SAC
import rclpy
import threading
from natsort import natsorted
from env_lab import GazeboEnv, Odom_subscriber, LaserScan_subscriber, DepthImage_subscriber, Image_fish_subscriber, Image_subscriber
from got_sac_network import GoTPolicy 



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
            istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)
        _, _, action = self.policy.sample([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()


    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

def evaluate(env,frame_stack,network, max_steps,state,timestep, linear_cmd_scale,angular_cmd_scale, max_action,  eval_episodes=10, epoch=0):
    obs_list = deque(maxlen=frame_stack)
    env.collision = 0
    ep = 0
    avg_reward_list = []
    while ep < eval_episodes:
        count = 0
        obs,x,y, goal = env.reset()
        done = False
        avg_reward = 0.0

        # for i in range(4):
        #     obs_list.append(obs)

        observation =obs # np.concatenate((obs_list[-4], obs_list[-3], obs_list[-2], obs_list[-1]), axis=-1)

        while not done and count < max_steps:
            
            if count == 0:
                action = network.choose_action(np.array(state), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
                # a_in = [action[0]/4 + 0.25 , action[1]*1.0]
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                last_goal = goal
                obs_,_, done, goal, target = env.step(a_in, timestep)     
                observation =obs_# np.concatenate((obs_, obs_, obs_, obs_), axis=-1)
                
                # for i in range(4):
                #     obs_list.append(obs_)           

                if done:
                    env.get_logger().info("\n..............................................")
                    env.get_logger().info("Bad Initialization, skip this episode.")
                    env.get_logger().info("..............................................")
                    ep -= 1
                    if not target :
                        env.collision -= 1
                    break

                count += 1
                continue
            
            act = network.choose_action(np.array(observation), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
            a_in = [(act[0] + 1) * linear_cmd_scale, act[1]*angular_cmd_scale]
            obs_,reward, done, goal, target = env.step(a_in, count)       
            avg_reward += reward
            observation = obs_ #np.concatenate((obs_list[-3], obs_list[-2], obs_list[-1], obs_), axis=-1)
            # obs_list.append(obs_)
            count += 1
        
        ep += 1
        avg_reward_list.append(avg_reward)
        env.get_logger().info("\n..............................................")
        env.get_logger().info("%i Loop, Steps: %i, Avg Reward: %f, Collision No. : %i " % (ep, count, avg_reward, env.collision))
        env.get_logger().info("..............................................")
    reward = statistics.mean(avg_reward_list)
    col = env.collision
    env.get_logger().info("\n..............................................")
    env.get_logger().info("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward: %f, Collision No.: %i" % (eval_episodes, epoch, reward, col))
    env.get_logger().info("..............................................")
    return reward, col



def plot_animation_figure(ep_real, block, head,
                          reward_list, reward_mean_list,model_name,desc, sensor):
    fig = plt.figure()
    plt.title(' desc : ' +str(desc) +' block = '+str(block)+ ' head = ' + str(head))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep_real), reward_list)
    plt.plot(np.arange(ep_real), reward_mean_list)
    plt.tight_layout()
    plt.savefig('/home/regmed/dregmed/vis_to_nav/results/data_'+sensor+'/extanded_64/plot_new_ideahh_'+ model_name +str(block)+str(head)+'_'+desc+'.png')
    plt.close(fig)
    
def main():
    rclpy.init(args=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    path = os.getcwd()
    yaml_path = os.path.join(path, 'src/vis_nav/vis_nav/config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ##### Individual parameters for each model ######
    model = 'GoT-SAC' #'GoT-SAC'
    mode_param = config[model]
    desc=config['DESC'] #describe what are you dowing
    l_f_size = config['LATENT_FEATURES_SIZE']
    model_name = mode_param['name']  # gtrl
    policy_type = mode_param['actor_type'] # GaussianTransformer
    critic_type = mode_param['critic_type'] # CNN
    transformer_block = mode_param['block'] # 2
    transformer_head = mode_param['head'] # 4

    ###### Default parameters for DRL ######
    max_steps = config['MAX_STEPS'] # 300
    max_episodes = config['MAX_EPISODES'] #500
    batch_size = config['BATCH_SIZE'] #32
    lr_a = config['LR_A'] # 0.001
    lr_c = config['LR_C'] # 0.001
    gamma = config['GAMMA'] # 0.999 
    tau = config['TAU'] # 0.005
    policy_freq = config['ACTOR_FREQ'] # 1
    buffer_size = config['BUFFER_SIZE'] # 20000
    frame_stack = config['FRAME_STACK'] # 4
    plot_interval = config['PLOT_INTERVAL'] # 1

    ##### Evaluation #####
    save_threshold = config['SAVE_THRESHOLD'] # 50
    reward_threshold = config['REWARD_THRESHOLD'] # 100
    eval_threshold = config['EVAL_THRESHOLD'] # 95

    ##### Attention #####
    policy_attention_fix = config['P_ATTENTION_FIX'] #False whether fix the weights and bias of policy attention
    critic_attention_fix = config['C_ATTENTION_FIX'] #False whether fix the weights and bias of value attention

    ##### Human Intervention #####
    pre_buffer = config['PRE_BUFFER'] #False Human expert buffer
    human_guidence = config['HUMAN_INTERVENTION'] #False whether need guidance from human driver

    ##### Entropy ######
    auto_tune = config['AUTO_TUNE'] #True
    alpha = config['ALPHA'] # 1.0
    lr_alpha = config['LR_ALPHA'] #0.0001

    ##### Environment ######
    seed = config['SEED'] #525
    linear_cmd_scale = config['L_SCALE'] # 0.5
    angular_cmd_scale = config['A_SCALE'] # 2
    pre_train = config['PRE_TRAIN']
    testing = config['IF_TEST']
    # Create the network storage folders
    env = GazeboEnv()

    odom_subscriber = Odom_subscriber()
    if config['VIS_SENSOR'] == 'image':
        image_subscriber = Image_subscriber()
    if config['VIS_SENSOR'] == 'depth_image': 
        image_subscriber = DepthImage_subscriber()
    if config['VIS_SENSOR'] == 'fish_image': 
        image_subscriber = Image_fish_subscriber()
    laserScan_subscriber = LaserScan_subscriber()
    sensor = config['VIS_SENSOR']
    env.get_logger().warn(f"Device used is : {device} and visual sensor is {sensor}")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(image_subscriber)
    executor.add_node(laserScan_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = odom_subscriber.create_rate(2)
    intervention = 0
    time.sleep(5)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env.set_seed(seed)
    action_dim = 2
    physical_state_dim = 2 # Polar coordinate
    max_action = 1.0

    # Initialize the agent
    # if pre_buffer:
    files = natsorted(glob.glob('/home/regmed/dregmed/vis_to_nav/Data/Bachelor/Regragui_depth_Image'+'/*.npz'))
    obs_list = []
    act_list = []
    goal_list = []
    r_list = []
    next_obs_list = []
    next_goal_list = []
    done_list = []
    
    for idx, file in enumerate(files):
        
        obs = np.load(file)['obs']
        act = np.load(file)['act']
        goal = np.load(file)['goal']
        r = np.load(file)['reward']
        next_obs = np.load(file)['next_obs']
        next_goal = np.load(file)['next_goal']
        done = np.load(file)['done']
        
        obs_list.append(np.array(obs))
        act_list.append(np.array(act))
        goal_list.append(np.array(goal))
        r_list.append(np.array(r))
        next_obs_list.append(np.array(next_obs))
        next_goal_list.append(np.array(next_goal))
        done_list.append(np.array(done))
    
    obs_dataset = np.concatenate(obs_list, axis=0)
    act_dataset = np.concatenate(act_list, axis=0)
    goal_dataset = np.concatenate(goal_list, axis=0)
    reward_dataset = np.concatenate(r_list, axis=0)
    next_obs_dataset = np.concatenate(next_obs_list, axis=0)
    next_goal_dataset = np.concatenate(next_goal_list, axis=0)
    done_dataset = np.concatenate(done_list, axis=0)
    print(f'obs_dataset={obs_dataset.shape} \n act_dataset={act_dataset.shape}\n goal_dataset={goal_dataset.shape}\n\
            reward_dataset ={reward_dataset.shape}\n next_obs_dataset={next_obs_dataset.shape}\n next_goal_dataset={next_goal_dataset.shape}\n\
            done_dataset={done_dataset.shape}')
    buffer_size_expert = obs_dataset.shape[0]
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
                critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
                buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
                head=transformer_head,l_f_size = l_f_size,buffer_size_expert=buffer_size_expert , automatic_entropy_tuning=auto_tune)
    # if pre_buffer:
    ego.initialize_expert_buffer(obs_dataset, act_dataset, goal_dataset[:,:2], 
                                        next_goal_dataset[:,:2], reward_dataset,
                                        next_obs_dataset, done_dataset)
    # ego_teacher =  SAC_teacher(transformer_block, transformer_head,l_f_size = l_f_size)
    # ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
    #             critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
    #             buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
    #             head=transformer_head,l_f_size = l_f_size, automatic_entropy_tuning=auto_tune)
    bh = transformer_block*10+transformer_head 
    name = f'gtrl{bh}_seed{seed}_{l_f_size}'
    # ego_teacher.load_actor(name,directory="/home/regmed/dregmed/vis_to_nav/metrics_data")
    if pre_train:
        name = 'gtrl_imita_depth8'
        ego.load_actor(name, directory="/home/regmed/dregmed/vis_to_nav/metrics_data")
    if testing :
        name = 'gtrl66_im_34_A_SCALE_1_583_reward152.6700250946039_seed2030'
        env.get_logger().info(f'Let s go with {name}!!')
        ego.load (name,directory="./final_models/data_depth_image/extanded_64")

    # Create evaluation data store
    evaluations = []

    ep_real = 0
    done = False
    target = False
    reward_list = []
    reward_mean_list = []


    plt.ion()
    total_timestep = 0
    ep_real = 0
    cntr = 0
    cntr2 = 0
    nb_col = 0
    indice = 0
    max_reward = -300
    teacher_freq = 300
    total_timestep_rel=max_episodes
    reward = 0.0
    # Set the parameters for the implementation
    try:
        start_time = time.time()
        for ep in tqdm(range(0, max_episodes), ascii=True):
            episode_reward = 0
            # s_list = deque(maxlen=frame_stack)
            # env.get_logger().info(f'goal number  = {ep}')
            s, x, y, goal = env.reset()
            # time.sleep(150)
            # env.get_logger().info(f'state shape = {s.shape}')
            # for i in range(4):
            #     s_list.append(s)

            # state = np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)
            state = s
            for timestep in range(max_steps):
                # On termination of episode
                if timestep ==0:
                    if testing :
                        action = ego.choose_action(np.array(state), np.array(goal[:2]), True)
                        # gini_coef_list.append(gini_coef)
                    else :
                        
                        action = ego.choose_action(np.array(state), np.array(goal[:2]))
                        
                    # env.get_logger().info(f'vitesse = \n[{action[0]},{action[1]}]')
                    action = action.clip(-max_action, max_action)
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                    last_goal = goal
                    s_,reward, done, goal, target = env.step(a_in, timestep)
                    # state = np.concatenate((s_, s_, s_, s_), axis=-1)
                    state = s_

                    # for i in range(1):
                    #     s_list.append(s_)           

                    if done:
                        total_timestep_rel -= 1
                        env.get_logger().warn("Bad Initialization, skip this episode.")
                        break

                    continue
                
                if done or timestep == max_steps-1:
                    ep_real += 1
        
                    done = False
                    reward_list.append(episode_reward)
                    avgr = np.mean(reward_list[-20:])
                    reward_mean_list.append(avgr)
                

                    if reward_mean_list[-1] >= reward_threshold and ep_real > eval_threshold and not testing:
                        reward_threshold = reward_mean_list[-1]
                        env.get_logger().warn("Evaluating the Performance.")
                        avg_reward, nb_col = evaluate(env,frame_stack,ego, max_steps,state,timestep, linear_cmd_scale,angular_cmd_scale, max_action)
                        evaluations.append(avg_reward)
                        if avg_reward > save_threshold or nb_col<6:
                            indice +=1
                            ego.save('eval_'+desc+'_'+str(cntr2), directory='final_models/data_'+sensor+'/extanded_64', reward=int(avg_reward), seed=seed, nb_col=nb_col)
                            np.save(os.path.join('final_curves/data_'+sensor+'/extanded_64', 'eval_reward_mean_'+desc+'_'+str(cntr2)), reward_mean_list, allow_pickle=True, fix_imports=True)
                            
                            save_threshold = avg_reward
                            

                    total_timestep += timestep 
                    if reward_mean_list[-1]>max_reward :
                        max_reward = reward_mean_list[-1]
                    env.get_logger().info("Reward: %.2f, Overak R: %.2f, Max average reward: %.2f" % (episode_reward, reward_mean_list[-1], max_reward))
                    

                    if ep_real % plot_interval == 0:
                        plot_animation_figure(ep_real, transformer_block, transformer_head, reward_list, reward_mean_list,model_name,desc,sensor)

                    break
                # if timestep % teacher_freq == 0:
                #     # action = ego_teacher.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                #     a_in = [action[0]*1.0, action[1]*1.0]
                #     env.get_logger().info("Stoooop!!")
                else:
                    # env.get_logger().info(f'l etat de depart est de taille = {state.shape}')
                    action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                
                action_exp = None
                

                last_goal = goal
                last_reward = reward
                s_,reward, done, goal,  target = env.step(a_in, timestep)

                episode_reward += reward

                # next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)
                next_state = s_#np.concatenate(, axis=-1)

                # Save the tuple in replay buffer
                if not testing:
                    ego.store_transition(state, action, last_goal[:2], goal[:2], reward, next_state, intervention, action_exp, done)
                cntr += 1
                if(cntr==buffer_size):
                    env.get_logger().warn('Buffer is already full we lose Data')
                # Train the SAC model
                if human_guidence or pre_buffer:
                    ego.learn_guidence(intervention, batch_size)
                else:
                    if not testing:
                        ego.learn(batch_size)

                # Update the counters
                state = next_state
                # s_list.append(s_)
                if(target):
                    cntr2=cntr2+1
                    env.get_logger().warn(f'Goal reached successfully : {cntr2} !!')
            # if ep % 100 ==0:
            #     teacher_freq = teacher_freq*2
        if not testing:            
            env.get_logger().warn(f'Saviiiiiiiiiiiiiing !!')
            np.save(os.path.join('final_curves/data_'+sensor+'/extanded_64/etat', 'reward_mean_' + desc), reward_mean_list, allow_pickle=True, fix_imports=True)
            ego.save(desc, directory='final_models/data_'+sensor+'/extanded_64', reward=int(avgr), seed=seed)   
        s_r = cntr2/total_timestep_rel
        env.get_logger().warn(f'Number total of success {transformer_block}{transformer_head} : {cntr2} with percentage : {s_r*100} % !!')
        # rclpy.shutdown()
        # executor_thread.join()
        # sys.exit(0)
    finally:
        end_time = time.time()
        with open(f'/home/regmed/dregmed/vis_to_nav/results/training_data.txt', 'a') as f:
            f.write("\n------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            f.write(f'Id = {desc} \t Sensor = {sensor} P_ATTENTION_FIX : {policy_attention_fix} New reward *50 goal* Auto-tune : {auto_tune}')
            f.write(f'\nModel imitation = {name} Pretrain = {pre_train} Prebuffer = {pre_buffer}\t seed = {seed}\n')
            f.write(f'critic_type : {critic_type} \t actor_type : {policy_type} \t lfs = {l_f_size} \t number of blocks = {transformer_block} \t number of heads = {transformer_head}\n')
            f.write(f'Number total of success : {cntr2} with percentage : {s_r*100} % and the max_reward = {max_reward} \t Duration = {end_time-start_time} (s)\n')
        rclpy.shutdown()
        executor_thread.join()
        odom_subscriber.destroy_node()
        image_subscriber.destroy_node()
        laserScan_subscriber.destroy_node()
if __name__ == '__main__':
    main() #call the main function
    