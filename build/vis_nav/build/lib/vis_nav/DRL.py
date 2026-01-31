#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:12:20 2023

@author: oscar
"""

# import sys
# sys.path.append('/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts')

import os
import numpy as np
from cpprb import PrioritizedReplayBuffer

import torch
from torch.optim import Adam
import torch.nn.functional as F

from utils import soft_update, hard_update
import copy

###### GoT-SAC #######
from got_sac_network import GaussianPolicy, QNetwork
from got_sac_network import DeterministicPolicy, set_seed
from got_sac_network import GoTPolicy as GaussianTransformerPolicy
from got_sac_network import GoTQNetwork as TransformerQNetwork
from got_sac_network import DeterministicGoTPolicy as DeterministicTransformerPolicy

###### ViT-SAC ######
# from SAC.vit_sac_network import GaussianTransformerPolicy, GaussianPolicy, TransformerQNetwork, QNetwork
# from SAC.vit_sac_network import  DeterministicTransformerPolicy, DeterministicPolicy, set_seed

class SAC(object):
    def __init__(self, action_dim, pstate_dim, policy_type, critic_type,
                 policy_attention_fix, critic_attention_fix, pre_buffer, seed,
                 LR_C = 1e-3, LR_A = 1e-3, LR_ALPHA=1e-4, BUFFER_SIZE=int(2e5), 
                 TAU=5e-3, POLICY_FREQ = 2, GAMMA = 0.99, ALPHA=0.05,
                 block = 2, head = 4,l_f_size=32, buffer_size_expert=10816,automatic_entropy_tuning=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = ALPHA

        self.pstate_dim = pstate_dim
        self.action_dim = action_dim
        
        self.itera = 0
        self.guidence_weight = 1.0
        self.engage_weight = 1.0
        self.buffer_size_expert = buffer_size_expert+1
        self.batch_expert = 0

        self.policy_type = policy_type
        self.critic_type = critic_type
        self.policy_freq = POLICY_FREQ
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.pre_buffer = pre_buffer # expert priors buffer
        self.seed = int(seed)

        self.block = block
        self.head = head
        self.l_f_size = l_f_size
        #TD3 parameters
        self.policy_delay = 2 
        self.target_noise_std = 0.4
        self.noise_clip = 0.5
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        set_seed(self.seed)

        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE,
                                          {"obs": {"shape": (128,160)},
                                           "act": {"shape":action_dim},
                                           "pobs": {"shape":pstate_dim},
                                           "next_pobs": {"shape":pstate_dim},
                                           "rew": {},
                                           "next_obs": {"shape": (128,160)},
                                           "engage": {},
                                           "done": {}},
                                          next_of=("obs"))

        if self.pre_buffer:
            self.replay_buffer_expert = PrioritizedReplayBuffer(self.buffer_size_expert,
                                                                {"obs": {"shape": (128,160)},
                                                                 "act_exp": {"shape":action_dim},
                                                                 "pobs": {"shape":pstate_dim},
                                                                 "next_pobs": {"shape":pstate_dim},
                                                                 "rew": {},
                                                                 "next_obs": {"shape": (128,160)},
                                                                 "done": {}},
                                                                next_of=("obs"))

        ################# Initialize Critic Network ##############
        if self.critic_type == "Transformer":
            #self.critic = TransformerQNetwork(self.action_dim, self.pstate_dim).to(device=self.device)
            self.critic = TransformerQNetwork(self.action_dim, self.pstate_dim,
                                                    self.block, self.head, self.l_f_size).to(device=self.device)
            if critic_attention_fix: #False
                params = list(self.critic.fc1.parameters()) + list(self.critic.fc2.parameters()) +\
                         list(self.critic.fc3.parameters()) + list(self.critic.fc11.parameters()) +\
                         list(self.critic.fc21.parameters()) + list(self.critic.fc31.parameters())
                self.critic_optim = Adam(params, LR_C)
            else:
                self.critic_optim = Adam(self.critic.parameters(), LR_C)

            self.critic_target = TransformerQNetwork(self.action_dim, self.pstate_dim,
                                                     self.block, self.head,self.l_f_size).to(self.device)
            # hard_update(self.critic_target, self.critic)
        else:
            self.critic = QNetwork(self.action_dim, self.pstate_dim).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), LR_C)
            self.critic_target = QNetwork(self.action_dim, self.pstate_dim).to(self.device)

        hard_update(self.critic_target, self.critic)

        ############## Initialize Policy Network ################
        if self.policy_type == "GaussianConvNet":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - self.action_dim
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=LR_ALPHA)

            self.policy = GaussianPolicy(self.action_dim, self.pstate_dim).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

        elif self.policy_type == "GaussianTransformer":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - self.action_dim
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=LR_ALPHA)

            ######### Initializing Transformer based Actor ##########
            self.policy = GaussianTransformerPolicy(self.action_dim, self.pstate_dim,
                                                    self.block, self.head, self.l_f_size).to(self.device)
            
            if policy_attention_fix:
                params = list(self.policy.fc1.parameters()) + list(self.policy.fc2.parameters()) +\
                         list(self.policy.mean_linear.parameters()) + list(self.policy.log_std_linear.parameters()) #+ 
                self.policy_optim = Adam(params, LR_A)
            else:
                self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

        elif self.policy_type == 'DeterministicTransformer':
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicTransformerPolicy(self.action_dim, self.pstate_dim,
                                                         self.block, self.head,self.l_f_size).to(self.device)
            
            if policy_attention_fix:
                params = list(self.policy.fc1.parameters()) + list(self.policy.fc2.parameters()) +\
                         list(self.policy.mean_linear.parameters()) + list(self.policy.log_std_linear.parameters())
                self.policy_optim = Adam(params, LR_A)
            else:
                self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.action_dim, self.pstate_dim).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)
        self.target_policy = copy.deepcopy(self.policy)
    def choose_action(self, istate, pstate, evaluate=False):
        if istate.ndim < 4:
            #print(f'istate.ndim = {istate.ndim}')
            istate = torch.FloatTensor(istate).float().permute(2,0,1).to(self.device)
            # istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)
        
        if evaluate is False:
            # print(f'self.policy.sample([istate, pstate]) {istate.shape}')
            action, _, _ = self.policy.sample([istate, pstate])
        else:
            _, _, action = self.policy.sample([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()

    def learn_guidence(self, engage, batch_size=64):

        agent_buffer_size = self.replay_buffer.get_stored_size()

        if self.pre_buffer:
            exp_buffer_size = self.replay_buffer_expert.get_stored_size()
            scale_factor = 1
            
            self.batch_expert = min(np.floor(exp_buffer_size/agent_buffer_size * batch_size / scale_factor), batch_size)

            batch_agent = batch_size
        
        if self.batch_expert > 0:
            expert_flag = True
            data_agent = self.replay_buffer.sample(batch_agent)
            data_expert = self.replay_buffer_expert.sample(self.batch_expert)

            istates_agent, pstates_agent, actions_agent, engages = \
                data_agent['obs'], data_agent['pobs'], data_agent['act'], data_agent['engage']
            rewards_agent, next_istates_agent, next_pstates_agent, dones_agent = \
                data_agent['rew'], data_agent['next_obs'], data_agent['next_pobs'], data_agent['done']

            istates_expert, pstates_expert, actions_expert = \
                data_expert['obs'], data_expert['pobs'], data_expert['act_exp']
            rewards_expert, next_istates_expert, next_pstates_expert, dones_expert = \
                data_expert['rew'], data_expert['next_obs'], data_expert['next_pobs'], data_expert['done']

            istates = np.concatenate((istates_agent, istates_expert), axis=0)
            pstates = np.concatenate([pstates_agent, pstates_expert], axis=0)
            actions = np.concatenate([actions_agent, actions_expert], axis=0)
            rewards = np.concatenate([rewards_agent, rewards_expert], axis=0)
            next_istates = np.concatenate([next_istates_agent, next_istates_expert], axis=0)
            next_pstates = np.concatenate([next_pstates_agent, next_pstates_expert], axis=0)
            dones = np.concatenate([dones_agent, dones_expert], axis=0)

        else:
            expert_flag = False
            data = self.replay_buffer.sample(batch_size)
            istates, pstates, actions, engages = data['obs'], data['pobs'], data['act'], data['engage']
            rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']
            
        istates = torch.FloatTensor(istates).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        engages = torch.FloatTensor(engages).to(self.device)
        next_istates = torch.FloatTensor(next_istates).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])  
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        ##### Pre buffer (expert) guidence loss, Optional #####
        if expert_flag:
            istates_expert = torch.FloatTensor(istates_expert).to(self.device)
            pstates_expert = torch.FloatTensor(pstates_expert).to(self.device)
            actions_expert = torch.FloatTensor(actions_expert).to(self.device)
            _, _, predicted_actions = self.policy.sample([istates_expert, pstates_expert]) 
            guidence_loss = self.guidence_weight * F.mse_loss(predicted_actions, actions_expert).mean()
        else:
            guidence_loss = 0.0

        ##### Real-time engage loss, Optional ######
        engage_index = (engages == 1).nonzero(as_tuple=True)[0]
        if engage_index.numel() > 0:
            istates_expert = istates[engage_index]
            pstates_expert = pstates[engage_index]
            actions_expert = actions[engage_index]
            _, _, predicted_actions = self.policy.sample([istates_expert, pstates_expert]) 
            engage_loss = self.engage_weight * F.mse_loss(predicted_actions, actions_expert).mean()
        else:
            engage_loss = 0.0

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() + guidence_loss + engage_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Automatic Entropy Adjustment #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone()
            alpha_loss = torch.tensor(0.).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)
        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        ##### update priorities #####
        # priorities = td_errors
        # priorities = priorities.cpu().numpy()
        # if expert_flag:
        #     self.replay_buffer.update_priorities(indexes_agent, priorities[0:batch_size])
        #     self.replay_buffer_expert.update_priorities(indexes_expert, priorities[-int(self.batch_expert):])
        # else:
        #     self.replay_buffer.update_priorities(indexes, priorities)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        return qf1_loss.item(), policy_loss.item()

    def learn_sac(self, batch_size=64):
        #SAC
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            print(f'qf1_next_target = {qf1_next_target}, qf2_next_target = {qf2_next_target} ')
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
                
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])
        
        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Automatic Entropy Adjustment #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        ##### update priorities #####
        # priorities = td_errors
        # priorities = priorities.cpu().numpy()
        # self.replay_buffer.update_priorities(indexes, priorities)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        return qf1_loss.item(), policy_loss.item() 
    def learn_td3(self, batch_size=64):
        #TD3    
        policy_loss = None
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Add noise to target actions for TD3
        with torch.no_grad():
            target_actions, b, _ = self.target_policy.sample([next_istates, next_pstates])
            # target_actions = self.target_policy(next_istates, next_pstates)
            noise = torch.clamp(
                torch.normal(0, self.target_noise_std, size=target_actions.shape).to(self.device), 
                -self.noise_clip, self.noise_clip
            )
            target_actions = torch.clamp(b + noise, -1.0, 1.0)
            # target_actions = torch.clamp((target_actions + noise)/100.0, -1.0, 1.0)
            # target_actions = (target_actions + noise)/100.0
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, target_actions])
            # print(f'qf1_next_target = {qf1_next_target}, qf2_next_target = {qf2_next_target} ')
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = rewards + self.gamma * (1 - dones) * min_qf_next_target

        # Update critics
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Delay policy updates
        if self.itera % self.policy_delay == 0:
            pi, _, _ = self.policy.sample([istates, pstates])
            qf1_pi, _ = self.critic([istates, pstates, pi])
            
            
            # pi = self.policy(istates, pstates)
            # qf1_pi = self.critic.q1(istates, pstates, pi)  # Use only q1 for the policy update
            policy_loss = -qf1_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Soft update target networks
            soft_update(self.critic_target, self.critic, self.tau)
            # soft_update(self.target_policy, self.policy, self.tau)

        self.itera += 1
        return qf1_loss.item(), policy_loss.item() if policy_loss is not None else None
    
  
    def learn_6(self, batch_size=64):
        # DDPG  
        policy_loss = None
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Pas de bruit pour les actions cibles en DDPG
        with torch.no_grad():
            target_actions, _, _ = self.target_policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, target_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = rewards + self.gamma * (1 - dones) * min_qf_next_target

        # Mise à jour des critiques
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Mise à jour de la politique sans délai (DDPG met à jour la politique à chaque itération)
        pi, _, _ = self.policy.sample([istates, pstates])
        qf1_pi, _ = self.critic([istates, pstates, pi])  # Utilisation uniquement de q1 pour la mise à jour de la politique

        policy_loss = -qf1_pi.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Mise à jour douce des réseaux cibles
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.target_policy, self.policy, self.tau)

        self.itera += 1
        return qf1_loss.item(), policy_loss.item() if policy_loss is not None else None

    
    
    def learn_4(self, batch_size=64):
        policy_loss = None
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0, 3, 1, 2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0, 3, 1, 2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Add noise to target actions for TD3
        with torch.no_grad():
            target_actions = self.target_policy([next_istates, next_pstates])
            noise = torch.clamp(
                torch.normal(0, self.target_noise_std, size=target_actions.shape).to(self.device),
                -self.noise_clip, self.noise_clip
            )
            target_actions = torch.clamp(target_actions + noise, -1.0, 1.0)
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, target_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = rewards + self.gamma * (1 - dones) * min_qf_next_target

        # Update critics
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Delay policy updates
        if self.itera % self.policy_delay == 0:
            pi = self.policy([istates, pstates])
            qf1_pi, _ = self.critic([istates, pstates, pi])
            policy_loss = -qf1_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Soft update target networks
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.target_policy, self.policy, self.tau)

        self.itera += 1
        policy_loss_value = policy_loss.item() if policy_loss is not None else 0.0
        return qf1_loss.item(), policy_loss_value
    def learn_5(self, batch_size=64):
        policy_loss = None
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0, 3, 1, 2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0, 3, 1, 2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Add noise to target actions for TD3
        with torch.no_grad():
            target_actions = self.target_policy.sample([next_istates, next_pstates])
            
            # Handle tuple output from the policy
            if isinstance(target_actions, tuple):
                target_actions = target_actions[0]  # Extract only the action tensor
            
            # Add noise and clip the target actions
            noise = torch.normal(0, self.target_noise_std, size=target_actions.size()).to(self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            target_actions = torch.clamp(target_actions + noise, -1.0, 1.0)

            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, target_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = rewards + self.gamma * (1 - dones) * min_qf_next_target

        # Update critics
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Delay policy updates
        if self.itera % self.policy_delay == 0:
            pi, _, _ = self.policy.sample([istates, pstates])  # Use the sampled action from the policy
            qf1_pi, _ = self.critic([istates, pstates, pi])

            # Policy loss
            policy_loss = -qf1_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Soft update target networks
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.target_policy, self.policy, self.tau)

        self.itera += 1
        return qf1_loss.item(), policy_loss.item() if policy_loss is not None else None

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']
        # print(f'istates.shape Av = {istates.shape}')
        istates = torch.FloatTensor(istates).to(self.device)
        # print(f'istates.shape Ap = {istates.shape}')
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        # print(f'pstates.shape Av = {pstates.shape}')
        # print(f'actions.shape Av = {actions.shape}')
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
                
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Automatic Entropy Adjustment #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        ##### update priorities #####
        # priorities = td_errors
        # priorities = priorities.cpu().numpy()
        # self.replay_buffer.update_priorities(indexes, priorities)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        return qf1_loss.item(), policy_loss.item()
    def store_transition(self,  s, ps, a, ae, i, r, s_, ps_, d=0):
        self.replay_buffer.add(obs=s,
                               pobs=ps,
                               act=a,
                               acte=ae,
                               intervene=i,
                               rew=r,
                               next_obs=s_,
                               next_pobs=ps_,
                               done=d)

    def store_transition(self, s, a, ps, ps_, r, s_, engage, a_exp, d=0):
        if a is not None:
            self.replay_buffer.add(obs=s,
                    act=a,
                    pobs=ps,
                    next_pobs=ps_,
                    rew=r,
                    next_obs=s_,
                    engage = engage,
                    done=d)
        else:
            self.replay_buffer.add(obs=s,
                    act=a_exp,
                    pobs=ps,
                    next_pobs=ps_,
                    rew=r,
                    next_obs=s_,
                    engage = engage,
                    done=d)

    def initialize_expert_buffer(self, s, a_exp, ps, ps_, r, s_, d=0):

        self.replay_buffer_expert.add(obs=s,
                act_exp=a_exp,
                pobs=ps,
                next_pobs=ps_,
                rew=r,
                next_obs=s_,
                done=d)

    # Save and load model parameters
    def load_model(self, output):
        if output is None: return
        self.policy.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.policy.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save(self, filename, directory, reward,seed, nb_col=100):
        torch.save(self.policy.state_dict(), '%s/%s_reward_%s_nbCol_%s_seed_%s_actor.pth' % (directory, filename, reward,nb_col, seed))
        torch.save(self.critic.state_dict(), '%s/%s_reward_%s_nbCol_%s_seed_%s_critic.pth' % (directory, filename, reward,nb_col, seed))

    def load(self, filename, directory):
        # gtrl_reward148_seed525
        # gtrl_reward148_seed525_critic
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))    

    def load_target(self):
        hard_update(self.critic_target, self.critic)

    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
