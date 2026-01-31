#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:23:22 2022

@author: 
"""

import math
import torch
import numpy as np 

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
def reward_function(action, last_action, distance, col,collision):
    target = False
    done = False
    r_target = 0.0
    r_collision = 0.0
    if distance < 0.8:
            target = True
            done = True
            r_target = 100
    if col:
            collision += 1
            r_collision = -100
            done = True      
    r_smooth = -abs(action[1]-last_action[1])
    r_speed = 2*action[0]  
    reward = r_target + r_collision + r_smooth + r_speed
    return reward, collision, target, done

def reward_function1(action, last_action, distance, col,collision):
    target = False
    done = False
    r_target = 0.0
    r_collision = 0.0
    if distance < 0.8:
            target = True
            done = True
            r_target = 100
    if col:
            collision += 1
            r_collision = -100
            done = True      
    r_smooth = -abs(action[1]-last_action[1])
    r_speed = 2*action[0]  
    reward = r_target + r_collision + r_smooth + r_speed
    return reward, collision, target, done

    
# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    if (3.6 < x < 5.5 and -3.5 < y < 4) or \
       (-4.5 < x < 4 and -3.5 < y < -1.8) or \
       (-3.5 < x < 3.3 and -1.6 < y < 2.5) or \
       (-5 < x < -4 and -3.5 < y < 0.3) or \
       (-5.5 < x < -4 and 2 < y < 4) or \
       (-4.5 < x < -5.5 and 0.2 < y < 2.1) or \
       (-4.1 < x < 0.1 and 3 < y < 4) or \
       (2.2 < x < 3.8 and 2.5 < y < 4) or \
       (0 < x < 2.3 and 2.5 < y < 4) or x > 5 or x < -5 or y > 3.7 or y < -3:
        return False
    else:
        return True
    
# Function to put the laser data in bins
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])
    