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
from env_lab import GazeboEnv, Odom_subscriber, LaserScan_subscriber, DepthImage_subscriber, Image_subscriber
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

# Constants and helpers
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
# Weight initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# PreNorm for Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
# Parallel for Transformer
class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])
# RMSNorm for Transformer
class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if (heads != 1 or dim_head != dim) else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn[attn > 0.995] = 0.5
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# FeedForward network
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
# Transformer encoder
class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        parallel: bool = True,
        dropout: float = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if parallel:
                attn_layer = Parallel(
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                )
                ff_layer = Parallel(
                    FeedForward(dim, mlp_dim, dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                )
            else:
                attn_layer = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
                ff_layer = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                
            self.layers.append(nn.ModuleList([attn_layer, ff_layer]))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x

# Goal-conditioned Transformer (GoT)
class GoT(nn.Module):
    """
    Goal-conditioned Transformer (GoT): Vision Transformer adapted to goal-conditional inputs.
    """
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        norm_type='RMS',
        pool='cls',
        channels=1,
        parallel = True,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.1
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either "cls" or "mean"'

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange ('b (h ph) (w pw) -> b (h w) (ph pw)', ph = patch_height, pw = patch_width), 
            nn.Linear(patch_dim, dim),)

        # Positional embedding + goal token (cls token replaced by goal)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, parallel, dropout=dropout)

        # Normalization
        self.layer_norm = RMSNorm(dim) if norm_type == 'RMS' else nn.LayerNorm(dim)

        # Classifier head
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        

    def forward(self, img: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            """
            img: Tensor of shape (B, H, W, C)
            goal: Tensor of shape (B, dim)
            """
            x = self.to_patch_embedding(img)  # (B, N, dim)
            b, n, _ = x.shape

            # Replace cls token with goal
            goal_token = goal.unsqueeze(1)  # (B, 1, dim)
            x = torch.cat((goal_token, x), dim=1)
            x = x + self.pos_embedding[:, :x.size(1)]
            x = self.dropout(x)

            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            x = self.layer_norm(x)
            return x  # Can use `self.mlp_head(x)` if classification is required

class GoTPolicy(nn.Module):
    def __init__(self, nb_actions, nb_pstate, block, head, l_f_size, parallel=True, norm_type='RMS', action_space=None):
            
        super().__init__()
        # Visual transformer encoder
        self.trans = GoT(
                    image_size=(128, 160),
                    patch_size=(16, 20),
                    num_classes=2,  # not used in final output, but can be handy for probing
                    dim=l_f_size,
                    depth=block,
                    heads=head,
                    mlp_dim=2048,
                    norm_type='RMS',
                    channels=1,
                    parallel=parallel
                )
        
        # Polar state encoder
        self.fc_embed = nn.Linear(nb_pstate, l_f_size)
        # Actor MLP
        self.fc1 = nn.Linear(l_f_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Output layers
        self.mean_linear = nn.Linear(128, nb_actions)
        self.log_std_linear = nn.Linear(128, nb_actions)

        # Action scaling
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1.0))
            self.register_buffer("action_bias", torch.tensor(0.0))
        else:
            action_scale = (action_space.high - action_space.low) / 2.0
            action_bias = (action_space.high + action_space.low) / 2.0
            self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

        self.apply(weights_init_)

    def forward(self, inputs):
        """
        Forward pass through the policy network.
        Returns the action mean and log_std (for sampling).
        """
        istate, pstate = inputs
        pstate_emb = self.fc_embed(pstate)                      # (B, nb_pstate) → (B, l_f_size)
        latent_features = self.trans(istate, pstate_emb)        # (B, C, H, W), (B, l_f_size)

        x = F.relu(self.fc1(latent_features))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, inputs):
        """
        Samples actions using the reparameterization trick.
        Returns:
            action: sampled action scaled to environment bounds
            log_prob: log probability of the sampled action
            mean: deterministic action (tanh-scaled)
        """
        mean, log_std = self.forward(inputs)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()                  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Compute log_prob with tanh adjustment
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

# Actor Agent
class Actor(object):
    def __init__(self, action_dim, pstate_dim, policy_attention_fix, seed,
                 LR_A=1e-3, block=2, head=4, l_f_size=64, parallel =False, norm_type='RMS'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.pstate_dim = pstate_dim
        self.seed = int(seed)
        self.block = block
        self.head = head
        self.l_f_size = l_f_size
        self.parallel = parallel
        self.norm_type= norm_type
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Policy network
        self.policy = GoTPolicy(self.action_dim, self.pstate_dim,
                                self.block, self.head, self.l_f_size, self.parallel,self.norm_type).to(self.device)
        
        # Optimizer for policy (with or without attention fix)
        if policy_attention_fix:
            params = list(self.policy.fc1.parameters()) + list(self.policy.fc2.parameters()) + \
                     list(self.policy.mean_linear.parameters()) + list(self.policy.log_std_linear.parameters())
            self.policy_optim = optim.Adam(params, LR_A)
        else:
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=LR_A)

    def choose_action(self, istate, pstate, evaluate=False):
        istate = torch.FloatTensor(istate).float().unsqueeze(0).to(self.device)
        pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        
        if evaluate:
            _, _, action = self.policy.sample([istate, pstate])
        else:
            action, _, _ = self.policy.sample([istate, pstate])
            
        return action.detach().squeeze(0).cpu().numpy()
    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

##################################################################################################

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
    ##### Entropy ######
    ego =  Actor(action_dim, physical_state_dim, policy_attention_fix, seed, 
              l_f_size=lfs, block=block, head=head, parallel=False, norm_type=norm_type)

    bh = block*10+head 
    name = 'eval_75_149_reward205__nbCol1_seed3407'
    ego.load_actor(name,directory="/home/regmed/Downloads")
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
    done = False
    try:
        for ep in tqdm(range(0, max_episodes), ascii=True):
            s_list = deque(maxlen=4)
            s, x, y, goal = env.reset()
            state = np.squeeze(s, axis=2)
            for timestep in range(max_steps):
                if timestep == 0:
                    action = ego.choose_action(np.array(state), np.array(goal[:2]))
                    action = action.clip(-max_action, max_action)
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                    last_goal = goal
                    s_,reward, done, goal, target = env.step(a_in, timestep)
                    state  = np.squeeze(s_, axis=2)
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
                
                s_,reward, done, goal,  target = env.step(a_in, timestep)
                s_ = np.squeeze(s_, axis=2)
                state = s_
                if(target):
                    cntr2=cntr2+1
                    env.get_logger().warn(f'Goal reached successfully : {cntr2} !!')
        s_r = cntr2/total_timestep_rel
        env.get_logger().warn(f'Number total of success {bh} {lfs} : {cntr2} with percentage : {s_r*100} % !!')
    finally:
        rclpy.shutdown()
        executor_thread.join()
        odom_subscriber.destroy_node()
        image_subscriber.destroy_node()
        laserScan_subscriber.destroy_node()
if __name__ == '__main__':
    main() #call the main function