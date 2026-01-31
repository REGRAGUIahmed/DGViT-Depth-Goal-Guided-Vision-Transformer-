
from cpprb import PrioritizedReplayBuffer
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Normal
import torch.optim as optim

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys
sys.path.append('/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav')

import os
import glob
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from natsort import natsorted
import copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn[attn > 0.995] = 0.5
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class GoT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        ####### Add LayerNormalization ########
        self.layer_norm = nn.LayerNorm(dim)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, goal):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = torch.unsqueeze(goal, dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.layer_norm(x)
        # return self.mlp_head(x)
        return x
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
class GoTPolicy(nn.Module):
    def __init__(self, nb_actions, nb_pstate, block, head,l_f_size, action_space=None):
        super(GoTPolicy, self).__init__()

        self.trans = GoT(
            image_size = (128, 160),
            patch_size = (16, 20),
            num_classes = 2,
            dim = l_f_size,
            depth = block,
            heads = head,
            mlp_dim = 2048,
            channels = 4
        )
        self.fc_embed = nn.Linear(nb_pstate, l_f_size) #  32 --> 64

        self.fc1 = nn.Linear(l_f_size,128)
        self.fc2 = nn.Linear(128,128)

        self.mean_linear = nn.Linear(128, nb_actions)
        self.log_std_linear = nn.Linear(128, nb_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, inp):
        istate, pstate = inp
        x1 = istate

        x2 = pstate
        x2 = self.fc_embed(x2) #2
        # print(f'verify x2.shape ={x2.shape}')
        latent_features = self.trans.forward(x1, x2)
        
        x = F.relu(self.fc1(latent_features))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, inp):
        mean, log_std= self.forward(inp) #1
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        # print(f'action ya khoya = {y_t}')
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GoTPolicy, self).to(device)
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

class QNetwork(nn.Module):
    def __init__(self, nb_actions, nb_pstate):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4,16,5, stride=2)
        self.conv2 = nn.Conv2d(16,64,5, stride=2)
        self.conv3 = nn.Conv2d(64,256,5, stride=2)
        
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Linear(256+32+nb_actions,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,nb_actions)
        
        self.fc_embed = nn.Linear(nb_pstate, 32)
        
        self.fc11 = nn.Linear(256+32+nb_actions,128)
        self.fc21 = nn.Linear(128,32)
        self.fc31 = nn.Linear(32,nb_actions)

        self.apply(weights_init_)

    def forward(self, inp):
        istate, pstate, a = inp

        x1 = istate
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = self.avg(x1)
        x1 = x1.view(x1.size(0), -1)
        
        x2 = pstate
        x2 = F.relu(self.fc_embed(x2))
        
        x = torch.cat([x1, x2, a], dim=1)
        
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc11(x))
        q2 = F.relu(self.fc21(q2))
        q2 = self.fc31(q2)

        return q1, q2
class SAC(object):
    def __init__(self, action_dim, pstate_dim, policy_type, critic_type,
                 policy_attention_fix, critic_attention_fix, pre_buffer, seed,
                 LR_C = 1e-3, LR_A = 1e-3, LR_ALPHA=1e-4, BUFFER_SIZE=int(2e5), 
                 TAU=5e-3, POLICY_FREQ = 2, GAMMA = 0.99, ALPHA=0.05,
                 block = 2, head = 4,l_f_size=32, automatic_entropy_tuning=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = ALPHA

        self.pstate_dim = pstate_dim
        self.action_dim = action_dim
        
        self.itera = 0
        self.guidence_weight = 1.0
        self.engage_weight = 1.0
        self.buffer_size_expert = 5e3
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
                                          {"obs": {"shape": (128,160,4)},
                                           "act": {"shape":action_dim},
                                           "pobs": {"shape":pstate_dim},
                                           "next_pobs": {"shape":pstate_dim},
                                           "rew": {},
                                           "next_obs": {"shape": (128,160,4)},
                                           "engage": {},
                                           "done": {}},
                                          next_of=("obs"))

        if self.pre_buffer:
            self.replay_buffer_expert = PrioritizedReplayBuffer(self.buffer_size_expert,
                                                                {"obs": {"shape": (128,160,4)},
                                                                 "act_exp": {"shape":action_dim},
                                                                 "pobs": {"shape":pstate_dim},
                                                                 "next_pobs": {"shape":pstate_dim},
                                                                 "rew": {},
                                                                 "next_obs": {"shape": (128,160,4)},
                                                                 "done": {}},
                                                                next_of=("obs"))

        ################# Initialize Critic Network ##############

        self.critic = QNetwork(self.action_dim, self.pstate_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), LR_C)
        self.critic_target = QNetwork(self.action_dim, self.pstate_dim).to(self.device)

        hard_update(self.critic_target, self.critic)

        ############## Initialize Policy Network ################

        #
        if self.automatic_entropy_tuning is True:
            self.target_entropy = - self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=LR_ALPHA)

        ######### Initializing Transformer based Actor ##########
        self.policy = GoTPolicy(self.action_dim, self.pstate_dim,
                                                self.block, self.head, self.l_f_size).to(self.device)
        
        if policy_attention_fix:
            params = list(self.policy.fc1.parameters()) + list(self.policy.fc2.parameters()) +\
                        list(self.policy.mean_linear.parameters()) + list(self.policy.log_std_linear.parameters()) #+ 
            self.policy_optim = Adam(params, LR_A)
        else:
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)
        #

        self.target_policy = copy.deepcopy(self.policy)
    def choose_action(self, istate, pstate, evaluate=False):
        if istate.ndim < 4:
            #print(f'istate.ndim = {istate.ndim}')
            istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)
        
        if evaluate is False:
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
            
        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        engages = torch.FloatTensor(engages).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
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
            istates_expert = torch.FloatTensor(istates_expert).permute(0,3,1,2).to(self.device)
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
            alpha_tlogs = self.alpha.clone()
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1
        return qf1_loss.item(), policy_loss.item()


    def learn(self, batch_size=64):
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

    def save(self, filename, directory, reward, seed):
        torch.save(self.policy.state_dict(), '%s/%s_reward%s_seed%s_actor.pth' % (directory, filename, reward, seed))
        torch.save(self.critic.state_dict(), '%s/%s_reward%s_seed%s_critic.pth' % (directory, filename, reward, seed))

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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Use:', device)

class DatasetWrapper(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        x3 = self.data3[index]
        return x1, x2, x3

def train(epoch):

    train_loss = 0.0

    for i, (obs_data, act_data, goal_data) in enumerate(train_loader):
        observation = obs_data.float().to(device)
        action = act_data.float().to(device)
        goal = goal_data.float().to(device)

        observation = observation.float().permute(0,3,1,2).to(device)
        goal = goal[:, :2]

        # Dist  = torch.minimum(goal[:,0]/15, torch.tensor(1.0))
        # heading = goal[:, 1] / np.pi
        # goal_normalized = torch.stack((Dist, heading), dim=1)
        predict, log_prob, mean = ego.policy.sample([observation, goal])

        mean = mean.clip(-max_action, max_action)
        optimizer.zero_grad()
        loss = torch.sqrt(torch.pow(mean - action, 2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ego.policy.parameters(), 10)
        optimizer.step()
        train_loss += loss.detach().cpu().numpy().mean()

    return round(train_loss/(i+1), 4)

def val(epoch):
    val_loss = 0.0

    with torch.no_grad():
        for i, (obs_data, act_data, goal_data) in enumerate(val_loader):
            observation = obs_data.float().to(device)
            action = act_data.float().to(device)
            goal = goal_data.float().to(device)
        
            observation = observation.float().permute(0,3,1,2).to(device)
            goal = goal[:, :2]
        
            # Dist  = torch.minimum(goal[:,0]/15, torch.tensor(1.0))
            # heading = goal[:, 1] / np.pi
            # goal_normalized = torch.stack((Dist, heading), dim=1)
            predict, log_prob, mean = ego.policy.sample([observation, goal])
        
            mean = mean.clip(-max_action, max_action)
            loss = torch.sqrt(torch.pow(mean - action, 2).mean())
            val_loss += loss.detach().cpu().numpy().mean()

    return round(val_loss/(i+1), 4)

if __name__ == "__main__":

    file_path = os.getcwd()
    env = 'RRC'
    driver = 'Oscar_GoT_augmentend'
    files = natsorted(glob.glob(file_path + '/Data/' + env + '/' + driver + '/*.npz'))
    seed = 1
    iteration = 600
    batch_size = 32  # Size of the mini-batch
    lr_a = 1e-3 # Actor learning rate
    lr_c = 1e-3 # Critic learning rate
    lr_il = 1e-3
    lr_alpha = 1e-4
    gamma = 0.999  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.005  # Soft target update variable (should be close to 0)
    buffer_size = 10 #int(2e4)  # Maximum size of the buffer
    file_name = "gtrl"  # name of the file to store the policy
    frame_stack = 4 # Number of Channel
    plot_interval = int(1)
    policy_type = "GaussianTransformer"
    critic_type = "CNN"
    policy_attention_fix = False # whether fix the weights and bias of transformer
    critic_attention_fix = False # # whether fix the weights and bias of transformer
    pre_buffer = False # Human expert buffer
    alpha = 1.0
    auto_tune = False
    action_dim = 2
    max_action = 1
    policy_freq = 1
    physical_state_dim = 2 # Polar coordinate
    transformer_head = 4
    transformer_block = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    obs_list = []
    act_list = []
    goal_list = []
    
    ######## Read the Dataset ###########
    for idx, file in enumerate(files):
        
        obs = np.load(file)['obs']
        act = np.load(file)['act']
        goal = np.load(file)['goal']
        
        obs_list.append(np.array(obs))
        act_list.append(np.array(act))
        goal_list.append(np.array(goal))
    
    ######### Split the dataset #########    
    obs_dataset = np.concatenate(obs_list, axis=0)
    obs_train_size = int(0.8*len(obs_dataset))
    obs_val_size = len(obs_dataset) - obs_train_size
    obs_train_set, obs_val_set = random_split(obs_dataset, [obs_train_size, obs_val_size])
    obs_train_idx = obs_train_set.indices
    obs_val_idx = obs_val_set.indices
    
    act_dataset = np.concatenate(act_list, axis=0)
    goal_dataset = np.concatenate(goal_list, axis=0)
    
    ######### Wrap to dataloader ########
    obs_train_sample = obs_dataset[obs_train_idx]
    obs_val_sample = obs_dataset[obs_val_idx]

    act_train_sample = act_dataset[obs_train_idx]
    act_val_sample = act_dataset[obs_val_idx]

    goal_train_sample = goal_dataset[obs_train_idx]
    goal_val_sample = goal_dataset[obs_val_idx]
    
    train_ensemble = DatasetWrapper(obs_train_sample, act_train_sample, goal_train_sample)
    val_ensemble = DatasetWrapper(obs_val_sample, act_val_sample, goal_val_sample)
    train_loader = \
        DataLoader(train_ensemble, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = \
        DataLoader(val_ensemble, batch_size=batch_size, shuffle=True, num_workers=4)

    ######### Initialize DRL agent #######
    # ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
    #           critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
    #           buffer_size, tau, policy_freq, gamma, alpha, auto_tune)
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
            critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
            buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
            head=transformer_head, automatic_entropy_tuning=auto_tune)

    optimizer = optim.Adam(ego.policy.parameters(), lr=lr_il)

    ######### Trainining ########
    fig = plt.figure()
    ax = plt.subplot()
    min_val_loss = 10
    val_low_idx = 0
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(0, iteration), ascii=True):
        train_loss_epoch = train(epoch)
        val_loss_epoch = val(epoch)

        train_loss_list.append(train_loss_epoch)
        val_loss_list.append(val_loss_epoch)
        
        print('Epoch:%i, Train and Validation loss are:%f, %f' % (epoch, train_loss_epoch, val_loss_epoch))

        if val_loss_list[-1] < min_val_loss:
            val_low_idx = epoch
            torch.save(ego.policy.state_dict(), '%s/%s_actor.pth' % ("./pytorch_models", file_name))
            torch.save(ego.policy, '%s/%s_actor_model.pth' % ("./pytorch_models", file_name))
            min_val_loss = val_loss_list[-1]

        if (int(epoch) + 1 == iteration):
            ax.scatter(val_low_idx, min_val_loss, marker='*', s=128, color='cornflowerblue', label='Lowest Validation Loss Epoch')

        if (int(epoch) + 1 == iteration):
            ax.plot(np.arange(len(train_loss_list)), train_loss_list, label='Train Loss', color='lightseagreen')
            ax.plot(val_loss_list, label='Validation Loss', color='tomato')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RMSE Loss')
            ax.legend(frameon=False)
            plt.show()