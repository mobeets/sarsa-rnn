#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:51:55 2022

@author: mobeets
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import gym

from neurogym.wrappers import pass_reward
from tasks import PerceptualDecisionMaking, SingleContextDecisionMaking

#%% collect data

def run_episode(env, model, trial=None):
    model.eval()
    obs = env.reset()
    obs = Variable(torch.from_numpy(obs).float())
    h = model.initial_hidden_state()
    done = False
    episode = []
    while not done:
        probs, h_next = model(obs, h)
        m = Categorical(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)
        if not(not m.sample().shape):
            action = action[0]
        next_obs, reward, done, info = env.step(action)
        done = done or (info.get('new_trial', False) == True)
        # if done:
        #     reward = 0
        episode.append((obs, float(action), reward, h.detach()))
        obs = Variable(torch.from_numpy(next_obs).float())
        h = h_next
    return episode, h

def discount_reward(rs, gamma):
    rs = np.array(rs)
    gs = np.zeros(len(rs))
    times = np.arange(len(gs))
    for t in times:
        ix = (times <= t)
        gs[ix] += rs[t] * (gamma ** (t - times[ix]))
    return gs

def discount_rewards(batch, gamma):
    rss = []
    for episode in batch:
        rs = [r for (s,a,r,h) in episode]
        rs = discount_reward(rs, gamma)
        rss.append(rs)
    return rss

def normalize(rs, mu, sd):
    return (rs - mu)/(sd + 1e-10)

def normalize_rewards(rss):
    rs = np.hstack(rss)
    mu, sd = rs.mean(), rs.std()
    return [normalize(rs, mu, sd) for rs in rss]

def train_batch(model, optimizer, batch):
    model.train()
    
    # discount rewards; then normalize across entire batch
    rss = discount_rewards(batch, model.gamma)
    rss = normalize_rewards(rss)
    
    optimizer.zero_grad()
    for episode, rs in zip(batch, rss):
        h = episode[0][-1]
        loss = 0
        for (s,a,_,_), r in zip(episode, rs):
            probs, h = model(s, h)
            m = Categorical(probs)
            action = Variable(torch.FloatTensor([a]))
            loss += (-m.log_prob(action) * r)
        loss.backward()
    optimizer.step()

#%% model

class PolicyGradientRNN(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_size=3, gamma=0.9):
      super(PolicyGradientRNN, self).__init__()
      self.input_size = input_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.gamma = gamma
      self.rnn = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=True)
      
      # self.fc_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
      # self.fc_h = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
      
      self.output = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
      self.reset()

    def forward(self, x, h):
        """
        n.b. assumes all inputs are Tensors
        """
        # x = F.relu(self.fc_x(x))
        # hc = F.relu(self.fc_h(h))
        # prefs = self.output(x) + hc
        # outs = F.softmax(prefs, dim=-1)
        # return outs, prefs
        
        h_next = self.rnn(x[None,:], h)
        prefs = self.output(h_next)
        outs = F.softmax(prefs, dim=-1)
        return outs, h_next
    
    def initial_hidden_state(self):
        h = torch.zeros((1,self.hidden_size))
        # h[0][0] = 10
        return h

    def checkpoint_weights(self):
        self.saved_weights = deepcopy(self.state_dict())
        return self.saved_weights
    
    def save_weights_to_path(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights_from_path(self, path):
        self.load_state_dict(torch.load(path))
    
    def reset(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        self.initial_weights = self.checkpoint_weights()

#%% initialize

# env = gym.make('CartPole-v0')
env = PerceptualDecisionMaking(abort=False, rewards={'abort': -0.1}, timing={'fixation': 200})
# env = SingleContextDecisionMaking(abort=True, rewards={'abort': -1})
# env = pass_reward.PassReward(env)

hidden_size = 10 # env.action_space.n
gamma = 0.9
model = PolicyGradientRNN(input_size=env.observation_space.shape[0],
                       output_size=env.action_space.n,
                       hidden_size=hidden_size, gamma=gamma)

#%% train

n_episodes = 2000
batch_size = 5
lr = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

batch = []
# stats = []
for e in range(n_episodes):
    episode, h = run_episode(env, model)
    
    probs, _ = model(torch.Tensor(env.reset()), model.initial_hidden_state())
    
    stats.append({
        'duration': len(episode),
        'correct': np.any([r > 0 for s,a,r,h in episode]),
        'initial_probs': probs.detach().numpy(),#[0],
        'mean_reward': np.mean([r for s,a,r,h in episode]),
        })
    batch.append(episode)

    if e > 0 and (e % batch_size) == 0:
        train_batch(model, optimizer, batch)
        batch = []
        cstats = stats[-1:-(batch_size+1):-1]
        durs = np.mean([item['duration'] for item in cstats])
        pcor = 100*np.mean([item['correct'] for item in cstats])
        ps = np.mean([item['initial_probs'][0][0] for item in cstats])
        rew = np.mean([item['mean_reward'] for item in cstats])
        print('Batch {}, avg. episode length: {}, % cor: {:0.1f}, init prob: {:0.2f}, r: {:0.2f}'.format(int(e / batch_size), durs, pcor, ps, rew))

#%% plot

moving_average = lambda x,w: np.convolve(x, np.ones(w), 'valid') / w

xs = np.array([item['correct'] for item in stats])
plt.plot(moving_average(xs, 20))
# plt.ylim([0, 1])

plt.plot([x['initial_probs'][0] for x in stats])
plt.plot(moving_average([x['mean_reward'] for x in stats], 20))

#%% probe model

cohs = np.arange(5, 51, 5)
probe_stats = []
nreps = 100
env.sigma = 0.1
for i in range(nreps):
    for coh in cohs:
        trial = {'coh': coh}
        env.new_trial(**trial)
        env.t = env.t_ind = 0
        episode, _ = run_episode(env, model)
        trial.update({
            'duration': len(episode),
            'correct': np.any([r > 0 for s,a,r,h in episode]),
            'action': episode[-1][1]
            })
        probe_stats.append(trial)

pts = []
for c in cohs:
    pcor = np.mean([x['correct'] for x in probe_stats if x['coh']==c])
    pts.append((c, pcor))
pts = np.array(pts)
plt.plot(pts[:,0], pts[:,1]), plt.ylim([0, 1])

