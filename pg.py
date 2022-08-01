#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:51:55 2022

@author: mobeets
"""

import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import gym

class PolicyGradient(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_size=3, gamma=0.9):
      super(PolicyGradient, self).__init__()
      self.input_size = input_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.gamma = gamma
      self.rnn = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
      self.output = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

      # self.fc1 = nn.Linear(input_size, 24)
      # self.fc2 = nn.Linear(24, 36)
      # self.fc3 = nn.Linear(36, output_size)  # Prob of Left
      self.reset()

    def forward(self, x, h):
        """
        n.b. assumes all inputs are Tensors
        """
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc3(x), dim=-1)
        # return x, h
        
        h_next = self.rnn(x[None,:], h)
        prefs = self.output(h_next)
        outs = F.softmax(prefs, dim=-1)
        return outs, h_next
    
    def initial_hidden_state(self):
        return torch.zeros((1,self.hidden_size))

    def checkpoint_weights(self):
        self.saved_weights = deepcopy(self.state_dict())
        return self.saved_weights
    
    def reset(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        self.initial_weights = self.checkpoint_weights()

#%% collect data

def run_episode(env, model):
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
        action = action.data.numpy().astype(int)[0]
        next_obs, reward, done, _ = env.step(action)
        if done:
            reward = 0
        episode.append((obs, float(action), reward, h.detach()))
        obs = Variable(torch.from_numpy(next_obs).float())
        h = h_next
    return episode, h

def discount_reward(rs, gamma):
    rs = np.array(rs)
    gs = np.zeros(len(rs))
    times = np.arange(len(gs))
    rinds = np.where(rs != 0.0)[0]
    for r in rinds:
        ix = (times <= r)
        gs[ix] += (gamma ** (r - times[ix]))
    return gs

def discount_rewards(batch, gamma):
    rss = []
    for episode in batch:
        rs = [r for (s,a,r,h) in episode]
        rs = discount_reward(rs, gamma)
        rss.append(rs)
    return rss

def normalize(rs):
    return (rs - rs.mean())/rs.std()

def train_batch(model, optimizer, batch):
    model.train()
    rss = discount_rewards(batch, model.gamma)
    rss = [normalize(rs) for rs in rss]
    optimizer.zero_grad()
    for episode, rs in zip(batch, rss):
        h = episode[0][-1]
        loss = 0
        for (s,a,_,_), r in zip(episode, rs):
            probs, h = model(s, h)
            m = Categorical(probs)
            action = Variable(torch.FloatTensor([a]))
            loss += -m.log_prob(action) * r
        loss.backward()
    optimizer.step()

#%% initialize

hidden_size = 20
gamma = 0.9
env = gym.make('CartPole-v0')
model = PolicyGradient(input_size=env.observation_space.shape[0],
                       output_size=env.action_space.n,
                       hidden_size=hidden_size, gamma=gamma)

#%%

n_episodes = 500
batch_size = 5
lr = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

batch = []
episode_durations = []
h = model.initial_hidden_state()
for e in range(n_episodes):
    episode, h = run_episode(env, model)
    episode_durations.append(len(episode))
    batch.append(episode)
        
    if e > 0 and (e % batch_size) == 0:
        train_batch(model, optimizer, batch)
        batch = []
        print('Batch {}, avg. episode length: {}'.format(int(e / batch_size), np.mean(episode_durations[-1:-(batch_size+1):-1])))
    