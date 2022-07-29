#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:31:52 2022

@author: mobeets
"""

import numpy as np
import matplotlib.pyplot as plt
import neurogym as ngym
import torch
import torch.nn as nn
from tasks import PerceptualDecisionMaking
from models import SarsaRNN

#%% initialize environment

env = PerceptualDecisionMaking()
_ = env.reset()

trial = env.new_trial()
ob, gt = env.ob, env.gt

print('Trial information', trial)
print('Observation shape is (N_time, N_unit) =', ob.shape)
print('Groundtruth shape is (N_time,) =', gt.shape)

#%% run using a random agent

env = PerceptualDecisionMaking(dt=20)
fig = ngym.utils.plot_env(env, num_trials=10)

#%% initialize agent

input_size = env.observation_space.shape[0] + env.action_space.n + 1
model = SarsaRNN(input_size=input_size, output_size=1,
                 hidden_size=3, nactions=env.action_space.n, gamma=0.9)

#%% train

lr = 0.003
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

model.train()

nepisodes = 100
ntrials_per_episode = 20
actions = np.arange(env.action_space.n)
# losses = []
for i in range(nepisodes):
    # new episode
    h = model.initial_hidden_state()
    train_loss = 0
    n = 0
    ob = env.reset()
    reward = 0
    cob = np.hstack([ob, [reward]])
    for j in range(ntrials_per_episode):
        # simulate trial
        continue_trial = True
        while continue_trial:
            action = model.sample_action(cob, actions, h, T=0.1)
            ob_next, reward, done, info = env.step(action)
            cob_next = np.hstack([ob_next, [reward]])
            continue_trial = info['new_trial'] == False
            
            # train using SARSA
            Q_cur, h_next = model.Q(cob, action, h)
            if not done:
                action_next = model.sample_action(cob_next, actions, h, T=0.1)
                Q_next, _ = model.Q(cob_next, action_next, h_next)
                Q_target = reward + model.gamma*Q_next.detach()
            else:
                Q_target = reward
            loss = loss_fn(Q_cur, Q_target)
            
            # log and prepare for next iteration
            train_loss += loss
            n += 1
            cob = cob_next
            h = h_next

    # gradient step
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
            
    train_loss = train_loss.item()/n
    losses.append(train_loss)
    print("Epoch {}: loss={:0.3f}".format(i, train_loss))

plt.plot(losses), plt.xlabel('# episodes'), plt.ylabel('loss')
