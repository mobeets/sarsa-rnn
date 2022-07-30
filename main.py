#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:31:52 2022

@author: mobeets
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import neurogym as ngym
from neurogym.wrappers import pass_reward

from tasks import PerceptualDecisionMaking
from models import SarsaRNN
from plotting import plot_loss

#%% initialize environment

env = PerceptualDecisionMaking()
env = pass_reward.PassReward(env)
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

input_size = env.observation_space.shape[0] + env.action_space.n
model = SarsaRNN(input_size=input_size, output_size=1,
                 hidden_size=3, actions=np.arange(env.action_space.n),
                 gamma=0.9, T=0.1)

#%% train

"""
to dos:
    - consider training with the normal task, just to see if I'm doing things right
        since currently, model might suck for reasons to do with the custom task
        and not my training methods/code
    - what's the right way to do a gradient step with the RNN? (see: DQRNN)
    - save best model in terms of loss, and reset to that one at the end
    - am I handling the ends of trials right? (is done ever True?)
""" 

lr = 0.003
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

model.train()

nepochs = 100
ntrials_per_episode = 20

losses = []
for i in range(nepochs):
    # new episode
    h = model.initial_hidden_state()
    train_loss = 0
    n = 0
    ob = env.reset()
    reward = 0
    for j in range(ntrials_per_episode):
        # simulate trial
        continue_trial = True
        while continue_trial:
            action, _ = model.predict(ob, h)
            ob_next, reward, done, info = env.step(action)
            continue_trial = info['new_trial'] == False
            
            # train using SARSA
            Q_cur, h_next = model.Q(ob, action, h)
            if not done:
                action_next, _ = model.predict(ob_next, h)
                Q_next, _ = model.Q(ob_next, action_next, h_next)
                Q_target = reward + model.gamma*Q_next.detach()
            else:
                Q_target = reward
            loss = loss_fn(Q_cur, Q_target)
            
            # log and prepare for next iteration
            train_loss += loss
            n += 1
            ob = ob_next
            h = h_next

    # gradient step
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
            
    train_loss = train_loss.item()/n
    losses.append(train_loss)
    print("Epoch {}: loss={:0.3f}".format(i, train_loss))

plot_loss(losses)

#%% run using trained agent

# model.T = 0.1
fig = ngym.utils.plot_env(env, num_trials=200, model=model)
