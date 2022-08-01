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
fig = ngym.utils.plot_env(env, num_trials=2)

#%% initialize agent

input_size = env.observation_space.shape[0] + env.action_space.n
model = SarsaRNN(input_size=input_size, output_size=1,
                 hidden_size=3, actions=np.arange(env.action_space.n),
                 gamma=0.9, T=0.1)

#%% train

lr = 0.003
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()

nepochs = 500
batch_size = 15
ntrials_per_episode = 1

losses = []
aborts = []
corrects = []
lens = []
for i in range(nepochs):
    # new epoch
    train_loss = 0
    n = 0
    ob = env.reset()
    reward = 0
    
    caborts = []
    ccorrects = []
    clens = []
    for _ in range(batch_size):
        # reset hidden state every episode
        h = model.initial_hidden_state()
        for _ in range(ntrials_per_episode):

            # simulate trial
            continue_trial = True
            rs = []
            nstart = n
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
                    assert False, "done happened?"
                    Q_target = reward
                loss = loss_fn(Q_cur, Q_target)
                
                # log and prepare for next iteration
                train_loss += loss
                n += 1
                ob = ob_next
                h = h_next
                rs.append(reward)
            rs = np.array(rs)
            caborts.append((rs < 0).any())
            ccorrects.append((rs > 0).any())
            clens.append(n-nstart)

    # gradient step
    train_loss = train_loss/n
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    losses.append(train_loss.item())
    aborts.append(np.mean(caborts))
    corrects.append(np.mean(ccorrects))
    lens.append(np.mean(clens))
    print("Epoch {}: loss={:0.3f}, aborts={:0.2f}, corrects={:0.2f}, lens={:0.2f}".format(i, train_loss, np.mean(caborts), np.mean(ccorrects), np.mean(clens)))

plot_loss(losses)

#%%

scores = np.vstack([aborts, corrects, lens]).T
plot_loss(scores)

#%% run using trained agent

# model.T = 0.1
fig = ngym.utils.plot_env(env, num_trials=100, model=model)
