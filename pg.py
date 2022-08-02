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
from sklearn.decomposition import PCA

from neurogym.wrappers import pass_reward
from tasks import PerceptualDecisionMaking, SingleContextDecisionMaking, PerceptualDecisionMakingSingleCue

#%% collect data

def run_episode(env, model, ntrials=1, trials=None):
    model.eval()
    obs = env.reset()
    obs = Variable(torch.from_numpy(obs).float())
    h = model.initial_hidden_state()
    episode = []
    
    for trial_index in range(ntrials):
        if trials is not None and trials[trial_index] is not None:
            trial = trials[trial_index]
            env.new_trial(**trial)
            env.t = env.t_ind = 0
            obs, _, _, _ = env.step(0)
            obs = Variable(torch.from_numpy(obs).float())
        
        done = False
        while not done:
            probs, h_next = model(obs, h)
            m = Categorical(probs)
            action = m.sample()
            action = action.data.numpy().astype(int)
            if not(not m.sample().shape):
                action = action[0]
            next_obs, reward, done, info = env.step(action)
            done = done or (info.get('new_trial', False) == True)
            cdata = {'trial_index': trial_index,
                     'obs': obs,
                     'action': float(action),
                     'reward': reward,
                     'hidden': h.detach(),
                     'probs': probs.detach()
                     }
            episode.append(cdata)
            obs = Variable(torch.from_numpy(next_obs).float())
            h = h_next
    return episode

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
        rs = discount_reward([x['reward'] for x in episode], gamma)
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
        h = episode[0]['hidden']
        loss = 0
        for x, r in zip(episode, rs):
            probs, h = model(x['obs'], h)
            m = Categorical(probs)
            action = Variable(torch.FloatTensor([x['action']]))
            loss += (-m.log_prob(action) * r)
        loss.backward()
    optimizer.step()

def summarize_trial(episode, index):
    trial = [x for x in episode if x['trial_index']==index]
    rews = np.array([x['reward'] for x in trial])
    iti = max([i for i,x in enumerate(trial) if x['obs'].numpy()[0]==1])
    return { # todo: only works if there is one trial per episode
        'trial_index': index,
        'iti': iti,
        'RT': len(trial)-iti,
        'duration': len(trial),
        'correct': np.any(rews > 0),
        'initial_probs': trial[iti]['probs'].numpy()[0],
        'mean_reward': np.mean(rews),
        }

def summarize_episode(episode, ntrials_per_episode=1):
    return [summarize_trial(episode, i) for i in range(ntrials_per_episode)]

def summarize_batch(stats, batch_size):
    cstats = stats[-1:-(batch_size+1):-1]
    
    durs = np.mean([item['RT'] for items in cstats for item in items])
    pcor = 100*np.mean([item['correct'] for items in cstats for item in items])
    p0 = np.mean([item['initial_probs'][0] for items in cstats for item in items])
    rew = np.mean([item['mean_reward'] for items in cstats for item in items])
    
    m1 = 'avg duration: {}'.format(durs)
    m2 = '% cor: {:0.1f}'.format(pcor)
    m3 = 'p(t=0): {:0.2f}'.format(p0)
    m4 = 'avg rew: {:0.2f}'.format(rew)
    return (m1, m2, m3, m4)

#%% model

class PolicyGradientRNN(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_size=3, gamma=0.9):
      super(PolicyGradientRNN, self).__init__()
      self.input_size = input_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.gamma = gamma
      self.rnn = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=True)
      self.readout = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
      self.output = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
      self.reset()

    def forward(self, x, h):
        """
        n.b. assumes all inputs are Tensors
        """
        h_next = self.rnn(x[None,:], h)
        z = F.relu(self.readout(h_next))
        prefs = self.output(z)
        outs = F.softmax(prefs, dim=-1)
        return outs, h_next
    
    def initial_hidden_state(self):
        h = torch.zeros((1,self.hidden_size))
        h[0][0] = 1
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
env = PerceptualDecisionMakingSingleCue(abort=False,
                                        rewards={'abort': -0.1, 'fail': -5},
                                        early_response=True,
                                        cohs=[12.8],
                                        sigma=2.0)

# env = SingleContextDecisionMaking(abort=True, rewards={'abort': -1})
# env = pass_reward.PassReward(env)

hidden_size = 5 # env.action_space.n
gamma = 0.98
model = PolicyGradientRNN(input_size=env.observation_space.shape[0],
                       output_size=env.action_space.n,
                       hidden_size=hidden_size, gamma=gamma)

model.load_state_dict(init_w)

#%% train

n_epochs = 2500
ntrials_per_episode = 5
batch_size = 1
lr = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

stats = []
for e in range(n_epochs):
    batch = []
    for _ in range(batch_size):
        episode = run_episode(env, model, ntrials=ntrials_per_episode)
        batch.append(episode)

    train_batch(model, optimizer, batch)

    if e % 10 == 0:
        stats.append(summarize_episode(episode, ntrials_per_episode))
        msgs = summarize_batch(stats, batch_size)
        print('Epoch {}: '.format(e) + ', '.join(msgs[1:]))

#%% plot accuracy during training

moving_average = lambda x,w: np.convolve(x, np.ones(w), 'valid') / w

key = 'correct'
# key = 'mean_reward'

xs = np.array([item[key] for items in stats for item in items])
plt.plot(moving_average(xs, 20), 'k-')
if key == 'correct':
    plt.ylim([-0.05, 1.05])
    plt.ylabel('% correct')
else:
    plt.ylabel(key)
plt.xlabel('# trials (every 10)')

#%% plot accuracy and RT per condition (on first trial per episode)

cohs = np.arange(5, 51, 5)
probe_stats = []
nreps = 25
ntrials_per_episode = 5
plotPerIndex = False

for i in range(nreps):
    for coh in cohs:
        trial = {'coh': coh}
        episode = run_episode(env, model, ntrials=ntrials_per_episode, trials=[trial]*ntrials_per_episode)
        for x in summarize_episode(episode, ntrials_per_episode):
            x.update(trial)
            if not plotPerIndex:
                x['trial_index'] = 0
            probe_stats.append(x)

inds = np.unique([x['trial_index'] for x in probe_stats])
pts = []
for i in inds:
    for c in cohs:
        ctrs = [x for x in probe_stats if x['coh']==c and x['trial_index']==i]
        pcor = np.mean([x['correct'] for x in ctrs])
        pcor_se = np.std([x['correct'] for x in ctrs])/np.sqrt(len(ctrs))
        
        dur = np.mean([x['RT'] for x in ctrs])
        dur_se = np.std([x['RT'] for x in ctrs])/np.sqrt(len(ctrs))
        
        pts.append((i, c, pcor, pcor_se, dur, dur_se))
pts = np.array(pts)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
for i in inds:
    ix = pts[:,0] == i
    h = plt.plot(pts[ix,1], pts[ix,2])
    plt.plot(pts[ix,1], pts[ix,2]-pts[ix,3], color=h[0].get_color(), alpha=0.5)
    plt.plot(pts[ix,1], pts[ix,2]+pts[ix,3], color=h[0].get_color(), alpha=0.5)
plt.ylim([-0.05, 1.05])
plt.xlabel('coherence')
plt.ylabel('P(correct)')
plt.subplot(1,2,2)
for i in inds:
    ix = pts[:,0] == i
    h = plt.plot(pts[ix,1], pts[ix,4])
    plt.plot(pts[ix,1], pts[ix,4]-pts[ix,5], color=h[0].get_color(), alpha=0.5)
    plt.plot(pts[ix,1], pts[ix,4]+pts[ix,5], color=h[0].get_color(), alpha=0.5)
plt.xlabel('coherence')
plt.ylabel('RT')
plt.tight_layout()

#%% probe model on noiseless trials (first trial per episode)

sigma = env.sigma
env.sigma = 0 # no noise for probe
early_response = env.early_response
env.early_response = False # make the network wait to respond

trials = []
for trial in env.all_trials([0.0, 3.2, 6.4, 12.8, 25.6, 51.2]):
    episode = run_episode(env, model, trials=[trial])
    iti = max([i for i,x in enumerate(episode) if x['obs'].numpy()[0] == 1])
    episode = episode[iti:] # skip iti
    trial.update({'episode': episode})
    trials.append(trial)

env.early_response = early_response
env.sigma = sigma

doNormalize = False # i.e., renormalize ignoring the "no response" probability

plt.figure(figsize=(6,3))
for caction in [0,1]:
    plt.subplot(1,2,caction+1)
    for trial in sorted(trials, key=lambda x: x['coh']*(2*(x['ground_truth']==0)-1), reverse=True):
        coh = trial['coh']
        gt = trial['ground_truth']
        episode = trial['episode']
        clr = np.array([1,0,0])*round(np.log(coh) if coh > 0 else 1)/np.log(60)
        if gt != 0:
            clr = np.roll(clr, -1)
        Probs = np.vstack([x['probs'] for x in episode])
        pc = Probs[:,caction+1]
        if doNormalize:
            ps = Probs[:,1:].sum(axis=1)
            pc = pc/ps
        # pc = Probs[:,0]
        plt.plot(pc, '.-', color=clr, label=coh if gt==0 else -coh)
    plt.xlabel('time in trial')
    plt.ylabel(('P(right)' if caction == 0 else 'P(left)') + (", normalized" if doNormalize else ""))
    plt.ylim([-0.05, 1.05])
plt.legend()
plt.tight_layout()

#%% visualize hidden activity

Zs = []
for trial in sorted(trials, key=lambda x: x['coh']*(2*(x['ground_truth']==0)-1), reverse=True):
    Z = np.vstack([x['hidden'].numpy() for x in trial['episode']])
    Zs.append(Z[1:])
Zs = np.vstack(Zs)

pca = PCA(n_components=Zs.shape[-1])
pca.fit(Zs)
plt.plot(pca.explained_variance_ratio_, '.-'), plt.show()

for trial in sorted(trials, key=lambda x: x['coh']*(2*(x['ground_truth']==0)-1), reverse=True):
    coh = trial['coh']
    gt = trial['ground_truth']
    clr = np.array([1,0,0])*round(np.log(coh) if coh > 0 else 1)/np.log(60)
    if gt != 0:
        clr = np.roll(clr, -1)
    Z = np.vstack([x['hidden'].numpy() for x in trial['episode']])
    Zc = pca.transform(Z)
    # plt.plot(Zc[0,0], Zc[0,1], '+', color=clr)
    plt.plot(Zc[1:,0], Zc[1:,1], '.-', color=clr)
plt.axis('equal')

#%% plot integration of signal

sigma = 2*env.sigma
N = 20

plt.figure(figsize=(9,5))
for i,coh in enumerate(env.cohs):
    plt.subplot(2,3,i+1)
    stim = np.cos(0) * (coh/200)
    y = stim + sigma*np.random.randn(N)
    plt.plot(y, '.', markersize=1, alpha=0.5)
    ymu = [np.mean(y[:i]) for i in range(1,len(y)+1)]
    plt.plot(ymu, '-', alpha=0.5)
    plt.plot([0,N],[0,0],'k-',alpha=0.5,linewidth=1)
    plt.ylim([-1, 1])
    plt.title(coh)
plt.tight_layout()
