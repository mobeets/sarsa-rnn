#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:43:13 2022

@author: mobeets
"""
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

def to_onehot(a, nclasses):
    b = np.zeros((nclasses,))
    b[a] = 1
    return b

class Policy:
    def __init__(self, model, actions, T):
        self.model = model
        self.actions = actions
        self.T = T
        self.h = model.initial_hidden_state()
        
    def predict(self, obs):
        action, h_next = self.model.sample_action(obs, actions, self.h, self.T)
        self.h = h_next
        return action

class SarsaRNN(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_size=3,
                 actions=None, gamma=0.9, T=0.1):
      super(SarsaRNN, self).__init__()

      self.gamma = gamma
      self.input_size = input_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.actions = actions
      self.rnn = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
      self.output = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
      self.T = T # temperature (for converting Q to policy)
      self.h = self.initial_hidden_state()
      self.reset()

    def forward(self, x, h_prev):
        """
        n.b. assumes all inputs are Tensors
        """
        h_next = self.rnn(x, h_prev)
        return self.output(h_next), h_next
    
    def Q(self, o, a, h_prev):
        """
        n.b. assumes all inputs are Tensors
        """
        return self.Q_inner(torch.Tensor(o), self.action_to_tensor(a), h_prev)
    
    def Q_inner(self, o, a, h_prev):
        """
        n.b. assumes all inputs are Tensors
        """
        return self.forward(torch.concat([o, a])[None,:], h_prev)
    
    def action_to_tensor(self, a):
        return torch.Tensor(to_onehot(a, len(self.actions)))
    
    def sample_action(self, obs, h_prev, T):
        """
        sample action using temperature T

        n.b. assumes all inputs are numpy arrays (excluding hprev)
        """
        qs = [self.Q(obs, a, h_prev) for a in self.actions]
        prefs = np.array([q[0].detach().numpy() for q,h in qs])
        pol = np.exp(prefs/T)/np.exp(prefs/T).sum()
        action = np.random.choice(self.actions, p=pol[:,0])
        h_next = qs[action][1].detach()
        return action, h_next
    
    def predict(self, obs, state=None, T=None):
        state = self.h if state is None else state
        T = self.T if T is None else T
        action, state_next = self.sample_action(obs, state, T)
        self.h = state_next
        return action, []
    
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
