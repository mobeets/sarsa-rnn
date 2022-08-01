#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:33:40 2022

@author: mobeets
"""
import numpy as np
import neurogym as ngym
from neurogym import spaces

class PerceptualDecisionMaking(ngym.TrialEnv):
    """Two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    A noisy stimulus is shown during the stimulus period. The strength (
    coherence) of the stimulus is randomly sampled every trial. Because the
    stimulus is noisy, the agent is encouraged to integrate the stimulus
    over time.

    Args:
        cohs: list of float, coherence levels controlling the difficulty of
            the task
        sigma: float, input noise level
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, cohs=None,
                 sigma=1.0, dim_ring=2):
        super().__init__(dt=dt)
        if cohs is None:
            self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])
        else:
            self.cohs = cohs
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and
            decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        coh = trial['coh']
        ground_truth = trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        # if self.in_period('fixation'):
        if not self.in_period('decision'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
                # new_trial = True
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

# class PerceptualDecisionMaking(ngym.TrialEnv):
#     """Two-alternative forced choice task in which the subject has to
#     integrate two stimuli to decide which one is higher on average.
    
#     Modified by JAH so the agent can respond at any time during the stimulus epoch

#     Args:
#         stim_scale: Controls the difficulty of the experiment. (def: 1., float)
#         sigma: float, input noise level
#         dim_ring: int, dimension of ring input and output
#     """
#     metadata = {
#         'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
#         'paper_name': '''The analysis of visual motion: a comparison of
#         neuronal and psychophysical performance''',
#         'tags': ['perceptual', 'two-alternative', 'supervised']
#     }

#     def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
#                  sigma=1.0, dim_ring=2):
#         super().__init__(dt=dt)
#         # The strength of evidence, modulated by stim_scale
#         self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale
#         self.sigma = sigma / np.sqrt(self.dt)  # Input noise

#         # Rewards
#         self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
#         if rewards:
#             self.rewards.update(rewards)

#         self.timing = {
#             'fixation': 100,
#             'stimulus': 2000}
#         if timing:
#             self.timing.update(timing)

#         self.abort = False

#         self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
#         self.choices = np.arange(dim_ring)

#         name = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
#         self.observation_space = ngym.spaces.Box(
#             -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32, name=name)
#         name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
#         self.action_space = ngym.spaces.Discrete(1+dim_ring, name=name)

#     def _new_trial(self, **kwargs):
#         # Trial info
#         trial = {
#             'ground_truth': self.rng.choice(self.choices),
#             'coh': self.rng.choice(self.cohs),
#         }
#         trial.update(kwargs)

#         coh = trial['coh']
#         ground_truth = trial['ground_truth']
#         stim_theta = self.theta[ground_truth]

#         # Periods
#         self.add_period(['fixation', 'stimulus'])

#         # Observations
#         self.add_ob(1, period=['fixation', 'stimulus'], where='fixation')
#         stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
#         self.add_ob(stim, 'stimulus', where='stimulus')
#         self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

#         # Ground truth
#         self.set_groundtruth(ground_truth, period='stimulus', where='choice')

#         return trial

#     def _step(self, action):
#         new_trial = False
#         # rewards
#         reward = 0
#         gt = self.gt_now
#         # observations
#         if self.in_period('fixation'):
#             if action != 0:  # action = 0 means fixating
#                 new_trial = self.abort
#                 reward += self.rewards['abort']
#         elif self.in_period('stimulus'):
#             if action != 0:
#                 new_trial = True
#                 if action == gt:
#                     reward += self.rewards['correct']
#                     self.performance = 1
#                 else:
#                     reward += self.rewards['fail']

#         return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
