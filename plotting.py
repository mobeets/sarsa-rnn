#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:58:26 2022

@author: mobeets
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from analysis import condition_matcher
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plot_loss(scores):
    plt.plot(scores)
    plt.xlabel('# epochs')
    plt.ylabel('loss')
