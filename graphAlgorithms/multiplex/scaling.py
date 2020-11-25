"""
Estimates different scaling factors, that can be used when mergin multiple same node networks.
"""

import networkx as nx
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
import statistics
import dask.bag as db
import pickle
import asyncio
import sys
import multiprocessing as mp

from .build_network import *








def get_median_weight(edge_list, reformat = False):

    """
    Estimates median weight from a weighted edge list.

    Parameters:
        edge_list (list): is either weighted edge list or list of weights.
        reformat (boolean): if True assumes edge_list is edge list. If False assumes edge_list is list of weights.
        
    Returns::
        median (float):
    """
    
    if reformat:
        edges, weights = reformat_edge_list(edge_list)
    else:
        weights = edge_list
    #get median of weights
    print("median is ", statistics.median(weights))
    print("mean is ", statistics.mean(weights))
    
    return statistics.median(weights)  



def estimate_scaling_factor(weights):
    
    """
    Estimates scaling factor for networks based on some weight parameter. 
    It is possible to provide a list of median (get_median_weight()) weights and scall all networks to have the same median.
    
    Parameters:
        weights (list): list of int/ floats to be scaled to the same value.
    Returns::
        scaling factors (list):
    """

    scale_to = min(weights)

    results = []

    for med in weights:
        scale = scale_to / med
        
        results.append(round(scale, 3))

    return results