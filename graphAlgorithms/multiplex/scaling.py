"""
multiplex.scaling contains possible scaling options to adjust layers & how they should be merged.
It allows users to select data driven scaling options when creating flat multilayer networks.
It can be used as input for multiplex.build_network.construct_multilayer_network_matrix.
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
    gets median of each layer, which can be used to estimate appropriate scaling values

    Input:
        edge_list needs to be either list of sublists or list of weights

        if it is list of sublists set reformat to True (default: False)

    Output:
        returns float
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
    helper function that estimates scaling values based on provided input weights
        it estimates the scaling factor to be applied to set all weights to the same value

    possible input is result of  get_median_weight for all layers, formatted as list
        scales all layers to the same value as min(weights)

    Input:
        list of float
    Output:
        list of float in same order as Input
    """

    scale_to = min(weights)

    results = []

    for med in weights:
        scale = scale_to / med
        
        results.append(round(scale, 3))

    return results