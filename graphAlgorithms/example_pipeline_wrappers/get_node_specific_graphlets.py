"""
This is a collection of wrapper functions to simplify how to estimate the similarity between multiple networks
based on their graphlet distribution.
"""

import networkx as nx
import pandas as pd
import csv
import random
import sys
import graphAlgorithms.distances.global_distances as global_distances
import graphAlgorithms.distances.local as local
import graphAlgorithms.simplification as simplification
import graphAlgorithms.distances.trees as trees
import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
import scipy


def generate_graphlets(nodes, min_size=3, max_size=7):
    """
    Generates all possible graphlets between min and max size, taking into account node IDs.

    Parameters:
        nodes (list): list of node IDs to be considered.
       
    Returns:
        graphlets (dict): key is graphlet size value is dict where key is graphlet ID and value is list of edges.
        
    """

    graphlets = {}

    for i in range(min_size, max_size+1):
        graphlets[i] = local.generate_node_specific_graphlets(nodes, graphlet_size=i)

    return graphlets


def get_graphlet_vector(networks, graphlets):
    """
    Generates binary vectors if a node specific graphlet exists in graph or not.

    Parameters:
        networks (list): list of networkX graph objects
        graphlets (dict): containing node specific graphlets to be identified as returned by generate_graphlet().
        
    Returns:
        binary (dict): key is network ID as ordered in networks and value is dict where key is graphlet size ID 
            and value is dict were key is graphlet ID as provided in graphlets and value is 1 if graphlet exists in the network and 0 if not.
        
    """

    results = {}

    for i in range(len(networks)):
        print("estimating network", i)
        net = networks[i]
        results[i] = {}

        for x in graphlets.keys():
            print("graphlets of size", x)
            current_graphlets = graphlets[x]

            res_graphlets = local.find_graphlet(net, current_graphlets)

            results[i][x] = res_graphlets

    return results


