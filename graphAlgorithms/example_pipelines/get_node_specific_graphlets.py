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
    generates dict of all possible graphlets of min to max size

    Input
        list of all possible nodes

    Output
        dict were key is number of nodes in graphlet as specified with min & max size and value is 
            dict were key is graphlet id and value is edge list
    """

    graphlets = {}

    for i in range(min_size, max_size+1):
        graphlets[i] = local.generate_node_specific_graphlets(nodes, graphlet_size=i)

    return graphlets


def get_graphlet_vector(networks, graphlets):
    """
    function that generates binary vectors if graphlet (node specific) exists in graph or not

    Input
        list of networkx graphs

        dict of graphlets as returned by generate_graphlets()

    Output:
        dict were key is networks id as ordered in networks
        were value is dict were key is graphlet size id
        were key is graphlet id as provided in graphlets and value is 1 if in graph and 0 if not
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


