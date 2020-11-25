"""
This is a collection of wrapper functions to simplify how to estimate the similarity between multiple networks
based on their (structural) similarities when converted into a binary tree.
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


def helper_tree_vector(networks, nodes, tree_type="level", edge_attribute="weight", cycle_weight = "max", initial_cycle_weight=True):
    """
    Estimate for each network a vector based on its tree properties. Computes a binary tree representation, where each specified node is selected as root.
    Compares same root trees between each other. 

    Parameters:
        networks (list): of networkX graph objects
        nodes (list): list of nodes that should be compared between the networks. Nodes need to be present in all networks.
        tree_type (str): if is "level" then a hierarchical tree is created. Paths from the root indicate how far each nodes are from the root node. Edge weights are not considered.
					if is "cycle" then a hierarchical tree is created where each node represents a cycle in the graph. 
						Tree leaves are the original cycles in the graph and are merged into larger cycles through edge removal until all have been merged into a single cycle.
						This method can be helpful to categorize cyclic graphs. The root parameter is not considered when this option is selected and only cyclic structures in the graph are considered.
		edge_attribute (str): name of the edge weights to be considered if type = "cycle".
		cycle_weight (str): sets how cycles are merged i.e. which edges are removed to merge cycles into larger ones.
							if is "max" then the edge with the highest weight is removed first. if is "min" then the edge with the smalles weight is removed first.
							if is "betweenness_max" the the edge with the highest betweenness value is removed first.
							if is "betweenness_min" the edge with the lowest betweenness value is removed first.
		initial_cycle_weight (boolean): if True the initial cycle basis is estimated based on edge weights with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.minimum_cycle_basis.html#networkx.algorithms.cycles.minimum_cycle_basis
										if False the initial cycles are estimated based on steps only with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis

    Returns:
        tree properties vector (list): each sublist contains a network specific vector, ordered as provided in networks.
        trees (dict): compared tree objects. Key is network ID and value is dict where key is node ID and value is tree object.
        
    """
    tree_vector = []
    trees_save = {}
    for i in range(len(networks)):
        trees_save[i] = {}




    for i in range(len(networks)):
        network = networks[i]
        temp_vector = []
        for node in nodes:
            #get tree
            tree = trees.construct_tree(network,  root=node, nr_trees=1, type=tree_type, edge_attribute=edge_attribute, cycle_weight=cycle_weight, initial_cycle_weight=initial_cycle_weight)[0]
            trees_save[i][node] = tree

            #get tree parameters
            depth = trees.tree_depth(tree)
            temp_vector.append(depth)

            paths = trees.leave_path_metrics(tree)
            mean = paths["mean path length"]
            median = paths["median path length"]
            std = paths["std path length"]
            skw = paths["skw path length"]
            kurt = paths["kurtosis path length"]
            alt = paths["altitude"]
            alt_mag = paths["altitude magnitude"]
            ext = paths["total exterior path length"]
            ext_mag = paths["total exterior magnitude"]

            temp_vector.append(mean)
            temp_vector.append(median)
            temp_vector.append(std)
            temp_vector.append(skw)
            temp_vector.append(kurt)
            temp_vector.append(alt)
            temp_vector.append(alt_mag)
            temp_vector.append(ext)
            temp_vector.append(ext_mag)

            asy = trees.tree_asymmetry(tree, trees.number_of_leaves(tree))
            asymmetry = asy["asymmetry"]
            temp_vector.append(asymmetry)

            branching = trees.strahler_branching_ratio(tree)
            mean = branching["mean branching ratio"]
            median = branching["median branching ratio"]
            std = branching["std branching ratio"]
            skw = branching["skw branching ratio"]
            kurt = branching["kurtosis branching ratio"]

            temp_vector.append(mean)
            temp_vector.append(median)
            temp_vector.append(std)
            temp_vector.append(skw)
            temp_vector.append(kurt)

            ext = trees.exterior_interior_edges(tree)
            ee = ext["EE"]
            ei = ext["EI"]
            ee_mag = ext["EE magnitude"]
            ei_mag = ext["EI magnitude"]

            temp_vector.append(ee)
            temp_vector.append(ei)
            temp_vector.append(ee_mag)
            temp_vector.append(ei_mag)

        tree_vector.append(temp_vector)


    return tree_vector, trees_save


def helper_tree_sim(networks, nodes, tree_type="level", edge_attribute="weight", cycle_weight = "max", initial_cycle_weight=True, return_all=False):
    """
    Estimates a similarity matrix between networks by comparing their tree structures node specific - are the same nodes the same distance apart from the root node?


    Parameters:
        networks (list): of networkX graph objects
        nodes (list): list of nodes that should be compared between the networks. Nodes need to be present in all networks.
        tree_type (str): if is "level" then a hierarchical tree is created. Paths from the root indicate how far each nodes are from the root node. Edge weights are not considered.
					if is "cycle" then a hierarchical tree is created where each node represents a cycle in the graph. 
						Tree leaves are the original cycles in the graph and are merged into larger cycles through edge removal until all have been merged into a single cycle.
						This method can be helpful to categorize cyclic graphs. The root parameter is not considered when this option is selected and only cyclic structures in the graph are considered.
		edge_attribute (str): name of the edge weights to be considered if type = "cycle".
		cycle_weight (str): sets how cycles are merged i.e. which edges are removed to merge cycles into larger ones.
							if is "max" then the edge with the highest weight is removed first. if is "min" then the edge with the smalles weight is removed first.
							if is "betweenness_max" the the edge with the highest betweenness value is removed first.
							if is "betweenness_min" the edge with the lowest betweenness value is removed first.
		initial_cycle_weight (boolean): if True the initial cycle basis is estimated based on edge weights with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.minimum_cycle_basis.html#networkx.algorithms.cycles.minimum_cycle_basis
										if False the initial cycles are estimated based on steps only with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis
        return_all (boolean): if True intermediate similarity results are returned as well else only the similarity matrices between the networks are returned.
    Returns:
        average node level overlap percentage (numpy matrix): 
        average node level SMC (numpy matrix):
        average node level Kendall Rank Correlation (numpy matrix):
        average node level jaccard index (numpy matrix):
        intermediate percentage scores (dict): if return_all is True. Key is tuple of network IDs and value is list of scores ordered as in nodes. If node does not exist in the network it is set to None.
        intermediate SMC scores (dict): if return_all is True. Key is tuple of network IDs and value is list of scores ordered as in nodes. If node does not exist in the network it is set to None.
        intermediate Kendall Rank correlation (dict): if return_all is True. Key is tuple of network IDs and value is list of scores ordered as in nodes. If node does not exist in the network it is set to None.
        intermediate jaccard indices (dict): if return_all is True. Key is tuple of network IDs and value is list of scores ordered as in nodes. If node does not exist in the network it is set to None.
    
    """

    results_percentage =  np.zeros((len(networks), len(networks)))
    results_correlation =  np.zeros((len(networks), len(networks)))
    results_smc =  np.zeros((len(networks), len(networks)))
    results_jaccard =  np.zeros((len(networks), len(networks)))

    if return_all:
        results_percentage_all =  {}
        results_correlation_all =  {}
        results_smc_all =  {}
        results_jaccard_all =  {}

   
    index_list = []
    for index, x in np.ndenumerate(results_percentage):
        temp = (index[1], index[0])
        if temp not in index_list and index not in index_list:
            index_list.append(index)


    for index in index_list:
        n1 = index[0]
        n2 = index[1]

        p = []
        c = []
        s = []
        j = []

        p_all = []
        c_all = []
        s_all = []
        j_all = []

        for node in nodes:
            if node in networks[n1].nodes() and node in networks[n2].nodes():
                t1 = trees.construct_tree(networks[n1],  root=node, nr_trees=1, type=tree_type, edge_attribute=edge_attribute, cycle_weight=cycle_weight, initial_cycle_weight=initial_cycle_weight)[0]

                t2 = trees.construct_tree(networks[n2],  root=node, nr_trees=1, type=tree_type, edge_attribute=edge_attribute, cycle_weight=cycle_weight, initial_cycle_weight=initial_cycle_weight)[0]



                per, u = trees.tree_node_level_similarity(t1, t2, type="percentage")
                cor, u = trees.tree_node_level_similarity(t1, t2, type="correlation")
                
                smc, u = trees.tree_node_level_similarity(t1, t2, type="smc")
                jac, u = trees.tree_node_level_similarity(t1, t2, type="jaccard")


                p.append(per)
                c.append(cor)
                s.append(smc)
                j.append(jac)

                if return_all:
                    p_all.append(per)
                    c_all.append(cor)
                    s_all.append(smc)
                    j_all.append(jac)

            else:
                if return_all:
                    p_all.append(None)
                    c_all.append(None)
                    s_all.append(None)
                    j_all.append(None)


        #remove nan values from lists

        p = [x for x in p if str(x) != 'nan']
        c = [x for x in c if str(x) != 'nan']
        s = [x for x in s if str(x) != 'nan']
        j = [x for x in j if str(x) != 'nan']


        mean_p = statistics.mean(p)
        mean_c = statistics.mean(c)
        mean_s = statistics.mean(s)
        mean_j = statistics.mean(j)

        results_percentage[n1][n2] = mean_p
        results_percentage[n2][n1] = mean_p

        results_correlation[n1][n2] = mean_c
        results_correlation[n2][n1] = mean_c

        results_smc[n1][n2] = mean_s
        results_smc[n2][n1] = mean_s

        results_jaccard[n1][n2] = mean_j
        results_jaccard[n2][n1] = mean_j

        if return_all:
            results_percentage_all[(n1, n2)] = p_all
            results_correlation_all[(n1, n2)] = c_all
            results_smc_all[(n1, n2)] = s_all
            results_jaccard_all[(n1, n2)] = j_all


    if return_all:

        return results_percentage, results_smc, results_correlation, results_jaccard, results_percentage_all, results_smc_all, results_correlation_all, results_jaccard_all


    else:

        return results_percentage, results_smc, results_correlation, results_jaccard






def get_node_levels(trees):
    """
    Estimates level of each node in a list of trees.

    Parameters:
        trees (dict): Key is network ID and value is dict where key is node ID and value is tree object. As returned as second item by helper_tree_vector().

    Returns:
        levels (dict): key is network ID  and value is dict where key is node ID of the root and value is dict where keys are node IDs of the tree and values are their level in the tree.
        
    """
    node_levels = {}
    for i in trees.keys(): #this is network id
        node_levels[i] = {}

    for i in trees.keys():
        for node in trees[i].keys(): #this are node specific trees
            tree = trees[i][node]

            #now get for each node its level in the tree
            node_levels[i][node] = {}

            for n in trees[i].keys():
                t = tree.get_node(n)
                level = tree.depth(t)

                node_levels[i][node][n] = level

    return node_levels

