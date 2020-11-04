"""
This is an example pipeline on how to estimate the similarity between multiple networks
based on mapping it to a tree & estimating similarities based on a tree
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
    Helper function to estimate for each network a vector based on its tree properties

    Input
        networks list of networkx graph objects

        nodes list of all nodes in all networks / or nodes to be investigated/ compared between networks

        the other parameters describe how the tree should be constructed, for a reference refer to its function declaration

    Output
        list of lists, were each lists contains "a network specific vector"
            network distances can be estimated by calcualting distances between these vectors
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
    Helper function that shows on how to create a similarity matrix between networks by comparing
    tree structures directly

    Input
        networks list of networkx graph objects

        nodes list of all nodes in all networks / or nodes to be investigated/ compared between networks

        if return all then for each network pair, its full similarity list are returned
            this can be used to estimate the node pairs with the highest similarity between each other

        the other parameters describe how the tree should be constructed, for a reference refer to its function declaration

    Output
        numpy matrix containing similarity scores between networks
            matrix index is index of network as provided in networks

        matrix pf percentage score
        matrix of smc score
        matrix of correlation score
        matrix of jaccard score

        if return_all
            then additional 4 dicts are returned, containing the node specific values
                key is tuple of networks ids as ordered in networks
                value is list of scores ordered in order of nodes
                    if node does not occure in one of the networks value is set to None

            dict of percentage scores
            dict of smc scores
            dict of correlation scores
            dict of jaccard scores
            
            these values can be used to find the "most similar node sub-areas" between two networks

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


def matrix_from_vector(tree_vector, normalize=False):
    """
    this is an example function on how to estimate similarity/ distance matrices based on computed vectors
    careful distances are only calculated one-sided (only half of matrix is calculated, other half is assumed to be the same)
    None values in vector are replaced with 0
    based on the vector parameters distance between the same networks may not be 0
        if this is necessary matrix diagonal needs to be set manually to 0

    this function provides eclidean, canberra, correlation, cosine & jaccard distance
        others can be added if needed

    Input
        list of vectors as returned by helper_tree_vector()

        if normalize then euclidean & canberra distance matrices are normalized

    Output
        returns numpy matrixes containing the similarity/ distance scores 
            matrix indices are ordered as provided in tree_vector

        euclidean distance #careful the distance may not be normalized!
        canberra #careful distance may not be normalized!
        correaltion
        cosine
        jaccard



    """

    results_euclidean =  np.zeros((len(tree_vector), len(tree_vector)))
    results_canberra =  np.zeros((len(tree_vector), len(tree_vector)))
    results_correlation =  np.zeros((len(tree_vector), len(tree_vector)))
    results_cosine =  np.zeros((len(tree_vector), len(tree_vector)))
    results_jaccard =  np.zeros((len(tree_vector), len(tree_vector)))

    results =  np.zeros((len(tree_vector), len(tree_vector)))

    index_list = []
    for index, x in np.ndenumerate(results):
        temp = (index[1], index[0])
        if temp not in index_list and index not in index_list:
            index_list.append(index)

    for i in index_list:
        print(i)
        v1 = tree_vector[i[0]]
        v2 = tree_vector[i[1]]
        
        while None in v1:
            ii = v1.index(None)
            v1[ii] = 0
            
        while None in v2:
            ii = v2.index(None)
            v2[ii] = 0
        
        
        e = scipy.spatial.distance.euclidean(v1, v2)
        
        results_euclidean[i[0]][i[1]] = e
        results_euclidean[i[1]][i[0]] = e
        
        
        e = scipy.spatial.distance.canberra(v1, v2)
        
        results_canberra[i[0]][i[1]] = e
        results_canberra[i[1]][i[0]] = e
        
        e = scipy.spatial.distance.correlation(v1, v2)
        
        results_correlation[i[0]][i[1]] = e
        results_correlation[i[1]][i[0]] = e
        
        e = scipy.spatial.distance.cosine(v1, v2)
        
        results_cosine[i[0]][i[1]] = e
        results_cosine[i[1]][i[0]] = e
        
        e = scipy.spatial.distance.jaccard(v1, v2)
        
        results_jaccard[i[0]][i[1]] = e
        results_jaccard[i[1]][i[0]] = e

    if normalize:
        xmax, xmin = results_canberra.max(), results_canberra.min()
        results_canberra = (results_canberra - xmin)/(xmax - xmin)

        xmax, xmin = results_euclidean.max(), results_euclidean.min()
        results_euclidean = (results_euclidean - xmin)/(xmax - xmin)


    return results_euclidean, results_canberra, results_correlation, results_cosine, results_jaccard



def get_node_levels(trees):
    """
    function that estimates level of each node in the tree for each networks and each starting node

    Input
        trees is outptut of helper_tree vector


    Output
        dict where key is network id and value is dict
            were key is node id of root node of tree
                value is dict were each key is node id and value is its level in the tree


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

