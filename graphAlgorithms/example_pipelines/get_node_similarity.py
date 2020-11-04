"""
This is an example pipeline on how to estimate the similarity between multiple networks
based on their similarity in shared nodes 
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
import graphAlgorithms.distances.node_edge_similarities as node_edge_similarities
import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
import scipy


def preprocess_graph(net_temp, attribute="weight", location = None, labels = None):
    """
    function to convert list of networkx graph objects into list of sublist format as needed by
    these functions

    Input
        net_temp list of networkx graph objects if location is not None
            else needs to be list of paths to networkx edgelists (files need to end on .edgelist)

        weight edge attribute to be converted

        location is str of were output should be saved
            if None then a list of edgelists is returned else their pickled location is returned

        labels is list of network names in same order as in net_temp
            only needed if location is not None

    Output
        list of graph objects in edge list format or if location is not None list of path location of saved objects
    """
    if location is None:
        networks = []
        for n in net_temp:
            temp = []
            edges = list(n.edges())
            for edge in edges:
                temp.append([edge[0], edge[1], n[edge[0]][edge[1]][attribute]])

            networks.append(temp)

        return networks

    else:
        networks = []
        for i in range(len(net_temp)):
            path = net_temp[i]
            name = labels[i]
            n=nx.read_weighted_edgelist(path)

            temp = []
            edges = list(n.edges())
            for edge in edges:
                temp.append([edge[0], edge[1], n[edge[0]][edge[1]][attribute]])

            
            #save converted
            print("save", name)
            with open(location + name+".pckl", "wb") as f:
                pickle.dump(temp, f, protocol=4)

            networks.append(location + name+".pckl")

        return networks

def preprocess_node_list(networks, is_file = False, location = None, names= None):
    """
    function to map nodes to ids for faster & easier computation

    Input
        networks is list of edgelists as outputed by preprocess_graph()
            or list of file locations to python pickles of these objects (as saved when using preprocess_graph() with location not None)

        is_file if False then networks is list of networkx objects
            if True then networks is list of file locations to python pickles
                if True converted networks are saved as pickles to location
        location str to where converted networks should be saved - will only be used if is_file is True

        names list of network names in same order as networks

    Output
        list of converted networks (or str to saved location)
        dict of node to id mapping which can be used to reverse mapping

    """
    if not is_file:
        for i in range(len(networks)):
            if i == 0:
                
                m, n = node_edge_similarities.map_node_to_id(networks[i], mapping={}, next_value=0)

            else:
                m, n = node_edge_similarities.map_node_to_id(networks[i], mapping=m, next_value=n)


        node_lists = []

        for net in networks:
            lst = list(dict.fromkeys(node_edge_similarities.construct_mapped_node(m, net)))
            node_lists.append(lst)
            

        return node_lists, m


    else:
        for i in range(len(networks)):
            #load file from disk
            
            with open(networks[i], "rb") as f:
                net = pickle.load(f)

            if i % 10 == 0:
                print("loaded ", i , "network from disk out of ", len(networks))

            if i == 0:
                
                m, n = node_edge_similarities.map_node_to_id(net, mapping={}, next_value=0)

            else:
                m, n = node_edge_similarities.map_node_to_id(net, mapping=m, next_value=n)

        
        node_lists = []
        for i in range(len(networks)):
            with open(networks[i], "rb") as f:
                net = pickle.load(f)

            lst = list(dict.fromkeys(node_edge_similarities.construct_mapped_node(m, net)))
            #save
            name = names[i]

            with open(location + name+".pckl", "wb") as f:
                pickle.dump(lst, f, protocol=4)

            print("saved")
            node_lists.append(location + name+".pckl")

            

        return node_lists, m


def sort_list_and_get_shared(node_lists, m, network_graphs, labels, degree=True, degree_centrality=True, closeness_centrality=True, betweenness=True, is_file = False):
    """
    preprocessing function to sort edge list after weight

    Input
        output of preprocess_node_list
        mapping as returned by preprocess_node_list
        original list of networkx graph objects
        labels is list of str containing names of each layer for later identification

        if is_file is True then network_graphs contains file location instead of object
            
            network_graphs contains paths to networkx weighted edge lists

    Output
        list of networks containing sorted node list after degree, degree centrality, closeness centrality, betweenness, & average of all
        list of shared edges between networks
        binary representation of networks nodes based on all possible nodes in all networks
    """

    
    shared_nodes = node_edge_similarities.compute_shared_layers(node_lists, labels, mapping = None, weight=False)

    binary = node_edge_similarities.compute_binary_layer(shared_nodes, layers=labels)

    sorted_nodes = []
    saved_values = []
    
    if not is_file:
        for net in network_graphs:
            s, v = node_edge_similarities.sort_node_list(net, m, degree=degree, degree_centrality=degree_centrality, closeness_centrality=closeness_centrality, betweenness=betweenness,as_str=False)
            sorted_nodes.append(s)
            saved_values.append(v)

    else:
        for path in network_graphs:
            net = nx.read_weighted_edgelist(path)
            s, v = node_edge_similarities.sort_node_list(net, m, degree=degree, degree_centrality=degree_centrality, closeness_centrality=closeness_centrality, betweenness=betweenness,as_str=False)
            sorted_nodes.append(s)
            saved_values.append(v)


    return sorted_nodes, shared_nodes, binary, saved_values


def estimate_similarities_nodes(node_lists, sorted_nodes, binary,  kendall_x=50):
    """
    function to estimate edge similarities

    Input
        output of preprocess_node_list

        sorted_nodes as returned by sort_list_and_get_shared

        binary as returned by sort_list_and_get_shared

        kendall_x number of edges to be considered in kendall ranking (top)

    Output
        numpy matrices containing

        jaccard similarity, jaccard distance, similarity
        kendall correlation for degree centrality, closeness centrality, betweenness, hamming distance
    """

    j, s = node_edge_similarities.shared_elements_multiple(node_lists, labels=None, percentage=True, jaccard=True, jaccard_similarity = True, penalize_percentage=False)
    jd = node_edge_similarities.to_distance(j)

    print("kendall top")
    #ranked after degree centrality
    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["dc"])
    kendall_dc_top ,b_dc_top,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="top", kendall_x = kendall_x)

    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["cc"])
    kendall_cc_top ,b_cc_top,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="top", kendall_x = kendall_x)

    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["betweenness"])
    kendall_betweenness_top ,b_b_top,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="top", kendall_x = kendall_x)


    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["average_mean"])
    kendall_avg_top ,b_avg_top,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="top", kendall_x = kendall_x)

    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["average_median"])
    kendall_med_top ,b_med_top,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="top", kendall_x = kendall_x)


    print("kendall bottom")
    #ranked after degree centrality
    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["dc"])
    kendall_dc_bottom ,b_dc_bottom,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)

    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["cc"])
    kendall_cc_bottom ,b_cc_bottom,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)

    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["betweenness"])
    kendall_betweenness_bottom ,b_b_bottom,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)


    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["average_median"])
    kendall_med_bottom ,b_med_bottom,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)

    current_sorted = []
    for di in sorted_nodes:
            current_sorted.append(di["average_mean"])
    kendall_avg_bottom ,b_avg_bottom,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)




    hamming, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="hamming")

    smc, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="smc")

    return j, jd, s, kendall_dc_top, b_dc_top, kendall_cc_top, b_cc_top, kendall_betweenness_top, b_b_top, kendall_avg_top, b_avg_top, hamming, kendall_dc_bottom , b_dc_bottom , kendall_cc_bottom , b_cc_bottom , kendall_betweenness_bottom , b_b_bottom , kendall_avg_bottom , b_avg_bottom , smc, kendall_med_top, b_med_top, kendall_med_bottom, b_med_bottom


