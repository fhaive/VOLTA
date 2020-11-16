"""
This is an example pipeline on how to estimate the similarity between multiple networks
based on their similarity in shared  edges
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
        net_temp list of networkx graph objects

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

def preprocess_edge_list(networks, is_file = False, location = None, names= None):
    """
    function to map edges to ids for faster & easier computation

    Input
        output of preprocess_graph
        is_file if False then networks is list of networkx objects
            if True then networks is list of file locations to python pickles
                if True converted networks are saved as pickles to location
        location str to where converted networks should be saved - will only be used if is_file is True

        names list of network names in same order as networks

    Output
        list of converted networks (or str to saved location)

        dict of edge to id mapping which can be used to reverse mapping
            in order to withdraw later information mapping needs to be kept
            key is str of node ids in format "node1, node2", which can be split into a list
                both edge directions are stored & the same value is assigned to them
            value is id assigned to the particular edge

    """

    if not is_file:
        for i in range(len(networks)):
            if i == 0:
                m, n = node_edge_similarities.map_edge_to_id(networks[i], mapping={}, next_value=0)

            else:
                m, n = node_edge_similarities.map_edge_to_id(networks[i], mapping=m, next_value=n)

        network_lists = []
        for net in networks:
            network_lists.append(node_edge_similarities.construct_mapped_edge(m, net))

        return network_lists, m


    else:
        for i in range(len(networks)):
            #load file from disk
            
            with open(networks[i], "rb") as f:
                net = pickle.load(f)

            if i % 10 == 0:
                print("loaded ", i , "network from disk out of ", len(networks))


            if i == 0:
                m, n = node_edge_similarities.map_edge_to_id(net, mapping={}, next_value=0)

            else:
                m, n = node_edge_similarities.map_edge_to_id(net, mapping=m, next_value=n)


        network_lists = []
        for i in range(len(networks)):
            with open(networks[i], "rb") as f:
                net = pickle.load(f)

            lst = list(dict.fromkeys(node_edge_similarities.construct_mapped_edge(m, net)))
            #save
            name = names[i]

            with open(location + name+".pckl", "wb") as f:
                pickle.dump(lst, f, protocol=4)

            print("saved")
            network_lists.append(location + name+".pckl")

        return network_lists, m

def sort_list_and_get_shared(networks, m, network_lists, labels, is_file = False, in_async=True):
    """
    preprocessing function to sort edge list after weight

    Input
        output of preprocess_graph
        mapping as returned by preprocess_edge_list
        network_list as returned by preprocess_edge_list
        labels is list of str containing names of each layer for later identification
        if is_file is True then networks is list of paths to pickled networks else contains python objects
        if in_async then run in async where applicable

    Output
        list of networks containing sorted edge list
        dict of shared edges between networks
            key is edge id as provided in m & value is list of labels in which this edge occures
        binary representation of networks edges based on all possible edges in all networks
    """

    sorted_networks = []
    if not is_file:
        for net in networks:
            sorted_networks.append(node_edge_similarities.sort_edge_list(net, m))

    else:
        for path in networks:
            with open(path, "rb") as f:
                net = pickle.load(f)

            sorted_networks.append(node_edge_similarities.sort_edge_list(net, m))

    shared_edges = node_edge_similarities.compute_shared_layers(network_lists, labels, mapping = None, weight=False, in_async=in_async)

    binary = node_edge_similarities.compute_binary_layer(shared_edges, layers=labels)

    return sorted_networks, shared_edges, binary

def sort_list(networks, m,  is_file = False, location=None):
    """
    preprocessing function to sort edge list after weight

    Input
        output of preprocess_graph
        mapping as returned by preprocess_edge_list
        
        if is_file is True then networks is list of paths to pickled networks else contains python objects

    Output
        list of networks containing sorted edge list or if_file then contains file location of pickled objects
        
    """

    sorted_networks = []
    if not is_file:
        for net in networks:
            sorted_networks.append(node_edge_similarities.sort_edge_list(net, m))

    else:
        for path in networks:
            with open(path, "rb") as f:
                net = pickle.load(f)
            name =  path.split("/")[-1].replace(".pckl", "")
            temp = node_edge_similarities.sort_edge_list(net, m)

            with open(location + name+".pckl", "wb") as f:
                pickle.dump(temp, f, protocol=4)

            sorted_networks.append(location + name+".pckl")

    

    return sorted_networks


def get_shared(network_lists, labels, is_file = False):
    """
    preprocessing function to sort edge list after weight

    Input
        
        
        network_list as returned by preprocess_edge_list
        labels is list of str containing names of each layer for later identification
        if is_file is True then network_lists is list of paths to pickled networks else contains python objects

    Output
        list of networks containing sorted edge list
        dict of shared edges between networks
            key is edge id as provided in m & value is list of labels in which this edge occures
        binary representation of networks edges based on all possible edges in all networks
    """

    

    shared_edges = node_edge_similarities.compute_shared_layers(network_lists, labels, mapping = None, weight=False, is_file=is_file)

    binary = node_edge_similarities.compute_binary_layer(shared_edges, layers=labels)

    return shared_edges, binary


def estimate_similarities_edges(network_lists, sorted_networks, binary,  kendall_x=50, is_file=True):
    """
    function to estimate edge similarities

    Input
        network_lists as returned by preprocess_edge_list()

        sorted_networks as returned by sort_list_and_get_shared

        binary as returned by sort_list_and_get_shared

        kendall_x number of edges to be considered in kendall ranking (top)

        if is_file then network_lists is list of file location of pickled network objects

    Output
        numpy matrices containing

        jaccard similarity, jaccard distance, similarity
        kendall correlation, hamming distance
    """

    j, percentage = node_edge_similarities.shared_elements_multiple(network_lists,  labels=None, percentage=True, jaccard=True, jaccard_similarity = True, penalize_percentage=False, is_file=is_file)
    jd = node_edge_similarities.to_distance(j)

    kendall_top ,b_top, x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(sorted_networks, compute="kendall", kendall_usage="top", kendall_x = kendall_x)

    kendall_bottom ,b_bottom, x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(sorted_networks, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)

    hamming, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="hamming")

    smc, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="smc")


    return j, jd, percentage, kendall_top,b_top, kendall_bottom, b_bottom, hamming, smc
