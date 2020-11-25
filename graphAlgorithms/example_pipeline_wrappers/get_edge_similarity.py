"""
This is a collection of wrapper functions to simplify how to estimate the similarity between multiple networks
based on their similarity in edges.
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
    Converts list of networkX graph objects into a list of sublist format, which is used by most functions in this package.

    Parameters:
        net_temp (list): list of networkX graph objects
        attribute (str): edge weight label to be converted
        location (str or None): if None the converted object is returned. Else it needs to be file location where the converted objects should be pickled and their locations will be returned instead.
        labels (list or None): list of network names in same order as net_temp. Only needs to be provided if location is not None.
        
    Return:
        converted objects (list): if location is None list of converted objects is returned else list of pickled locations is returned.
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
    Maps edges to IDs.

    Parameters:
        networks (list): list of converted networks or list of pickled file locations as returned by preprocess_graph()
        is_file (boolean): if False then networks is list converted objects. If True then networks is list of file locations to the pickled objects instead.
        location (str or None): if is_file is True then output of this function will be pickled to location.
        names (list or None): list of network names in same order as networks. If is_file is True then names will be used to store pickled objects.

    Returns:
        networks with edge IDs or their pickled location (list):
        edge ID mapping (dict): keys are str of node IDs building an edge in format "node1, node2" and values are assigned ID.
        
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
    Preprocessing function to sort edge list after weight, convert to a binary format and claculate shared edges.

    parameters:
        networks (list): list of converted networks or list of pickled file locations as returned by preprocess_graph().
        m (dict): edge to ID mapping as returned by preprocess_edge_list().
        is_file (boolean): if False then networks is list converted objects. If True then networks is list of file locations to the pickled objects instead.
        network_lists (list): list of converted edge IDs as returned by  preprocess_edge_list()
        labels (list): list of network names in same order as networks. 
        in_async (boolean): if True then run in async where applicable.

    Returns:
        sorted networks (list): contains sorted edge list.
        shared edges (dict): key is edge ID as provided in m and value is list of network labels containing this edge.
        binary (list): binary representation of network edges based on the union of edges in all provided networks.
        
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

    shared_edges = node_edge_similarities.compute_shared_layers(network_lists, labels, in_async=in_async)

    binary = node_edge_similarities.compute_binary_layer(shared_edges, layers=labels)

    return sorted_networks, shared_edges, binary

def sort_list(networks, m,  is_file = False, location=None):
    """
    Sorts edge list after weight.

    Parameters:
        networks (list): networks (list): list of converted networks or list of pickled file locations as returned by preprocess_graph().
        m (dict): edge to ID mapping as returned by preprocess_edge_list().
        is_file (boolean): if False then networks is list converted objects. If True then networks is list of file locations to the pickled objects instead and output will be saved to file as well.
        location (str or None): location where to pickle output to if is_file is True.
        
    Returns:
        sorted (list): list of networks containing sorted edge list or if_file is True then contains the file locations of the pickled objects.
        
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
    Gets shared edges between networks.

    Parameters:
        network_lists (list): list of converted edge IDs as returned by  preprocess_edge_list().
        labels (list): list of network names in same order as networks. 
        is_file (boolean): if False then network_lists is list of converted objects. If True then networks is list of file locations to the pickled objects instead.
       
    Returns:
        shared edges (dict): key is edge ID as provided in m and value is list of network labels containing this edge.
    """

    

    shared_edges = node_edge_similarities.compute_shared_layers(network_lists, labels,  is_file=is_file)

    binary = node_edge_similarities.compute_binary_layer(shared_edges, layers=labels)

    return shared_edges, binary


def estimate_similarities_edges(network_lists, sorted_networks, binary,  kendall_x=50, is_file=True, in_async=True):
    """
    Wrapper function to estimate similarity between networks based on their edges.

    Parameters:
        network_lists (list): list of converted edge IDs as returned by preprocess_edge_list().
        sorted_networks (list): list of edges sorted by weight as returned by sort_list_and_get_shared () or sort_list().
        binary (list): list of binary edge representation as returned by sort_list_and_get_shared() or node_edge_similarities.compute_binary_layer().
        kendall_x (int): top/bottom number of edges to be considered when estimating kendall rank correlation.
        is_file (boolean): if False then network_lists is list of converted objects. If True then network_lists is list of file locations to the pickled objects instead.
        in_async (boolean): if True then run in async where applicable.

    Returns:
        jaccard similarity (numpy matrix):
        jaccard distance (numpy matrix):
        percentage of shared edges (numpy matrix):
        kendall correlation coefficient based on top edges (numpy matrix):
        kendall p value based on top edges (numpy matrix):
        kendall correlation coefficient based on bottom edges (numpy matrix):
        kendall p value based on bottom edges (numpy matrix):
        hamming distance (numpy matrix):
        SMC (numpy matrix):

    """

    j, percentage = node_edge_similarities.shared_elements_multiple(network_lists,  labels=None, percentage=True, jaccard=True, jaccard_similarity = True, penalize_percentage=False, is_file=is_file, in_async=in_async)
    jd = node_edge_similarities.to_distance(j)

    kendall_top ,b_top, x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(sorted_networks, compute="kendall", kendall_usage="top", kendall_x = kendall_x)

    kendall_bottom ,b_bottom, x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(sorted_networks, compute="kendall", kendall_usage="bottom", kendall_x = kendall_x)

    hamming, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="hamming")

    smc, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="smc")


    return j, jd, percentage, kendall_top,b_top, kendall_bottom, b_bottom, hamming, smc
