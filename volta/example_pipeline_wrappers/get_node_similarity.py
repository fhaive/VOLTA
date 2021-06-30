"""
This is a collection of wrapper functions to simplify how to estimate the similarity between multiple networks
based on their similarity in nodes.
"""

import networkx as nx
import pandas as pd
import csv
import random
import sys
import volta.distances.global_distances as global_distances
import volta.distances.local as local
import volta.simplification as simplification
import volta.distances.trees as trees
import volta.distances.node_edge_similarities as node_edge_similarities
import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
import scipy


def preprocess_graph(net_temp, attribute="weight", location = None, labels = None):
    """
    Converts list of networkX graph objects into a list of sublist format, which is used by most functions in this package.
    A directed graph object can be provided as input, however each edge will be interpreted as bidirectional by most downstream functions
        that require this converted network format.

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

def preprocess_node_list(networks, is_file = False, location = None, names= None):
    """
    Maps nodes to IDs.

    Parameters:
        networks (list): list of converted networks or list of pickled file locations as returned by preprocess_graph()
        is_file (boolean): if False then networks is list converted objects. If True then networks is list of file locations to the pickled objects instead.
        location (str or None): if is_file is True then output of this function will be pickled to location.
        names (list or None): list of network names in same order as networks. If is_file is True then names will be used to store pickled objects.

    Returns:
        networks (list): with nodes IDs or their pickled location as network list format
        node ID mapping (dict): keys are node IDs and values are assigned ID.
    

    """
    if not is_file:
        for i in range(len(networks)):
            if i == 0:
                
                m, n = node_edge_similarities.map_node_to_id(networks[i], mapping={}, next_value=0)

            else:
                m, n = node_edge_similarities.map_node_to_id(networks[i], mapping=m, next_value=n)


        node_lists = []

        for net in networks:
            lst = list(dict.fromkeys(node_edge_similarities.__construct_mapped_node__(m, net)))
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

            lst = list(dict.fromkeys(node_edge_similarities.__construct_mapped_node__(m, net)))
            #save
            name = names[i]

            with open(location + name+".pckl", "wb") as f:
                pickle.dump(lst, f, protocol=4)

            print("saved")
            node_lists.append(location + name+".pckl")

            

        return node_lists, m


def sort_list_and_get_shared(node_lists, m, network_graphs, labels, degree=True, degree_centrality=True, closeness_centrality=True, betweenness=True, weight=None, is_file = False, in_async =True, k=None):
    """
    Preprocessing function to sort node list after their attributes, convert to a binary format and claculate shared nodes.
    A directed graph object can be provided as input.

    parameters:
        node_lists (list): list of converted node IDs as returned by  preprocess_node_list()
        m (dict): edge to ID mapping as returned by preprocess_node_list().
        network_graphs (list): list of networkX graph objects. This needs to be the original networks before conversion. If is_file is True then it is list locations to the pickled graph objects.
        labels (list): list of network names in same order as networks.
        degree (boolean): if True nodes are sorted after degree. If multiple values are set to True a combined ranking is calculated.
        degree_centrality (boolean): if True nodes are sorted after degree centrality. If multiple values are set to True a combined ranking is calculated.
        closeness_centrality (boolean): if True nodes are sorted after closeness centrality. If multiple values are set to True a combined ranking is calculated.
        betweenness (boolean): if True nodes are sorted after betweenness. If multiple values are set to True a combined ranking is calculated.
        weight (str or None): for weighted networks name of edge attribute to be used. If None all edges are considered equal.
            Instead of node degree strength of degree will be calculated if not None, betweenness centrality will be calculated based on
            weighted edges as well as closeness centrality (weight is distance). Has no impact on degree centrality.
        is_file (boolean): if False then network_graphs is list converted objects. If True then network_graphs is list of file locations to the pickled objects instead.
        in_async (boolean): if True then run in async where applicable.
        k (float [0,1] or None): approximation of betweenness, if float then k percentage of nodes are used to approximate the betweenness values. If None all nodes are used.

    Returns:
        sorted networks (list): contains dicts where keys are degree, dc, cc, betweenness, average_mean and average_median, values are list of ranked node ids. If key is set to False an empty list is returned.
        shared nodes (dict): key is node ID as provided in m and value is list of network labels containing this node.
        binary (list): binary representation of network nodes based on the union of nodes in all provided networks.

    """

    
    shared_nodes = node_edge_similarities.compute_shared_layers(node_lists, labels, in_async=in_async)

    binary = node_edge_similarities.compute_binary_layer(shared_nodes, layers=labels)

    sorted_nodes = []
    saved_values = []
    
    if not is_file:
        for net in network_graphs:
            s, v = node_edge_similarities.sort_node_list(net, m, degree=degree, degree_centrality=degree_centrality, closeness_centrality=closeness_centrality, betweenness=betweenness, weight=weight, as_str=False, k=k)
            sorted_nodes.append(s)
            saved_values.append(v)

    else:
        for path in network_graphs:
            net = nx.read_weighted_edgelist(path)
            s, v = node_edge_similarities.sort_node_list(net, m, degree=degree, degree_centrality=degree_centrality, closeness_centrality=closeness_centrality, betweenness=betweenness, weight = weight, as_str=False, k=k)
            sorted_nodes.append(s)
            saved_values.append(v)


    return sorted_nodes, shared_nodes, binary, saved_values

def estimate_similarities_nodes(node_lists, sorted_nodes, binary,  kendall_x=50, is_file=False, in_async=True):
    """
    Wrapper function to estimate similarity between networks based on their nodes.

    Parameters:
        node_lists (list): list of converted node IDs as returned by preprocess_node_list().
        sorted_nodes (list): list of node sorted by weight as returned object by sort_list_and_get_shared () or sort_node_list().
        binary (list): list of binary node representation as returned by sort_list_and_get_shared() or node_edge_similarities.compute_binary_layer().
        kendall_x (int): top/bottom number of nodes to be considered when estimating kendall rank correlation.
        is_file (boolean): if False then node_lists is list of converted objects. If True then node_lists is list of file locations to the pickled objects instead.
        in_async (boolean): if True then run in async where applicable.

    Returns:
        results (dict) where keys and values are:
            jaccard similarity (numpy matrix): between network pairs
            jaccard distance (numpy matrix): between network pairs
            percentage of shared nodes (numpy matrix): between network pairs
            kendall_dc_top (numpy matrix): kendall correlation coefficient based on top nodes ranked by degree centrality between network pairs
            b_dc_top (numpy matrix): kendall p value based on top nodes ranked by degree centrality between network pairs
            kendall_cc_top (numpy matrix): kendall correlation coefficient based on top nodes ranked by closeness centrality between network pairs
            b_cc_top (numpy matrix): kendall p value based on top nodes ranked by closeness centrality between network pairs
            kendall_betweenness_top (numpy matrix): kendall correlation coefficient based on top nodes ranked by betweenness centrality  between network pairs
            b_b_top (numpy matrix): kendall p value based on top nodes ranked by degree betweenness between network pairs
            kendall_avg_top (numpy matrix): kendall correlation coefficient based on top nodes ranked by mean ranking between network pairs
            b_avg_top (numpy matrix): kendall p value based on top nodes ranked by mean ranking  between network pairs
            hamming (numpy matrix): distance between network pairs
            kendall_dc_bottom (numpy matrix): kendall correlation coefficient based on bottom nodes ranked by degree centrality between network pairs
            b_dc_bottom (numpy matrix): kendall p value based on bottom nodes ranked by degree centrality between network pairs
            kendall_cc_bottom (numpy matrix): kendall correlation coefficient based on bottom nodes ranked by closeness centrality between network pairs
            b_cc_bottom (numpy matrix): kendall p value based on bottom nodes ranked by closeness centrality between network pairs
            kendall_betweenness_bottom (numpy matrix): kendall correlation coefficient based on bottom nodes ranked by betweenness centrality between network pairs
            b_b_bottom (numpy matrix): kendall p value based on bottom nodes ranked by degree betweenness between network pairs
            kendall_avg_bottom (numpy matrix): kendall correlation coefficient based on bottom nodes ranked by mean ranking between network pairs
            b_avg_bottom (numpy matrix): kendall p value based on bottom nodes ranked by mean ranking between network pairs
            smc (numpy matrix): between network pairs
            kendall_med_top (numpy matrix): kendall correlation coefficient based on top nodes ranked by median ranking between network pairs
            b_med_top (numpy matrix):  kendall p value based on top nodes ranked by median ranking  between network pairs
            kendall_med_bottom (numpy matrix): kendall correlation coefficient based on bottom nodes ranked by median ranking  between network pairs
            b_med_bottom (numpy matrix): kendall p value based on bottom nodes ranked by meadianranking between network pairs

    """

    j, s = node_edge_similarities.shared_elements_multiple(node_lists, labels=None, percentage=True, jaccard=True, jaccard_similarity = True, penalize_percentage=False, is_file=is_file, in_async=in_async)
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

    res_dict = {}
    res_dict["jaccard distance"] = jd
    res_dict["jaccard similarity"] = j
    res_dict["percentage of shared nodes"] = s
    res_dict["kendall_dc_top"] = kendall_dc_top
    res_dict["b_dc_top"] = b_dc_top
    res_dict["kendall_cc_top"] = kendall_cc_top
    res_dict["b_cc_top"] = b_cc_top
    res_dict["kendall_betweenness_top"] = kendall_betweenness_top
    res_dict["b_b_top"] = b_b_top
    res_dict["kendall_avg_top"] = kendall_avg_top
    res_dict["b_avg_top"] = b_avg_top

    res_dict["hamming"] = hamming
    res_dict["kendall_dc_bottom"] = kendall_dc_bottom
    res_dict[" b_dc_bottom"] =  b_dc_bottom
    res_dict["kendall_cc_bottom"] = kendall_cc_bottom
    res_dict["b_cc_bottom"] = b_cc_bottom
    res_dict["kendall_betweenness_bottom"] = kendall_betweenness_bottom
    res_dict["b_b_bottom"] = b_b_bottom 
    res_dict["kendall_avg_bottom"] = kendall_avg_bottom
    res_dict["b_avg_bottom"] = b_avg_bottom
    res_dict["smc"] = smc

    res_dict["kendall_med_top"] = kendall_med_top
    res_dict["b_med_top"] = b_med_top
    res_dict["kendall_med_bottom"] = kendall_med_bottom
    res_dict["b_med_bottom"] = b_med_bottom



    return res_dict


