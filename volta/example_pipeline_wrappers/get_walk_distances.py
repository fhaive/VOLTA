"""
This is a collection of wrapper functions to simplify how to estimate the similarity between multiple networks
based on random walks.
"""

import networkx as nx
import pandas as pd
import csv
import random
import sys
#sys.path.insert(1, '../distances/')
import volta.distances.global_distances as global_distances
import volta.distances.local as local
import volta.simplification as simplification
import volta.distances.trees as trees

import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
import scipy
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial


def helper_walks(networks, nodes, network_ids, steps=10, number_of_walks=10, degree=True,  probabilistic=True, weight="weight"):

    """
    Estimates for networks number_of_walks walks of size steps.

    Parameters:
        networks (list): of networkX graph objects
        nodes (list): of nodes (areas) to be compared.
        network_ids (list): list of network IDs.
        steps (int): is size of random walk
        number_of_walks (int): how many random walks are performed on G
        degree (boolean): if True then the number of random walks performed for each starting node is dependent on its degree and is estimated as degree*number_of_walks.
        probabilisitc (boolean): if True edge weights are taken into account else all edges are considered equal.  If true then weight needs to be set
        weight (str): edge attribute name as contained in G. Weight is evaluated as a similarity

    Returns:
        walks (dict): key is network IDs and value is dict where key is starting node and value is list of performed walks.
                    Each walk is a sublist and contains the node IDs in order of their visit by the random walk.
        
    """

    performed_walks = {}
   
    for net_id in network_ids:
        performed_walks[net_id] = {}
        
        
    cn = 0
    for node in nodes:
        
        if cn % 100 == 0:
            print("walks for node ", cn, "outof", len(nodes))
        cn = cn + 1

        walks = []
        

        for i in range(len(networks)):
            net = networks[i]
            network_id = network_ids[i]
            
            if node in net.nodes():
                
                if not nx.is_isolate(net, node):
                    if degree:
                        nw = int(number_of_walks * net.degree[node])
                        print("running walks", nw, "for node", node)
                    else:
                        nw = number_of_walks

                        
                    walks  = global_distances.perform_random_walks(net, steps=steps, number_of_walks=nw, start=node, probabilistic=probabilistic, weight=weight)
                    



            #save
            performed_walks[network_id][node] = walks
            


    return performed_walks

def helper_get_counts(labels, networks, performed_walks):
    """
    Count number of appearenses of nodes & edges in walks performed on the same starting nodes. Also estimates the fraction of appearens w.r.t. to all nodes/ edges visited from the same strating node.

    Parameters:
        labels (list): network labels as provided to helper_walks().
        networks (list): of networkX graph objects on which the random walks have been performed. Needs to be in same order as labels.
        performed_walks (dict): as returned by helper_walks().
     
    Returns:
        node counts (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Node ID and value is its counts.
        edge counts (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Edge ID and value is its counts.
        node fraction (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Node ID and value is its fraction w.r.t to all visited nodes from that start node.
        edge fraction (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Edge ID and value is its fraction w.r.t to all visited edges from that start node..
    """


    edges = {}
    nodes = {}
    edges_percentage = {}
    nodes_percentage = {}
    for i in labels:
        edges[i] = {}
        nodes[i]= {}
        edges_percentage[i] = {}
        nodes_percentage[i]= {}
        for s in performed_walks[i].keys():
            edges[i][s] = []
            nodes[i][s] = []
            edges_percentage[i][s] = []
            nodes_percentage[i][s] = []


    for ii in range(len(labels)):
        i = labels[ii]
        for s in performed_walks[i].keys():
            walk_list = performed_walks[i][s]
            nodes_cnt, edges_cnt = global_distances.__rank_walks__(networks[ii], walk_list)
            edges[i][s] = edges_cnt
            nodes[i][s] = nodes_cnt

            #compute fraction values
            nodes_frc = {}
            for key in nodes_cnt.keys():
                nodes_frc[key] = nodes_cnt[key] / len(nodes_cnt.keys())

            edges_frc = {}
            for key in edges_cnt.keys():
                edges_frc[key] = edges_cnt[key] / len(edges_cnt.keys())


            edges_percentage[i][s] = edges_frc
            nodes_percentage[i][s] = nodes_frc



            


    return nodes, edges, nodes_percentage, edges_percentage


def perform_walks_compute_counts(networks, nodes, network_ids, steps=10, number_of_walks=10, degree=True,  probabilistic=True, weight="weight"):
    
    """
    Estimates for networks number_of_walks walks of size steps.

    Parameters:
        networks (list): of networkX graph objects
        nodes (list): of nodes (areas) to be compared.
        network_ids (list): list of network IDs.
        steps (int): is size of random walk
        number_of_walks (int): how many random walks are performed on G
        degree (boolean): if True then the number of random walks performed for each starting node is dependent on its degree and is estimated as degree*number_of_walks.
        probabilisitc (boolean): if True edge weights are taken into account else all edges are considered equal.  If true then weight needs to be set
        weight (str): edge attribute name as contained in G. Weight is evaluated as a similarity

    Returns:
        walks (dict): key is network IDs and value is dict where key is starting node and value is list of performed walks.
                    Each walk is a sublist and contains the node IDs in order of their visit by the random walk.
        node counts (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Node ID and value is its counts.
        edge counts (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Edge ID and value is its counts.
        node fraction (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Node ID and value is its fraction w.r.t to all visited nodes from that start node.
        edge fraction (dict): key is network ID ordered as in labels. Value is dict where key is start node and value is dict where key is Edge
        
    """

    #call helper_walks() & helper_get_counts and chunks of nodes for each network & then merge dicts
    walks = {}
    node_cnt = {}
    edge_cnt = {}
    node_frct = {}
    edge_frct = {}

    for i in range(len(networks)):
        #provide networks[i] & networks_ids[i]
        walks_temp = {}

        node_cnt_temp = {}
        edge_cnt_temp = {}
        node_frct_temp = {}
        edge_frct_temp = {}
        #split nodes into chunks
        for c in [nodes[i:i + 10] for i in range(0, len(nodes), 10)]:
            walks_temp2 = helper_walks([networks[i]], c, [network_ids[i]], steps=steps, number_of_walks=number_of_walks, degree=degree,  probabilistic=probabilistic, weight=weight)

            #merge with walks_temp
            walks_temp.update(walks_temp2[network_ids[i]])

            #call counter
            node_cnt_temp2, edge_cnt_temp2, node_frct_temp2, edge_frct_temp2 = helper_get_counts([network_ids[i]], [networks[i]], walks_temp2)

            #update dict
            node_cnt_temp.update(node_cnt_temp2[network_ids[i]])
            edge_cnt_temp.update(edge_cnt_temp2[network_ids[i]])
            node_frct_temp.update(node_frct_temp2[network_ids[i]])
            edge_frct_temp.update(edge_frct_temp2[network_ids[i]])

        #append to main dict
        walks[network_ids[i]] = walks_temp

        node_cnt[network_ids[i]] = node_cnt_temp
        edge_cnt[network_ids[i]] = edge_cnt_temp
        node_frct[network_ids[i]] = node_frct_temp
        edge_frct[network_ids[i]] = edge_frct_temp


    return walks, node_cnt, edge_cnt, node_frct, edge_frct






    

def helper_walk_sim(networks, performed_walks, nodes, network_ids, undirected=True, top=10, return_all=False, nodes_ranked=None, edges_ranked=None):

    """
    Compares random walks based on their similarity of visited nodes/ edges. Estimates for each network pair a correlation score based on the mean of each node pairs random walks.
    
    Parameters:
        networks (list): of networkX graph objects
        performed_walks (dict): as returned by helper_walks().
        nodes (list): of nodes (areas) to be compared.
        network_ids (list): list of network IDs as contained in performed_walks().
        top (int): top x nodes & edges are considered when calculating the correlation 
        undirected (boolean): if True then edge traversal is not taken into account
        return_all (boolean): if True then for each network pair its full correlation list is returned as well.
        nodes_ranked (dict): as returned by helper_get_counts()
        edges_ranked (dict): as returned by helper_get_counts())

    Returns:
        correlation edges (numpy matrix): between network pairs
        correlation nodes (numpy matrix): between network pairs
        correlation edges p-value (numpy matrix): between network pairs
        correlation nodes p-value (numpy matrix): between network pairs
        intermediate correaltion scores edges (dict): if return_all is True. Key is tuple of network IDs and value is list of scores ordered as in nodes.
        intermediate correaltion scores nodes (dict):  if return_all is True. Key is tuple of network IDs and value is list of scores ordered as in nodes.
        intermediate p-values edges (dict): if return_all is True. Key is tuple of network IDs and value is list of p-values ordered as in nodes.
        intermediate p-values edges (dict):  if return_all is True. Key is tuple of network IDs and value is list of p-values ordered as in nodes.

    """
                
    results_nodes =  np.zeros((len(networks), len(networks)))
    results_edges =  np.zeros((len(networks), len(networks)))

    results_nodes_p =  np.zeros((len(networks), len(networks)))
    results_edges_p =  np.zeros((len(networks), len(networks)))


    if return_all:
        results_nodes_all =  {}
        results_edges_all =  {}

        results_nodes_p_all =  {}
        results_edges_p_all =  {}

    index_list = []
    for index, x in np.ndenumerate(results_nodes):
        temp = (index[1], index[0])
        if temp not in index_list and index not in index_list:
            index_list.append(index)


    for index in index_list:
        n1 = network_ids[index[0]]
        n2 = network_ids[index[1]]
        #print("n1", n1)
        #print("n2", n2)

        nodes_sim = []
        edges_sim = []

        nodes_sim_p = []
        edges_sim_p = []

        nodes_sim_all = []
        edges_sim_all = []
        nodes_sim_p_all = []
        edges_sim_p_all = []

        for node in nodes:
            #get consensus walks
            #since prefiously if node is not in networks has been set to an empty list this does not need to be checked for here
            #print("len c1 network", len(performed_walks[n1]))
            c1 = performed_walks[n1][node]
            c2 = performed_walks[n2][node]

            if len(c1) > 0 and len(c2) > 0:

               
                kendall = global_distances.compare_walks(networks[index[0]], [nodes_ranked[n1][node], edges_ranked[n1][node]], walk2=[nodes_ranked[n2][node], edges_ranked[n2][node]], G2=networks[index[1]],undirected=undirected, comparison="ranked", top=top)
                
                e_t = kendall["edges_tau"]
                n_t = kendall["nodes_tau"]
                e_p = kendall["edges_p"]
                n_p = kendall["nodes_p"]

                nodes_sim.append(n_t)
                edges_sim.append(e_t)
                nodes_sim_p.append(n_p)
                edges_sim_p.append(e_p)

                if return_all:
                    nodes_sim_all.append(n_t)
                    edges_sim_all.append(e_t)
                    nodes_sim_p_all.append(n_p)
                    edges_sim_p_all.append(e_p)


            else:
                print("no walk similarities can be estimated", node, n1, n2)
                if return_all:
                    nodes_sim_all.append(None)
                    edges_sim_all.append(None)
                    nodes_sim_p_all.append(None)
                    edges_sim_p_all.append(None)


        #estiamte mean and write to results matrix
        if len(nodes_sim) > 1:
            mean_nodes = statistics.mean(nodes_sim)
            mean_nodes_p = statistics.mean(nodes_sim_p)
        else:
            print("no mean value can be estimated & value is set to None for", n1, n2)
            mean_nodes = None
            mean_nodes_p=None

        if len(edges_sim) > 1:
            mean_edges = statistics.mean(edges_sim)
            mean_edges_p = statistics.mean(edges_sim_p)
        else:
            print("no mean value can be estimated & value is set to None for", n1, n2)
            mean_edges = None
            mean_edges_p = None


        results_edges[index[0]][index[1]] = mean_edges
        results_edges[index[1]][index[0]] = mean_edges
        results_edges_p[index[0]][index[1]] = mean_edges_p
        results_edges_p[index[1]][index[0]] = mean_edges_p

        results_nodes[index[0]][index[1]] = mean_nodes
        results_nodes[index[1]][index[0]] = mean_nodes
        results_nodes_p[index[0]][index[1]] = mean_nodes_p
        results_nodes_p[index[1]][index[0]] = mean_nodes_p

        if return_all:
            results_nodes_all[(n1, n2)] = nodes_sim_all
            results_nodes_p_all[(n1, n2)] = nodes_sim_p_all

            results_edges_all[(n1, n2)] = edges_sim_all
            results_edges_p_all[(n1, n2)] = edges_sim_p_all


    if return_all:
        return results_edges, results_nodes, results_edges_p, results_nodes_p, results_edges_all, results_nodes_all, results_edges_p_all, results_nodes_p_all

    else:

        return results_edges, results_nodes, results_edges_p, results_nodes_p


def __walks_multi__(nodes, net=None, network_id=None, steps=10, number_of_walks=10, degree=True, start=None, probabilistic=True, weight="weight"):

    performed_walks = {}
    cn = 0
    for node in nodes:
        
        if cn % 100 == 0:
            print("walks for node ", cn, "outof", len(nodes))
        cn = cn + 1

        walks = []
        

        
            
            
        if node in net.nodes():
            
            if not nx.is_isolate(net, node):
                if degree:
                    nw = int(number_of_walks * net.degree[node])
                    print("running walks", nw, "for node", node)
                else:
                    nw = number_of_walks

                    
                walks  = global_distances.perform_random_walks(net, steps=steps, number_of_walks=nw, start=node, probabilistic=probabilistic, weight=weight)
                #print("count nodes / edges in walk")
                #nodes_cnt, edges_cnt = global_distances.rank_walks(net, walks)
                performed_walks[node] = walks


    return performed_walks



def helper_walks_multi(net, nodes, network_id=0, steps=10, number_of_walks=10, degree=True, start=None, probabilistic=True, weight="weight", nr_processes=20):

    """
    Estimates random walks for specific or random select starting node on multiple cores.

    Parameters:
        net (NetworkX graph object): graph to estimate on
        nodes (list): nodes to be investigated.
        network_id (str or int): name of network that can be set custom.
        steps (int): is size of random walk
        number_of_walks (int): how many random walks are performed on net
        degree (boolean): if True then the number of random walks performed for each starting node is dependent on its degree and is estimated as degree*number_of_walks.
        start (node ID): if is None start node is selected at random from nodes else start needs to be node ID as contained in net
        probabilisitc (boolean): if True edge weights are taken into account else all edges are considered equal.  If true then weight needs to be set
        weight (str): edge attribute name as contained in G. Weight is evaluated as a similarity
        nr_processes (int): how many processes should be created.

    Returns:
        walks (dict): where key is Network ID and value is dict where key is starting node and value is list of performed walks.

    """

    performed_walks = {}
    performed_walks[network_id] = {}
    


    #split into chunks and initialize multiprocessing
    chunks = list(np.array_split(np.array(nodes), nr_processes))

    func = partial(__walks_multi__, net=net, network_id=network_id, steps=steps, number_of_walks=number_of_walks, degree=degree, start=start, probabilistic=probabilistic, weight=weight)

    pool = Pool(processes=nr_processes)

    result = pool.map(func, chunks)

    print("terminate multiprocesses")
    pool.terminate()
    pool.join()

    #combine results
    print("merging result ")
   
    temp = {}
    for r in result:
        #print("r", r)
        temp.update(r)
    
        
    
    performed_walks[network_id] = temp


    


    return performed_walks