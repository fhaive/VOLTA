"""
This is an example pipeline on how to estimate the similarity between multiple networks
based on a multitude of random walks
"""

import networkx as nx
import pandas as pd
import csv
import random
import sys
#sys.path.insert(1, '../distances/')
import graphAlgorithms.distances.global_distances as global_distances
import graphAlgorithms.distances.local as local
import graphAlgorithms.simplification as simplification
import graphAlgorithms.distances.trees as trees

import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
import scipy
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial


def helper_walks(networks, nodes, network_ids, steps=10, number_of_walks=10, degree=True, start=None, probabilistic=True, weight="weight"):

    """
    Helper function to estimate for each node per network x random walks of size r

    Input
        networks list of networkx graph objects

        nodes list of all nodes in all networks / or nodes to be investigated/ compared between networks

        if degree then number of walks is dependen on a nodes degree and is estimated as degree*number_of_walks

        the other parameters describe how the random walks should be performed, for a reference refer to its function declaration

    Output
        dict where key is network id as contained in nodes
            each node dict contains dict where key is node as provided in networks
            which value is list of performed walks 
                each walk is a sublist containing the node ids in order as visited by the random walk

        returns dict in same orders for node and edge counts
    """

    performed_walks = {}
    node_counts = {}
    edge_counts = {}
    for net_id in network_ids:
        performed_walks[net_id] = {}
        node_counts[net_id] = {}
        edge_counts[net_id] = {}
        
    cn = 0
    for node in nodes:
        
        if cn % 100 == 0:
            print("walks for node ", cn, "outof", len(nodes))
        cn = cn + 1

        walks = []
        nodes_cnt = {}
        edges_cnt = {}

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
                    #print("count nodes / edges in walk")
                    #nodes_cnt, edges_cnt = global_distances.rank_walks(net, walks)



            #save
            performed_walks[network_id][node] = walks
            #node_counts[network_id][node] = nodes_cnt
            #edge_counts[network_id][node] = edges_cnt


    return performed_walks, node_counts, edge_counts


def helper_get_counts(networks, performed_walks):
    """
    helper function to count number of appearenses of nodes & edges in walks performed on the same starting nodes


    Input
        list of networks

        dict of walks as returned by helper_walks

        

    Output
        edge & node dict containing counts 
        key is network id as ordered in networks
            contains dict where key is starting node to identify walk
                contains dict where either node or edge id is key and value is its counts
    """


    edges = {}
    nodes = {}
    for i in range(len(networks)):
        edges[i] = {}
        nodes[i]= {}
        for s in performed_walks.keys():
            edges[i][s] = []
            nodes[i][s] = []

    for i in range(len(networks)):
        for s in performed_walks.keys():
            walk_list = performed_walks[s][i]
            nodes_cnt, edges_cnt = global_distances.rank_walks(networks[i], walk_list)
            edges[i][s] = edges_cnt
            nodes[i][s] = nodes_cnt

    return nodes, edges




def helper_walk_sim(networks, performed_walks, nodes, network_ids, undirected=True, top=10, return_all=False, ranked=False, nodes_ranked=None, edges_ranked=None):

    """
    helper function to compare random walks based on their similarity of visited nodes/ edges
    estimates for each network pair a correlation score based on the mean of each node pairs x walks

    Input
        list of networks

        dict of walks as returned by helper_walks

        nodes list of all nodes in all networks / or nodes to be investigated/ compared between networks

        network_ids is list of ids as used to create walks

        if return all then for each network pair, its full correlation list are returned
            this can be used to estimate the node pairs with the highest similarity between each other

        other parameters are description parameters of compare_walks()
            please refer to its declaration for more information

        if ranked then it is assumed that ranks are already provided
            provide ranked dicts in nodes_ranked and edges_ranked

    Output
        correlation matrix for ranked nodes & edges
            ranked after their occurenc in random walks (based on specific start nodes)
        and separate matrices for their p values

        if return_all
            then additional 4 dicts are returned, containing the node specific values
                key is tuple of networks ids as ordered in networks
                value is list of scores ordered in order of nodes
                    if node does not occure in one of the networks value is set to None

            dict of edges tau
            dict of nodes tau
            dict of edges p val
            dict of nodes p val
            
            these values can be used to find the "most similar node sub-areas" between two networks
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
        print("n1", n1)
        print("n2", n2)

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

                if not ranked:
                    #compare the 2 walks
                    kendall = global_distances.compare_walks(networks[index[0]], c1, walk2=c2, G2=networks[index[1]], comparison="ranked", undirected=undirected, top=top)
                else:
                    kendall = global_distances.compare_walks(networks[index[0]], [nodes_ranked[index[0]][n1], edges_ranked[index[0]][n1]], walk2=[nodes_ranked[index[1]][n2], edges_ranked[index[1]][n2]], G2=networks[index[1]], comparison="ranked", undirected=undirected, top=top)
                    if n1 != n2:
                        print("1", [nodes_ranked[index[0]][n1][21].keys()])
                        print("2", [nodes_ranked[index[1]][n2][21].keys()])
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


def walks_multi(nodes, net=None, network_id=None, steps=10, number_of_walks=10, degree=True, start=None, probabilistic=True, weight="weight"):

    performed_walks = {}
    cn = 0
    for node in nodes:
        
        if cn % 100 == 0:
            print("walks for node ", cn, "outof", len(nodes))
        cn = cn + 1

        walks = []
        nodes_cnt = {}
        edges_cnt = {}

        
            
            
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



def helper_walks_multi(net, nodes, network_id, steps=10, number_of_walks=10, degree=True, start=None, probabilistic=True, weight="weight", nr_processes=20):

    """
    Helper function to estimate for each node per network x random walks of size r

    Input
        network single networkx graph object

        nodes list of all nodes in all networks / or nodes to be investigated/ compared between networks

        if degree then number of walks is dependen on a nodes degree and is estimated as degree*number_of_walks

        nr_processes how many processes should be used

        the other parameters describe how the random walks should be performed, for a reference refer to its function declaration

    Output
        dict where key is network id as contained in nodes
            each node dict contains dict where key is node as provided in networks
            which value is list of performed walks 
                each walk is a sublist containing the node ids in order as visited by the random walk

        returns dict in same orders for node and edge counts
    """

    performed_walks = {}
    performed_walks[network_id] = {}
    


    #split into chunks and initialize multiprocessing
    chunks = list(np.array_split(np.array(nodes), nr_processes))

    func = partial(walks_multi, net=net, network_id=network_id, steps=steps, number_of_walks=number_of_walks, degree=degree, start=start, probabilistic=probabilistic, weight=weight)

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