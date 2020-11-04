'''
community detection & evaluation algorithms
Functionalities are mainly build on top of CDlib - https://github.com/GiulioRossetti/cdlib

only a few community detection algorithms are exposed here, for more algorithms please refer to the official documentation

Node Partitioning
    node clustering
    fuzzy clustering
    attribute node clustering 

edge partitioning
'''

import pandas as pd
import glob
import sys
import os
import datetime
import math
import networkx as nx
import collections
import matplotlib.pyplot as plt
import random
import treelib as bt
import pickle
import cdlib
from cdlib import algorithms
from operator import itemgetter
import itertools
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy
from collections import Counter
import numpy as np
import pquality as pq
import dask.bag as db
import asyncio
import markov_clustering as mc
import sklearn
from sklearn.cluster import AgglomerativeClustering






#Node clustering


#unweighted

def async_fluid(G, k=5, return_object=True):
    """
    based on the idea of fluids (communitites) 
    propagation-based algorithm, were the desired number of communities needs to be set
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.async_fluid.html#cdlib.algorithms.async_fluid

    Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés, Toyotaro Suzumura T. 
    Fluid Communities: A Competitive and Highly Scalable Community Detection Algorithm.
    
    Input

        a networkx/ igraph object G

        number of desired communities k

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.async_fluid(G,k)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def edmot(G, component_count=2, cutoff=10, return_object=True):
    """
    based on the idea of fluids (communitites) 
    propagation-based algorithm, were the desired number of communities needs to be set
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.edmot.html#cdlib.algorithms.edmot


    Li, Pei-Zhen, et al. “EdMot: An Edge Enhancement Approach for Motif-aware Community Detection.” 
    Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.
    
    Input

        a networkx/ igraph object G

        component_count  Number of extracted motif hypergraph components

        cutoff  Motif edge cut-off value

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.edmot(G,component_count=component_count, cutoff=cutoff )

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def em(G, k=5, return_object=True):
    """
    based on a mixture model
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.em.html#cdlib.algorithms.em

    Newman, Mark EJ, and Elizabeth A. Leicht. Mixture community and exploratory analysis in networks. 
    Proceedings of the National Academy of Sciences 104.23 (2007): 9564-9569.
    
    Input

        a networkx/ igraph object G

        number of desired communities k

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.em(G,k)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def sbm_dl(G, B_min=None, B_max=None, deg_corr=True, return_object=True):
    """
    based on mote carlo heuristic
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.sbm_dl.html#cdlib.algorithms.sbm_dl

    Tiago P. Peixoto, “Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models”, 
    Phys. Rev. E 89, 012804 (2014), DOI: 10.1103/PhysRevE.89.012804
    
    Input

        a networkx/ igraph object G

        B_min  minimum number of communities that are allowed

        B_max  maximum number of communities that are allowed

        if deg_corr  use degree corrected version

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.sbm_dl(G,B_min= B_min, B_max=B_max, deg_corr=deg_corr)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def newman_modularity(G, return_object=True):
    """
    based (maximising) modularity
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.eigenvector.html#cdlib.algorithms.eigenvector

    Newman, Mark EJ. Finding community structure in networks using the eigenvectors of matrices. Physical review E 74.3 (2006): 036104.
    
    Input

        a networkx/ igraph object G

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.eigenvector(G)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def gdmp2(G, min_threshold=0.75, return_object=True):
    """
    based on finding dense subgraphs
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.gdmp2.html#cdlib.algorithms.gdmp2

    Chen, Jie, and Yousef Saad. Dense subgraph extraction with application to community detection. 
    IEEE Transactions on Knowledge and Data Engineering 24.7 (2012): 1216-1230.
    
    Input

        a networkx/ igraph object G

        min_threshold minimum density threshold to control the density of the detected communities

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.gdmp2(G, min_threshold=min_threshold)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def infomap(G, return_object=True):
    """
    based on random walks
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.infomap.html#cdlib.algorithms.infomap

    Rosvall M, Bergstrom CT (2008) Maps of random walks on complex networks reveal community structure. 
    Proc Natl Acad SciUSA 105(4):1118–1123
    
    Input

        a networkx/ igraph object G

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    
    """

    communities = cdlib.algorithms.infomap(G)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def label_propagation(G, return_object=True):
    """
    based network structure based community detection
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.label_propagation.html#cdlib.algorithms.label_propagation

    Raghavan, U. N., Albert, R., & Kumara, S. (2007). 
    Near linear time algorithm to detect community structures in large-scale networks. Physical review E, 76(3), 036106.
    
    Input

        a networkx/ igraph object G

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    
    """

    communities = cdlib.algorithms.label_propagation(G)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def walktrap(G, return_object=True):
    """
    based on random walks
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.walktrap.html#cdlib.algorithms.walktrap

    Pons, Pascal, and Matthieu Latapy. Computing communities in large networks using random walks. 
    J. Graph Algorithms Appl. 10.2 (2006): 191-218.
    
    Input

        a networkx/ igraph object G

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    
    """

    communities = cdlib.algorithms.walktrap(G)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m





###########################################3
#the graphs need to be similarity graphs
#the algorithms prefer putting nodes with strong weights into the same communities
#weighted

def cpm(G, initial_membership=None, weights="weight", node_sizes=None, resolution_parameter=1, return_object=True):
    """
    finds communities of a particular density,
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.cpm.html#cdlib.algorithms.cpm

    Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011). 
    Narrow scope for resolution-limit-free community detection. Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114
    
    Input

        a networkx/ igraph object G

        initial_membership  list of int Initial membership for the partition.
            If None then defaults to a singleton partition.

        weights list of double, or edge attribute Weights of edges. 
            Can be either an iterable or an edge attribute.

        node_sizes list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. 
            Usually this is set to 1 for all nodes, but in specific cases this could be changed. 

        resolution_parameter double >0 A parameter value controlling the coarseness of the clustering. 
            Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.cpm(G, initial_membership=initial_membership, weights=weights, node_sizes=node_sizes, resolution_parameter=resolution_parameter)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def greedy_modularity(G, weights="weight", return_object=True):
    """
    based on modularity
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.greedy_modularity.html#cdlib.algorithms.greedy_modularity

    Clauset, A., Newman, M. E., & Moore, C. 
    Finding community structure in very large networks. Physical Review E 70(6), 2004
    
    Input

        a networkx/ igraph object G

        weights list of double, or edge attribute Weights of edges. 
            Can be either an iterable or an edge attribute.

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.greedy_modularity(G, weight=weights)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def leiden(G, initial_membership=None, weights="weight", return_object=True):
    """
    based on the louvain algorithm
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.leiden.html#cdlib.algorithms.leiden

    Traag, Vincent, Ludo Waltman, and Nees Jan van Eck. 
    From Louvain to Leiden: guaranteeing well-connected communities. arXiv preprint arXiv:1810.08473 (2018).
    
    Input

        a networkx/ igraph object G

        weights list of double, or edge attribute Weights of edges. 
            Can be either an iterable or an edge attribute.

        initial_membership list of int Initial memberships.
            If None then defaults to a singleton partition.

        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.leiden(G, weights=weights, initial_membership=initial_membership)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def louvain(G, weights="weight", resolution=1.0, randomize=False, return_object=True):
    """
    based on modularity
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.louvain.html#cdlib.algorithms.louvain

    Blondel, Vincent D., et al. Fast unfolding of communities in large networks.
    Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.
    
    Input

        a networkx/ igraph object G

        weight list of double, or edge attribute Weights of edges. 
            Can be either an iterable or an edge attribute.

        resolution float, changes the size of the communities, default to 1.

        randomize boolean, randomizes the node evaluation order and the community evaluation order 
            to get different partitions at each call, default False
        
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.louvain(G, weight=weights, resolution=resolution, randomize=randomize)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def rber_pots(G, initial_membership=None, weights="weight", node_sizes=None, resolution_parameter=1, return_object=True):
    """
    optimizes a quality function
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.rber_pots.html#cdlib.algorithms.rber_pots

    Reichardt, J., & Bornholdt, S. (2006). Statistical mechanics of community detection. 
    Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110
    
    Input

        a networkx/ igraph object G

        weights list of double, edge attribute or None. 
            Can be either an iterable or an edge attribute.

        initial_membership  list of int Initial membership for the partition. 
            If None then defaults to a singleton partition

        node_sizes list of int, or vertex attribute Sizes of nodes 
            Usually set to 1 for all nodes, but in specific cases this could be changed.
            e
        resolution_parameter float >0 
            Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.
        
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.rber_pots(G, initial_membership=initial_membership, weights=weights, node_sizes=node_sizes, resolution_parameter=resolution_parameter)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def surprise(G, initial_membership=None, weights="weight", node_sizes=None, return_object=True):
    """
    optimizes a quality function
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.surprise_communities.html#cdlib.algorithms.surprise_communities

    Traag, V. A., Aldecoa, R., & Delvenne, J.-C. (2015). 
    Detecting communities using asymptotical surprise. Physical Review E, 92(2), 022816. 10.1103/PhysRevE.92.022816
    
    Input

        a networkx/ igraph object G

        weights list of double, edge attribute or None. 
            Can be either an iterable or an edge attribute.

        initial_membership  list of int Initial membership for the partition. 
            If None then defaults to a singleton partition

        node_sizes list of int, or vertex attribute Sizes of nodes 
            Usually set to 1 for all nodes, but in specific cases this could be changed.
            
        
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.surprise_communities(G, initial_membership=initial_membership, weights=weights, node_sizes=node_sizes)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m



def alisas_communities(H, weights="weight", t=0.2, max_iter=1000, by_degree=True, r=5, std=0.2, min_community=10, graph=False):
    """
    algorithm that tries to find weak links between node groups, while ensuring that no new isolates occure

    for each node a probabilistic selection of neighbors is run for r rounds based on their edge weights
        neigbors with less than t occurenses will be disconnected but only if this link is selected as a weak link
            in both directions
        based on this it is ensured that a small nodegroup/ single node connected to a high degree node via a weak link is NOT
            disconnected, since this is the only community it belongs to

    this algorithm is tuned to similarity graphs, especially where edge weights encode aa strong association between nodes and where it
        is preferred that weak links are kept in the same community if there is no "better option" to assign them

    the algorithm will stop when max_iter is reached or no edges can be removed anymore
    
    Input

        a networkx/ igraph object G

        weights edge attribute. 
            
        t treshold on which edges are "selected as weak" and to be removed if they occure <= in fraction
            of selected neighbors 

        max_iter max number of iterations if a convergence is not reached beforehand

        by_degree if true number of samplings for each node is based on its degree
            each node will samper from its neighbor list r*node_degree times

            if false then each nodes neighbors will be sampled r times

        r how often a nodes neighbors are sampled, based on by_degree
            
        std based on which standard deviation in the neighboring edges selection (after sampling)
            removal should be performed
            this avoides that weak edges are removed when all edges are counted equal

        min_community number of nodes of min community size
            if pre existing communities of that size exist some communities still may be smaller than min_community
            but no new ones of that size will be created
            if None not taken into account

        graph if true new graph object is also returned

    Output

        dict, were each node ID is key and value is list of communities

        if graph
            dict, networkx graph object

        

    """
    G = H.copy()
    init_edges = len(G.edges())
    cnt = 0
    converged = False
    
    print("min community is", min_community)

    while ((cnt < max_iter) and (not converged)):
        weak_edges = {}
        ncnt = 0
        for n in G.nodes():
            if ncnt % 1000 == 0:
                print("node", ncnt, "out of", len(G.nodes()))

            ncnt = ncnt + 1
            ngbs = list(G.neighbors(n))

            #if node belongs to a community larger than min_communities
            
            #if nodes has neighbors
            if len(ngbs) > 0:
                #get all its edge weights in same order as ngbs
                temp_weights = []
                for ng in ngbs:
                    temp_weights.append(G[n][ng][weights])
                
                #do probabilistic sampling
                if by_degree:
                    k = r * G.degree(n)
                    selected = random.choices(ngbs, weights=temp_weights, k=k)
                else:
                    k = r
                    selected = random.choices(ngbs, weights=temp_weights, k=k)

                #for each choice get their fraction to estimate its weak links
                temp_cnts = Counter(selected)
                if len(temp_cnts.values()) > 1: #dont need an else case since std will be 0 anyways
                    if statistics.stdev(temp_cnts.values()) > std:
                        for key in temp_cnts.keys():
                            if (temp_cnts[key] / k) <= t:
                                #possible removable edge
                                k1 = (n, key)
                                k2 = (key, n)
                                if k1 not in weak_edges.keys() and k2 not in weak_edges.keys():
                                    weak_edges[(n, key)] = 1
                                else:
                                    if k1 in weak_edges.keys():
                                        cur = weak_edges[k1]
                                        weak_edges[k1] = cur + 1
                                    else:
                                        cur = weak_edges[k2]
                                        weak_edges[k2] = cur + 1
                    else:
                        print("std too small no edges are removed for", n)
            
            #print("counts", temp_cnts, k)
            #print("weak edges", weak_edges)
        #find and remove weak edgegs
        to_remove = []
        cntk = 0
        for key in weak_edges:
            if cntk % 100 == 0:
                print("removing edge with min community", cntk, "out of", len(weak_edges))
            cntk = cntk + 1
            if weak_edges[key] > 1:
                #only remove if removal would not create too small communities
                if min_community is not None:
                    T = G.copy()
                    T = T.subgraph(nx.node_connected_component(T, key[0]))
                    T=nx.Graph(T)
                    T.remove_edges_from([key])
                    if len(sorted(nx.connected_components(T), key=len, reverse=False)[0]) >= min_community:
                        #print(sorted(nx.connected_components(T), key=len, reverse=False))
                        #print(key)
                        to_remove.append(key)
                        G.remove_edges_from([key])
                    T = None
                else:
                    to_remove.append(key)

        #remove edges
        if min_community is None:
            print("removing weak edges", len(to_remove), "in iteration", cnt)
            G.remove_edges_from(to_remove)
        cnt = cnt +1
        if len(to_remove) == 0:
            converged = True
            break

    if converged:
        print("graph has converged after iteration", cnt)
        
    else:
        print("graph has not converged but max iterations was reached")

    print("new number of edges", len(G.edges()), "our of originally ", init_edges)

    #create partition dict based on all subgraphs within the current graph
    com_temp = list(nx.connected_component_subgraphs(G))

    communities = convert_graph_to_community_dict(com_temp)

    if graph:
        return communities, G
    else:
        return communities

def disconnect_high_degree(G, nodes=10, percentage=None, weight="weight"):
    """
    function that takes the x highest degree nodes in the graph and removes all edges to disconnect them
    this can be used as an input for a community detection function

    Input
        networkx graph H

        number of high degree nodes that should be disconnected

        percentage percentage of high degree nodes that should be disconnected
        percentage needs to be in [0,1]

        either number of percentage needs to be set to None, if both are not None nodes is used

        if weight is not None then degree is estimated based on edge weight
            degree is sum of edge weights adjacent to the node

    Output new disconnected graph
    """
    #G = H.copy()

    #get high degree nodes
    degrees = {node:val for (node, val) in G.degree(weight=weight)}

    #sort dict
    s = {k: v for k, v in sorted(degrees.items(), key=lambda item: item[1], reverse=True)}

    #select to be disconnected nodes
    if nodes is not None:
        n = list(s.keys())[:nodes]
    elif percentage is not None:
        t = int(len(list(s.keys())) * percentage)
        n = list(s.keys())[:t]


    print("find edges to separate nodes", len(n))

    cnt = 0

    pcnt = 0
    #create all possible node pairs
    for pair in itertools.combinations(n, 2):
        if pcnt % 100 == 0:
            print("remoinving edges for pair", pcnt)
        pcnt = pcnt +1

        edges = nx.minimum_edge_cut(G, s=pair[0], t=pair[1])
        #remove edges
        G.remove_edges_from(edges)
        cnt = cnt + len(edges)


    print("in total removed edges", cnt)

    return G



def disconnect_high_degree_nodes(H, nodes=10, percentage=None, weight="weight", graph = False):
    """
    function that takes the x highest degree nodes in the graph and removes all edges to disconnect them
    

    Input
        networkx graph H

        number of high degree nodes that should be disconnected

        percentage percentage of high degree nodes that should be disconnected
        percentage needs to be in [0,1]

        either number of percentage needs to be set to None, if both are not None nodes is used

        if weight is not None then degree is estimated based on edge weight
            degree is sum of edge weights adjacent to the node

    Output 
        dict, were each node ID is key and value is list of communities

        if graph
            dict, networkx graph object
    """

    H = disconnect_high_degree(H, nodes=nodes, percentage=percentage, weight=weight)

    com_temp = list(nx.connected_component_subgraphs(H))

    communities = convert_graph_to_community_dict(com_temp)

    if graph:
        return communities, H
    else:
        return communities


def girvan_newman(G, valuable_edge="max_weight", k=1, is_leve=False, attribute="weight", w_max=True):
    """
    Networkx girvan newman implementation
    removes "most valuable edge" from the graph to partition it
    https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html

    Input

        networkx graph G

        valuable_edge  Function that takes a graph as input and outputs an edge. 
            options are:
                max_weight : based on max edge weight
                min_weight : based on min edge weight
                betweenness : based on betweenness centrality
                current_flow_betweenness :  based on current flow betweenness centrality
                load : based on load centrality

        attribute 
            edge attribute to be used for max_weight/ min_weight

        w_max
            if True the highest scoring edge is returned else the lowest scoring edge
            only relevant if valuable_edge is betweenness, current_flow_betweenness or load

        k int, levels of the algorithm it should return results to
            i.e. 1 - top most level, 3 - 3 top most levels or exactly the specified level
            if None then all levels are returned

        if is level this indicates that k is level index & that specific level is returned
            if false then k is seen to be a "range" and k top levels are returned
            level index starts at 0

    Output

        dict, were each node ID is key and value is list of communities it belongs to at each level of the algorithm
        first value is top most level & last value is most granular level
    """

    if valuable_edge == "max_weight":
        #this is a very shitty implementation but networkx does not allow to provide the edge function with parameters
        def temp(G, attribute=attribute):
            return by_weight(G, w_max=True, attribute=attribute)


    elif valuable_edge == "min_weight":
        def temp(G, attribute=attribute):
            return by_weight(G, w_max=False, attribute=attribute)

    elif valuable_edge == "betweenness":
        def temp(G, attribute=attribute, w_max=w_max):

            return by_centrality(G, w_max=w_max, attribute=attribute, type="betweenness")

    elif valuable_edge == "current_flow_betweenness":
        def temp(G, attribute=attribute, w_max=w_max):

            return by_centrality(G, w_max=w_max, attribute=attribute, type="current_flow_betweenness")

    elif valuable_edge == "load":
        def temp(G, attribute=attribute, w_max=w_max):

            return by_centrality(G, w_max=w_max, attribute=attribute, type="load")

    else:
        print("not implemented, will set to default of betweenness centrality")
        temp = None


    community = nx.algorithms.community.centrality.girvan_newman(G, most_valuable_edge=temp)



    d = {}

    
    if k is not None:
        for com in itertools.islice(community, k):
            counter = 0
            
            for c in com:
                for node in c:
                    if is_leve:
                        if counter == k:
                            if node in d.keys():
                                temp = d[node]
                                temp.append(counter)
                                d[node] = temp
                            else:
                                d[node] = [counter]
                    else:
                        if node in d.keys():
                                temp = d[node]
                                temp.append(counter)
                                d[node] = temp
                        else:
                                d[node] = [counter]

                counter = counter + 1
        return d

    else:
        for com in community:
            counter = 0
            for c in com:
                for node in c:
                    if node in d.keys():
                        temp = d[node]
                        temp.append(counter)
                        d[node] = temp
                    else:
                        d[node] = [counter]
                counter = counter + 1
        return d

def agglomerative_clustering(G, is_distance=True, linkage="ward", distance_threshold=0.2):
    """
    clusters the adjacency matrix (which needs to contain similarity or distance weights between nodes)

    Input
        networkx graph G

        if is distance values are used direct
        else its assumed to be a similarity matrix and values are converted into distance values

        linkage input to sklearn agglomerativeClustering
            “ward”, “complete”, “average”, “single”



    Output
        community dict
    """
    nodes = list(G.nodes())
    
    print("creating adjacency matrix")
    A = nx.to_numpy_matrix(G, nodelist=nodes)

    if not is_distance:
        print("convert to distance matrix")
        T = np.ones(A.shape)
        A = T - A
        T = None
    print("start clustering")
    agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage = linkage, distance_threshold=distance_threshold)
    clus = agg.fit_predict(A)  
    #print(clus)
    #convert to community dict
    com = {}
    for node in nodes:
        com[node] = []


    for i in range(len(clus)):
        com[nodes[i]] = [clus[i]]

    return com





def markov_clustering(G, inflation=1.1):
    """
    creates markov clustering on adjacency matrix

    Input 
        networkx graph object G

        inflation - clustering parameter

    Output
        clustering dict
    """
    nodes = list(G.nodes())
    print("creating adjacency matrix")
    A = nx.to_numpy_matrix(G, nodelist=nodes)

    print("run clustering")
    result = mc.run_mcl(A, inflation=inflation)
    clusters = mc.get_clusters(result)

    #convert to community dict
    com = {}
    for node in nodes:
        com[node] = []

    for i in range(len(clusters)):
        c = clusters[i]
        for n in c:
            #get node name
            
            nd = nodes[n]
            com[nd] = [i]

    return com





############################################################33
################################################################

#overlapping communities

def angel(G, treshold=0.5, min_community_size=3, return_object=True):
    """
    node centrinc bottom-up
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.angel.html#cdlib.algorithms.angel

    Rossetti, Giulio. “Exorcising the Demon: Angel, Efficient Node-Centric Community Discovery.”
     International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.
    
    Input

        a networkx/ igraph object G

        treshold merging treshold, float between 0 & 1

        min_community_size int 

        node_sizes list of int, or vertex attribute Sizes of nodes 
            Usually set to 1 for all nodes, but in specific cases this could be changed.
            
        
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.angel(G, treshold, min_community_size=min_community_size)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def lemon(G,  min_com_size=20, max_com_size=50, expand_step=6, subspace_dim=3, walk_steps=3, biased=False, return_object=True):
    """
    based on local expansion
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.lemon.html#cdlib.algorithms.lemon

    Yixuan Li, Kun He, David Bindel, John Hopcroft Uncovering the small community structure in large networks: A local spectral approach. 
    Proceedings of the 24th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2015.
    
    Input

        a networkx/ igraph object G

       
        min_com_size the minimum size of a single community in the network

        max_com_size the maximum size of a single community in the network

        expand_step the step of seed set increasement during expansion process

        subspace_dim dimension of the subspace; choosing a large dimension is undesirable because it would increase the computation cost of generating local spectra 

        walk_steps the number of step for the random walk

        if biased random walk startsfrom seed nodes
            
        
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.lemon(G, seeds=list(G.nodes()), min_com_size=min_com_size, max_com_size=max_com_size, expand_step=expand_step, subspace_dim=subspace_dim, walk_steps=walk_steps, biased=biased)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m



def congo(G, number_communities=3, height=2, return_object=True):
    """
    optimization of conga, which is based on girvan newman
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.congo.html#cdlib.algorithms.congo

    but seems to be extremly slow also for small graphs

    Gregory, Steve. A fast algorithm to find overlapping communities in networks. 
    Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2008.
    
    Input

        a networkx/ igraph object G

        number_communities int, number of desired communities

        height int, longest shortest path
            
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.congo(G, number_communities, height=height)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def ego_networks(G, level=1, return_object=True):
    """
    ego network based
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.ego_networks.html#cdlib.algorithms.ego_networks

    
    Input

        a networkx/ igraph object G

        level communities with neighbors of distance <= level
            
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.ego_networks(G, level=level)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


#########################################################
##weighted

def lais2(G, edge_weight="weight"):
    """
    adjusted implementation of the lais2 algorithm, where node importance is based on edge weights
    based on density
    https://github.com/kritishrivastava/CommunityDetection-Project2GDM/blob/master/main.py

    Baumes, Jeffrey, Mark Goldberg, and Malik Magdon-Ismail. 
    Efficient identification of overlapping communities. 
    International Conference on Intelligence and Security Informatics. Springer, Berlin, Heidelberg, 2005.

    Input

        a networkx/ igraph object G

        edge_weight str edge attribute to be used for node ranking, based on pagerank
            if None pagerank runs with equal weights
            this is a modification to the original algorithm
            
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each node ID is key and value is list of communities

    """

    initalClusters = LA(G)
    #Get final clusters using Improved Iterative Scan Algorithm
    finalClusters = []
    initalClustersWithoutDuplicates = []
    for cluster in initalClusters:
        cluster = sorted(cluster)
        if cluster not in initalClustersWithoutDuplicates:
            initalClustersWithoutDuplicates.append(cluster)
            updatedCluster = IS2(cluster,G)
            finalClusters.append(updatedCluster.nodes())
    d = {}
    #finalClustersWithoutDuplicates = []
    counter = 0
    for cluster in finalClusters:
        cluster = sorted(cluster)
        for node in cluster:
            d[node] = [counter]
        #if cluster not in finalClustersWithoutDuplicates:
        #    finalClustersWithoutDuplicates.append(cluster)
        counter = counter + 1  

    return d



#########################################################33
###########################################################

#fuzzy communities


def fuzzy_rough(G, theta= 1, eps=0.5, r=2, return_object=True):
    """
    nodes are assigend to each community based on their probability
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.frc_fgsn.html#cdlib.algorithms.frc_fgsn

    Kundu, S., & Pal, S. K. (2015). Fuzzy-rough community in social networks. Pattern Recognition Letters, 67, 145-152.

    Input

        a networkx/ igraph object G

        theta float, community density coefficient

        eps float in [0,1] coupling coefficient of the community, small values result in smaller communities

        r int, radius of the granule
            
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each edge ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.frc_fgsn(G, theta, eps, r)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


################################################333
######################################################3

#edge clustering

def agdl(G, number_communities=3, number_neighbors=3, kc=3, a=1, return_object=True):
    """
    agglomerative clustering of edges
    https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.agdl.html

    Zhang, W., Wang, X., Zhao, D., & Tang, X. (2012, October). Graph degree linkage: Agglomerative clustering on a directed graph. 
    In European Conference on Computer Vision (pp. 428-441). Springer, Berlin, Heidelberg.

    Input

        a networkx/ igraph object G

       number_communities int, 

        number_neighbors int, Number of neighbors to use for KNN

        kc int, size of the neighbor set for each cluster

        a – float
            
        if return_object is True then the resulting CDlib NodeClustering object is returned, for all options refer to 
            https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html
            this allows to retrieve a multitude of community fitness parameters

    Output

        dict, were each edge ID is key and value is list of communities

        if return_object then also the whole community object is returned

    """

    communities = cdlib.algorithms.agdl(G, number_communities=number_communities, number_neighbors=number_neighbors, kc=kc, a=a)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

#evaluate partion quality based on pquality package




###########################################################3
#############################################################3

#estimate clusering consensus

def create_initial_consensus(graph, communities, thresh = 0.2):
    """
    creates initial consensus graph out of provided community detection algorithms
    can be used as input of fast_consensus

    networkx graph graph, provide copy of graph if needed

        communities list of initial clustering based on x community detection algorithms
            values of list are assumed to be dict in format as returned by the community detection algorithms

    thresh if a node pair occures less than tresh in a fraction of the communities they will not be added to the same community

    """

    for u,v in graph.edges():
            graph[u][v]['weight'] = 0.0

    consensus_com, consensus_graph = create_consensus_graph_dict(graph, communities)

    

    print("transforming initial consensus graph in its communities based on selected treshold")
    remove_edges = []
    for u,v in consensus_graph.edges():
        if consensus_graph[u][v]['weight'] / len(communities) < thresh:
            remove_edges.append((u, v))
    consensus_graph.remove_edges_from(remove_edges)
    print("initial removed edges", len(remove_edges))

    com_temp = list(nx.connected_component_subgraphs(consensus_graph))

    initial_communities = convert_graph_to_community_dict(com_temp)

    return initial_communities, consensus_com, consensus_graph


def fast_consensus(graph, communities, algorithms = [], parameters=[], thresh = 0.2, delta = 0.02, max_iter=100, initial=None):
    """
    this is an adapted implementation of the fast_consensus algorithm
    https://github.com/adityat/fastconsensus/blob/master/fast_consensus.py
    https://arxiv.org/pdf/1902.04014.pdf

    

    Input
        networkx graph graph, provide copy of graph if needed

        communities list of initial clustering based on x community detection algorithms
            values of list are assumed to be dict in format as returned by the community detection algorithms

        algorithms list of algorithms to be used during the consensus estimation
            this do not need to be the same algorithms/ or same number of algorithms used to compute the initial clustering
            if an algorithm should be run multiple times, then add it multiple times to the list

        parameters list dicts of parameters as to be used in algorithms
            needs to be in same order as algorithms

        thresh if a node pair occures less than tresh in a fraction of the communities they will not be added to the same community

        delta between 0.02 & 0.1
            end condition -  defines how granular the final clustering should be

        max_iter max number of iterations if consensus is not reached

        if initial is not None 
            provide initial consensus graph to be used instead of the communities contained in communities
            in that case communities can be set to None

    Output
        returns disconected components as communities
        dict were each key is a node ID and its value is a list contianing the communities the node is part of

        returns dict of initial consensus communities where weight is number of community algorithms support this clustering

        returns community dict of initial consensus as community dict, where weak edges have been removed based on the initial treshold

    
    """
    #graph = G.copy()
    L = graph.number_of_edges()
    #N = G.number_of_nodes()

    print("start fast consensus")
    #create initial consensus graph
    if initial is None:
        print("no initial graph provided create from communities")
        for u,v in graph.edges():
            graph[u][v]['weight'] = 0.0

        consensus_com, consensus_graph = create_consensus_graph_dict(graph, communities)

        

        print("transforming initial consensus graph in its communities based on selected treshold")
        remove_edges = []
        for u,v in consensus_graph.edges():
            if consensus_graph[u][v]['weight'] / len(communities) < thresh:
                remove_edges.append((u, v))
        consensus_graph.remove_edges_from(remove_edges)
        print("initial removed edges", len(remove_edges))

        com_temp = list(nx.connected_component_subgraphs(consensus_graph))

        initial_communities = convert_graph_to_community_dict(com_temp)

        graph = consensus_graph.copy()

        print("giving initial graph free")
        consensus_graph = None

    else:
        graph = initial.copy()

        print("giving initial graph free")
        initial = None
        initial_communities = None
        consensus_com = None


    print("start convergence on  reduced graph")

    cnt = 0
    converged = False
    while((not converged) and (cnt < max_iter)):
        cnt = cnt + 1

        nextgraph = graph.copy()
        
        #for u,v in nextgraph.edges():
        #    nextgraph[u][v]['weight'] = 0.0

        communities = []
        for a in range(len(algorithms)):
            algo = algorithms[a]
            param = parameters[a]
            
            print("estimating communities ", algo.__name__)
            communities.append(algo(nextgraph, **param))

        #estimate communities based on desired community detection algorithm
        #communities need to be represented as nested lists

        

        #this updates the graph object of the consensus clustering
        tmp , nextgraph = create_consensus_graph_dict(graph, communities)
                    
        tmp = None
        print("before removal", len(nextgraph.edges()))
        L = nextgraph.number_of_edges()

        remove_edges = []
        for u,v in nextgraph.edges():
            if nextgraph[u][v]['weight'] / len(communities) < thresh:
                remove_edges.append((u, v))
        nextgraph.remove_edges_from(remove_edges)
        print("after removal", len(nextgraph.edges()))

        
        #this now randomly adds some new edges, for edges that are not existing in the consensus graph anymore
        #this provides triadic closure
        #helps to improve the consensus clustering
        print("adding random edges for", L/4)
        cntx = 0
        for _ in range(int(L/4)):
            if cntx % 10000 == 0:
                print("checking random", cntx)
            cntx = cntx + 1
            node = np.random.choice(nextgraph.nodes())
            neighbors = [a[1] for a in nextgraph.edges(node)]

            if (len(neighbors) >= 2):
                a, b = random.sample(set(neighbors), 2)

                if not nextgraph.has_edge(a, b):
                    nextgraph.add_edge(a, b, weight = 0)

                    for i in range(len(communities)):
                        if a in communities[i].keys() and b in communities[i].keys():
                            nextgraph[a][b]['weight'] += (1/len(communities))
        

        graph = nextgraph.copy()

        if check_consensus_graph(nextgraph, n_p = len(communities), delta = delta):
            converged = True
            break
        if cnt >= max_iter:
                break
            

    if not converged:
        print("max iter has been reached but consensus graph has not converged")

    #i think it would be best to convert the consensus graph into its communities based on its connected components
    com_temp = list(nx.connected_component_subgraphs(graph))

    communities = convert_graph_to_community_dict(com_temp)

    return communities, consensus_com, initial_communities


    


#########################33
############################3

#compare communities/ evaluate communities

def get_number_of_communities(communities):
    """
    function to estimate the number of detected communities

    Input
        dict as returned by the community functions

    Output
        int, number of detected communities
    """

    c = list(communities.values())
    c_flat = [item for sublist in c for item in sublist]

    return max(c_flat) + 1 #return plus 1 since community indexing starts with 0

def convert_communities(communities):
    """
    helper function to convert community dict as returned by functions
    into dict, with community id as list and nodes as values

    Input
        communities as returned by community functions, dict

    Output
        dict where community id is key and values is list of node ids that are part of this community
    """
    c = list(communities.values())
    c_flat = [item for sublist in c for item in sublist]
    #all community ids
    c_flat = list(dict.fromkeys(c_flat))

    nodes = list(communities.keys())


    dict_communities = {}

    for node in nodes:
        #get its community
        com = communities[node][0]
        if com in dict_communities:
            dict_communities.setdefault(com, []).append(node)
        else:
            dict_communities[com] = [node]

    return dict_communities

def get_number_of_nodes_community(communities, in_detail = False):
    """
    function to estimate the distribution of nodes per community

    Input 
        dict as returned by the community functions

        if in_detail then for each community its number of nodes is reported

    Output
        dict containing distribution parameters of number of nodes in communities

        if in_detail then a dict containing number of nodes per community is returned as well

    """

    dict_communities = convert_communities(communities)

    number_of_nodes = []
    dict_nodes = {}

    for key in dict_communities:
        nr_nodes = len(dict_communities[key])
        dict_nodes[key] = nr_nodes
        number_of_nodes.append(nr_nodes)

    #get distributional values
    mean = statistics.mean(number_of_nodes)
    median = statistics.median(number_of_nodes)
    std = statistics.stdev(number_of_nodes)
    kurt = kurtosis(number_of_nodes)
    sk = skew(number_of_nodes)

    if in_detail:

        return {"mean":mean, "median":median, "std":std, "kurt":kurt, "skewness":sk}, dict_nodes
    else:
        return {"mean":mean, "median":median, "std":std, "kurt":kurt, "skewness":sk}



def average_internal_degree(communities, G):
    """
    estimates average internal degree of a communities member nodes
    community specific & reports an average score
    degree within a community should be high, indicating a thight connectivity between members

    Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). 
    Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.average_internal_degree(subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}


def internal_edge_density(communities, G):
    
    """
    estimates internal edge density (existing edges out of possible edges)
    community specific & reports an average score
    within a community the score should be high, indicating thight connectivity

    Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004).
    Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    


    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.internal_edge_density(subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def triangle_ratio(communities, G):
    
    """
    estimates triangle participation ratio - fraction of nodes within a community that belong to a triangle - 
    community specific & reports an average score
    a high score indicates thighly connected nodes within a community

    Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. 
    Knowledge and Information Systems 42(1), 181–213 (2015)
    

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.triangle_participation_ratio(subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def internal_edge_count(communities, G):
    
    """
    estimates number of edges inside the community
    community specific & reports an average score

    Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. 
    Proceedings of the National Academy of Sciences, 101(9), 2658-2663.

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.edges_inside(subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def conductance(communities, G):
    
    """
    estimates conductance, which is the fraction of edges leaving the community
    community specific & reports an average score
    a small fraction indicates a "more closed up " community

    Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.conductance(G, subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def max_outgoing_edge_fraction(communities, G):
    
    """
    estimates fraction of a nodes's edges that leave the community
    community specific & reports an average score
    a small value indicates a "closed" community

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: 
    Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.max_odf(G, subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def mean_outgoing_edge_fraction(communities, G):
    
    """
    estimates average of a nodes's edges that leave the community
    community specific & reports an average score
    a small value indicates that most nodes only have "internal edges" and are weakly connected to the outside

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: 
    Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.avg_odf(G, subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def fraction_of_weak_members(communities, G):
    
    """
    estimates fraction of nodes within a community that have fewer edges inwards than outwards
    community specific & reports an average score
    a low score indicates that the community mainly consists out of "strong members", while a low value indicates
        that a large amount of members are more tightly connected to the "outside" than the inside

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: 
    Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.flake_odf(G, subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def community_modularity(communities, G):
    
    """
    estimates a communities modularity - fraction of edges within a community w.r.t to the expected number of such edges based on a null model - 
    community specific & reports an average score
    a high score indicates that the communiy structure "is higher than changes suggests" & therefore more likely to have arisen due 
        to external factors

    1. Newman, M.E.J. & Girvan, M. `Finding and evaluating community structure in networks. 
    Physical Review E 69, 26113(2004).

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on        

    Output
        modularity score of partition
    """
    com = {}
    for key in communities:
        com[key] = communities[key][0]


    val = pq.PartitionQuality.community_modularity(com, G)
    

    return val

def modular_density(communities, G):
    
    """
    estimates modularity while considering community size
    community specific & reports an average score

    Li, Z., Zhang, S., Wang, R. S., Zhang, X. S., & Chen, L. (2008). Quantitative function for community detection. 
    Physical review E, 77(3), 036109.
    

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        density modularity score for whole partitioning

        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    val = 0
    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        degree = []
        graph_degree = []

        for node in nodes:
            degree.append(subgraph.degree(node))
            graph_degree.append(G.degree(node) - subgraph.degree(node))

        if len(nodes) > 0:
            new_val = (1/len(nodes)) * (statistics.mean(degree) - statistics.mean(graph_degree))
            val = val + new_val

       
        quality.append(new_val)
        quality_dict[com] = new_val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return val, quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}


def fraction_of_median_degree(communities, G):
    
    """
    estimates fraction of nodes that have a higher degree than the median degree within the community
    community specific & reports an average score
    a high score indicates that the community contains hub nodes

    Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. 
    Knowledge and Information Systems 42(1), 181–213 (2015)

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.fraction_over_median_degree(subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def outgoing_edges(communities, G):
    
    """
    estimates fraction of edges leaving the community (expansion)
    community specific & reports an average score
    a small score indicates a "closed up community"

    Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). 
    Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = pq.PartitionQuality.expansion(G, subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def cut_ratio(communities, G, normalized=True):
    
    """
    estimates the (normalized) cut ratio - fraction of edges (of all possible edges) that leave the community
    community specific & reports an average score
    a small value is prefered

    1. Fortunato, S.: Community detection in graphs. Physics reports 486(3-5), 75–174 (2010)

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        if normalized return normalized cut ratio else cut ratio is estimated

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)
        if normalized:
            val = pq.PartitionQuality.normalized_cut(G, subgraph)
        else:
            val = pq.PartitionQuality.cut_ratio(G, subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def community_density_to_graph(communities, G):
    """
    estimates the community density with respect to the complete graph density
    community specific & reports an average score
    a high value indicates a "strongly connected" community w.r.t. the whole graph

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}
    graph_density = nx.density(G)

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = nx.density(subgraph) / graph_density
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}


def community_average_shortest_path(communities, G, weight="weight"):
    
    """
    estimates average shortest path within a community
    community specific & reports an average score
    a low value indicates a strongly connected community

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        weight name of edge attribute to be considered
            if None all edges are considered to be equal

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = nx.average_shortest_path_length(subgraph, weight=weight)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def community_average_shortest_path_fraction(communities, G, weight="weight"):
    
    """
    estimates average shortest path within a community w.r.t avg shortest path in the whole graph
    community specific & reports an average score
    a low value indicates that nodes within a community are stronger connected that "the main graph"

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        weight name of edge attribute to be considered
            if None all edges are considered to be equal

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G, weight=weight)

    else:
        print("avg shortest paths can only be estimated on connected graph, GCC will be used instead")
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        
        avg_path = nx.average_shortest_path_length(G0, weight=weight)

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        if nx.is_connected(subgraph):
            avg_path_sub = nx.average_shortest_path_length(subgraph, weight=weight)

        else:
            print("avg shortest paths can only be estimated on connected graph, GCC will be used instead for subgraph")
            Gcc = sorted(nx.connected_components(subgraph), key=len, reverse=True)
            G0 = subgraph.subgraph(Gcc[0])
            
            avg_path_sub = nx.average_shortest_path_length(G0, weight=weight)


        val = avg_path_sub / avg_path
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def mean_edge_weight_fraction(communities, G, weight="weight"):
    
    """
    estimates average edge weight within a community w.r.t edge weight in the whole graph
    community specific & reports an average score
    a low value indicates that nodes within a community are stronger connected that "the main graph"

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        weight name of edge attribute to be considered
            if None all edges are considered to be equal

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    
    global_weights = []
    for edge in G.edges():
        global_weights.append(G[edge[0]][edge[1]][weight])

    mean_global_weight = statistics.mean(global_weights)

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        local_weights = []
        for edge in subgraph.edges():
            local_weights.append(subgraph[edge[0]][edge[1]][weight])

        if len(local_weights) > 1:
            mean_local_weight = statistics.mean(local_weights)
            val = mean_local_weight / mean_global_weight
            quality.append(val)
            quality_dict[com] = val
        elif len(local_weights) == 1:
            mean_local_weight = local_weights[0]   
            
            val = mean_local_weight / mean_global_weight
            quality.append(val)
            quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}


def mean_edge_weight(communities, G, weight="weight"):
    
    """
    estimates average edge weight within a community 
    community specific & reports an average score
    a low value indicates that nodes within a community are stronger connected that "the main graph"

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        weight name of edge attribute to be considered
            if None all edges are considered to be equal

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}


    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        local_weights = []
        for edge in subgraph.edges():
            local_weights.append(subgraph[edge[0]][edge[1]][weight])

        if len(local_weights) > 1:
            mean_local_weight = statistics.mean(local_weights)
            quality.append(mean_local_weight)
            quality_dict[com] = mean_local_weight
        elif len(local_weights) == 1:
            mean_local_weight = local_weights[0]


        
            quality.append(mean_local_weight)
            quality_dict[com] = mean_local_weight

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}


def hub_dominace(communities, G):
    
    """
    estimates hub dominace
        which is the ratio of the communities most connected node w.r.t. to the theoretically highest possible degree within a community
    community specific & reports an average score
    a high value indicates that the community is at least tightly connected around a hub node

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        if len(nodes) > 1:

            val = max(x[1] for x in list(nx.degree(subgraph))) / (len(nodes) -1)
        else:
            print("community only contains 1 node, its hub dominace is set to 1")
            val = 1
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

def clustering_coefficient(communities, G):
    
    """
    estimates average clustering coefficient of the communities
    community specific & reports an average score
    a high value indicates a strong connectivity
        a comple graph has a CC of 1

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = nx.average_clustering(subgraph)
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}


def node_embeddedness(communities, G):
    
    """
    estimates community embeddedness 
        which is a nodes  mean degree within the community with respect to its overall degree in the graph
    a high value indicates that a node has most of its edges within the community instead of outside

    community specific & reports an average score

    Input 
        community dict as returned by community functions

        networkx graph object that community was detected on

        

    Output
        dict containing score for each community

        dict containing mean distributional scores
            if graph contains less than 2 communities distributional parameters of None are returned
    """

    dict_communities = convert_communities(communities)

    quality = []
    quality_dict = {}

    

    for com in dict_communities.keys():
        nodes = dict_communities[com]

        #get subgraph of community
        subgraph = nx.subgraph(G, nodes)

        val = statistics.mean([float(nx.degree(subgraph)[n]) / nx.degree(G)[n] for n in nodes])
        quality.append(val)
        quality_dict[com] = val

    #estimate mean values
    if len(quality) > 1:
        mean = statistics.mean(quality)
        median = statistics.median(quality)
        std = statistics.stdev(quality)
        kur = kurtosis(quality)
        sk = skew(quality)

    else:
        mean = None
        median = None
        std = None
        kur = None
        sk = None

    return quality_dict, {"mean":mean, "median":median, "std":std, "kurt":kur, "skewness":sk}

###############################
################################3
############################
#helper functions




def by_weight(G, w_max=True, attribute="weight"):
    """
    helper function to find the most valuable edge for girvan newman based partitioning

    Input

        networkx graph G

        w_max
            if true edge with the highest weight attribute is returned
            else edge with the smallest edge attribute is returned

        attribute
            str of edge attribute name
    """
    if w_max:
        u, v, w = max(G.edges(data=attribute), key=itemgetter(2))
    else:
        
        u, v, w = min(G.edges(data=attribute), key=itemgetter(2))

    return (u, v)


def by_centrality(G, w_max=True, attribute="weight", type="betweenness"):
    """
    helper function to find the most valuable edge for girvan newman based partitioning
    returns the edge with the highest/ lowest betweenness score

    Input

        networkx graph G

        w_max
            if true edge with the highest weight attribute is returned
            else edge with the smallest edge attribute is returned

        attribute
            str of edge attribute name

        type
            what centrality measure should be used
            options
                betweenness : based on betweenness centrality
                current_flow_betweenness :  based on current flow betweenness centrality
                load : based on load centrality 

    """
    if type == "betweenness":
        centrality = nx.edge_betweenness_centrality(G, weight=attribute)

    elif type == "current_flow_betweenness":
        centrality = nx.edge_current_flow_betweenness_centrality(G, weight=attribute)
    
    elif type == "load":
        centrality = nx.algorithms.centrality.edge_load_centrality(G)


    else:
        print("method not implemented, please define your own function")
        return None


    if w_max:
        return max(centrality, key=centrality.get)
    else:
        return min(centrality, key=centrality.get)


###this are the helper functions of the lais2 algorithm with an adaption to use edge weights

def weight(community):
    #Input: a subgraph/community in the graph
    #Output: weight of the community (using the formula mentioned in the paper)

    ##check if possible here to add edge weights 
    #i think we could make as additional condition that the new edge weight needs to be larger than the average edge weight
    if nx.number_of_nodes(community) == 0:
        return 0
    else:
        return float(2*nx.number_of_edges(community)/nx.number_of_nodes(community))


def orderNodes(graph, edge_weight="weight"):
    #Input: a networkx graph
    #Output: list of nodes sorted in the decreasing order of their page rank
    dictOfNodes = nx.pagerank(graph, weight=edge_weight)
    orderedNodes = dictOfNodes.items()
    orderedNodes = sorted(orderedNodes, reverse=True, key=get_key)
    return orderedNodes

def get_key(node):
    #Input: list containing node name and its page rank
    #Output: return rank of the node
    return node[1]

def LA(graph, edge_weight="weight"):
    #Input: a networkx graph
    #Output: a group of clusters (initial guesses to be fed into the second algorithm)
    #Order the vertices using page rank
    orderedNodes = orderNodes(graph, edge_weight=edge_weight)
    C = []
    for i in orderedNodes:
        added = False
        for c in C:
            #Add the node and see if the weight of the cluster increases
            temp1 = graph.subgraph(c)
            cc = list(c)
            cc.append(i[0])
            temp2 = graph.subgraph(cc)
            #If weight increases, add the node to the cluster
            if weight(temp2) > weight(temp1):
                added = True
                c.append(i[0])
        if added == False:
            C.append([i[0]])
    return C

def IS2(cluster, graph):
    #Input: cluster to be improved and the networkx graph
    #Output: improved cluster
    C = graph.subgraph(cluster)
    intialWeight = weight(C)
    increased = True
    while increased:
        listOfNodes = cluster
        for vertex in C.nodes():
            #Get adjacent nodes
            adjacentNodes = graph.neighbors(vertex)
            #Adding all adjacent nodes to the cluster
            listOfNodes = list(set(listOfNodes).union(set(adjacentNodes)))
        for vertex in listOfNodes:
            listOfNodes = list(C.nodes())
            #If the vertex was a part of inital cluster
            if vertex in C.nodes():
                #Remove vertex from the cluster
                listOfNodes.remove(vertex)
            #If the vertex is one of the recently added neighbours
            else:
                #Add vertex to the cluster
                listOfNodes.append(vertex)
            CDash = graph.subgraph(listOfNodes)
            if weight(CDash) > weight(C):
                C = CDash.copy()
        newWeight = weight(C)
        if newWeight == intialWeight:
            increased = False
        else:
            intialWeight = newWeight
    return C
        
###################################################
#this are the helper functions for the fast consensus

def refactor_communities(community):
    """
    function to refactor the resulting community dicts into a list of sublists
    this format is needed when used with the fastconsensus option

    Input
        community dict in the format as provided by the community detection algorithms, except fuzzy and agdl

    Ouput
        list of sublists [[community] [community]], where each community list contains node Ids 
    """
    result_temp = {}

    for node in community.keys():
        c = community[node][0]
        if c in result_temp:
            result_temp.setdefault(c, []).append(node)
        else:
            result_temp[c] = [node]

    result = []
    for key in result_temp:
        result.append(result_temp[key])

    return result




async def estimate_consensus_async(graph, n_p, node, nbr, communities):
    edges_to_add = []
    for i in range(n_p):
        for c in communities[i]:
            if node in c and nbr in c:
                if not graph.has_edge(node,nbr):
                    edges_to_add.append([node, nbr, 0])
                cur = graph[node][nbr]['weight']
                edges_to_add.append([node, nbr, cur + 1])

    return edges_to_add


def create_consensus_graph_dict(graph, communities):
    """
    assume that communities is list of dicts as returned by the community detection algorithms

    graph is graph communities have been detected on
    """

    res = {}
    
    counter = 0
    for edge in itertools.combinations(list(graph.nodes()), 2):
        #count how often the nodes in edge are in the same community
        if counter % 10000 == 0:
            print("evaluating edge", counter)
        counter = counter+1

        cnt = 0
        for com in communities:
            if com[edge[0]][0] == com[edge[1]][0]:
                cnt = cnt + 1
        
        res[edge] = cnt

    #create graph
    print("create consensus graph object")

    G = nx.Graph()

    for edge in res.keys():
        G.add_edge(edge[0], edge[1], weight=res[edge])
        #print("edge weight", edge, res[edge])

   

    return res, G


def create_consensus_graph(graph, communities, n_p, nextgraph=None, in_async=False):
    """
    helper function of fast_consensus which creates a consensus graph based on the clustering provided in communities

    Input
        networkx graph object

        list of community partitions

        n_p number of community partitions

        if nextgraph is None then it creates the initial consensus graph else it is converged consensus graph

        in_async
            if True computation is run in parallel

    Ouptut
        networkx graph object

    """
    print("creating consensus graph")
    E = len(graph.edges())
    print("graph has edges", E)
    cnt = 0
    if in_async:
        print("in async")
        tasks = []
        loop = asyncio.new_event_loop()

        if nextgraph is None:
            for node, nbr in graph.edges():
                cnt = cnt + 1
                if cnt % 10000 == 0:
                    print("computing", cnt, "out of", E)

                re = tasks.append(loop.create_task(estimate_consensus_async(graph, n_p, node, nbr, communities)))

            loop.run_until_complete(asyncio.wait(tasks))

            loop.close()
            # print(tasks)
            # use tasks (returns index and result as tuple) to fill result matrix
            temp_edges = []
            for r in tasks:
                temp_edges.append(r.result())

            #flatten
            new_edges = [item for sublist in temp_edges for item in sublist]
            cnt2 = 0
            for e in new_edges:
                cnt2 = cnt2 + 1
                if cnt2 % 10000 == 0:
                    print("computing", cnt2, "out of", len(new_edges))
                graph[e[0]][e[1]]['weight'] = e[2]


            return graph
        else:
            for node, nbr in graph.edges():
                cnt = cnt + 1
                if cnt % 10000 == 0:
                    print("computing", cnt, "out of", E)

                re = tasks.append(loop.create_task(estimate_consensus_async(nextgraph, n_p, node, nbr, communities)))

            loop.run_until_complete(asyncio.wait(tasks))

            loop.close()
            # print(tasks)
            # use tasks (returns index and result as tuple) to fill result matrix
            temp_edges = []
            for r in tasks:
                temp_edges.append(r.result())

            #flatten
            new_edges = [item for sublist in temp_edges for item in sublist]
            cnt2 = 0
            for e in new_edges:
                cnt2 = cnt2 + 1
                if cnt2 % 10000 == 0:
                    print("computing", cnt2, "out of", len(new_edges))
                nextgraph[e[0]][e[1]]['weight'] = e[2]


            return nextgraph

    else:
        if nextgraph is None:
            for node, nbr in graph.edges():
                cnt = cnt + 1
                if cnt % 10000 == 0:
                    print("computing", cnt, "out of", E)

                for i in range(n_p):
                    for c in communities[i]:
                        if node in c and nbr in c:
                            if not graph.has_edge(node,nbr):
                                graph.add_edge(node, nbr, weight = 0)
                            graph[node][nbr]['weight'] += 1

            return graph
        else:
            for node, nbr in graph.edges():
                cnt = cnt + 1
                if cnt % 10000 == 0:
                    print("computing", cnt, "out of", E)

                for i in range(n_p):
                    for c in communities[i]:
                        if node in c and nbr in c:
                            if not nextgraph.has_edge(node,nbr):
                                nextgraph.add_edge(node, nbr, weight = 0)
                            nextgraph[node][nbr]['weight'] += 1

            return nextgraph





def convert_graph_to_community_dict(communities):
    """
    helper function of fast_consensus, which converts the final community structure (disconnected components)
    into the dict structure of communities 

    Input
        netowrkx list og connected components

    Output
        dict were each key is a node ID, and values are list of communities this node belongs to
    """

    d = {}
    for i in range(len(communities)):
        c = communities[i]
        nodes = list(c.nodes())
        for node in nodes:
            d[node] = [i]

    return d

def check_consensus_graph(G, n_p, delta):
    '''
    This function checks if the networkx graph has converged. 
    helper function of fast_consensus

    Input
        networkx graph object
        n_p  number of partitions/community algorithms while creating G
        delta if more than delta fraction of the edges have weight != n_p then returns False, else True

    Output
        boolean
    '''



    count = 0
    
    for wt in nx.get_edge_attributes(G, 'weight').values():
        if wt != 0 and wt != n_p:
            count += 1

    if count > delta*G.number_of_edges():
        return False

    return True