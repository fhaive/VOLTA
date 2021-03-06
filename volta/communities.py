'''
Community detection & evaluation algorithms.
Many algorithms are imported from CDlib - https://github.com/GiulioRossetti/cdlib. But only a small fraction is exposed here, for more algorithms please refer to the official documentation

'''

import pandas as pd
import glob
import sys
import os
import datetime
import math
import networkx as nx
import collections
#import matplotlib.pyplot as plt
import random
#import treelib as bt
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
import pyintergraph
from itertools import chain

from networkx.utils import pairwise, not_implemented_for
import scipy
import statsmodels.stats.multitest as multi
from scipy.stats import hypergeom





#Node clustering


#unweighted

def async_fluid(G, k=5, return_object=True):
    """
    
    Propagation-based algorithm, were the desired number of communities needs to be set.

    Reference:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.async_fluid.html#cdlib.algorithms.async_fluid

        Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés, Toyotaro Suzumura T. 
        Fluid Communities: A Competitive and Highly Scalable Community Detection Algorithm.
    
    Parameters:
        G (networkX graph object): to estimate on
        k (int): number of desired communities.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

        

    """

    communities = cdlib.algorithms.async_fluid(G,k)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m




def gdmp2(G, min_threshold=0.75, return_object=True):
    """
    Based on finding dense subgraphs.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.gdmp2.html#cdlib.algorithms.gdmp2

        Chen, Jie, and Yousef Saad. Dense subgraph extraction with application to community detection. 
        IEEE Transactions on Knowledge and Data Engineering 24.7 (2012): 1216-1230.
    
    Parameters:
        G (networkX graph object): to estimate on
        min_threshold (float): minimum density threshold. This is used to control the density of the detected communities.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.
    
    """
    

    communities = cdlib.algorithms.gdmp2(pyintergraph.nx2igraph(G), min_threshold=min_threshold)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def infomap(G, return_object=True):
    """
    Based on random walks.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.infomap.html#cdlib.algorithms.infomap

        Rosvall M, Bergstrom CT (2008) Maps of random walks on complex networks reveal community structure. 
        Proc Natl Acad SciUSA 105(4):1118–1123
    
    Parameters:
        G (networkX graph object): to estimate on
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.
    

    
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
    Based on network structure.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.label_propagation.html#cdlib.algorithms.label_propagation

        Raghavan, U. N., Albert, R., & Kumara, S. (2007). 
        Near linear time algorithm to detect community structures in large-scale networks. Physical review E, 76(3), 036106.
    
    Parameters:
        G (networkX graph object): to estimate on
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

    
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
    Based on random walks. This algorithm can be given directed or undirected graph objects as input. 

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.walktrap.html#cdlib.algorithms.walktrap

        Pons, Pascal, and Matthieu Latapy. Computing communities in large networks using random walks. 
        J. Graph Algorithms Appl. 10.2 (2006): 191-218.
    
    Parameters:
        G (networkX graph object): to estimate on
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

    
    """
    #convert to an igraph object
    

    communities = cdlib.algorithms.walktrap(pyintergraph.nx2igraph(G))

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

def cpm(G, initial_membership=None, weights="weight", resolution_parameter=1, return_object=True):
    """
    Finds communities of a particular density. As input directed or undirected graph objects can be provided.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.cpm.html#cdlib.algorithms.cpm

        Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011). 
        Narrow scope for resolution-limit-free community detection. Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114
    
    Parameters:
        G (networkX graph object): to estimate on
        initial_membership (list or None):  list of iInitial membership for the partition. If None then defaults to a singleton partition.
        weights (str): edge attribute to be used.
        resolution_parameter (float):  > 0. Controls the community resolution. Higher values lead to more communities, while lower values lead to fewer communities.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.
    """

    communities = cdlib.algorithms.cpm(pyintergraph.nx2igraph(G), initial_membership=initial_membership, weights=weights, node_sizes=None, resolution_parameter=resolution_parameter)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m

def greedy_modularity(G, weights="weight", return_object=True):
    """
    Based on modularity.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.greedy_modularity.html#cdlib.algorithms.greedy_modularity

        Clauset, A., Newman, M. E., & Moore, C. 
        Finding community structure in very large networks. Physical Review E 70(6), 2004
    
    Parameters:
        G (networkX graph object): to estimate on
        weights (str): edge attribute to be used.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

    """

    communities = cdlib.algorithms.greedy_modularity(G, weight=weights)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def leiden(G,  weights="weight", return_object=True):
    """
    Based on the louvain algorithm. This algorithm can be given directed or undirected graph objects as input.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.leiden.html#cdlib.algorithms.leiden

        Traag, Vincent, Ludo Waltman, and Nees Jan van Eck. 
        From Louvain to Leiden: guaranteeing well-connected communities. arXiv preprint arXiv:1810.08473 (2018).
    
    Parameters:
        G (networkX graph object): to estimate on
        weights (str): edge attribute to be used.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

    """

    communities = cdlib.algorithms.leiden(pyintergraph.nx2igraph(G), weights=weights, initial_membership=None)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def louvain(G, weights="weight", resolution=1.0,  return_object=True):
    """
    Based on modularity.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.louvain.html#cdlib.algorithms.louvain

        Blondel, Vincent D., et al. Fast unfolding of communities in large networks.
        Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.
    
    Parameters:
        G (networkX graph object): to estimate on
        weights (str): edge attribute to be used.
        resolution (float): controls community size.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

    """

    communities = cdlib.algorithms.louvain(G, weight=weights, resolution=resolution, randomize=False)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m


def weak_link_communities(H, weights="weight", t=0.2, max_iter=1000, by_degree=True, r=5, std=0.2, min_community=10, graph=False):
    """
    Tries to find weak links between node groups, while ensuring that no new isolates occure.
    For each node a probabilistic selection of neighbors is run for r rounds based on their edge weights.
    Neigbors with less than t occurenses will be disconnected but only if this link is selected as a weak link in both directions.
    The algorithm will stop when max_iter is reached or no edges can be removed anymore.
    
    Parameters:

        G (networkX graph object): to estimate on
        weights (str): edge attribute to be used.
        t (float): in [0,1]. Treshold on which edges are "selected as weak" and to be removed if they occure <= in of samples neighbors.
        max_iter (int): maximum number of iterations. 
        by_degree (boolean): if True then the number of samplings for each node is based on its degree. Each node will sample from its neighbors r*node_degree times.
            If is false then each nodes neighbors will be sampled r times.
        r (int) how often a nodes neighbors are sampled.  
        std (float): treshold of a nodes neighboring sampling distribution standardiviationf. If it is above std then edge removal will be performed.
        min_community (int or None): minimum community size allowed. If already disconnected components of size < min_community exist, smaller communities can still occure. 
                If None will be ignored.
        graph (boolean): if is true then the paritioned graph object will be returned as well.

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        graph (networkX graph object): if graph is True.
                

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
            #if cntk % 100 == 0:
            #   print("removing edge with min community", cntk, "out of", len(weak_edges))
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
            #print("removing weak edges", len(to_remove), "in iteration", cnt)
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
    #com_temp = list(nx.connected_component_subgraphs(G))
    com_temp = [G.subgraph(c).copy() for c in nx.connected_components(G)]


    communities = __convert_graph_to_community_dict__(com_temp)

    if graph:
        return communities, G
    else:
        return communities

def __disconnect_high_degree__(G, nodes=10, percentage=None, weight="weight"):
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



def girvan_newman(G, valuable_edge="max_weight", k=1, is_leve=False, attribute="weight", w_max=True):
    """
    NetworkX girvan newman implementation. Removes "most valuable edge" from the graph to partition it.
    As input directed or undirected graph objects can be provided.

    References:
        https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html

    Parameters:
        G (networkX graph object): to estimate on
        valuable_edge (str): determines how the most valuable edge is selected. If "max_weight" then the edge with the highest weight is removed.
            If "min_weight" the edge with the lowest weight is removed. If "betweenness"  the edge with the highest/ lowest betweenness centrality is removed first.
            If is "current_flow_betweenness" the edge with the highest/ lowest current flow betweenness centrality is removed first. If is "load" the edge with the highest/lowest load centrality is removed.
        k (int or None): which level(s) of the algorithms should be returned. If None all levels are returned.
        is_level (boolean): if is True then k is assumed to be the level index and only that level is returned. If it is False the the k top levels are returned. 
        attribute (str): name of the edge attribute to be used.
        w_max (boolean): only applies if valuable_edge is not "max_weight" or "min_weight. If True then the edge with the heigest value is remvoed. If False the edge with the lowest value is removed.     
        
    Returns:
        communities (dict): key are node IDs and value is list of communities that the node belongs to. First value is the top most level and last value is the most granular level.

    """

    if valuable_edge == "max_weight":
        #this is a very shitty implementation but networkx does not allow to provide the edge function with parameters
        def temp(G, attribute=attribute):
            return __by_weight__(G, w_max=True, attribute=attribute)


    elif valuable_edge == "min_weight":
        def temp(G, attribute=attribute):
            return __by_weight__(G, w_max=False, attribute=attribute)

    elif valuable_edge == "betweenness":
        def temp(G, attribute=attribute, w_max=w_max):

            return __by_centrality__(G, w_max=w_max, attribute=attribute, type="betweenness")

    elif valuable_edge == "current_flow_betweenness":
        def temp(G, attribute=attribute, w_max=w_max):

            return __by_centrality__(G, w_max=w_max, attribute=attribute, type="current_flow_betweenness")

    elif valuable_edge == "load":
        def temp(G, attribute=attribute, w_max=w_max):

            return __by_centrality__(G, w_max=w_max, attribute=attribute, type="load")

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

def agglomerative_clustering(G, is_distance=True, linkage="complete", distance_threshold=0.2):
    """
    Clusters on the adjacency matrix of a graph. Cells need to contain distance or similarity values. Based on sklearn.
    As input directed or undirected graph objects can be provided.

    Parameters:
        G (networkX graph object): to estimate on
        is_distance (boolean): if True the values contained in the adjacency matrix are used for clustering. If False they are converted into a distance first with 1-x.
        linkage (str): method to be used. Options are  "complete", "average", "single".
        distance_treshold (float): treshold above which clusters will be merged.
       
    Returns:
        communities (dict): key are node IDs and value is list of communities that the node belongs to.
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
    Markov clustering on adjacency matrix. Based on the markovclustering package.
    As input directed or undirected graph objects can be provided.

    Parameters:
        G (networkX graph object): to estimate on
        inflation (float): clustering parameter.
        
   Returns:
        communities (dict): key are node IDs and value is list of communities that the node belongs to.
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
    As input directed or undirected graph objects can be provided.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.angel.html#cdlib.algorithms.angel

        Rossetti, Giulio. “Exorcising the Demon: Angel, Efficient Node-Centric Community Discovery.”
        International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.
        
    Parameters:
        G (networkX graph object): to estimate on
        treshold (float): in  [0,1]. Merging treshold.
        min_community_size (int): minimum community size.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.

    """

    communities = cdlib.algorithms.angel(pyintergraph.nx2igraph(G), treshold, min_community_size=min_community_size)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m



def ego_networks(G, level=1, return_object=True):
    """
    Based on ego networks.
    As input directed or undirected graph objects can be provided.

    References:
        https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.ego_networks.html#cdlib.algorithms.ego_networks

    
    Parameters:
        G (networkX graph object): to estimate on
        level (int): from a node its neighbors within a community have distance <= level.
        return_object (boolean): if is True then the resulting CDlib NodeClustering object is returned which includes fitness parameters. For all options refer to https://cdlib.readthedocs.io/en/latest/reference/classes/node_clustering.html

    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        fitness parameters (NodeClustering object): if return_object is True.


    """

    communities = cdlib.algorithms.ego_networks(G, level=level)

    m = communities.to_node_community_map()
    #j = communities.to_json()

    if return_object:
        return m,  communities

    else:
        return m








###########################################################3
#############################################################3

#estimate clusering consensus

def create_initial_consensus(graph, communities, thresh = 0.2):
    """
    Creates initial consensus graph out of provided community detection algorithms.
    Can be used as input of fast_consensus().

    Parameters:
        graph (networkX graph object): to estimate on
        communities (list): list of initial clusterings. Items are assumed to be dicts as returned by the community detection algorithms.
        tresh (float): threshold on if to add a node pair to the same community or not.
    
    Returns:
        initial communities (dict): merged communities detected. Keys are node IDs and value is list of communities that node belongs to.
        consensus communities (dict): key is edge ID and value is how often the edge is in the same community
        consensus graph (networkX graph object): graph object of consensus communities. Edge weights indicate how often the 2 nodes are in the same community.

    """

    for u,v in graph.edges():
            graph[u][v]['weight'] = 0.0

    consensus_com, consensus_graph = __create_consensus_graph_dict__(graph, communities)

    

    print("transforming initial consensus graph in its communities based on selected treshold")
    remove_edges = []
    for u,v in consensus_graph.edges():
        if consensus_graph[u][v]['weight'] / len(communities) < thresh:
            remove_edges.append((u, v))
    consensus_graph.remove_edges_from(remove_edges)
    print("initial removed edges", len(remove_edges))

    #com_temp = list(nx.connected_component_subgraphs(consensus_graph))
    com_temp = [consensus_graph.subgraph(c).copy() for c in nx.connected_components(consensus_graph)]

    initial_communities = __convert_graph_to_community_dict__(com_temp)

    return initial_communities, consensus_com, consensus_graph


def fast_consensus(G, communities, algorithms = [], parameters=[], thresh = 0.2, delta = 0.02, max_iter=100, initial=None):
    """
    This is an adapted implementation of the fast_consensus algorithm.

    References:
        https://github.com/adityat/fastconsensus/blob/master/fast_consensus.py
        https://arxiv.org/pdf/1902.04014.pdf
        https://doi.org/10.1103/PhysRevE.99.042301

    Parameters:
        G (networkX graph object): to estimate on
        communities (list): list of initial clusterings. Items are assumed to be dicts as returned by the community detection algorithms.
        algorithms (list): list of algorithms to be used during the consensus estimation. If an algorithm needs to be run multiple times it needs to be added multiple times.
            Algorithms that need a connected graph to work (such as async_fluid) cannot be provided as input.
        parameters (list): items are dicts containing the parameters of the algorithms listed in algorithms. Needs to be in the same order as algorithms.
            Make sure that for the algorithms where applicable return_object is set to False.
        tresh (float): threshold on if to add a node pair to the same community or not.
        delta (float): in [0.02, 0.1].
        max_iter (int): maximum number of iterations allowed if consensus is not reached beforehand.
        initial (None or networkX graph object): instead of using predicted communities provided in communities algorithm can be started on a consensus graph.
    
    Returns:
        communities (dict): key are node IDs and value is list of communities that node belongs to.
        consensus communities (dict): key is edge ID and value is how often the edge is in the same community. This is the consensus communities infered from the provided communities only. After one round of the algorithm.
        consensus graph (networkX graph object): graph object of consensus communities. Edge weights indicate how often the 2 nodes are in the same community. This is the consensus graph infered from the provided communities only. After one round of the algorithm.


    
    """
    graph = G.copy()
    L = graph.number_of_edges()
    #N = G.number_of_nodes()

    print("start fast consensus")
    #create initial consensus graph
    if initial is None:
        print("no initial graph provided create from communities")
        for u,v in graph.edges():
            graph[u][v]['weight'] = 0.0

        consensus_com, consensus_graph = __create_consensus_graph_dict__(graph, communities)

        

        print("transforming initial consensus graph in its communities based on selected treshold")
        remove_edges = []
        for u,v in consensus_graph.edges():
            if consensus_graph[u][v]['weight'] / len(communities) < thresh:
                remove_edges.append((u, v))
        consensus_graph.remove_edges_from(remove_edges)
        print("initial removed edges", len(remove_edges))

        #com_temp = list(nx.connected_component_subgraphs(consensus_graph))
        com_temp = [consensus_graph.subgraph(c).copy() for c in nx.connected_components(consensus_graph)]

        initial_communities = __convert_graph_to_community_dict__(com_temp)

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
        tmp , nextgraph = __create_consensus_graph_dict__(graph, communities)
                    
        tmp = None
        #print("before removal", len(nextgraph.edges()))
        L = nextgraph.number_of_edges()

        remove_edges = []
        for u,v in nextgraph.edges():
            if nextgraph[u][v]['weight'] / len(communities) < thresh:
                remove_edges.append((u, v))
        nextgraph.remove_edges_from(remove_edges)
        #print("after removal", len(nextgraph.edges()))

        
        #this now randomly adds some new edges, for edges that are not existing in the consensus graph anymore
        #this provides triadic closure
        #helps to improve the consensus clustering
        #print("adding random edges for", L/4)
        cntx = 0
        for _ in range(int(L/4)):
            #if cntx % 10000 == 0:
            #    print("checking random", cntx)
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

        if __check_consensus_graph__(nextgraph, n_p = len(communities), delta = delta):
            converged = True
            break
        if cnt >= max_iter:
                break
            

    if not converged:
        print("max iter has been reached but consensus graph has not converged")

    #i think it would be best to convert the consensus graph into its communities based on its connected components
    #com_temp = list(nx.connected_component_subgraphs(graph))
    com_temp = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    communities = __convert_graph_to_community_dict__(com_temp)

    return communities, consensus_com, initial_communities


    


#########################33
############################3

#compare communities/ evaluate communities

def get_number_of_communities(communities):
    """
    Returns the number of detected communities.

    Parameters:
        communities(dict): as returned by the community functions.
    Returns:
        number of communities (int): detected

    """

    c = list(communities.values())
    c_flat = [item for sublist in c for item in sublist]

    return max(c_flat) + 1 #return plus 1 since community indexing starts with 0

def convert_communities(communities):
    """
    Converts the community object returned by the community detection algorithms into another dict format. 
    Where keys are community IDs and values are lists of nodes in that community.

    Parameters:
        communities (dict): as returned by the community detection algorithms.

    Returns:
        communities (dict): transformed object.
    """

    return __convert_communities__(communities)


def __convert_communities__(communities):
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
    Estimate the distribution of nodes between the communities.

    Parameters:
        communities(dict): as returned by the community functions.
        in_detail (boolean): if True then for each community its number of nodes is reported. If False only distributional parameters are returned.
        
    Returns:
        distribution (dict): returns mean and median number of nodes, the standard deviation, kurtoisis and skewness. Keys are mean, median, std, kurt, skewness.
        individual nodes (dict): if in_detail is True. key is community ID and value is number of nodes in that community.

    """

    dict_communities = __convert_communities__(communities)

    number_of_nodes = []
    dict_nodes = {}

    for key in dict_communities:
        nr_nodes = len(dict_communities[key])
        dict_nodes[key] = nr_nodes
        number_of_nodes.append(nr_nodes)

    #get distributional values
    try:
        mean = statistics.mean(number_of_nodes)
        median = statistics.median(number_of_nodes)
        std = statistics.stdev(number_of_nodes)
        kurt = kurtosis(number_of_nodes)
        sk = skew(number_of_nodes)
    except:
        mean = None
        median = None
        std=None
        kurt = None
        sk = None
        


    if in_detail:

        return {"mean":mean, "median":median, "std":std, "kurt":kurt, "skewness":sk}, dict_nodes
    else:
        return {"mean":mean, "median":median, "std":std, "kurt":kurt, "skewness":sk}



def average_internal_degree(communities, G):
    """
    Estimates the average internal degree of a communities member nodes. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high within community degree is prefered, indicating a strong connectivity within its members. Based on pquality.

    References:
        Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). 
        Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value is its average internal degree
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates thr internal edge density. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high within community score is prefered. Based on pquality.
    
    References:
        Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004).
        Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.

    """

    dict_communities = __convert_communities__(communities)

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
    Estimates thr triangle participation ratio. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high within community score is prefered. Based on pquality.
   
    References:
        Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. 
        Knowledge and Information Systems 42(1), 181–213 (2015)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates the number of edges inside the community.This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high within community score is prefered. Based on pquality.

    References:
        Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. 
        Proceedings of the National Academy of Sciences, 101(9), 2658-2663.

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates conductance, which is the fraction of edges leaving the community. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small within community score is prefered. Based on pquality.

    References:
        Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Maximum fraction of a nodes's edges that leave the community.This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small within community score is prefered. Based on pquality.

    References:
        1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: 
        Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates average fraction of a nodes's edges that leave the community.This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small within community score is prefered. Based on pquality.

    References:
        1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: 
        Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates fraction of nodes within a community that have fewer edges going inwards than outwards. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small within community score is prefered. Based on pquality.
    
    References:
        1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: 
        Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates a communities modularity. A high score is prefered. Based on pquality.

    References:
        Newman, M.E.J. & Girvan, M. `Finding and evaluating community structure in networks. 
        Physical Review E 69, 26113(2004).

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        score (float): modularity score 
    """
    com = {}
    for key in communities:
        com[key] = communities[key][0]


    val = pq.PartitionQuality.community_modularity(com, G)
    

    return val

def modular_density(communities, G):
    
    """
    Estimates modularity while considering community size. Based on pquality.
    
    References:
        Li, Z., Zhang, S., Wang, R. S., Zhang, X. S., & Chen, L. (2008). Quantitative function for community detection. 
        Physical review E, 77(3), 036109.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        density (float): modularity score for whole partitioning.
        individual scores (dict): key is community ID and value is modularity score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates thr fraction of nodes that have a higher degree than the median degree within the community. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high score indicates that the community contains hub nodes. Based on pquality.

    References:
        Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. 
        Knowledge and Information Systems 42(1), 181–213 (2015)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates fraction of edges leaving the community (expansion). This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small score indicates closed up communities. Based on pquality.
    
    References:
        Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). 
        Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates the (normalized) cut ratio. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small score is preferred. Based on pquality.
    
    References:
        Fortunato, S.: Community detection in graphs. Physics reports 486(3-5), 75–174 (2010)

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        normalized (boolean): if True the normalized cut ratio is returned.

     Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates the community density with respect to the complete graph density. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high score is preferred.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        
     Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates thr average shortest path within a community. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small score is preferred.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        weight (str or None): edge attribute to be considered for shortest paths. If is None all edges are considered equal.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates the average shortest path within a community w.r.t avgerage shortest path in the whole graph.This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A small score is preferred.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        weight (str or None): edge attribute to be considered for shortest paths. If is None all edges are considered equal.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates average edge weight within a community w.r.t thr edge weights in the whole graph. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    Depending on of edge weights are distances or similarities a small or high score may be prefered.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        weight (str or None): edge attribute to be considered. If is None all edges are considered equal.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates thr average edge weight within a community. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    Depending on of edge weights are distances or similarities a small or high score may be prefered.
   
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        weight (str or None): edge attribute to be considered. If is None all edges are considered equal.

    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates hub dominace. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high value is preferred.
    
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        
    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates thr average clustering coefficient of the different communities. This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high value is preferred.

    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        
    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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
    Estimates community embeddedness. This is a nodes mean degree within the community w.r.t. its overall degree in the graph.This measurement is community specific and an average score + distributional parameters over all communities is reported.
    A high value is preferred.
   
    Parameters:
        communities(dict): as returned by the community functions.
        G (networkX graph object): the graph the communities where detected on.
        
    Returns:
        individaul scores (dict): key is community ID and value its score.
        mean score (dict): Returns distributional parameters of mean, median, standard deviation, kurtosis and skewness. Keys are mean, median , std, kurt and skewness.
            If there are less than 2 communities None is returned.
    """

    dict_communities = __convert_communities__(communities)

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




def __by_weight__(G, w_max=True, attribute="weight"):
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


def __by_centrality__(G, w_max=True, attribute="weight", type="betweenness"):
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

def __weight__(community):
    #Input: a subgraph/community in the graph
    #Output: weight of the community (using the formula mentioned in the paper)

    ##check if possible here to add edge weights 
    #i think we could make as additional condition that the new edge weight needs to be larger than the average edge weight
    if nx.number_of_nodes(community) == 0:
        return 0
    else:
        return float(2*nx.number_of_edges(community)/nx.number_of_nodes(community))


def __orderNodes__(graph, edge_weight="weight"):
    #Input: a networkx graph
    #Output: list of nodes sorted in the decreasing order of their page rank
    dictOfNodes = nx.pagerank(graph, weight=edge_weight)
    orderedNodes = dictOfNodes.items()
    orderedNodes = sorted(orderedNodes, reverse=True, key=__get_key__)
    return orderedNodes

def __get_key__(node):
    #Input: list containing node name and its page rank
    #Output: return rank of the node
    return node[1]

def __LA__(graph, edge_weight="weight"):
    #Input: a networkx graph
    #Output: a group of clusters (initial guesses to be fed into the second algorithm)
    #Order the vertices using page rank
    orderedNodes = __orderNodes__(graph, edge_weight=edge_weight)
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
            if __weight__(temp2) > __weight__(temp1):
                added = True
                c.append(i[0])
        if added == False:
            C.append([i[0]])
    return C

def __IS2__(cluster, graph):
    #Input: cluster to be improved and the networkx graph
    #Output: improved cluster
    C = graph.subgraph(cluster)
    intialWeight = __weight__(C)
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
            if __weight__(CDash) > __weight__(C):
                C = CDash.copy()
        newWeight = __weight__(C)
        if newWeight == intialWeight:
            increased = False
        else:
            intialWeight = newWeight
    return C
        
###################################################
#this are the helper functions for the fast consensus

def refactor_communities(community):
    """
    Refactors the community dicts into a list of sublists. This format is needed when using the fastconsensus function.
    

    Parameters:
        community (dict): in the format as provided by the community detection algorithms, except fuzzy and agdl

    Returns:
        transformed communities (list): list of sublists [[community] [community]], where each community list contains node Ids 
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




async def __estimate_consensus_async__(graph, n_p, node, nbr, communities):
    edges_to_add = []
    for i in range(n_p):
        for c in communities[i]:
            if node in c and nbr in c:
                if not graph.has_edge(node,nbr):
                    edges_to_add.append([node, nbr, 0])
                cur = graph[node][nbr]['weight']
                edges_to_add.append([node, nbr, cur + 1])

    return edges_to_add


def __create_consensus_graph_dict__(graph, communities):
    """
    assume that communities is list of dicts as returned by the community detection algorithms

    graph is graph communities have been detected on
    """

    res = {}
    
    counter = 0
    for edge in itertools.combinations(list(graph.nodes()), 2):
        #count how often the nodes in edge are in the same community
        #if counter % 10000 == 0:
        #    print("evaluating edge", counter)
        counter = counter+1

        cnt = 0
        for com in communities:
            if (len(com[edge[0]])>0 and len(com[edge[1]])>0):
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

'''
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

                re = tasks.append(loop.create_task(__estimate_consensus_async__(graph, n_p, node, nbr, communities)))

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

                re = tasks.append(loop.create_task(__estimate_consensus_async__(nextgraph, n_p, node, nbr, communities)))

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

'''



def __convert_graph_to_community_dict__(communities):
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

def __check_consensus_graph__(G, n_p, delta):
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


def __metric_closure__(G, weight="weight"):
    
    """
    Helper function of steiner tree, taken from Networkx:
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/approximation/steinertree.html#steiner_tree
    """
    
    M = nx.Graph()

    Gnodes = set(G)

    # check for connected graph while processing first node
    all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
    u, (distance, path) = next(all_paths_iter)
    if Gnodes - set(distance):
        msg = "G is not a connected graph. metric_closure is not defined."
        raise nx.NetworkXError(msg)
    Gnodes.remove(u)
    for v in Gnodes:
        M.add_edge(u, v, distance=distance[v], path=path[v])

    # first node done -- now process the rest
    for u, (distance, path) in all_paths_iter:
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])

    return M

################## active module detection 

def __steiner_tree__(G, terminal_nodes, weight="weight"):
    """ 
    This function has been taken from NetworkX: 
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/approximation/steinertree.html#steiner_tree
    
    Return an approximation to the minimum Steiner tree of a graph.

    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes`
    is a tree within `G` that spans those nodes and has minimum size
    (sum of edge weights) among all such trees.

    The minimum Steiner tree can be approximated by computing the minimum
    spanning tree of the subgraph of the metric closure of *G* induced by the
    terminal nodes, where the metric closure of *G* is the complete graph in
    which each edge is weighted by the shortest path distance between the
    nodes in *G* .
    This algorithm produces a tree whose weight is within a (2 - (2 / t))
    factor of the weight of the optimal Steiner tree where *t* is number of
    terminal nodes.

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    
    """
    # H is the subgraph induced by terminal_nodes in the metric closure M of G.
    M = __metric_closure__(G, weight=weight)
    H = M.subgraph(terminal_nodes)
    # Use the 'distance' attribute of each edge provided by M.
    mst_edges = nx.minimum_spanning_edges(H, weight="distance", data=True)
    # Create an iterator over each edge in each shortest path; repeats are okay
    edges = chain.from_iterable(pairwise(d["path"]) for u, v, d in mst_edges)
    # For multigraph we should add the minimal weight edge keys
    if G.is_multigraph():
        edges = (
            (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in edges
        )
    T = G.edge_subgraph(edges)
    return T



def active_modules(G, active_nodes, algorithms=[[louvain],[girvan_newman]], algorithm_parameters=[[{"weights":"weight","return_object": False}],[{"valuable_edge":"betweenness"}]], initialclustering=None, pval1 = 0.3, pval2=0.05):
    """
    Adjusted implementation of the DOMINO algorithm.

    Levi, Hagai, Ran Elkon, and Ron Shamir. 
    "DOMINO: a network‐based active module identification algorithm with reduced rate of false calls." 
    Molecular Systems Biology 17.1 (2021): e9593.
    
    Parameters:
        G (NetworkX Graph): Graph object active modules are identified of
        active_nodes (list): list of node IDs that are classified as "active"
        algorithms (list): list of sublists, where first list contains the community detection algorithms
            used in the first module detection round and the second list contains the algorithms to be used for
            the second round. If more than one algorithm is provided in each sublist a consensus partitioning
            is calculated.
        algorithm_parameters(list): list of sublists, corresponding to the algorithms stated in algorithms.
            parameters need to be provided as dict.
        initialclustering (dict or None): if None then the provided graph will be partitioned based on the stated algorithms. If dict
            then instead of estimating an initial partitioning the provided partitioning will be used. Key is node ID and value is list of communities
            node belongs to. The format correponds to the output of the provided community algorithms. If not None then in algorithms and algorithm_parameter
            the first list can be subsidiced with an empty list - e.g. [[], [girvan_newman]] 
        pval1 (float): in [0,1] pval cutoff to be applied in the first round of module selection
        pval2 (float): in [0,1] pval cutoff to be applied in the secondround of module selection
            
    Output:
            active_modules (list): list of sublists. Each sublists represents an active module and its items represents
                the nodes contained in it.
            active_pval (list): contains the corresponding corrected pvalues to active_modules. 
                list is in same order as active_modules.
            all_modules (list): list of sublists. Each sublists represents a module (active or not active)
                and its items represents the nodes contained in it.
            pvals (list): contains the corresponding corrected pvalues to all_modules. 
                list is in same order as all_modules.
            initialclustering (dict): the partitioning used on the provided graph in the initial partitioning step.
            
    """
    
    
    #do first round of module detection
    
    if initialclustering is None:
        if len(algorithms[0]) > 1:
            #consensus
            print("A consensus partitioning is calculated")
            #calculate initial partitionings
            community_list = []
            for a in range(len(algorithms[0])):
                algo = algorithms[0][a]
                param = algorithm_parameters[0][a]
                
                print("estimating communities ", algo.__name__)
                community_list.append(algo(G, **param))
            #TODO expose parameters
            partitioning, cons1, consgraph  = fast_consensus(G, communities = community_list, algorithms=algorithms[0], parameters=algorithm_parameters[0], thresh=0.5, max_iter=50)
            
            
        else:
            #do single partitioning
            community_list = []
            for a in range(len(algorithms[0])):
                algo = algorithms[0][a]
                param = algorithm_parameters[0][a]

                print("estimating communities ", algo.__name__)
                community_list.append(algo(G, **param))
            partitioning = community_list[0]

    else:
        partitioning = initialclustering
    #save as return
    initialclustering = partitioning
    #convert partitioning into list of lists format
    temp = convert_communities(partitioning)
    partitioning = []
    for key in temp:
        partitioning.append(temp[key])
        
        
    print("find enriched modules")
    temp_pvals = []

    for par in partitioning:
        M = len(G.nodes()) #number of nodes
        n = len(active_nodes) #nuber of active nodes
        N = len(par) #cluster size

        x = 0
        for no in par:
            if no in active_nodes:
                x = x+1
        #number of active nodes in this partitioning

        v = hypergeom.sf(x-1, M, n, N)
        temp_pvals.append(v)



    #do correction
    #TODO expose parameters
    adj_pval = multi.multipletests(temp_pvals, method="fdr_bh")[1]

    subgraphs = []
    #if below pval treshold create subgraphs of module

    for i in range(len(adj_pval)):
        if adj_pval[i] <= pval1:
            subgraphs.append(G.subgraph(partitioning[i]))

    
    
    #TODO add node treshold as parameter
    
    #apply prize steiner tree
    print(len(subgraphs), "modules are enriched")
    if len(subgraphs) < 1:
        print("no enriched modules have been detected in the first partitioning")
    else:
        print("calculating Prize Steiner Tree")

    trees = []
    for H in subgraphs:
        #get active nodes
        terminal = []
        for no in H.nodes():
            if no in active_nodes:
                terminal.append(no)
        #TODO expose weight parameter
        trees.append(__steiner_tree__(H, terminal, weight='weight'))
    
    
    all_modules = []
    all_pval = []
    enriched_active_modules=[]
    enriched_pval = []


    for tree in trees:

        if len(algorithms[1]) > 1:
            #consensus
            print("A consensus partitioning is calculated")
            #calculate initial partitionings
            community_list = []
            for a in range(len(algorithms[1])):
                algo = algorithms[1][a]
                param = algorithm_parameters[1][a]

                print("estimating communities ", algo.__name__)
                community_list.append(algo(tree, **param))
            #TODO expose parameters
            partitioning, cons1, consgraph  = fast_consensus(tree, communities = community_list, algorithms=algorithms[0], parameters=algorithm_parameters[0], thresh=0.5, max_iter=50)


        else:
            #do single partitioning
            community_list = []
            for a in range(len(algorithms[1])):
                algo = algorithms[1][a]
                param = algorithm_parameters[1][a]

                print("estimating communities ", algo.__name__)
                community_list.append(algo(tree, **param))
            partitioning = community_list[0]

        #convert partitioning into list of lists format
        temp = convert_communities(partitioning)
        partitioning = []
        for key in temp:
            partitioning.append(temp[key])

        #TODO add community size treshold as parameter
        print("find enriched modules")
        temp_pvals = []

        #DISCUSS against all nodes or only in the tree??
        for par in partitioning:
            M = len(G.nodes()) #number of nodes
            n = len(active_nodes) #nuber of active nodes
            N = len(par) #cluster size

            x = 0
            for no in par:
                if no in active_nodes:
                    x = x+1
            #number of active nodes in this partitioning

            v = hypergeom.sf(x-1, M, n, N)
            temp_pvals.append(v)



        #do correction
        #TODO expose method parameter 
        adj_pval = multi.multipletests(temp_pvals, method="bonferroni")[1]



        for i in range(len(adj_pval)):
            if adj_pval[i] <= pval2:
                enriched_active_modules.append(partitioning[i])
                enriched_pval.append(adj_pval[i])

            all_modules.append(partitioning[i])
            all_pval.append(adj_pval[i])

    print(len(enriched_active_modules), "active modules have been detected")
    print("returning enriched and all modules")
    
    return enriched_active_modules, enriched_pval, all_modules, all_pval, initialclustering