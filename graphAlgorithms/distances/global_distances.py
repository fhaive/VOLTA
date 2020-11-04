"""
functions to estimate multiple global measurements of a graph object

"""

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

import itertools
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy
from collections import Counter



def perform_random_walks(G, steps=10, number_of_walks=100, start=None, probabilistic=True, weight="weight"):
    """
    performs x random walks of size n

    Input
        G is networkx graph, steps is size of random walk

        number of walks is number of how many random walks are performed on G

        if start is None start node is selected at random from G
            else start needs to be node ID contained in G

        if probabilisitc edge weights are taken into account 
            else all edges are considered equal
            start node is always evaluated at random without taking any properties into account

    if probabilistic weight & nodelist need to be set
    weight indicates edge weight attribute, weights need to be similarities

    Output
        returns list of lists containing random walk sequence

    
    """

    walks = []
    
    for i in range(number_of_walks):
        #start node id, select from graph at random if start is None
        if start is None:
            start = random.choice(list(G.nodes()))
        
        visited = perform_walk(G, start, steps, probabilistic=probabilistic, weight=weight)
        walks.append(visited)
        
    return walks


	
def perform_walk(G, start, length, probabilistic=True, weight="weight"):
    """
    helper function of perform_random_walk()
    for parameters refer to parent function

    
    """
    current = start
    visited = []
    visited.append(current)

    for i in range(length):
        #get neighbors of current
        neighbors = list(G.neighbors(current))
        if probabilistic:
            #get edge weights:
            path_weights = []
            #print(len(neighbors))
            for node in neighbors:
                
                    path_weights.append(G[current][node][weight])
        #pick next node at random
        if probabilistic:
            current = random.choices(neighbors, weights=path_weights, k=1)[0]
        else:
            current = random.choice(neighbors)
        visited.append(current)
    return visited

def rank_walks(G, walks, undirected=True):
    """
    takes list of random walks and computes counts of appearing nodes & edges 
    

    Input
        G is networkx graph that random walks have been performed on
        walks needs to be list of sublists, where each sublists contains node sequence of a single random walk
            as returned by perform random walk
        if undirected then edges are counted disregarding direction

    Output
        dict of ranked nodes, sorted after dict values
        dict of ranked edges
    
    """

    nodes = {}
    edges = {}

    #initialize node and edge counts
    for node in list(G.nodes()):
        nodes[node] = 0

    for edge in list(G.edges()):
        edges[str(edge[0])+"-"+str(edge[1])] = 0


    for walk in walks:
        start = None
        for node in walk:
            
            current = nodes[node]
            nodes[node] = current + 1

            if start is None:
                #this is the first node in the walk and therefor no edge
                start = node
            else:
                #get edge
                edge1 = str(start)+"-"+str(node)
                edge2 = str(node)+"-"+str(start)
                if undirected:
                    if edge1 in edges.keys():
                        current = edges[edge1]
                        edges[edge1] = current + 1
                    elif edge2 in edges.keys():
                        current = edges[edge2]
                        edges[edge2] = current + 1
                    else :
                        print("edges not found in graph", edge1, edge2)
                else:
                    if edge1 in edges.keys():
                        current = edges[edge1]
                        edges[edge1] = current + 1
                    
                    else :
                        print("edges not found in graph", edge1)
                #set current node to start
                start = node

    #sort nodes and edges
    nodes_sorted = {k: v for k, v in sorted(nodes.items(), key=lambda item: item[1], reverse=True)}
    edges_sorted = {k: v for k, v in sorted(edges.items(), key=lambda item: item[1], reverse=True)}
    return nodes_sorted, edges_sorted
            
def get_walk_consensus(walks, G):
    """
    estimate for one network a consensus walk based on all performed walks
    this is only possible if the walks have been performed with the same start node

    Input
        walks are list of sublists including node ids in order performed for each single walk

        G is networkx graph object walks have been computed on 

    Output
        returns list, containing node ids of consensus random walk or None if start node is not the same
    """
    consensus = []
    for i in range(len(walks[0])):
        if i == 0:
            #make sure that all start nodes are the same
            firsts = []
            for walk in walks:
                firsts.append(walk[i])
            firsts = list(dict.fromkeys(firsts))
            if len(firsts) > 1:
                print("walks have been performed with different start nodes, consensus cannot be estimated")
                consensus = None
                break
            else:
                consensus.append(firsts[0])
        else:
            temp = []
            for walk in walks:
                temp.append(walk[i])
            #estimate most used value
            c = Counter(temp)
            c_sorted = {k: v for k, v in sorted(c.items(), key=lambda item: item[1])}
            
            #get latest node added to consensus
            latest = consensus[-1]
            #check if path is possible in G
            found = False
            for node in list(c_sorted.keys()):
                if G.has_edge(node, latest) and not found:
                    consensus.append(node)
                    found = True
                    break




    return consensus





def compare_walks(G, walk1, walk2=None, G2=None, comparison="ranked", undirected=True, top=100):
    """
    compare list of random walks (for same and multiple graphs)
    s. perform_random_walks()

    Input
        walk1 and walk2 need to be list of sublists containing node lists orderd as walks
            if walks performed on the same graph should be compared set walk1 & walk2 to the same values  & their similarity between each other 
                will be estimated

        G and G2 are networkx graph random walks are performed on

        comparison states what comparison is performed
            if comaprison ranked and walk1 & walk2 are provided, nodes & edges are ranked (for each subwalk after usage)
                and correlation between rankings is estimated
                    based on kendall tau, top value needs to be set max to number of nodes/edges in smaller graph
                this requires that walk1 and walk2 are of same lenght

            

            undirected states if edge direction needs to be taken into account or not

    Output
        dict

    """

    if comparison == "ranked":
        if walk2 is not None and G2 is not None:
            nodes1, edges1 = rank_walks(G, walk1, undirected=undirected)
            nodes2, edges2 = rank_walks(G2, walk2, undirected=undirected)

            #rank_walks already returns sorteddicts
            nodes1 = list(nodes1.keys())
            nodes2 = list(nodes2.keys())
            
            edges1 = list(edges1.keys())
            edges2 = list(edges2.keys())

            
            if undirected:
                #reformat edge names if necessary
                

                for edge in edges1:
                    if edge not in edges2:
                        temp = edge.split("-")
                        new = temp[1]+"-"+temp[0]

                        if new in edges2:
                            #replace with edge value
                            ind = edges2.index(new)
                            edges2[ind] = edge

            #perform kendall
            if ((len(nodes1) >= top) or  (len(nodes2) >= top)):
                nodes_tau, nodes_p = kendalltau(nodes1[:top], nodes2[:top])
            else:
                
                print("ranked node lists are shorter than top value, please adjust value, list lengths are", len(nodes1), len(nodes2))
                nodes_tau=None
                nodes_p = None
            
            if len(edges1) >= top or len(edges2) >= top:
                edges_tau, edges_p = kendalltau(edges1[:top], edges2[:top])

            else:
                
                print("ranked edge lists are shorter than top value, please adjust value, list lengths are", len(edges1), len(edges2))
                edges_tau=None
                edges_p = None
        else:
            print("comparison for one walk currently not implemented")
            return None

        return {"nodes_tau":nodes_tau, "nodes_p":nodes_p, "edges_tau":edges_tau, "edges_p":edges_p}


    else:
        print("currently not implemented")

        return None

            


def node_degree_distribution(G):

    """
    Input
        G is networkx object

    Output
        returns different mesures to describe node degree distribution
    """

    degrees = list(dict(G.degree()).values())

    mean_degree = statistics.mean(degrees)
    median_degree = statistics.median(degrees)
    std_degree = statistics.stdev(degrees)	
    skw_degree = skew(degrees)
    kurt_degree = kurtosis(degrees)



    return mean_degree, median_degree, std_degree, skw_degree, kurt_degree

def is_connected(G):
    """
    Input
        networkx graph object

    Output
        boolean
    """
    return nx.is_connected(G)

def graph_size(G):
    """
    estimates graph radius, diameter, number of nodes, number of edges
    only works for connected graph

    Input
        networkx graph object

    Output
        dict
    """

    if is_connected(G):
        radius = nx.radius(G)
        diameter = nx.diameter(G)

    else:
        print("graph size can only be estimated on connected graph, GCC will be used instead")
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        radius = nx.radius(G0)
        diameter = nx.diameter(G0)


    nr_nodes = nx.number_of_nodes(G)
    nr_edges = nx.number_of_edges(G)

    return {"radius": radius, "diameter": diameter, "nodes": nr_nodes, "edges":nr_edges}

def density(G):
    """
    Input
        networkx graph object

    Output
        graph density
    """
    return nx.density(G)

def average_clustering(G):
    """
    Input
        networkx graph object

    Output
        average clustering
    """
    return nx.average_clustering(G)

def graph_edges(G):
    """
    provides estimate on how many of all posible edges exist in G (100% implies that G is a complete graph)

    Input
        networkx graph object

    Output
        dict
    """

    nr_nodes = nx.number_of_nodes(G)
    nr_edges = nx.number_of_edges(G)

    #number of non existent edges
    non_edges = len(list(nx.non_edges(G)))
    expected_number_of_edges = (nr_nodes * (nr_nodes - 1))/2
    #missing edges in comparison to a complete graph
    non_edges_percentage = (non_edges / expected_number_of_edges)*100
    existing_edges_percentage = (nr_edges / expected_number_of_edges) *100


    return {"missing_edges": non_edges, "max_edges":expected_number_of_edges, "missing_edges_percentage": non_edges_percentage, "existing_edges_percentage": existing_edges_percentage}

def cycle_attributes(G):
    """
    helper function of cycle_attributes()
    """
    cycle_length=[]

    for cycle in nx.cycle_basis(G):
        cycle_length.append(len(cycle))
        
    return cycle_length


def cycle_distribution(G):
    """
    estimates size distributions of cycles (how many steps to form a cycle) in G
    gives insight in how "structured" graph is - and what biological elements it contains, i.e. multiple feedback loops

    Input
        networkx graph object

    Output
        returns distribution parameters in dict
    """

    cycles_lengths = cycle_attributes(G)	

    nr_cycles = len(cycles_lengths)
    if nr_cycles >= 2:
        median_cycles = statistics.median(cycles_lengths)		
        mean_cycles = statistics.mean(cycles_lengths)
        std_cycles = statistics.stdev(cycles_lengths)	
        skw_cycles = skew(cycles_lengths)
        kurt_cycles = kurtosis(cycles_lengths)

    else:
        print("less than two cycles, statistical parameters cannot be estimated")
        median_cycles = None
        mean_cycles = None
        std_cycles =  None
        skw_cycles = None
        kurt_cycles = None

    return {"number_of_cycles":nr_cycles, "median_cycle_length":median_cycles, "mean_cycle_length":mean_cycles, "std_cycle_length": std_cycles, "skw_cycle_length":skw_cycles, "kurtosis_cycle_length":kurt_cycles}

def path_length_distribution(H):

    """
    estimates pathlength distribution between all node pairs
        if graph is not connected uses giant component

    Input
        networkx graph object

    Output
        dict contianing distirubtion parameters
    """
    G = H.copy()

    if not is_connected(H):
        print("graph is not connected, GC will be used instead")
        #use largest connected component       
        #G  = sorted(nx.connected_components(G), key=len, reverse=True)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])


    pathlengths = []


    for v in G.nodes():
        
        spl = dict(nx.single_source_shortest_path_length(G, v))
        for p in spl:
            pathlengths.append(spl[p])	
        
        
    avg_path_length = (sum(pathlengths) / len(pathlengths))
    median_path = statistics.median(pathlengths)

    std_path = statistics.stdev(pathlengths)	
    skw_path = skew(pathlengths)
    kurt_path = kurtosis(pathlengths)


            
    return {"mean path length":avg_path_length, "median path length":median_path, "std path length":std_path, "skw path length":skw_path, "kurtosis path length": kurt_path}


def clustering_coefficient(G, nodes=None, weight=None, count_zeros=True):
    """
    computes clustering coefficient for networkx graph
    a complete graph has CC of 1

    Input
        networkx graph object

        nodes (container of nodes, optional (default=all nodes in G)) – Compute average clustering for nodes in this container.

        weight (string or None, optional (default=None)) – The edge attribute that holds the numerical value used as a weight. If None, then each edge has weight 1.

        count_zeros (bool) – If False include only the nodes with nonzero clustering in the average.

    Output
        float
    """

    return nx.average_clustering(G, nodes=nodes, weight=weight, count_zeros=count_zeros)

def contains_triangles(G, nodes=None):
    """
    computes number of triangles contained in networkx G

    Input
        networkx graph object

        optional specific nodes can be set in nodes as list

    Output
        dict
    """

    return nx.triangles(G, nodes=nodes)


def degree_centrality(G, distribution=True):
    """
    Compute the degree centrality for nodes.
        The degree centrality for a node v is the fraction of nodes it is connected to.

    Input

        G (graph)
            A networkx graph
        distribution (optional)
            if distributional parameters should be returned

    Output

        Dictionary 

    """

    degree_centrality = nx.degree_centrality(G)

    centralities = list(degree_centrality.values())

    mean = None
    median = None
    std = None
    skw = None
    kurt = None

    if distribution:
        if len(centralities) > 1:
            mean = statistics.mean(centralities)
            median = statistics.median(centralities)
            std = statistics.stdev(centralities)
            skw = skew(centralities)
            kurt = kurtosis(centralities)
    else:
        print("not engough values to estimate distribution") 

    return {"centrality":degree_centrality, "mean_centrality":mean, "median_centrality":median, "std_centrality":std, "skew_centrality":skw, "kurtosis_centrality":kurt}



def eigenvector_centrality(G, distribution=True, max_iter=100, tol=1e-06, nstart=None, weight=None):
    """
    Compute the degree centrality for nodes.
        The degree centrality for a node v is the fraction of nodes it is connected to.

    Input

        G (graph) – A networkx graph
        max_iter (integer, optional (default=100)) – Maximum number of iterations in power method.

        tol (float, optional (default=1.0e-6)) – Error tolerance used to check convergence in power method iteration.

        nstart (dictionary, optional (default=None)) – Starting value of eigenvector iteration for each node.

        weight (None or string, optional (default=None)) – If None, all edge weights are considered equal. Otherwise holds the name of the edge attribute used as weight.

        distribution (optional, default = True)
            if distributional parameters should be returned

    Output

        dict
    """

    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol, nstart=nstart, weight=weight)

    centralities = list(eigenvector_centrality.values())

    mean = None
    median = None
    std = None
    skw = None
    kurt = None

    if distribution:
        if len(centralities) > 1:
            mean = statistics.mean(centralities)
            median = statistics.median(centralities)
            std = statistics.stdev(centralities)
            skw = skew(centralities)
            kurt = kurtosis(centralities)
    else:
        print("not engough values to estimate distribution") 

    return {"centrality":eigenvector_centrality, "mean_centrality":mean, "median_centrality":median, "std_centrality":std, "skew_centrality":skw, "kurtosis_centrality":kurt}



def closeness_centrality(G, distribution=True, u=None, distance=None, wf_improved=True):
    """
    Compute the degree centrality for nodes.

    The degree centrality for a node v is the fraction of nodes it is connected to.

    Input

        G (graph) – A networkx graph
        max_iter (integer, optional (default=100)) – Maximum number of iterations in power method.

        u (node, optional) – Return only the value for node u

        distance (edge attribute key, optional (default=None)) – Use the specified edge attribute as the edge distance in shortest path calculations

        wf_improved (bool, optional (default=True)) – If True, scale by the fraction of nodes reachable. This gives the Wasserman and Faust improved formula. For single component graphs it is the same as the original formula.
    
        distribution (optional, default = True)
            if distributional parameters should be returned

    Output
        dict
    """

    closeness_centrality = nx.closeness_centrality(G, u=u, distance=distance, wf_improved=wf_improved)

    centralities = list(closeness_centrality.values())

    mean = None
    median = None
    std = None
    skw = None
    kurt = None

    if distribution:
        if len(centralities) > 1:
            mean = statistics.mean(centralities)
            median = statistics.median(centralities)
            std = statistics.stdev(centralities)
            skw = skew(centralities)
            kurt = kurtosis(centralities)
    else:
        print("not engough values to estimate distribution") 

    return {"centrality":closeness_centrality, "mean_centrality":mean, "median_centrality":median, "std_centrality":std, "skew_centrality":skw, "kurtosis_centrality":kurt}



def betweeness_centrality(G, distribution=True, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    """
    Compute the degree centrality for nodes.

    The degree centrality for a node v is the fraction of nodes it is connected to.

    Input

        G (graph) – A networkx graph

        k (int, optional (default=None)) – If k is not None use k node samples to estimate betweenness. The value of k <= n where n is the number of nodes in the graph. Higher values give better approximation.

        normalized (bool, optional) – If True the betweenness values are normalized by 2/((n-1)(n-2)) for graphs, and 1/((n-1)(n-2)) for directed graphs where n is the number of nodes in G.

        weight (None or string, optional (default=None)) – If None, all edge weights are considered equal. Otherwise holds the name of the edge attribute used as weight.

        endpoints (bool, optional) – If True include the endpoints in the shortest path counts.

        seed (integer, random_state, or None (default)) – Indicator of random number generation state. See Randomness. Note that this is only used if k is not None.

        distribution (optional, default = True)
            if distributional parameters should be returned

    Output
        dict

    """

    betweenness_centrality = nx.betweenness_centrality(G,k=k, normalized=normalized, weight=weight, endpoints=endpoints, seed=seed)

    centralities = list(betweenness_centrality.values())

    mean = None
    median = None
    std = None
    skw = None
    kurt = None

    if distribution:
        if len(centralities) > 1:
            mean = statistics.mean(centralities)
            median = statistics.median(centralities)
            std = statistics.stdev(centralities)
            skw = skew(centralities)
            kurt = kurtosis(centralities)
    else:
        print("not engough values to estimate distribution") 

    return {"centrality":betweenness_centrality, "mean_centrality":mean, "median_centrality":median, "std_centrality":std, "skew_centrality":skw, "kurtosis_centrality":kurt}

