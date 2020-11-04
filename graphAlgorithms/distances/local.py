"""
LOCAL GRAPH METRICS

RIGHT NOW ONLY CONTAINS NON NODE SPECIFIC GRAPHLET COUNT OPTIONS
MULTIPLE OPTIONS AND NODE SPECIFIC OPTION NEED TO BE INTEGRATED 
NEED TO BE ADJUSTED TO ASYNC AND MULTIPROCESSES
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
from scipy.stats import kurtosis, skew
import statistics
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic as isomorphic
from networkx.generators.atlas import graph_atlas_g



def connected_component_subgraphs(G):
    """
    replaces deprecated networkx function, which is still used in networkx generate_motifs code
    """
    for c in nx.connected_components(G):
        yield G.subgraph(c)




def generate_motifs(x=208):
    """ 
    Return the atlas of all connected graphs of 5 nodes or less.
        Attempt to check for isomorphisms and remove.

    uses networkx graph atlas, which returns all possible graphs with up to 7 nodes
    based on https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_atlas.html

    Input
        x: which motifs of graph atlas are returned
    Output
        graph atlas
    """

    #uses networkx graph atlas, which returns all possible graphs with up to 6 nodes
    #based on https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_atlas.html
    Atlas = graph_atlas_g()[0:x]  
    # remove isolated nodes, only connected graphs are left
    U = nx.Graph()  # graph for union of all graphs in atlas
    for G in Atlas:
        zerodegree = [n for n in G if G.degree(n) == 0]
        for n in zerodegree:
            G.remove_node(n)
        U = nx.disjoint_union(U, G)

    # list of graphs of all connected components
    C = (U.subgraph(c) for c in nx.connected_components(U))

    UU = nx.Graph()
    # do quick isomorphic-like check, not a true isomorphism checker
    nlist = []  # list of nonisomorphic graphs
    for G in C:
        # check against all nonisomorphic graphs so far
        
        
            if not iso(G, nlist):
                nlist.append(G)
                UU = nx.disjoint_union(UU, G)  # union the nonisomorphic graphs
    return UU


def iso(G1, glist):
    """
    helper function of generate motifs
    """
    for G2 in glist:
        if isomorphic(G1, G2):
            return True
    return False
	
	
	
def get_neighbors(nodes, G):
    """
    get all neighbors of a node in G

    Input
        nodes
            list of node ids

        G
            networkx graph object

    Output
        list of neighbors 
    """
    neighbors = []
    for node in nodes:
        neighbors_node = list(G[node])
        
        for neighbor in neighbors_node:
            neighbors.append(neighbor)
            
    return neighbors
	
	
def count_graphlets(G, H, estimate_on=300, edge_attribute=None):
    
    """
    counts graphlets occuring in G
        iterate over network graph G and get all possible subgraphs of size k
    
    Input
        G is networkx graph
        H (networkx graph) is graphlet (generated with generate motifs to cmpare graph with)

        graphlets are estimated based on a random selection of x nodes which cannot be larger than the number of nodes in G
            which are specified in estimate_on

        if edge attribute is not None, then based on the provided edge weight the size of the graphlets will be returned as list
            which can be used to estimate its size distributions

    Output
        count of graphlets
        list of the found graphlets in G
        list of size of the estimated graphlets
            if edge_attribute = None list only contains 0s


    """
    nr_graphlets = 0
    graphlets = []
    length_graphlet = []
    
    #pick 300 nodes at random
    count = estimate_on

    for l in range(count):
        subgraph_list = []
        #get all nodes k steps apart from node n (size of H)
        #select node at random
        temp = list(nx.nodes(G))
        node = random.choice(temp)
        neighbors = [node]
        for k in range(nx.number_of_nodes(H)):
            for neighbor in neighbors:
                subgraph_list.append(neighbor)
            neighbors = get_neighbors(neighbors, G)

        #build subgraph to investigate all possible structures
        motif_neighborhood = G.subgraph(subgraph_list)

        #find possible sets
        for sub_nodes in itertools.combinations(motif_neighborhood.nodes(),len(H.nodes())):
            subgraph = G.subgraph(sub_nodes)
            if nx.is_connected(subgraph) and nx.is_isomorphic(subgraph, H) and (subgraph.edges() not in graphlets):
                graphlets.append(subgraph.edges())
                nr_graphlets = nr_graphlets + 1
                #print(nr_graphlets)
                length_temp = 0
                if edge_attribute is not None:
                    for edge in list(nx.get_edge_attributes(subgraph, edge_attribute).keys()):
                        length_temp = length_temp + nx.get_edge_attributes(subgraph, edge_attribute)[edge]
                
                #sum of edges per graphlet
                #print(nx.get_edge_attributes(subgraph, "thickness"))
                length_graphlet.append(length_temp)
                #average edge radius per graphlet
                

    return nr_graphlets, graphlets, length_graphlet


def iterate_graphlets(G, estimate_on=300, edge_attribute=None, motif_min_size=3, motif_max_size=6):

    """
    function to count over all defined motifs in graph G

    Input
        networkx graph object

        graphlets are estimated based on a random selection of x nodes which cannot be larger than the number of nodes in G
            which are specified in estimate_on

        if edge attribute is not None, then based on the provided edge weight the size of the graphlets will be returned as list
            which can be used to estimate its size distributions

        motif_min_siz: number of nodes in smallest graphlet to be counted, min permitted value is 2

        motif_max_size: number of nodes in largest graphlets, max permitted value is 6

    Output
        counted graphlets (dict)

        all graphlets found (dict)

        size of all graphlets (dict), all 0 if edge_attribute = None

        motifs (list): list index corresponds to dict keys

    """
    if motif_max_size <= 6 and motif_min_size >= 2:

        #get all motifs
        H = generate_motifs()
        motifs = [H.subgraph(c) for c in nx.connected_components(H)]

        graphlet_counter = {}
        graphlets_found = {}
        graphlet_size = {}

        for i in range(len(motifs)):

            
            motif = motifs[i]

            if int(nx.number_of_nodes(motif)) >= motif_min_size and int(nx.number_of_nodes(motif)) <= motif_max_size:
                print("counting graphlet", i)
                nr_graphlets, graphlets, length_graphlet = count_graphlets(G, motif, estimate_on=estimate_on, edge_attribute=edge_attribute)

                graphlet_counter[i] = nr_graphlets
                graphlets_found[i] = graphlets
                graphlet_size[i] = length_graphlet

    else:
        print("motif min and/or max value not permitted")
        graphlet_counter = {}
        graphlets_found = {}
        graphlet_size = {}
        motifs = None


    return graphlet_counter, graphlets_found, graphlet_size, motifs



def generate_node_specific_graphlets(nodes, graphlet_size=3):
    """
    function to generate all node specific graphlets of given size between all given nodes

    Input
        nodes is list of all possible nodes

        graphlet size is number of nodes of to be generated graphlets

    Output
        dict of graphlet id and values is edge list of graphlet
    """
    
    pos_graphlets = {}
    cnt = 0


    #generate all possible node combinations
    node_comb = itertools.combinations(nodes, graphlet_size)

    for g in node_comb:
        #get all possible edges
        pos_edges = list(itertools.combinations(g, 2))
        #min edges to have a connected graph
        min_edges = len(g) - 1
        print("generating graphlets ", len(g), "nodes")
        #create all possible graphs with min min_edges
        for x in range(min_edges, len(pos_edges)+1):
            
            temp_edges = itertools.combinations(pos_edges, x)

            #print("temp edges", temp_edges)

            for e in temp_edges:
            #create graph and check if connected
                X  = nx.Graph()
                
                X.add_edges_from(e)

                if nx.is_connected(X):
                    #save
                    #print("is connected")
                    pos_graphlets[cnt] = e
                    cnt = cnt + 1

    return pos_graphlets





def find_graphlet(G, graphlets):
    """
    function that checks if graphlets (node/edge specific) are in graph G

    Input
        networkx graph G

        dict of graphlets where key is graphlet id and value is edge list

    Output
        dict containing graphlet id and value of 0 (False) or 1 (True) if graphlet is in Graph
    """
    in_graph = {}

    for g in graphlets.keys():
        #construct a graph from edge list
        e = graphlets[g]
        #print("len e", len(e))
        X  = nx.Graph()   
        X.add_edges_from(e)

        nodes = list(X.nodes())

        #get subgraph from G with this nodes
        S = G.subgraph(nodes)


        #check if S & X are the same
        if nx.is_isomorphic(X, S):
            #print("edges S", S.edges())
            #print("edges X", X.edges())
            if g in in_graph.keys():
                print("error")
            in_graph[g] = 1
        else:
            in_graph[g] = 0

    return in_graph


