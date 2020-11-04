
"""
functions to simplify graph object
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
#sys.path.insert(1, '../../')
#import communities
import graphAlgorithms.communities as communities



def add_opposit_weight(G, current_weight="weight", new_weight="distance"):
    """
    function to convert distance weights into similarity weights and vice versa

    Input
        networkx graph object

        str of current weight attribute

        str of to be set weight attribute

    Output
        networkx graph object containing attributes for new weight attribute
    """

    for edge in G.edges():
            #update current edge
            G[edge[0]][edge[1]][new_weight] = 1 - G[edge[0]][edge[1]][current_weight]


    return G

def add_absolute_weight(G, current_weight="weight", new_weight="absolute"):
    """
    function to convert correlation weights into absolute weights

    Input
        networkx graph object

        str of current weight attribute

        str of to be set weight attribute

    Output
        networkx graph object containing attributes for new weight attribute
    """

    for edge in G.edges():
            #update current edge
            G[edge[0]][edge[1]][new_weight] = abs(G[edge[0]][edge[1]][current_weight])


    return G


def get_min_spanning_tree(G, weight="weight", is_distance=True, new_weight="distance"):
    """
    returns min spanning tree for G with minimum edge weight

    Input
        networkx graph object
            if graph is distance graph weights can be used directly else need to be converted => set is_distance = False

        weight specifies name of edge attribute to be used

        if is_distance is False weights are assumed to be similarities and distances are calculated from them
            then a new paramter new_weight is added to the graph
            
    Output
        returns min spanning tree 
    """


    if is_distance:
        T=nx.minimum_spanning_tree(G, weight=weight)


    else:
        #estimate distance values from similarity values
        G = add_opposit_weight(G, current_weight=weight, new_weight=new_weight)
        T=nx.minimum_spanning_tree(G, weight=new_weight)


    return T


def remove_edges_per_node(G, treshold=None, percentage=None, direction="bottom", attribute="weight"):
    """
    removes for each node its weakest or strongest edges, while keeping at least 1 edge per node
    if all are below treshold then none are removed

    Input
        networkx graph object G

        treshold or percentage need to be set not both
            either all edges below or above treshold are removed or top/ bottom percentage of a nodes 
            edges are removed, percentage needs to be in [0,1]

        direction
            if bottom weakest edges are removed
            if top, strongest edges are removed

        attribute
            edge attribute to be used to determine edge strength

    Output
        networkx graph object



    """
    if treshold is None and percentage is None:
        print("please set treshold or percentage ")
    elif treshold is not None and percentage is not None:
        print("please set either treshold or percentage not both")

    else:

        to_remove = {}

        for node in G.nodes():
            #get nodes edges
            edges = G.edges(node)
            #print(len(edges))

            if percentage is not None:

                if (len(edges) - int(len(edges) * percentage)) >= 1:
                    #remove
                    edges=sorted(G.edges(node,data=True), key=lambda t: t[2].get(attribute, 1))
                    #print(edges)
                    if direction == "top":
                        
                        all_edges = len(edges)
                        remove = edges[-int(all_edges*percentage):]
                    
                    else:
                        
                        all_edges = len(edges)
                        remove = edges[:int(all_edges*percentage)+1]
                        
                        #print(remove)
                        
                    print("possible removing ", len(remove), "edges for ", node)
                    #G.remove_edges_from(remove)
                    for e in remove:
                        e1 = e[0]
                        e2 = e[1]

                        if (e1, e2) in to_remove.keys() or (e2, e1) in to_remove.keys():
                            if (e1, e2) in to_remove.keys():
                                c = to_remove[(e1, e2)]
                                to_remove[(e1, e2)] = c+1
                            else:
                                c = to_remove[(e2, e1)]
                                to_remove[(e2, e1)] = c+1

                        else:
                            to_remove[(e1, e2)] = 1

            

                else:
                    print("too less edges for", node)

            elif treshold is not None:
                if direction == "top":
                    
                    remove = [(i,j) for i,j in G.edges(node) if G[i][j][attribute] > treshold]
                else:
                    
                    remove = [(i,j) for i,j in G.edges(node) if G[i][j][attribute] < treshold]

                if len(remove) == len(edges):
                    print("all edges are below treshold none are removed", node)
                else:
                    print("removing ", len(remove), "edges for ", node)
                    G.remove_edges_from(remove)

        if percentage is not None:
            print("start removing")
            remove = []
            for e in to_remove.keys():
                if to_remove[e] > 1:
                    remove.append(e)

            print("removing ", len(remove))
            G.remove_edges_from(remove)
                            


    return G






def remove_edges(H, treshold=None, percentage=None, based_on="weight", direction="bottom", attribute="weight"):

    """
    removes edges from H

    Input
        networkx graph object

        if based_on is weight then treshold (float) needs to be set to weight parameter and all edges taht are
            below (if direction is bottom) or above (if direction is top)
            attribute str specifiying which edge attribute should be used

        if based_on is betweenness
            edges based on betweenes are removed, either based on treshold value or percentage in the direction of direction
            percentage needs to be value between 0 and 1 top/bottm percentage edges are removed
            attribute str specifiying which edge attribute should be used

        
        direction states if top or bottom scoring edges are removed
            top: top scoring edges are removed or edges that are above treshold
            bottom: bottom scoring edges are removed or edges that are below treshold

        

    Output
        networkx graph object
    """

    G = H.copy()

    if type(based_on) == type([]):
        print("removing all edges contained in based_on")

        print("removing ", len(based_on), "edges")
        G.remove_edges_from(based_on)


    elif based_on == "weight":
        if treshold is not None and percentage is not None:
            print("please only set treshold or percentage not both")

        else:
            if treshold is not None:
                #remove edges that have weight larger or smaller tran defined in treshold
                if direction == "top":
                    print("edges with weight larger than ", treshold, "will be removed")
                    remove = [(i,j) for i,j in G.edges() if G[i][j][attribute] > treshold]
                else:
                    print("edges with weight smaller than ", treshold, "will be removed")
                    remove = [(i,j) for i,j in G.edges() if G[i][j][attribute] < treshold]

                print("removing ", len(remove), "edges")
                G.remove_edges_from(remove)

            elif percentage is not None:
                edges=sorted(G.edges(data=True), key=lambda t: t[2].get(attribute, 1))
                #print(edges)
                if direction == "top":
                    print("top ", percentage, "percent of edges are removed")
                    all_edges = len(edges)
                    remove = edges[-int(all_edges*percentage):]
                    

                    
                else:
                    print("bottom ", percentage, "percent of edges are removed")
                    all_edges = len(edges)
                    remove = edges[:int(all_edges*percentage)]
                    
                print("removing ", len(remove), "edges")
                G.remove_edges_from(remove)

            else:
                print("treshold or percentage need to be set to float, no edges will be removed")

    elif based_on == "betweenness":
        betweenness = nx.edge_betweenness_centrality(G, weight=attribute)
        #print(betweenness)

        #sort after values
        s = {k: v for k, v in sorted(betweenness.items(), key=lambda item: item[1])} #small to high
        edges = list(s.keys())

        if treshold is not None and percentage is not None:
            print("please only set treshold or percentage not both")

        else:
        

            if treshold is not None:
                #remove edges that have weight larger or smaller tran defined in treshold
                if direction == "top":
                    print("edges with betweenness larger than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] > treshold]
                else:
                    print("edges with betweenness smaller than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] < treshold]

                print("removing ", len(remove), "edges")
                G.remove_edges_from(remove)

            elif percentage is not None:

                if direction == "top":
                    print("top ", percentage, "percent of edges are removed")
                    all_edges = len(edges)
                    remove = edges[-int(all_edges*percentage):]
        
                else:
                    print("bottom ", percentage, "percent of edges are removed")
                    all_edges = len(edges)
                    remove = edges[:int(all_edges*percentage)]
                    
                print("removing ", len(remove), "edges")
                G.remove_edges_from(remove)

            else:
                print("please set treshold or percentage")

    else:
        print("simiplification method not known")

    return G



def remove_nodes(H, treshold=None, percentage=None, based_on="degree", direction="bottom"):

    """
    removes nodes from H

    Input
        networkx graph object

        if based_on is degree nodes are removed based on degree value

        if based_on is list all nodes in that list are removed

        if based_on is betweenness nodes are removed based on betweenes value

        if based_on is closeness nodes are removed based on closeness centrality value


        either treshold or percentage needs to be provided
            treshold is direct value on which nodes are removed while percentage removes x percentage of nodes (ranked by value)

        direction states if top or bottom scoring nodes are removed
            top: top scoring nodes are removed or nodes that are above treshold
            bottom: bottom scoring nodes are removed or nodes that are below treshold

    Output
        networkx graph object
    """

    G = H.copy()

    if type(based_on) == type([]):
        print("removing all nodes contained in based_on")

        print("removing ", len(based_on), "nodes")
        G.remove_nodes_frome(based_on)

    elif based_on == "degree":
        degree = nx.degree_centrality(G)
        #print(degree)
        #sort after values
        s = {k: v for k, v in sorted(degree.items(), key=lambda item: item[1])} #small to high
        nodes = list(s.keys())

        if treshold is not None and percentage is not None:
            print("please only set treshold or percentage not both")

        else:

        

            if treshold is not None:
                #remove nodes that have weight larger or smaller tran defined in treshold
                if direction == "top":
                    print("nodes with degree larger than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] > treshold]
                else:
                    print("nodes with degree smaller than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] < treshold]

                print("removing ", len(remove), "nodes")
                G.remove_nodes_from(remove)

            elif percentage is not None:

                if direction == "top":
                    print("top ", percentage, "percent of nodes are removed")
                    all_nodes = len(nodes)
                    remove = nodes[-int(all_nodes*percentage):]
        
                else:
                    print("bottom ", percentage, "percent of nodes are removed")
                    all_nodes = len(nodes)
                    remove = nodes[:int(all_nodes*percentage)]
                    
                print("removing ", len(remove), "nodes")
                G.remove_nodes_from(remove)

            else:
                print("please set treshold or percentage value")




    elif based_on == "betweenness":
        betweenness = nx.betweenness_centrality(G)
        #print(betweenness)
        #sort after values
        s = {k: v for k, v in sorted(betweenness.items(), key=lambda item: item[1])} #small to high
        nodes = list(s.keys())

        if treshold is not None and percentage is not None:
            print("please only set treshold or percentage not both")

        else:

            if treshold is not None:
                #remove nodes that have weight larger or smaller tran defined in treshold
                if direction == "top":
                    print("nodes with betweenness larger than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] > treshold]
                else:
                    print("nodes with betweenness smaller than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] < treshold]

                print("removing ", len(remove), "nodes")
                G.remove_nodes_from(remove)

            elif percentage is not None:

                if direction == "top":
                    print("top ", percentage, "percent of nodes are removed")
                    all_nodes = len(nodes)
                    remove = nodes[-int(all_nodes*percentage):]
        
                else:
                    print("bottom ", percentage, "percent of nodes are removed")
                    all_nodes = len(nodes)
                    remove = nodes[:int(all_nodes*percentage)]
                    
                print("removing ", len(remove), "nodes")
                G.remove_nodes_from(remove)

            else:
                print("please set treshold or percentage value")

    elif based_on == "closeness":
        closeness = nx.closeness_centrality(G)
        #print(closeness)
        #sort after values
        s = {k: v for k, v in sorted(closeness.items(), key=lambda item: item[1])} #small to high
        nodes = list(s.keys())

        if treshold is not None and percentage is not None:
            print("please only set treshold or percentage not both")

        else:

            if treshold is not None:
                #remove nodes that have weight larger or smaller tran defined in treshold
                if direction == "top":
                    print("nodes with closeness larger than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] > treshold]
                else:
                    print("nodes with closeness smaller than ", treshold, "will be removed")
                    remove = [key for key in s.keys() if s[key] < treshold]

                print("removing ", len(remove), "nodes")
                G.remove_nodes_from(remove)

            elif percentage is not None:

                if direction == "top":
                    print("top ", percentage, "percent of nodes are removed")
                    all_nodes = len(nodes)
                    remove = nodes[-int(all_nodes*percentage):]
        
                else:
                    print("bottom ", percentage, "percent of nodes are removed")
                    all_nodes = len(nodes)
                    remove = nodes[:int(all_nodes*percentage)]
                    
                print("removing ", len(remove), "nodes")
                G.remove_nodes_from(remove)
            else:
                print("please set treshold or percentage value")

    else:
        print("simiplification method not known")

    return G



def simplify_weak_community_connections(H, weights="weight", t=0.2, by_degree=True, r=5, min_community=None, std=0.2):
    """
    simplification method that siplifies the graph based on one run of alisas_communities()

    algorithm that tries to find week links between node groups, while ensuring that no new isolates occure

    for each node a probabilistic selection of neighbors is run for r rounds based on their edge weights
        neigbors with less than t occurenses will be disconnected but only if this link is selected as a weak link
            in both directions
        based on this it is ensured that a small nodegroup/ single node connected to a high degree node via a weak link is NOT
            disconnected, since this is the only community it belongs to

    this algorithm is tuned to similarity graphs, especially where edge weights encode aa strong association between nodes and where it
        is preferred that weak links are kept in the same community if there is no "better option" to assign them

    for graph simplification only one iteration is performed
    
    Input

        a networkx/ igraph object G

        weights edge attribute. 
            
        t treshold on which edges are "selected as weak" and to be removed if they occure <= in fraction
            of selected neighbors 


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

    Output

        networkx graph object

    """
   
    x, G = communities.alisas_communities(H, weights=weights, t=t, max_iter=1, by_degree=by_degree, r=r, std=std, min_community=min_community, graph=True)

    return G


