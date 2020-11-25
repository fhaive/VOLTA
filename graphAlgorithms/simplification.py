
"""
Functions to simplify graph objects.
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
    Function to convert distance weights into similarity weights and vice versa. Adds a new edge attribute that contains 1-x.

    Parameters:
        G (networkX graph object):
        current_weight (str): edge attribute to be converted
        new_weight (str): name of to be added edge attribute.

    Returns:
        graph (networkX graph object):
    """

    for edge in G.edges():
            #update current edge
            G[edge[0]][edge[1]][new_weight] = 1 - G[edge[0]][edge[1]][current_weight]


    return G

def add_absolute_weight(G, current_weight="weight", new_weight="absolute"):
    """
    Function to convert weights to absolute values.

    Parameters:
        G (networkX graph object):
        current_weight (str): edge attribute to be converted
        new_weight (str): name of to be added edge attribute.

    Returns:
        graph (networkX graph object):
    """

    for edge in G.edges():
            #update current edge
            G[edge[0]][edge[1]][new_weight] = abs(G[edge[0]][edge[1]][current_weight])


    return G


def get_min_spanning_tree(G, weight="weight", is_distance=True, new_weight="distance"):
    """
    Returns min spanning tree for G with minimum edge.

    Parameters:
        G (networkX graph object): needs to contains distances or similarities as weights.
        weight (str): edge attribute to be used
        is_distance (boolean): if True edge attributes are assumed to be distances. Else weights are assumend to be similarities.
        new_weight (str): if is_distance is False the estimated distances are added with this name to the graph.
                    
    Returns:
        minimum spanning tree (networkX graph object): 
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
    Removes for each node its weakest or strongest edges, while keeping at least 1 edge per node. If all are below thr treshold then none are removed.

    Parameters:
        G (networkX graph object):
        treshold (float or None): in [0,1]. Edges below or above this treshold are removed. If treshold is not None percentage needs to be None.
        percentage (float or None): in [0,1]. Percentage of top/ bottom ranked edges are removed for each node individually. If percentage is not None then treshold needs to be None.
        direction (str): options are "bottom" (edges below treshold or weakest percentage edges are removed) and "top" (edges above the treshold or highest ranked perecentage edges are removed).
        attribute (str): name of edge attribute to be used.
        
    Returns:
        simplified graph (networkX graph object)



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
    Removes edges from a graph.

    Parameters:
        H (networkX graph object):
        treshold (float or None): in [0,1]. Edges below or above this treshold are removed. If treshold is not None percentage needs to be None.
        percentage (float or None): in [0,1]. Percentage of top/ bottom ranked edges are removed. If percentage is not None then treshold needs to be None.
        based_on (str or list): if "weight" then treshold needs to be not None and all edges in the graph below or above this value are removed.
                If is list then items need to be edge IDs and all edges contained are removed.
                If is "betweenness" then either treshold or percentage can be set and top/ bottom edges are removed based on their edge betweenness scores.
        direction (str): options are "bottom" (edges below treshold or weakest percentage edges are removed) and "top" (edges above the treshold or highest ranked perecentage edges are removed).
        attribute (str): name of edge attribute to be used when based_on is "weight".       

    Returns:
        simplified graph (networkX graph object):
        
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
    Removes nodes from a graph.

        H (networkX graph object):
        treshold (float or None): in [0,1]. Nodes below or above this treshold are removed. If treshold is not None percentage needs to be None.
        percentage (float or None): in [0,1]. Percentage of top/ bottom ranked nodes are removed. If percentage is not None then treshold needs to be None.
        based_on (str or list): if is "degree" nodes are removed based on their degree centrality. If is "betweenness" nodes are removed on their betweenness centrality scores.
                If is "closeness" nodes are removed based on their closeness centrality scores. If is list then items need to be node IDs and these nodes are removed from the graph.
        direction (str): options are "bottom" (nodes below treshold or weakest percentage of nodes are removed) and "top" (nodes above the treshold or highest ranked perecentage nodes are removed).
        
    Returns:
        simplified graph (networkX graph object):
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
    Simplification method that siplifies the graph based on one run of communities.weak_link_communities().
    Tries to find weak links between node groups, while ensuring that no new isolates occure.
    For each node a probabilistic selection of neighbors is run for r rounds based on their edge weights.
    Neigbors with less than t occurenses will be disconnected but only if this link is selected as a weak link in both directions.
    The algorithm will stop when max_iter is reached or no edges can be removed anymore.
    Only one iteration of  communities.weak_link_communities() is performed.
    
    Parameters:
        G (networkX graph object):
        weights (str): edge attribute to be used.
        t (float): in [0,1]. Treshold on which edges are "selected as weak" and to be removed if they occure <= in of samples neighbors.
        by_degree (boolean): if True then the number of samplings for each node is based on its degree. Each node will sample from its neighbors r*node_degree times.
            If is false then each nodes neighbors will be sampled r times.
        r (int) how often a nodes neighbors are sampled.  
        std (float): treshold of a nodes neighboring sampling distribution standardiviationf. If it is above std then edge removal will be performed.
        min_community (int or None): minimum community size allowed. If already disconnected components of size < min_community exist, smaller communities can still occure. 
                If None will be ignored.
        
    Returns:
        simplified graph (networkX graph object):

    """
   
    x, G = communities.weak_link_communities(H, weights=weights, t=t, max_iter=1, by_degree=by_degree, r=r, std=std, min_community=min_community, graph=True)

    return G


