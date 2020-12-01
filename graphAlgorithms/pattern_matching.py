'''
Functions to find common structures / communities between a group of graphs or statistical overrepresented structures / communities.
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
import itertools
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
from collections import Counter
import markov_clustering as mc
import pickle
from netneurotools import cluster
from netneurotools import plotting
from sklearn.cluster import AgglomerativeClustering
import sklearn
from pyclustering.cluster.optics import optics, ordering_analyser, ordering_visualizer
import pyclustering
from random import randrange
from pyclustering.cluster.kmedoids import kmedoids
import itertools
#sys.path.insert(1, 'distances/')
import graphAlgorithms.simplification as simplification
import graphAlgorithms.clustering as clustering
import bct
import community as community_louvain
from sklearn.utils.validation import check_random_state
from collections import Counter
import scipy
import statsmodels.stats.multitest as multi
from scipy.stats import hypergeom
from itertools import combinations


def get_common_subgraph(matrices, p=0.5):
    """
    Retrieve subgraph structure based on edges that are common in at least p fraction of the provided graphs.
    
    Parameters:
        matrices (list): list of adjacency matrices. Nodes need to be ordered the same in all matrices. 1 indicates that the edge exists and 0 that the edge does not exist.
        p (float): in [0,1]: fraction in how many networks an edge should be present to be kept.
        
    Returns:
        common subgraph (matrix): An adjacency matrix where a 1 indicates that this edge is present in at least p fraction networks.
       
    """
    
    m = len(matrices)
    
    for i in range(len(matrices)):
        if i == 0:
            M = matrices[i]
        else:
            M = M + matrices[i]
            
            
    cut = m * p
    
    M[M < cut] = 0
    M[M>= cut] = 1
    
    return M



def get_statistical_overrepresented_edges(clusters):
    """
    Finds substructures (edges) that are statistically overrepresented in a cluster (group of networks) based on a hypergeometric function and Benjamin-Hochberg correction.
    All networks need to have the same nodes.

    Parameters:
        clusters (dict): key is cluster id and value is list of adjacency matrices (containing only 0s and 1s). Nodes need to be in the same order in the matrices.
 
    Returns:
        p values (dict): non corrected p-values as estimated by the hypergeometric function. Keys are cluster IDs and values are adjacency matrices where cell values are p-values.
        corrected p values (dict): p-values corrected with a Benjamin-Hochberg correction. Keys are cluster IDs and values are adjacency matrices where cell values are p-values.
        
    """
    
    pval_matrices = {}
    adjpval_matrices = {}
    #build background by adding up all matrices
    cnt = 0
    
    for key in clusters.keys():
        for m in clusters[key]:
            if cnt == 0:
                B = m
            else:
                B = B + m
                
            cnt = cnt + 1
            
                       
            
    B = np.array(B) 
   
    #perform for each edge in each cluster hypergeometric test
    for cl in clusters.keys():
        for i in range(len(clusters[cl])):
            if i == 0:
                C = clusters[cl][i]
            else:
                C = C + clusters[cl][i]
        C = np.array(C)
         
        #go through each edge and build pval matrix
        pval_matrix = np.zeros(C.shape)
        
        ids = []
        for i in range(len(pval_matrix)):
            ids.append(i)

        indices = list(combinations(ids, 2))
        
        for ind in indices:
            M = cnt #number networks
            n = B[ind[0]][ind[1]] #how often does this edge exist in all graphs
            N = len(clusters[cl])
            x = C[ind[0]][ind[1]]
            
            v = hypergeom.sf(x-1, M, n, N)
            
            pval_matrix[ind[0]][ind[1]] = v
            
            
        pval_matrices[cl] = pval_matrix
        
    #adjust values
    #combine all pval matrices into a flat list
    f1 = []
    for m in pval_matrices.values():
        f1.append(m.flatten())
        x = len(m.flatten())
        
    fl = [item for sublist in f1 for item in sublist]


    adj_pval = multi.multipletests(fl, method="fdr_bh")[1]
    
    #split into different clusters
    tt = [adj_pval[i:i+x] for i in range(0, len(adj_pval), x)]
    
    
    for cl in range(len(tt)):
        #transform back into matrix
        tr = [tt[cl][i:i+len(pval_matrix)] for i in range(0, len(tt[cl]), len(pval_matrix))]
        #print(tr)
        t = []
        for r in tr:
            t.append(r.tolist())

        adjpval_matrices[cl+1] = np.matrix(t)
        

        
        
        
    return pval_matrices, adjpval_matrices

def build_graph_remove_isolates(mat):
    """
    Constructs a networkx graph from an adj matrix and removes isolated nodes.
    
    Parameters:
        mat (matrix): adjacency matrix
    
    Returns:
        graph (networkX graph object):
        
    """
    
    G = nx.from_numpy_matrix(mat)
    
    iso = list(nx.isolates(G))
    G.remove_nodes_from(iso)
    
    return G


def get_consensus_community(networks, nodes, rep_network=10, seed=123, threshold=0.75, per_node=True, rep=10):
    """
    Finds a consensus community distribution between all provided networks. For each network communities are identified based on the Louvain algorithm.
    A consensus is constructed from these based on clustering.consensus_clustering(). All networks need to contain the same nodes.
    
    Parameters:
        networks (list): list of networkX graph objects.
        nodes (list): of all nodes. Ordering will be used in the output.
        rep_network (int): how often the Louvain algorithm should be applied to each network during the community detection stage.
        seed (int): random seed to be used for any random processes in clustering.consensus_clustering()
        treshold (float or str): if float needs to be in [0,1]. If float and not per_node all edges with a weight lower than the treshold are removed
            in the agreement graph. If float and per_node is True then for each node the weakes treshold % of edges are removed, if they are weak for 
                both nodes making up the edge. If treshold is "matrix" then the teshold is estimated based on a permutation of the clusterings as implemented in netneurotools.
        per_node (boolean): if True treshold is applied on a per node pasis else it is applied gloablly.
        rep (int): how often louvain clustering is reapeated on the agreement graph.
                
    Returns:
        communities (list): of community IDs ordered as in nodes
        
    """
    node_dict = {}
    for n in nodes:
        node_dict[n] = 0
    
    
    clusterings = []
    
    for net in networks:
        for i in range(rep_network):
            partion = community_louvain.best_partition(net, weight="weight")
            
            #sort to be in same order as nodes
            temp_dict = node_dict.copy()
            for key in partion.keys():
                temp_dict[key] = partion[key]
                
            clusterings.append(np.array(list(temp_dict.values())))
            
    
            
    #call consensus algorithm
    cons = clustering.consensus_clustering(np.column_stack(clusterings), seed=seed, threshold=threshold, per_node=per_node, rep=rep)
    
    
    return cons
    

def get_statistical_overrepresented_communities(clusters_networks, nodes, pval=0.05, partions = None):
    """
    Estimates a background distribution on how likely it is for each node pair to fall in the same community.
    Based on this it can be calculated if specific communities are overrepresented within a cluster (group of networks). 
    P values are estimated based on a hypergeometric function + a Benjamin Hochberg correction. Communities are estimated on a louvain parition.

    Parameters:
        cluster_networks (dict): key is cluster ID and value is list of networkX graph objects. All networks need to have the same nodes
        nodes (list): of node IDs. 
        p (float): [0,1]. P value cutoff applied, which estimates if a node pair is statistically enriched.
        partions (dict or None): if None the community detection based on Louvain is performed on each network provided in clusters_networks.
            If it is not None a custom partioning can be provided. Key needs to be cluster ID same as in clusters_networks. 
            Value needs to be list of dicts where key is node ID and value is Community ID of performed partitionings order as the networks provided in clusters_networks.
        
    Returns:
        communities (dict): key is cluster ID and value is the community ID. Ordered the same way as nodes. 
        Communities with a single node, indicates that that node was not statistically significant connectet to any other node.
    """

    statistical_communities_back = {}

    
        
    A = np.zeros((len(nodes), len(nodes)))


        
    for cl in clusters_networks.keys():
        node_dict = {}
        for n in nodes:
            node_dict[n] = 0
        print(cl)
        communities = []
        for ii in range(len(clusters_networks[cl])):
                net = clusters_networks[cl][ii]
                if partions is None:
                    partion = community_louvain.best_partition(net, weight="weight")
                else:
                    partion = partions[cl][ii]
                
                #sort to be in same order as nodes
                temp_dict = node_dict.copy()
                for key in partion.keys():
                    temp_dict[key] = partion[key]
                    
                partion2 = np.array(list(temp_dict.values()))
                #print("communities found", max(partion2)+1)
                #print("length", len(partion2))
                #transform into adjacency matrix
                B = A.copy()
                pairs = itertools.combinations(range(len(partion2)), 2)
                for p in pairs:
                    i1 = int(p[0])
                    i2 = int(p[1])
                    
                    if partion2[i1] == partion2[i2]:
                        B[i1][i2] = 1
                        B[i2][i1] = 1
                    
                communities.append(B)
                
        
        statistical_communities_back[cl] = communities

        pval_matrix_back, adj_pval_matrix_back = get_statistical_overrepresented_edges(statistical_communities_back)


        statistical_communities = {}

        for cl in adj_pval_matrix_back.keys():
            node_dict = {}
            
            #print("cluster statistical overrepresented", cl)
            
            M = adj_pval_matrix_back[cl].copy()
            M[M >= pval] = 0
            #M[M < 0.05] = 1
            
            #build graph
            G = nx.from_numpy_matrix(M)
            
            #disconnected components are communities
            
            graphs = list(nx.connected_component_subgraphs(G))
            
            for i in range(len(graphs)):
                T = graphs[i]
                for node in list(T.nodes()):
                    node_dict[node] = i
                    
            com = list(node_dict.values())
            
            statistical_communities[cl] = com



    return statistical_communities



def generate_community_overlap_background_distribution(communities, runs=1000, steps=5):
    """
    Estimates how likely it is that two graphs have communities with x% overlapping nodes. All graphs need ot have the same nodes.     
    Runs times, two communities are selected at random and their node % overlap is estimated.
    
    Parameters:
        communities (list): of dicts were keys are node IDs and values are community IDs.
        runs (int): how often random sampling should be performed to estimate the background distribution.
        steps (int): in [1,100]. Results are summarized into sections. For example steps = 5 would return results for sections 1-5%; 6-10%; 11-15% ... .
        
    Returns:
        background (dict): key is % overlap (rounded and summaized to steps) and value is how often this value has been scored.
       
    """
    
    s = int(100 / steps)
    
    back = {}
    cur = 1
    for i in range(s):
        if i == 0:
            back[cur] = 0
        else:
            back[cur+steps] = 0
            cur = cur + steps
            
            
    for x in range(runs):
        #select two networks at random 
        
        n1 = random.choice(range(len(communities)))
        n2 = random.choice(range(len(communities)))
        
        #select random community id from these two networks
        c1 = random.choice(list(Counter(communities[n1].values())))
        c2 = random.choice(list(Counter(communities[n2].values())))
        
        #get nodes of these communities
        nodes1 = []
        
        for key, value in communities[n1].items():
            if value == c1:
                nodes1.append(key)
                
                
        nodes2 = []
        
        for key, value in communities[n2].items():
            if value == c2:
                nodes2.append(key)
                
                
        flat_list = [item for sublist in [nodes1, nodes2] for item in sublist]


        #get overlap score
        total = len(list(Counter(flat_list)))
        
        #overlapping
        cnt = 0
        for n in nodes1:
            if n in nodes2:
                cnt = cnt +1
                
        p = int((100/ total) * cnt)
        
        
        for i in range(s):
            if p + i in back.keys():
                sec = p+i
        
        temp = back[sec] 
        back[sec] = temp + 1
        
        
    return back
        
        
    
    
def significantly_similar_partitionings(all_networks, back, is_file=False, steps=5):
    """
    Compares networks pairwise, based on if their community-node-overlap is statistically significant.
    For each network community detection based on louvain is perforemd. Statistical significant overlap is determained based on a provided background distribution as computed by
        generate_community_overlap_background_distribution().
    For each network pair a mean & median p-value (and adjusted p-value based on a benjamin hochberg correction) is estimated based on the p-value overlap score of each community pair between these networks.

    Parameters:
        all_networks (list): of networkX graph objects or file locations if is_file is True.
        back (dict): background community overlap distribution as computed by generate_community_overlap_background_distribution().
        is_file (boolean): if True all_networks should contain file locations to pickled graph objects. If False should contain the networkX graph objects.
        steps (int): in [1,100]. Results are summarized into sections. Needs to be the same value as used to generate the background distribution.

    Returns:
        mean p value (matrix): ordered as in all_networks.
        median p value (matrix): ordered as in all_networks.
        mean corrected p value (matrix): correction performed based on a Benjamin Hochberg correction. Ordered as in all_networks.
        median corrected p value (matrix): correction performed based on a Benjamin Hochberg correction. Ordered as in all_networks.
        
    """

    s = int(100 / steps)

    results_mean = np.zeros((len(all_networks), len(all_networks)))
    results_median = np.zeros((len(all_networks), len(all_networks)))

    results_mean_adj = np.zeros((len(all_networks), len(all_networks)))
    results_median_adj = np.zeros((len(all_networks), len(all_networks)))


    for index in itertools.combinations(range(len(all_networks)), 2):
        n1 = all_networks[index[0]]
        n2 = all_networks[index[1]]
        #get communities for both

        if is_file:
            G1 = nx.read_edgelist(n1)
            G2 = nx.read_edgelist(n2)
        else:
            G1 = n1
            G2 = n2
        
        c1 = community_louvain.best_partition(G1, weight="weight")
        c2 = community_louvain.best_partition(G2, weight="weight")
        
        pairs = list(itertools.product(list(Counter(c1.values())), list(Counter(c2.values()))))
        #for each community pair compare node overlap & get adj pval
        
    
        
        temp_p = []
        for p in pairs:
            x1 = p[0]
            x2 = p[1]
            
            #get overlap
            nodes1 = []
            
            for key, value in c1.items():
                if value == x1:
                    nodes1.append(key)
                    
                    
            nodes2 = []
            
            for key, value in c2.items():
                if value == x2:
                    nodes2.append(key)
                    
                    
            flat_list = [item for sublist in [nodes1, nodes2] for item in sublist]


            #get overlap score
            total = len(list(Counter(flat_list)))
            
            #overlapping
            cnt = 0
            for n in nodes1:
                if n in nodes2:
                    cnt = cnt +1
                    
            p = int((100/ total) * cnt)
            
            for i in range(s):
                if p + i in back.keys():
                    sec = p+i
            
            temp_p.append(sec)
            
        
        pvals = []
        for p in Counter(temp_p).keys():
        


            #get pval
            M = sum(back.values()) #number of samplings
            n = back[sec] #how often does such a overlap occure
            N = len(temp_p)
            x = Counter(temp_p)[p]

            v = hypergeom.sf(x-1, M, n, N)

            pvals.append(v)
            
        
        results_mean[index[0]][index[1]] = statistics.mean(pvals)
        results_mean[index[1]][index[0]] = statistics.mean(pvals)
        
        
        results_median[index[0]][index[1]] = statistics.median(pvals)
        results_median[index[1]][index[0]] = statistics.median(pvals)
        
        #adjust pvals
        adj_pval = multi.multipletests(pvals, method="fdr_bh")[1]
        
        results_mean_adj[index[0]][index[1]] = statistics.mean(adj_pval)
        results_mean_adj[index[1]][index[0]] = statistics.mean(adj_pval)
        
        
        results_median_adj[index[0]][index[1]] = statistics.median(adj_pval)
        results_median_adj[index[1]][index[0]] = statistics.median(adj_pval)



    return results_mean, results_median, results_mean_adj, results_median_adj