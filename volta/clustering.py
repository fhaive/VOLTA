'''
Clustering algorithms that take as input a distance matrix.
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
import volta.simplification as simplification
import bct
import community as community_louvain
from sklearn.utils.validation import check_random_state



def consensus_clustering(clusterings, seed=123, threshold=0.75, per_node=True, rep=10):
    """
    Finds a consensus clustering based on multiple provided clusterings.
    Based on netneurotools & bct for whole graph thresholds   https://github.com/netneurolab/netneurotools , https://github.com/aestrivex/bctpy.
    Builds an agreement graph from different clusterings, removes weak edges & performs community detection rep times (louvain) until convergance.
    Multiple options on how to identify weak edges are provided.

    Parameters:
        clusterings (list): list of arrays containing numeric class labels
        seed (int): seed to be used for random processes
        treshold (float or str): if float needs to be in [0,1]. If float and not per_node all edges with a weight lower than the treshold are removed
            in the agreement graph. If float and per_node is True then for each node the weakes treshold % of edges are removed, if they are weak for 
                both nodes making up the edge. If treshold is "matrix" then the teshold is estimated based on a permutation of the clusterings as implemented in netneurotools.
        per_node (boolean): if True treshold is applied on a per node pasis else it is applied gloablly.
        rep (int): how often louvain clustering is reapeated on the agreement graph.

    Returns:
        consensus class labels (array): of clustering
      
    
    """
    clusterings = np.column_stack(clusterings)

    rs = check_random_state(seed)
    samp, comm = clusterings.shape


    agreement = bct.clustering.agreement(clusterings, buffsz=samp) / comm
    
    null_assign = np.column_stack([rs.permutation(i) for i in clusterings.T])
    null_agree = bct.clustering.agreement(null_assign, buffsz=samp) / comm
    
    if threshold is None:
        threshold = np.mean(null_agree)
        print(threshold)
        
    if threshold == "matrix":
        #do on a per node basis
        print("matrix")
        cons = __get_consensus_node__(agreement, null_agree, rs, seed=seed, rep = rep)
        
            
        cons = np.array(cons)[0]
                
    else:
        if per_node:
            cons = __get_consensus_node_threshold__(agreement, node_threshold=threshold, seed=seed ,rep = rep)
            cons = np.array(cons)[0]
        else:
    
            consensus = bct.clustering.consensus_und(agreement, threshold, rep)


            cons = consensus.astype(int)
    
    return cons


def __get_consensus_node_threshold__(agreement, node_threshold=0.75, seed=123 ,rep = 10):

    """
    called by consensus_clustering in a per node case

    for explanation and parameters refer to consensus_clustering
    """
    
    #node treshold is mean of all connecting edges of node_treshold matrix
    
    #create treshold graph and node treshold
    
    #build graph from agreement matrix
    T = nx.from_numpy_matrix(agreement)
    #remove edges for each node that are below threshold %
    T = simplification.remove_edges_per_node(T, treshold=None, percentage=node_threshold, direction="bottom", attribute="weight")
        
    #run weighted louvain on T rep times
    clusterings = []
    for x in range(rep):
        partion = community_louvain.best_partition(T, weight="weight")
        clusterings.append(list(partion.values()))
        
    T = None
    #while clusterings are different re-run
    ct = np.zeros(agreement.shape)
    
    matrices = ct.copy()
    for c in clusterings:
        cm = ct.copy()
        
        for index in itertools.combinations(list(partion.keys()), 2):
            i0 = index[0]
            i1 = index[1]
            
            if c[i0] == c[i1]:
                cm[i0][i1] = 1
                cm[i1][i0] = 1
            else:
                cm[i0][i1] = 0
                cm[i1][i0] = 0
                
                
        matrices = matrices + cm
        
    matrices = matrices / len(clusterings)
    #if all clusterings are the same matrices should only have 0 and 1 values 
    #- set 1 values to 0 and check if vlaues > 0 are left
    ind = matrices == 1
    matrices[ind] = 0
    
    rerun = np.any(matrices[:, 0] > 0 )
    
    while rerun:
        print("rerun")
        clusterings = np.column_stack(clusterings)
        samp, comm = clusterings.shape

        agreement = bct.clustering.agreement(clusterings, buffsz=samp) / comm

        T = nx.from_numpy_matrix(agreement)
        #remove edges for each node that are below threshold %
        T = simplification.remove_edges_per_node(T, treshold=None, percentage=node_threshold, direction="bottom", attribute="weight")

        #run weighted louvain on T rep times
        clusterings = []
        for x in range(rep):
            partion = community_louvain.best_partition(T, weight="weight")
            clusterings.append(list(partion.values()))

        T = None
        #while clusterings are different re-run
        ct = np.zeros(agreement.shape)

        matrices = ct.copy()
        for c in clusterings:
            cm = ct.copy()

            for index in itertools.combinations(list(partion.keys()), 2):
                i0 = index[0]
                i1 = index[1]

                if c[i0] == c[i1]:
                    cm[i0][i1] = 1
                    cm[i1][i0] = 1
                else:
                    cm[i0][i1] = 0
                    cm[i1][i0] = 0


            matrices = matrices + cm

        matrices = matrices / len(clusterings)
        #if all clusterings are the same matrices should only have 0 and 1 values 
        #- set 1 values to 0 and check if vlaues > 0 are left
        ind = matrices == 1
        matrices[ind] = 0
        
        rerun = np.any(matrices[:, 0] > 0 )
 
        
    
    #print(clusterings)
    #add one
   
    return clusterings
        
    
        
def __get_consensus_node__(agreement, node_threshold, rs, seed=123 ,rep = 10):

    """
    called by consensus_clustering in a per node case

    for explanation and parameters refer to consensus_clustering
    """
    
    #node treshold is mean of all connecting edges of node_treshold matrix
    
    #create treshold graph and node treshold
    T = nx.from_numpy_matrix(node_threshold)
    
    t = {}
    for i in range(len(node_threshold)):
        w = []
        for e in T.edges(i):
            w.append(T[e[0]][e[1]]["weight"])
        t[i] = statistics.mean(w)
        
    #build graph from agreement matrix
    T = nx.from_numpy_matrix(agreement)
    #remove edges for each node that are below threshold
    for node in t.keys():
        th = t[node]
        
        d = []
        #get all edges and select ones to be deleted
        for e in T.edges(node):
            if T[e[0]][e[1]]["weight"] < th:
                d.append(e)
                
        #remove edges
        T.remove_edges_from(d)
        
    #run weighted louvain on T rep times
    clusterings = []
    for x in range(rep):
        partion = community_louvain.best_partition(T, weight="weight")
        clusterings.append(list(partion.values()))
        
    T = None
    #while clusterings are different re-run
    ct = np.zeros(agreement.shape)
    
    matrices = ct.copy()
    for c in clusterings:
        cm = ct.copy()
        
        for index in itertools.combinations(list(partion.keys()), 2):
            i0 = index[0]
            i1 = index[1]
            
            if c[i0] == c[i1]:
                cm[i0][i1] = 1
                cm[i1][i0] = 1
            else:
                cm[i0][i1] = 0
                cm[i1][i0] = 0
                
                
        matrices = matrices + cm
        
    matrices = matrices / len(clusterings)
    #if all clusterings are the same matrices should only have 0 and 1 values 
    #- set 1 values to 0 and check if vlaues > 0 are left
    ind = matrices == 1
    matrices[ind] = 0
    
    rerun = np.any(matrices[:, 0] > 0 )
    
    while rerun:
        print("rerun")
        clusterings = np.column_stack(clusterings)
        samp, comm = clusterings.shape

        agreement = bct.clustering.agreement(clusterings, buffsz=samp) / comm

        null_assign = np.column_stack([rs.permutation(i) for i in clusterings.T])
        node_threshold = bct.clustering.agreement(null_assign, buffsz=samp) / comm
        
        T = nx.from_numpy_matrix(node_threshold)
    
        t = {}
        for i in range(len(node_threshold)):
            w = []
            for e in T.edges(i):
                w.append(T[e[0]][e[1]]["weight"])
            t[i] = statistics.mean(w)

        #build graph from agreement matrix
        T = nx.from_numpy_matrix(agreement)
        #remove edges for each node that are below threshold
        for node in t.keys():
            th = t[node]

            d = []
            #get all edges and select ones to be deleted
            for e in T.edges(node):
                if T[e[0]][e[1]]["weight"] < th:
                    d.append(e)

            #remove edges
            T.remove_edges_from(d)

        #run weighted louvain on T rep times
        clusterings = []
        for x in range(rep):
            partion = community_louvain.best_partition(T, weight="weight")
            clusterings.append(list(partion.values()))

        T = None
        #while clusterings are different re-run
        ct = np.zeros(agreement.shape)

        matrices = ct.copy()
        for c in clusterings:
            cm = ct.copy()

            for index in itertools.combinations(list(partion.keys()), 2):
                i0 = index[0]
                i1 = index[1]

                if c[i0] == c[i1]:
                    cm[i0][i1] = 1
                    cm[i1][i0] = 1
                else:
                    cm[i0][i1] = 0
                    cm[i1][i0] = 0


            matrices = matrices + cm

        matrices = matrices / len(clusterings)
        #if all clusterings are the same matrices should only have 0 and 1 values 
        #- set 1 values to 0 and check if vlaues > 0 are left
        ind = matrices == 1
        matrices[ind] = 0
        
        rerun = np.any(matrices[:, 0] > 0 )
 
        
    
    #print(clusterings)
    #add one
   
    return clusterings
        
    
        

    
    


def multiobjective(X, labels, min_number_clusters=2, max_number_clusters=None, min_cluster_size = 10, max_cluster_size=None, local =True, bet=True, e=0.5, s=0.5, cluster_size_distribution = True):
    
    """
        Multi objective function to evaluate clusterings. 
        
        Parameters:
            X (matrix): distance matrix, as used for clustering.
            labels (list): list of predicted labels. Needs to be in the same order as X.
            min_number_clusters (int or None): minimum number of allowed clusters, if less than a penalty is applied. If None will not be taken into account. 
            max_number_clusters (int or None): maximum number of clusters, if more then penalty is applied. If None will not be taken into account.
            min_cluster_size (int or None): for each cluster with less than x items a penalty is applied. If None will not be taken into account .
            max_cluster_size (int or None): for each cluster with more than x items a penalty is applied. If None will not be taken into account.
            local (boolean): if is True then objective aims at minimizing within cluster similarity based on thr data provided in X. If is False will be ignored.
            bet (boolean): if True objective aims at maximizing dissimilarity between clusters. If False will be ignored.
            e (float): in [0,1]. For each cluster with a mean similarity less than e an additional penalty is applied. If None will be ignored.
            s (float): in [0,1]. Between each cluster pair where thr distance is less than s an additional penalty is applied. If None will be ignored.
            cluster_size_distribution (boolean): if True mean difference of cluster size for each cluster to "most equal" partitioning is applied. Most equal partitioning is len(labels) / number of clusters. If False will be ignored.
            
        Returns:
            clustering score (float): the closer to 0 the better the clustering is with regards to the selected objectives.            
            
    """
    
    
    obj = 0
    
    clus = Counter(labels)
    
    nr_clusters = max(labels)+1
    
    #transform labels into list of lists for each cluster one with index of items
    
    ll = []
    
    for r in range(nr_clusters):
        indices = [i for i, x in enumerate(labels) if x == r]
        ll.append(indices)
        
    #print(ll)
    if min_number_clusters is not None:
        if min_number_clusters > nr_clusters:
            
            obj = obj + min_number_clusters/nr_clusters
            
            #print("min number of clusters penalty is",  min_number_clusters/nr_clusters)
            
    if max_number_clusters is not None:
        if max_number_clusters > nr_clusters:
            obj = obj +  max_number_clusters/nr_clusters
            
            #print("max number of clusters penalty is", max_number_clusters/nr_clusters)
            
            
    if min_cluster_size is not None:
        t = sum(i < min_cluster_size for i in list(clus.values()))
        obj = obj + (t/nr_clusters)
        
        #print("min_cluster_size penalty is", t/nr_clusters)
        
        
    if max_cluster_size is not None:
        t = sum(i > max_cluster_size for i in list(clus.values()))
        obj = obj + (t/nr_clusters)
                  
        #print("max_cluster_size penalty is", t/nr_clusters)

    if cluster_size_distribution:
        m = len(labels) / nr_clusters
        t = []
        cl = Counter(labels)

        for i in cl.values():
            t.append(abs(i/m))

        t = statistics.mean(t)

        obj = obj + t


        
        
    if local:
        #get mean cluster similarity scores
        m = []
        pen = 0
        
        for c in ll:
            temp = []
            if len(c) > 1:
                pairs = list(itertools.combinations(c, 2))

                for p in pairs:
                    temp.append(1-X[p[0]][p[1]])


            else:
                temp.append(0)
                
            #print("local", temp)
            if e is not None:
                if statistics.mean(temp) < e:
                    pen = pen + 1
            m.append(statistics.mean(temp))

        if len(m) > 1:
            obj = obj + (1-statistics.mean(m))
                  
            #print("local penalty is", (1-statistics.mean(m)))
        elif len(m) > 0:
            obj = obj + 1-m[0]
                  
                  
            #print("local penalty is", 1-m[0])
            
        obj = obj + (pen/nr_clusters)
                  
                  
        
        
        
        
        
    if bet:
        #get mean dissimilarity between all clusters 
        #this is calculated on an item by item base
        
        m = []
        pen = 0
        
        for cc in list(itertools.combinations(range(nr_clusters),2)):
            c1 = ll[cc[0]]
            c2 = ll[cc[1]]
            
            temp = []
            for i1 in c1:
                for i2 in c2:
                    temp.append(X[i1][i2])
                    
            #print("bet", temp)
            if len(temp) > 0:
                m.append(statistics.mean(temp))
                if s is not None:
                    if statistics.mean(temp) < s:
                        pen = pen +1
                        
                        
            
            elif len(temp) > 0:
                m.append(temp[0])
                
                if s is not None:
                    if temp[0] < s:
                        pen = pen +1
                    
                    
        if len(m) > 1:
            obj = obj + (1-statistics.mean(m))
                  
            #print("bet penalty is", 1-statistics.mean(m))
        elif len(m) > 0:
            obj = obj + (1-m[0])
                  
            #print("bet penalty is", (1-m[0]))
            
            
        if s is not None:
            obj = obj + (pen/len(list(itertools.combinations(range(nr_clusters),2))))

            #print("bet penalty s is", (pen/len(list(itertools.combinations(range(nr_clusters),2)))))
        
        
        
        
        
    return obj
                
                
            
        
            
    
def create_mean_distance_matrix(matrices, set_diagonal = True):
    """
    Creates a mean distance matrix out of individual distance matrices.

    Parameters:
        matrices (list): of numpy matrices. Items need to be in the same order in all the matrices. Matrices are assumed to be "distance amtrices".
        set_diagonal (boolean): if True diagonal values are set to 0 automatically - this may be helpful if the distance measures applied do not return 0 distance for the same object.
        
    Returns:
        mean distance matrix (matrix): of input matrices
    """

    if len(matrices) < 2:
        print("matrices needs to contain at least two items in order to estimate its median")
        return None
    else:

        mean_dist = matrices[0].copy()

        for index, x in np.ndenumerate(mean_dist):
            i0 = index[0]
            i1 = index[1]
            
            temp = []
            for m in matrices:
                temp.append(m[i0][i1])

            d = statistics.mean(temp)
            
            mean_dist[index[0]][index[1]] = d
            
            if set_diagonal:
                if index[0] == index[1]:
                    mean_dist[index[0]][index[1]] = 0

        return mean_dist


def create_median_distance_matrix(matrices, set_diagonal = True):
    """
    Creates a median distance matrix out of individual distance matrices.

    Parameters:
        matrices (list): of numpy matrices. Items need to be in the same order in all the matrices. Matrices are assumed to be "distance amtrices".
        set_diagonal (boolean): if True diagonal values are set to 0 automatically - this may be helpful if the distance measures applied do not return 0 distance for the same object.
        
    Returns:
        median distance matrix (matrix): of input matrices
    """
    if len(matrices) < 2:
        print("matrices needs to contain at least two items in order to estimate its median")
        return None
    else:

        median_dist = matrices[0].copy()

        for index, x in np.ndenumerate(median_dist):
            i0 = index[0]
            i1 = index[1]
            
            temp = []
            for m in matrices:
                temp.append(m[i0][i1])

            d = statistics.median(temp)
            
            median_dist[index[0]][index[1]] = d
            
            if set_diagonal:
                if index[0] == index[1]:
                    median_dist[index[0]][index[1]] = 0

        return median_dist


def hierarchical_clustering(distance, n_clusters=2, linkage="complete"):
    """
    Hierarchical clustering of distance matrix. Based on sklearn.

    Parameters:
        distance (matrix): distance matrix to be used for clustering.
        n_clusters (int): number of to be computed clusters. Maximum allowed value is thr number of items in distance. Minimum allowed value is 2.
        linkage (str): linkage method to be used. Options are "average", "complete" or "single".
            
    Returns:
        labels (array): of clustering
    """

    if n_clusters < 2 or n_clusters > len(distance):
        print("n_clusters is either too small or too large, n_clusters must be in range 2 ", len(distance))
        return None

    elif linkage not in ["average", "complete", "single"]:
        print("linkage has to be average, complete or single")
        return None

    else:

        ag= AgglomerativeClustering(n_clusters = n_clusters, affinity="precomputed", distance_threshold=None, linkage=linkage).fit(distance)
        labels = ag.labels_

        return labels



def affinityPropagation_clustering(distance):
    """
    Affinity propagation clustering on distance matrix. Based on sklearn

    Parameters:
        distance (matrix): distance matrix to be used for clustering.
        
    Returns:
        labels (array): of clustering
    """

    

    db= sklearn.cluster.AffinityPropagation(affinity="precomputed").fit(distance)
    labels = db.labels_

    return labels

def __generate_empty__(x):
    """
    generates empty dict to be used as input for other functions

    Input
        x int, lenght of keys to be generated

    Output
        dict
    """

    empty = {}
    for i in range(x):
        empty[i] = 0

    return empty

def convert_clusters(clusters, v):

    """
    Converts list of sublists into array of cluster labels,

    Parameters:
        clusters (list): list of sublists where each list contains indices of the items in this cluster and each sublist is a different cluster.
        v (list): of item IDs as contained in clusters.

    Returns:
        labels (array): of clustering
    """
    d = {}
    for vv in v:
        d[vv] = 0

    empty = d.copy()
    for i in range(len(clusters)):
        for k in clusters[i]:
            empty[k] = i
            
    return list(empty.values())

def optics_clustering(distance, radius=2, neighbors=2, n_clusters=2):
    """
    Optics clustering on a provided distance matrix. Based on pyclustering https://pyclustering.github.io/docs/0.8.2/html/de/d3b/classpyclustering_1_1cluster_1_1optics_1_1optics.html
    
    Parameters:
        distance (matrix): distance matrix to be used for clustering.
        radius (int): connectivity radius. 
        neighbors (int): in [1, number of samples -1]
        n_clusters (int): amount of clusters that should be found.

    Returns:
            labels (array): of clustering
    """

    if n_clusters < 2 or n_clusters > len(distance):
        print("n_clusters is either too small or too large, n_clusters must be in range 2 ", len(distance))
        return None

    elif neighbors < 1 or neighbors > len(distance)-1:
        print("neighbors has to be in range 1", len(distance)-1)
        return None

    else:

        optics_instance = optics(distance, radius, neighbors, n_clusters, data_type="distance_matrix")
        # Performs cluster analysis
        optics_instance.process()
        # Obtain results of clustering
        clusters = optics_instance.get_clusters()

        #converts output into labeled array
        empty = __generate_empty__(len(distance))
        

        labels = convert_clusters(clusters, empty)

        return labels



def kmedoids_clustering(distance, n_clusters=2):
    """
    Kmediods clustering on a provided distance matrix. Based on pyclustering https://pyclustering.github.io/docs/0.8.2/html/de/d3b/classpyclustering_1_1cluster_1_1optics_1_1optics.html
    
    Parameters:
        distance (matrix): distance matrix to be used for clustering.
        n_clusters (int): amount of clusters that should be found.
        
    Returns:
        labels (array): of clustering
        created mediods (list): random created mediods used for clustering
    """

    if n_clusters < 2 or n_clusters > len(distance):
        print("n_clusters is either too small or too large, n_clusters must be in range 2 ", len(distance))
        return None

    

    else:

        #generate random mediods
        initial_medoids = []
    
        for ii in range(n_clusters):
            initial_medoids.append(randrange(len(distance)))

        kmedoids_instance = kmedoids(distance, initial_medoids, data_type='distance_matrix')
        # run cluster analysis and obtain results
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()

        empty = __generate_empty__(len(distance))

        labels = convert_clusters(clusters, empty)


        return np.array(labels),  initial_medoids