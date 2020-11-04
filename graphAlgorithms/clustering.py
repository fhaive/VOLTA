'''
network clustering algorithms for distance matrices as computed by distance section
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
import bct
import community as community_louvain
from sklearn.utils.validation import check_random_state



def consensus_clustering(clusterings, seed=123, threshold=0.75, per_node=True, rep=10):
    """
    finds a consensus clustering based on multiple provided clusterings
    based on netneurotools & bct for whole graph thresholds, per node thresholds are new
    https://github.com/netneurolab/netneurotools
    https://github.com/aestrivex/bctpy

    build agreement graph from different clusterings, removes weak edges & performs community detection until convergance

    Input
        clustering list of arrays containing numeric class labels

        seed random seed to be used for random processes

        threshold either float [0,1] or "matrix"
            if "matrix" an automatic threshold based on permutation of the clusterings is applied on a per node basis
            if float and not per_node all edges lower threshold are removed in agreement graph during consensus clustering
            if float and per_node then on a per node basis threshold percentage of edges are removed 
                but only if they are "weak" for both nodes making the edges

        per_node boolean if True threshold is interpreted as a percentage value and applied on a per node basis
            else is interpreted as a static edge attribute and applied to the whole graph

        rep int how often louvain clustering is repeated on agreement graph


    Output
       array with consensus class labels

    
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
        cons = get_consensus_node(agreement, null_agree, rs, seed=seed, rep = rep)
        
            
        cons = np.array(cons)[0]
                
    else:
        if per_node:
            cons = get_consensus_node_threshold(agreement, node_threshold=threshold, seed=seed ,rep = rep)
            cons = np.array(cons)[0]
        else:
    
            consensus = bct.clustering.consensus_und(agreement, threshold, rep)


            cons = consensus.astype(int)
    
    return cons


def get_consensus_node_threshold(agreement, node_threshold=0.75, seed=123 ,rep = 10):

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
        
    
        
def get_consensus_node(agreement, node_threshold, rs, seed=123 ,rep = 10):

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
        multi objective function to evaluate clusterings
        
        Input:
            X distance matrix, as used for clustering
            
            labels list of predicted labels - needs to be in same order as X
            
            min_number_clusters minimum number of allowed clusters, if less than a penalty is applied ,
                if None will not be taken into account
                
            max_number_clusters maximum number of clusters, if more then penalty is applied,
                if None will not be taken into account
                
            min_cluster_size for each cluster with less than x items a penalty is applied, 
                if None will not be taken into account
                
            max_cluster_size for each cluster with more than x items a penalty is applied,
                if None will not be taken into account
                
            local if True then objective aims at minimizing within cluster similarity based on data provided in X
                if False will be ignored
                
            bet if True objective aims at high dissimilarity between clusters, if False will be ignored
            
            e [0,1] for each cluster with a mean similarity less than e an additional penalty is applied, if None 
                will be ignored
            
            s [0,1] between each cluster pair were distance is less than s an additional penalty is applied,
                if None will be ignored

            cluster_size_distribution if True mean difference of cluster size for each cluster to "most equal" partitioning is applied
                most equal partitioning is len(labels) / number of clusters

        Output
            clustering score (float), the closer to 0 the better the clustering is in regards to the selected objectives
            
            
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
    creates a mean distance matrix out of individual distance matrices

    Input
        list of numpy matrices, items need to be in the same order in all matrices, assumed all to be distance amtrices
        set_diagonal if True diagonal values are set to 0 automatically - 
            this may be helpful if data is not a full distance but should be treated as one

    Output
        matrix
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
    creates a median distance matrix out of individual distance matrices

    Input
        list of numpy matrices, items need to be in the same order in all matrices, assumed all to be distance matrices
        set_diagonal if True diagonal values are set to 0 automatically - 
            this may be helpful if data is not a full distance but should be treated as one

    Output
        matrix, each cell is median of the same cell provided in matrices
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


def hierarchical_clustering(distance, n_clusters=2, linkage="completed"):
    """
    hierarchical clustering of distance matrix, optimal number of clusters can be tuned with multiobjective
    based on sklearn

    Input
        distance distance matrix
        n_clusters int, number of to be computed clusters (max is number of items in distance, min 2)
        linkage linkage methods to be used
            average uses the average distance
            complete uses the maximum distance
            single uses the minimum of the distance

        Output
            array of class labels in same order as items in distance
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
    affinity propagation clustering on distance matrix, has no paramter settings
    based on sklearn

    Input
        distance distance matrix
       
        Output
            array of class labels in same order as items in distance
    """

    

    db= sklearn.cluster.AffinityPropagation(affinity="precomputed").fit(distance)
    labels = db.labels_

    return labels

def generate_empty(x):
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
    function to convert list of list of items into array of cluster labels

    Input
        clusters list of lists were each list contains index of item in this cluster
        v dictionary were keys are all elements in clusters and all values are 0 as returned by generate_empty

    Output
        array of cluster labels
    """

    empty = v.copy()
    for i in range(len(clusters)):
        for k in clusters[i]:
            empty[k] = i
            
    return list(empty.values())

def optics_clustering(distance, radius=2, neighbors=2, n_clusters=2):
    """
    optics clustering of distance matrix, optimal number of clusters, neighbors and radius can be tuned with multiobjective
    based on pyclustering: https://pyclustering.github.io/docs/0.8.2/html/de/d3b/classpyclustering_1_1cluster_1_1optics_1_1optics.html
    Input
        distance distance matrix
        radius connectivity radius, needs to be larger than real 
        neighbors int [1, number of samples -1]
        n_clusters amount of clusters aiming to be found

        Output
            array of class labels in same order as items in distance
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
        empty = generate_empty(len(distance))
        

        labels = convert_clusters(clusters, empty)

        return labels



def kmedoids_clustering(distance, n_clusters=2):
    """
    kmediods clustering of distance matrix, optimal number of clusters can be tuned with multiobjective
    based on pyclustering: 
    Input
        distance distance matrix
        
        n_clusters amount of clusters aiming to be found

        Output
            array of class labels in same order as items in distance
            list of initial random created mediods
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

        empty = generate_empty(len(distance))

        labels = convert_clusters(clusters, empty)


        return labels,  initial_medoids