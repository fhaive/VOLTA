"""
This is a collection of wrapper functions to simplify how to estimate the similarity between multiple networks
based on their similarity in structural descriptors.
"""

import networkx as nx
import pandas as pd
import csv
import random
import sys
import volta.distances.global_distances as global_distances
import volta.distances.local as local
import volta.simplification as simplification
import volta.distances.trees as trees
import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np
import scipy


def graphlet_helper(network, estimate_on=50, edge_attribute="weight", motif_min_size=2, motif_max_size=4):
    """
    Estimates graphlets of different sizes  on a networks based on a random selection of nodes.

    Parameters:
        network (networkX graph object): graph object to estimate on.
        estimate_on (int): graphlets are estimated based on a random selection of estimate_on nodes which cannot be larger than the number of nodes in G
        edge_attribute (str or None): if not None, then based on the provided edge attribute the size of the graphlets will be returned as list, which can be used to estimate its size distributions
        motif_min_siz (int): nodes size of smallest graphlet to be counted. Minimum permitted value is 2.
        motif_max_size (int): node size of largest graphlets. Maximum permitted value is 6.

    Returns:
        graphlets (list): list of sublists where each sublist contains counts and size distribution of a specific graphlet. Graphlets are orderd by ID as stated in the graph atlas.
        counts (dict): key is graphlet ID and value is counts.
        
    """

    temp = []
    

    graphlets_counted, graphlets, graphlets_size, motifs = local.iterate_graphlets(network, estimate_on=estimate_on, edge_attribute=edge_attribute, motif_min_size=motif_min_size, motif_max_size=motif_max_size)
    #print("graphlets", graphlets_counted)
    #print("graphlets size", graphlets_size)
    
    for key in graphlets_counted:
        temp.append(graphlets_counted[key])
        

    for key in graphlets_size:
        if len(graphlets_size[key]) > 1:
            mean = statistics.mean(graphlets_size[key])
            median = statistics.median(graphlets_size[key])
            std = statistics.stdev(graphlets_size[key])
            skw = skew(graphlets_size[key])
            kurt = kurtosis(graphlets_size[key])
        else:
            
            mean = graphlets_size[key]
            if len(mean) == 0:
                mean = 0
            else:
                mean = mean[0]
            median = graphlets_size[key]
            if len(median) == 0:
                median = 0
            else:
                median = median[0]
            std = 0
            skw = 0
            kurt = 0


        temp.append(mean)
        temp.append(median)
        temp.append(std)
        temp.append(skw)
        temp.append(kurt)

    return temp, graphlets_counted, graphlets, motifs




def estimate_vector(networks, edge_attribute="weight", is_file=False):
    """
    Wrapper that estimates a feature vector on different structural descriptors.
    Includes graph size parameters, density, clustering, cycles, degree/ closeness/ betweenness and shortest path distributions.
    
    Parameters:
        networks (list): list of networkX graph objects or their pickled locations if is_file is True.
        edge_attribute (str): name of the edge attribute to be taken into account.
        is_file (boolean): if True then networks contains locations to pickled graph objects, else contains graph objects directly.
        
    Returns:
        structural description vector (list): list of sublists. Each sublists contains a feature vector order as in networks.
    """



    vectors = []

    for nn in networks:
        if is_file:
            network = nx.read_weighted_edgelist(nn)
        else:
            network = nn
        temp_vector = []
       
        print("global")


        size = global_distances.graph_size(network)
        radius = size["radius"]
        diameter = size["diameter"]
        nodes = size["nodes"]
        edges = size["edges"]
        
        temp_vector.append(radius)
        temp_vector.append(diameter)
        temp_vector.append(nodes)
        temp_vector.append(edges)

        print("density")

        density = global_distances.density(network)
        temp_vector.append(density)

        print("clustering")
        clustering = global_distances.average_clustering(network)
        temp_vector.append(clustering)

        print("graph edges")

        edges = global_distances.graph_edges(network)
        non_edges = edges["missing_edges_percentage"]
        ex_edges = edges["existing_edges_percentage"]
        temp_vector.append(non_edges)
        temp_vector.append(ex_edges)

        
        print("cycles")

        cycles = global_distances.cycle_distribution(network)
        nr_cycles = cycles["number_of_cycles"]
        median_cycles = cycles["median_cycle_length"]
        mean_cycles = cycles["mean_cycle_length"]
        std_cycles = cycles["std_cycle_length"]
        skw_cycles = cycles["skw_cycle_length"]
        kurt_cycles = cycles["kurtosis_cycle_length"]
        temp_vector.append(nr_cycles)
        temp_vector.append(mean_cycles)
        temp_vector.append(median_cycles)
        temp_vector.append(std_cycles)
        temp_vector.append(skw_cycles)
        temp_vector.append(kurt_cycles)

        print("shortest path distribution")
        paths = global_distances.path_length_distribution(network)
        path_mean = paths["mean path length"]
        path_median = paths["median path length"]
        path_std = paths["std path length"]
        path_skw = paths["skw path length"]
        path_kurt = paths["kurtosis path length"]
        temp_vector.append(path_mean)
        temp_vector.append(path_median)
        temp_vector.append(path_std)
        temp_vector.append(path_skw)
        temp_vector.append(path_kurt)
        

        print("cc")
        cc = global_distances.clustering_coefficient(network)
        temp_vector.append(cc)
        
        print("degree dist")

        degree = global_distances.degree_centrality(network)
        mean_centrality = degree["mean_centrality"]
        median_centrality = degree["median_centrality"]
        std_centrality = degree["std_centrality"]
        skw_centrality = degree["skew_centrality"]
        kurt_centrality = degree["kurtosis_centrality"]
        temp_vector.append(mean_centrality)
        temp_vector.append(median_centrality)
        temp_vector.append(std_centrality)
        temp_vector.append(skw_centrality)
        temp_vector.append(kurt_centrality)

        
        print("cc dist")
        close = global_distances.closeness_centrality(network)
        mean_centrality = close["mean_centrality"]
        median_centrality = close["median_centrality"]
        std_centrality = close["std_centrality"]
        skw_centrality = close["skew_centrality"]
        kurt_centrality = close["kurtosis_centrality"]
        temp_vector.append(mean_centrality)
        temp_vector.append(median_centrality)
        temp_vector.append(std_centrality)
        temp_vector.append(skw_centrality)
        temp_vector.append(kurt_centrality)

        print("betweenness dist")

        bet = global_distances.betweeness_centrality(network, weight=edge_attribute)
        mean_centrality = bet["mean_centrality"]
        median_centrality = bet["median_centrality"]
        std_centrality = bet["std_centrality"]
        skw_centrality = bet["skew_centrality"]
        kurt_centrality = bet["kurtosis_centrality"]
        temp_vector.append(mean_centrality)
        temp_vector.append(median_centrality)
        temp_vector.append(std_centrality)
        temp_vector.append(skw_centrality)
        temp_vector.append(kurt_centrality)
        
        '''
        #local
        graphlets, graphlets_counted, g, motifs= graphlet_helper(network)
        for item in graphlets:
            temp_vector.append(item)
        '''
        print(temp_vector)
        
        vectors.append(temp_vector)


    return vectors


def matrix_from_vector(vectors, normalize=False):
    """
    Wrapper function that estimates a similarity/ distance matrices based on computed vectors. 
    It assumes distances are bidirectional and therefore only estimates one triangle of the matrix and infers the other one.
    If the vector contains None values they are replaced with 0. 
    Currently this wrapper computes the euclidean, canberra, correlation, cosine and jaccard distance.
    
    Parameters:
        vectors (list): list of sublist, each sublist containing the feature vectore of a network.
        normalize (boolean): if True euclidean and canberra distance are normalized to be in [0,1].

    Returns:
        euclidean (numpy matrix): the matrix indices are in the same order as the networks provided in vectors.
        canberra (numpy matrix): the matrix indices are in the same order as the networks provided in vectors.
        correlation (numpy matrix): the matrix indices are in the same order as the networks provided in vectors.
        cosine (numpy matrix): the matrix indices are in the same order as the networks provided in vectors.
        jaccard (numpy matrix): the matrix indices are in the same order as the networks provided in vectors.



    """

    results_euclidean =  np.zeros((len(vectors), len(vectors)))
    results_canberra =  np.zeros((len(vectors), len(vectors)))
    results_correlation =  np.zeros((len(vectors), len(vectors)))
    results_cosine =  np.zeros((len(vectors), len(vectors)))
    results_jaccard =  np.zeros((len(vectors), len(vectors)))

    results =  np.zeros((len(vectors), len(vectors)))

    index_list = []
    for index, x in np.ndenumerate(results):
        temp = (index[1], index[0])
        if temp not in index_list and index not in index_list:
            index_list.append(index)

    for i in index_list:
        print(i)
        v1 = vectors[i[0]]
        v2 = vectors[i[1]]
        
        while None in v1:
            ii = v1.index(None)
            v1[ii] = 0
            
        while None in v2:
            ii = v2.index(None)
            v2[ii] = 0
        
        
        e = scipy.spatial.distance.euclidean(v1, v2)
        
        results_euclidean[i[0]][i[1]] = e
        results_euclidean[i[1]][i[0]] = e
        
        
        e = scipy.spatial.distance.canberra(v1, v2)
        
        results_canberra[i[0]][i[1]] = e
        results_canberra[i[1]][i[0]] = e
        
        e = scipy.spatial.distance.correlation(v1, v2)
        
        results_correlation[i[0]][i[1]] = e
        results_correlation[i[1]][i[0]] = e
        
        e = scipy.spatial.distance.cosine(v1, v2)
        
        results_cosine[i[0]][i[1]] = e
        results_cosine[i[1]][i[0]] = e
        
        e = scipy.spatial.distance.jaccard(v1, v2)
        
        results_jaccard[i[0]][i[1]] = e
        results_jaccard[i[1]][i[0]] = e

    if normalize:
        xmax, xmin = results_canberra.max(), results_canberra.min()
        results_canberra = (results_canberra - xmin)/(xmax - xmin)

        xmax, xmin = results_euclidean.max(), results_euclidean.min()
        results_euclidean = (results_euclidean - xmin)/(xmax - xmin)


    return results_euclidean, results_canberra, results_correlation, results_cosine, results_jaccard