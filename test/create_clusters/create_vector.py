import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
import global_distances
import local
import simplification
import trees
import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G1 = pickle.load(f)


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph2.pckl", "rb") as f:
    G2 = pickle.load(f)


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph3.pckl", "rb") as f:
    G3 = pickle.load(f)

with open("test_distance_G11.pckl", "rb") as f:
    G11 = pickle.load(f)
with open("test_distance_G12.pckl", "rb") as f:
    G12 = pickle.load(f)
with open("test_distance_G13.pckl", "rb") as f:
    G13 = pickle.load(f)

with open("test_distance_G111.pckl", "rb") as f:
    G111 = pickle.load(f)
with open("test_distance_G112.pckl", "rb") as f:
    G112 = pickle.load(f)
with open("test_distance_G131.pckl", "rb") as f:
    G131 = pickle.load(f)

with open("test_distance_G21.pckl", "rb") as f:
    G21 = pickle.load(f)
with open("test_distance_G22.pckl", "rb") as f:
    G22 = pickle.load(f)

with open("test_distance_G31.pckl", "rb") as f:
    G31 = pickle.load(f)

networks = [G1, G2, G3, G11, G12, G13, G111, G112, G131, G21, G22, G31]

#estimate vectors
#estimate euclidean distance on vectors
#estimate similarity matrices

def graphlet_helper(network, estimate_on=50, edge_attribute="weight", motif_min_size=2, motif_max_size=4):

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

    return temp




def estimate_vector(networks):
    vectors = []

    for network in networks:
        temp_vector = []
        #mean_degree, median_degree, std_degree, skw_degree, kurt_degree = global_distances.node_degree_distribution(network)
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

        density = global_distances.density(network)
        temp_vector.append(density)
        clustering = global_distances.average_clustering(network)
        temp_vector.append(clustering)

        edges = global_distances.graph_edges(network)
        non_edges = edges["missing_edges_percentage"]
        ex_edges = edges["existing_edges_percentage"]
        temp_vector.append(non_edges)
        temp_vector.append(ex_edges)

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



        cc = global_distances.clustering_coefficient(network)
        temp_vector.append(cc)

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

        '''
        eigen = global_distances.eigenvector_centrality(network, weight="weight")
        mean_centrality = eigen["mean_centrality"]
        median_centrality = eigen["median_centrality"]
        std_centrality = eigen["std_centrality"]
        skw_centrality = eigen["skew_centrality"]
        kurt_centrality = eigen["kurtosis_centrality"]
        temp_vector.append(mean_centrality)
        temp_vector.append(median_centrality)
        temp_vector.append(std_centrality)
        temp_vector.append(skw_centrality)
        temp_vector.append(kurt_centrality)
        '''

        close = global_distances.closeness_centrality(network, distance="weight")
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

        bet = global_distances.betweeness_centrality(network, weight="weight")
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

        #local
        graphlets = graphlet_helper(network)
        for item in graphlets:
            temp_vector.append(item)

        print(temp_vector)
        vectors.append(temp_vector)


    return vectors


vectors = estimate_vector(networks)

#save
with open("feature_vector.pckl", "wb") as f:
    pickle.dump(vectors, f, protocol=4)


        

        
 







       





