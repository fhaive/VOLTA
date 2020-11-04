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
import node_edge_similarities
import pickle
from scipy.stats import kurtosis, skew, kendalltau
import statistics
import numpy as np


with open("test_distance_complete_G1.pckl", "rb") as f:
    G1 = pickle.load(f)

with open("test_distance_complete_G2.pckl", "rb") as f:
    G2 = pickle.load(f)

with open("test_distance_complete_G3.pckl", "rb") as f:
    G3 = pickle.load(f)

with open("test_distance_complete_G11.pckl", "rb") as f:
    G11 = pickle.load(f)
with open("test_distance_complete_G12.pckl", "rb") as f:
    G12 = pickle.load(f)
with open("test_distance_complete_G13.pckl", "rb") as f:
    G13 = pickle.load(f)

with open("test_distance_complete_G111.pckl", "rb") as f:
    G111 = pickle.load(f)
with open("test_distance_complete_G112.pckl", "rb") as f:
    G112 = pickle.load(f)
with open("test_distance_complete_G131.pckl", "rb") as f:
    G131 = pickle.load(f)

with open("test_distance_complete_G21.pckl", "rb") as f:
    G21 = pickle.load(f)
with open("test_distance_complete_G22.pckl", "rb") as f:
    G22 = pickle.load(f)

with open("test_distance_complete_G31.pckl", "rb") as f:
    G31 = pickle.load(f)

net_temp = [G1, G2, G3, G11, G12, G13, G111, G112, G131, G21, G22, G31]


#convert networks into list of sublist format
networks = []
for n in net_temp:
    temp = []
    edges = list(n.edges())
    for edge in edges:
        temp.append([edge[0], edge[1], n[edge[0]][edge[1]]["weight"]])

    networks.append(temp)




for i in range(len(networks)):
    if i == 0:
        m, n = node_edge_similarities.map_edge_to_id(networks[i], mapping={}, next_value=0)

    else:
        m, n = node_edge_similarities.map_edge_to_id(networks[i], mapping=m, next_value=n)

network_lists = []
for net in networks:
    network_lists.append(node_edge_similarities.construct_mapped_edge(m, net)) 

print("jaccard and similarity")

j, s = node_edge_similarities.shared_elements_multiple(network_lists,  labels=None, percentage=True, jaccard=True, jaccard_similarity = True, penalize_percentage=False)
jd = node_edge_similarities.to_distance(j)

#save pickles and plot

with open("results_complete/edges_jaccard_distance.pckl", "wb") as f:
    pickle.dump(jd, f, protocol=4)
with open("results_complete/edges_percentage_similarity.pckl", "wb") as f:
    pickle.dump(s, f, protocol=4)

print("get smc, hamming, kendall, edc")

#get sorted list
sorted_networks = []
for net in networks:
    sorted_networks.append(node_edge_similarities.sort_edge_list(net, m))

labels = ["G1", "G2", "G3", "G11", "G12", "G13", "G111", "G112", "G131", "G21", "G22", "G31"]
shared_edges = node_edge_similarities.compute_shared_layers(network_lists, labels, mapping = None, weight=False)

binary = node_edge_similarities.compute_binary_layer(shared_edges, layers=labels)

a,b, x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(sorted_networks, compute="kendall", kendall_usage="top", kendall_x = 100)

with open("results_complete/edges_kendall_top_50.pckl", "wb") as f:
    pickle.dump(a, f, protocol=4)


print("compute hamming")
a, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="hamming")

with open("results_complete/edges_hamming.pckl", "wb") as f:
    pickle.dump(a, f, protocol=4)


print("nodes")

for i in range(len(networks)):
    if i == 0:
        m, n = node_edge_similarities.map_node_to_id(networks[i], mapping={}, next_value=0)

    else:
        m, n = node_edge_similarities.map_node_to_id(networks[i], mapping=m, next_value=n)


node_lists = []

for net in networks:
    node_lists.append(list(dict.fromkeys(node_edge_similarities.construct_mapped_node(m, net))))


j, s = node_edge_similarities.shared_elements_multiple(node_lists, labels=None, percentage=True, jaccard=True, jaccard_similarity = True, penalize_percentage=False)
jd = node_edge_similarities.to_distance(j)
#save pickles and plot
with open("results_complete/nodes_jaccard_distance.pckl", "wb") as f:
    pickle.dump(jd, f, protocol=4)
with open("results_complete/nodes_percentage_similarity.pckl", "wb") as f:
    pickle.dump(s, f, protocol=4)

shared_nodes = node_edge_similarities.compute_shared_layers(node_lists, labels, mapping = None, weight=False)

binary = node_edge_similarities.compute_binary_layer(shared_nodes, layers=labels)

sorted_nodes = []
options_keys = ["degree", "dc", "cc", "betweenness", "average"]
for net in net_temp:
    sorted_nodes.append(node_edge_similarities.sort_node_list(net, m, degree=True, degree_centrality=True, closeness_centrality=True, betweenness=True,as_str=False))


for key in options_keys:
    current_sorted = []
    for di in sorted_nodes:
        current_sorted.append(di[key])

    a,b,x = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(current_sorted, compute="kendall", kendall_usage="top", kendall_x = 50)
    with open("results_complete/nodes_kendall_top_50_"+key+".pckl", "wb") as f:
        pickle.dump(a, f, protocol=4)


print("compute hamming")
a, p = node_edge_similarities.build_similarity_matrix_for_binary_and_ranked(binary, compute="hamming")

with open("results_complete/nodes_hamming.pckl", "wb") as f:
    pickle.dump(a, f, protocol=4)
