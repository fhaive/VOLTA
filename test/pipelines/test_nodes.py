import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
sys.path.insert(1, '../../')
sys.path.insert(1, '../../example_pipelines/')
import global_distances
import get_node_similarity
import pickle


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)

with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph2.pckl", "rb") as f:
    G2 = pickle.load(f)


networks_graphs = [G, G, G, G2]

labels=["G1", "G12", "G13", "G2"]


print("convert graphs")

networks = get_node_similarity.preprocess_graph(networks_graphs, attribute="weight")


print("map nodes to id")

network_lists, mapping = get_node_similarity.preprocess_node_list(networks)




print("sort nodes")

sorted_nodes, shared_nodes, binary, saved = get_node_similarity.sort_list_and_get_shared(network_lists, mapping, networks_graphs, labels)

print("saved nodes", saved)