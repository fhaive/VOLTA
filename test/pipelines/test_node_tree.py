import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
sys.path.insert(1, '../../')
sys.path.insert(1, '../../example_pipelines/')
import global_distances
import get_tree_distances
import pickle


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)

with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph2.pckl", "rb") as f:
    G2 = pickle.load(f)



networks = [G, G2]

#get all nodes in all networks
nodes = []
for net in networks:
    for node in net.nodes():
        if node not in nodes:
            nodes.append(node)

print(len(nodes))

print("estimate level trees and their structural vectors")

vectors, trees = get_tree_distances.helper_tree_vector(networks, nodes, tree_type="level", edge_attribute="weight", cycle_weight = "max", initial_cycle_weight=True)
print("vectors", len(vectors))
print("trees", len(trees))


print("get levels in tree for each node")
levels = get_tree_distances.get_node_levels(trees)
print("levels", len(levels[0]))


print("create similarity matrices")
results_percentage, results_smc, results_correlation, results_jaccard, results_percentage_all, results_smc_all, results_correlation_all, results_jaccard_all = get_tree_distances.helper_tree_sim(networks, nodes, tree_type="level", edge_attribute="weight", cycle_weight = "max", initial_cycle_weight=True, return_all=True)

print("percentage", results_percentage)
print("correlation", results_correlation)