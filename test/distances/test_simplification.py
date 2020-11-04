import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
import simplification
import pickle
sys.path.insert(1, '../../')
import communities


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)

print("nodes G", len(G.nodes()))
print("edges G", len(G.edges()))

T = simplification.remove_edges_per_node(G, treshold=None, percentage=0.02, direction="bottom", attribute="weight")

print("nodes T", len(T.nodes()))
print("edges T", len(T.edges()))
#G1 = simplification.add_opposit_weight(G, current_weight="weight", new_weight="similarity")
#G2 = simplification.add_absolute_weight(G)
'''
T = simplification.get_min_spanning_tree(G, weight="weight", is_distance=True, new_weight="distance")

print("nodes T", len(T.nodes()))
print("edges T", len(T.edges()))


G1 = simplification.remove_edges(G, treshold=0.2, percentage=None, based_on="weight", direction="top")

print("nodes G1", len(G1.nodes()))
print("edges G1", len(G1.edges()))

G1 = simplification.remove_edges(G, treshold=0.3, percentage=None, based_on="betweenness", direction="top")

print("nodes G1", len(G1.nodes()))
print("edges G1", len(G1.edges()))


G1 = simplification.remove_nodes(G, treshold=0.3, percentage=None, based_on="closeness", direction="top")

print("nodes G1", len(G1.nodes()))
print("edges G1", len(G1.edges()))

'''
