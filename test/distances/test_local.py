import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
import local
import pickle


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)


#graphlets_counted, graphlets, graphlets_size, motifs = local.iterate_graphlets(G, estimate_on=10, edge_attribute="weight", motif_min_size=2, motif_max_size=3)

#print(graphlets_counted)
#print(graphlets)
#print(graphlets_size)
#print(motifs)

nodes = list(G.nodes())[:10]
print(len(nodes))

pos_graphlets = local.generate_node_specific_graphlets(nodes, graphlet_size=3)

#print(pos_graphlets)
print("is graphlet in graph")

in_graph = local.find_graphlet(G, pos_graphlets)

#print(in_graph)