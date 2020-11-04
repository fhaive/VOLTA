import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
import global_distances
import pickle


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_complete_graph.pckl", "rb") as f:
    G = pickle.load(f)

#print(list(G.neighbors(60)))

#perform random walks
walks1 = global_distances.perform_random_walks(G, steps=30, number_of_walks=100, start=10, probabilistic=True, weight="weight")
walks2 = global_distances.perform_random_walks(G, steps=30, number_of_walks=100, start=10, probabilistic=True, weight="weight")




nodes, edges = global_distances.rank_walks(G, walks1, undirected=True)

ranked = global_distances.compare_walks(G, walks1, walk2=walks2, G2=G, comparison="ranked", undirected=True, top=50)
print(ranked)

consensus = global_distances.get_walk_consensus(walks1, G)
print(consensus)

degrees = global_distances.node_degree_distribution(G)
print(degrees)

print(global_distances.is_connected(G))

size = global_distances.graph_size(G)
print(size)

print(global_distances.density(G))

print(global_distances.graph_edges(G))

cycles = global_distances.cycle_distribution(G)
print(cycles)

paths = global_distances.path_length_distribution(G)
print(paths)

print(global_distances.clustering_coefficient(G))

print(global_distances.contains_triangles(G))

degree = global_distances.degree_centrality(G)
print(degree)

eigen = global_distances.eigenvector_centrality(G)
print(eigen)

close = global_distances.closeness_centrality(G)
print(close)

bet = global_distances.betweeness_centrality(G)
print(bet)