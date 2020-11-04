import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../multiplex/')
import build_network
import pickle

df = pd.read_csv('test_100_nodes.csv', delimiter=' ')

# Or export it in many ways, e.g. a list of tuples
tuples = [list(x) for x in df.values]

#add distance values to edges
edges = []
#random.seed(1)
for item in tuples:
    d = random.random()
    edges.append([item[0], item[1], d])

#create networkx graph

G = build_network.construct_single_layer_network(edges)

with open("test_distance_graph.pckl", "wb") as f:
    pickle.dump(G, f, protocol=4)

#estimate complete network

C, D, S = build_network.build_complete_network(G, infer = False)

#pickle D
with open("test_distance_complete_graph.pckl", "wb") as f:
    pickle.dump(D, f, protocol=4)
