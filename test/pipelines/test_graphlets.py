import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
sys.path.insert(1, '../../example_pipelines/')
import local
import pickle
import get_node_specific_graphlets



with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph2.pckl", "rb") as f:
    G2 = pickle.load(f)

networks = [G, G2]

nodes = list(G.nodes())[:10]

graphlets = get_node_specific_graphlets.generate_graphlets(nodes, min_size=3, max_size=4)

res = get_node_specific_graphlets.get_graphlet_vector(networks, graphlets)

print(len(res[0]))