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
sys.path.insert(1, '../../multiplex/')
import build_network
sys.path.insert(1, '../../supporting/')
import pickle4reducer
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
labels =  ["G1", "G2", "G3", "G11", "G12", "G13", "G111", "G112", "G131", "G21", "G22", "G31"]
networks_dist = []

for net in networks:
    #convert to a distance graph
    n = simplification.add_opposit_weight(net, current_weight="weight", new_weight="weight")
    networks_dist.append(n)






for i in range(len(networks_dist)):
    net = networks_dist[i]
    label = labels[i]
    C, D, S = build_network.build_complete_network(net, infer = False, edge_attribute="weight", manual_distance=0.9999, manual_sim=0.0001)

    #pickle D
    with open("test_distance_complete_"+label+".pckl", "wb") as f:
        pickle.dump(S, f, protocol=4)