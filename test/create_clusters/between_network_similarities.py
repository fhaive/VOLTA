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
import numpy as np


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
#networks_name = ["G1", "G2", "G3", "G11", "G12", "G13", "G111", "G112", "G131", "G21", "G22", "G31"]


#get possible nodes, since all are based on G1 nodes of G1 are enough
nodes = list(G1.nodes())

def helper_walks(networks, nodes):

    performed_walks = {}
    for node in nodes:
        performed_walks[node] = {}

    for node in nodes:
        for i in range(len(networks)):
            net = networks[i]
            if not nx.is_isolate(net, node):
                walks  = global_distances.perform_random_walks(net, steps=20, number_of_walks=30, start=node, probabilistic=True, weight="weight")
                #get consensus walk
                

            else:
                walks = []

            #save
            performed_walks[node][i] = walks


    return performed_walks


performed_walks = helper_walks(networks, nodes)

#save consensus
with open("temp_performed_walks.pckl", "wb") as f:
    pickle.dump(performed_walks, f, protocol=4)

def helper_walk_sim(networks, performed_walks):

    print("estimate networks similarities based on random consensus walks")
                
    results_nodes =  np.zeros((len(networks), len(networks)))
    results_edges =  np.zeros((len(networks), len(networks)))

    index_list = []
    for index, x in np.ndenumerate(results_nodes):
        temp = (index[1], index[0])
        if temp not in index_list and index not in index_list:
            index_list.append(index)


    for index in index_list:
        n1 = index[0]
        n2 = index[1]

        nodes_sim = []
        edges_sim = []

        for node in nodes:
            #get consensus walks
            c1 = performed_walks[node][n1]
            c2 = performed_walks[node][n2]

            if len(c1) > 0 and len(c2) > 0:

                #compare the 2 walks
                kendall = global_distances.compare_walks(networks[n1], c1, walk2=c2, G2=networks[n2], comparison="ranked", undirected=True, top=10)

                e_t = kendall["edges_tau"]
                n_t = kendall["nodes_tau"]

            else:
                print("no walk similarities can be estimated", node, n1, n2)
                e_t = 0
                n_t = 0

            nodes_sim.append(n_t)
            edges_sim.append(e_t)


        #estiamte mean and write to results matrix
        mean_nodes = statistics.mean(nodes_sim)
        mean_edges = statistics.mean(edges_sim)

        results_edges[n1][n2] = mean_edges
        results_edges[n2][n1] = mean_edges

        results_nodes[n1][n2] = mean_nodes
        results_nodes[n2][n1] = mean_nodes

    return results_edges, results_nodes

        
results_edges, results_nodes = helper_walk_sim(networks, performed_walks)

#save consensus
with open("walks_sim_edges.pckl", "wb") as f:
    pickle.dump(results_edges, f, protocol=4)

with open("walks_sim_nodes.pckl", "wb") as f:
    pickle.dump(results_nodes, f, protocol=4)
        
