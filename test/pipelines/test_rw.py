import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
sys.path.insert(1, '../../')
sys.path.insert(1, '../../example_pipelines/')
import global_distances
import get_walk_distances
import pickle


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)

with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph2.pckl", "rb") as f:
    G2 = pickle.load(f)

nodes = []

for node in G.nodes():
    if node not in nodes:
        nodes.append(node)

performed_walks = get_walk_distances.helper_walks_multi(G, nodes, "g", steps=10, number_of_walks=10, degree=True, start=None, probabilistic=True, weight="weight", nr_processes=2)

print(performed_walks)






'''
networks = [G, G2]
labels = ["G1", "G2"]

#get all nodes in all networks
nodes = []
for net in networks:
    for node in net.nodes():
        if node not in nodes:
            nodes.append(node)

print(len(nodes))

total_nodes = []
total_edges=[]

print("perform walks")
#use this to create a vector for each network for each node on how many nodes & how often occure in RW
#do same for edges
for i in range(len(networks)):
    net = [networks[i]]
    l = [labels[i]]

    performed_walks, node_counts, edge_counts = get_walk_distances.helper_walks(net, nodes, l)

    total_edges.append(edge_counts)
    total_nodes.append(node_counts)
    #print("edges", edge_counts)
    #print("nodes", node_counts[l[0]][0])

    if i == 0:
        total_walks = performed_walks
    else:
        total_walks.update(performed_walks)

    #rint("performed walks", len(performed_walks))
    #print(node_counts)
print(total_walks.keys())

nodes_count, edges_count = get_walk_distances.helper_get_counts(networks, performed_walks)

print("nodes", len(nodes_count[0]))
print("edges", len(edges_count[1]))


print("compute ranked kendall for top 10")
results_edges, results_nodes, results_edges_p, results_nodes_p, results_edges_all, results_nodes_all, results_edges_p_all, results_nodes_p_all = get_walk_distances.helper_walk_sim(networks, total_walks, nodes, labels, top=10, return_all=True, ranked=True, nodes_ranked=total_nodes, edges_ranked=total_edges)
print("edges", results_edges)
print("nodes", results_nodes)
'''