import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../')
import communities
import pickle
import statistics


with open("../test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)

#try to set node ids to string since int cannot be convert to igraph
re_label = {}
for n in list(G.nodes()):
    re_label[n] = str(n)

H = nx.relabel_nodes(G, re_label)

partitionings = []

#all are weighted community detection algorithms

m, c = communities.rber_pots(H, return_object=True)
print(m)
partitionings.append(m)

m, c = communities.surprise(H, return_object=True)
print(m)
partitionings.append(m)

m, c = communities.cpm(H, return_object=True)
print(m)
partitionings.append(m)

m, c = communities.greedy_modularity(H, return_object=True)
print(m)
partitionings.append(m)

m, c = communities.leiden(H, return_object=True)
print(m)
partitionings.append(m)

m, c = communities.louvain(H, return_object=True)
print(m)
partitionings.append(m)

def get_sorted(d, reverse=False):
    #print("tosort", d)
    s = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    #print("s", s)
    if reverse:
        #print("s keys", list(s.keys()))
        t = list(s.keys())
        t.reverse()
        return t

        
    else:
        return list(s.keys())

def rank_partitions(partitions, G):

    ranks = []

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        
        temp[i] = (communities.get_number_of_communities(com)) #for our fingerprint we prefer a high value
    ranks.append(get_sorted(temp, reverse=True))
    

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        y = communities.get_number_of_nodes_community(com, in_detail = False) #would prefer a pretty equal distribution / a small std value
        temp[i]= y["std"]
    ranks.append(get_sorted(temp, reverse=False))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x, y = communities.average_internal_degree(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.internal_edge_density(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.triangle_ratio(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.conductance(com, G) #prefer a small mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=False))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.mean_outgoing_edge_fraction(com, G) #prefer a small mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=False))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.fraction_of_weak_members(com, G) #prefer a small mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=False))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        y = communities.community_modularity(com, G) #prefer a high mean value
        temp[i] = y
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.cut_ratio(com, G, normalized=True) #prefer a small mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=False))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.community_density_to_graph(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.community_average_shortest_path_fraction(com, G, weight="weight") #prefer a small mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=False))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.hub_dominace(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.clustering_coefficient(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    temp = {}
    for i in range(len(partitions)):
        com = partitions[i]
        x,y = communities.node_embeddedness(com, G) #prefer a high mean value
        temp[i] = y["mean"]
    ranks.append(get_sorted(temp, reverse=True))

    return ranks
    

ranks = rank_partitions(partitionings, H)

#get average rank for each algorithm

mean_rank = {}

for a in range(len(partitionings)):
    temp = []
    for i in range(len(ranks)):
        cur = ranks[i]
        temp.append(cur.index(a))

    mean_rank[a] = statistics.mean(temp)


print(mean_rank)
        
    
        