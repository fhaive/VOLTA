import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
import global_distances
import pickle


def add_noise(H, p=0.2, iter=100, pw=0.1, weight="weight"):
    """
    takes networkx G and randomly rewires edges

    Input

        networkx Graph H (undirected)

        p: probability that edge x is rewired to a random choosen node

        iter: how many iterations are performed

        pw: probability of assigning a new weight to an edge (weight is choosen randomly between 0 & 1)

        weight: str attribute of edge weight that should be reassigned
    """

    G = H.copy()
    edges = list(G.edges())
    nodes = list(G.nodes())

    print("G has", len(edges), "edges and ", len(nodes), "nodes")

    countr = 0
    countw = 0
    for i in range(iter):
        edges = list(G.edges())
        nodes = list(G.nodes())
        #randomly select edge from edges
        edge = random.choice(edges)
        #print(G[int(edge[0])][int(edge[1])])
        w = G[int(edge[0])][int(edge[1])][weight]

        #rewire?
        rewire = random.choices([True, False], weights=[p, 1-p], k=1)[0]
        weig = random.choices([True, False], weights=[pw, 1-pw], k=1)[0]
        if weig:
            countw = countw + 1
            w = random.random()
            G[int(edge[0])][int(edge[1])][weight] = w

        if rewire:
            #print("rewire", edge)
            countr = countr + 1
            #edge is rewired to a random choosen node
            node = random.choice([edge[0], edge[1]])
            node2 = random.choice(nodes)

            #if there is an edge between nodes already remove
            if G.has_edge(node, node2) and node2 != node and node2 != edge[0] and node2 != edge[1]:
                G.remove_edge(node, node2)
            
            G.add_edge(node, node2)
            G[node][node2][weight] = w
            G.remove_edge(*edge)

    edges = list(G.edges())
    nodes = list(G.nodes())

    print("G has", len(edges), "edges and ", len(nodes), "nodes")
    print(countr, "edge have been rewired", countw, "edge weights have been updated")

    return G








with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G1 = pickle.load(f)

G2 = add_noise(G1, p=0.2, iter=100, pw=0.1, weight="weight")
with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph2.pckl", "wb") as f:
    pickle.dump(G2, f, protocol=4)

G3 = add_noise(G1, p=0.6, iter=100, pw=0.4, weight="weight")
with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph3.pckl", "wb") as f:
    pickle.dump(G3, f, protocol=4)




#permutate networks by rewirering nodes

G11 = add_noise(G1, p=0.01, iter=100, pw=0.1, weight="weight")
with open("test_distance_G11.pckl", "wb") as f:
    pickle.dump(G11, f, protocol=4)
G12 = add_noise(G1, p=0.1, iter=100, pw=0.01, weight="weight")
with open("test_distance_G12.pckl", "wb") as f:
    pickle.dump(G12, f, protocol=4)
G13 = add_noise(G1, p=0.4, iter=100, pw=0.4, weight="weight")
with open("test_distance_G13.pckl", "wb") as f:
    pickle.dump(G13, f, protocol=4)

G111 = add_noise(G11, p=0.2, iter=100, pw=0.1, weight="weight")
with open("test_distance_G111.pckl", "wb") as f:
    pickle.dump(G111, f, protocol=4)
G112 = add_noise(G11, p=0.2, iter=100, pw=0.3, weight="weight")
with open("test_distance_G112.pckl", "wb") as f:
    pickle.dump(G112, f, protocol=4)

G131 = add_noise(G13, p=0.2, iter=100, pw=0.6, weight="weight")
with open("test_distance_G131.pckl", "wb") as f:
    pickle.dump(G131, f, protocol=4)


G21 = add_noise(G2, p=0.41, iter=100, pw=0.1, weight="weight")
with open("test_distance_G21.pckl", "wb") as f:
    pickle.dump(G21, f, protocol=4)
G22 = add_noise(G2, p=0.41, iter=100, pw=0.1, weight="weight")
with open("test_distance_G22.pckl", "wb") as f:
    pickle.dump(G22, f, protocol=4)


G31 = add_noise(G3, p=0.2, iter=100, pw=0.7, weight="weight")
with open("test_distance_G31.pckl", "wb") as f:
    pickle.dump(G31, f, protocol=4)
G32 = add_noise(G3, p=0.3, iter=100, pw=0.1, weight="weight")
with open("test_distance_G32.pckl", "wb") as f:
    pickle.dump(G32, f, protocol=4)

