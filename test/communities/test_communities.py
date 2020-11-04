import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../')
import communities
import pickle


with open("../test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)
'''
G = nx.watts_strogatz_graph(200, 5,0.3)
#assigne random edge weights
for edge in G.edges():
    G[edge[0]][edge[1]]["weight"] = random.random()
#try to set node ids to string since int cannot be convert to igraph
'''
re_label = {}
for n in list(G.nodes()):
    re_label[n] = str(n)

H = nx.relabel_nodes(G, re_label)



m = communities.agglomerative_clustering(H, is_distance=False, linkage="average", distance_threshold=0.2)
print(m)

m = communities.markov_clustering(H, inflation=1.4)

print(m)


m, c = communities.async_fluid(H, return_object=True)
print(m)
'''
m, c = communities.cpm(H, return_object=True)
print(m)

m, c = communities.em(H, k=3, return_object=True)
print(m)

m, c = communities.newman_modularity(H, return_object=True)
print(m)

m, c = communities.gdmp2(H, min_threshold=0.25, return_object=True)
print(m)

m, c = communities.greedy_modularity(H, return_object=True)
print(m)

m, c = communities.infomap(H, return_object=True)
print(m)

m, c = communities.label_propagation(H, return_object=True)
print(m)

m, c = communities.leiden(H, return_object=True)
print(m)

print(len(G.nodes()))
m, c = communities.lemon(G,  min_com_size=1, max_com_size=200, expand_step=10, subspace_dim=3, walk_steps=10, biased=True, return_object=True)
print(m)
print(c)

m, c = communities.lemon(G,  min_com_size=20, max_com_size=50, expand_step=6, subspace_dim=3, walk_steps=3, biased=False, return_object=True)
print(m)

m, c = communities.lemon(G,  min_com_size=10, max_com_size=20, expand_step=6, subspace_dim=3, walk_steps=3, biased=False, return_object=True)
print(m)


m, c = communities.lemon(G,  min_com_size=20, max_com_size=50, expand_step=6, subspace_dim=3, walk_steps=10, biased=False, return_object=True)
print(m)


m, c = communities.lemon(G,  min_com_size=10, max_com_size=20, expand_step=6, subspace_dim=3, walk_steps=10, biased=False, return_object=True)
print(m)





m, c = communities.rber_pots(H, return_object=True)
print(m)

m, c = communities.surprise(H, return_object=True)
print(m)

m, c = communities.walktrap(H, return_object=True)
print(m)

m = communities.girvan_newman(G, valuable_edge="betweenness")
print(m)

m = communities.girvan_newman(G, valuable_edge="max_weight")
print(m)

m = communities.girvan_newman(G, valuable_edge="min_weight")
print(m)

m = communities.girvan_newman(G, valuable_edge="current_flow_betweenness")
print(m)

m = communities.girvan_newman(G, valuable_edge="load")
print(m)

m, c = communities.angel(H)
print(m)

m, c = communities.congo(H)
print(m)


m = communities.lais2(G, edge_weight="weight")
print(m)

m, c = communities.ego_networks(G, level = 4)
print(m)


m, c = communities.fuzzy_rough(H)
print(m)

m, c = communities.agdl(H, number_neighbors=1, kc=1)
print(m)


m = communities.refactor_communities(communities.girvan_newman(G, valuable_edge="max_weight"))
print(m)

m2, c = communities.louvain(H, return_object=True)
#m2 = communities.refactor_communities(m2)
print(m2)

print("starting new algo by degree")
mxy = communities.alisas_communities(H, weights="weight", t=0.2, max_iter=1000, by_degre=True, r=5)
print(mxy)

print("starting new algo with 100 iterations without min community")
c = communities.alisas_communities(H, weights="weight", t=0.2, max_iter=1000, by_degree=True, r=5, std=0.05, min_community=None)
print(c)

print("starting new algo with 100 iterations with min community")
c = communities.alisas_communities(H, weights="weight", t=0.2, max_iter=1000, by_degree=True, r=5, std=0.05, min_community=10)
print(c)

G = H.copy()
print("starting disconnecting")
print("G edges", len(G.edges()))
c = communities.disconnect_high_degree_nodes(G, nodes=20, percentage=None, weight="weight", graph = False)
print(c)

G = H.copy()
print("starting disconnecting no weight")
print("G edges", len(G.edges()))
c = communities.disconnect_high_degree_nodes(G, nodes=20, percentage=None, weight=None, graph = False)
print(c)




m3, c = communities.walktrap(H, return_object=True)
#m3 = communities.refactor_communities(m3)
print(m3)

ini, com, ini_graph = communities.create_initial_consensus(H, [m2, m3], thresh = 0.6)

communities, consensus_com, initial_communities = communities.fast_consensus(H, [m2, m3], algorithms = [communities.louvain,communities.walktrap], parameters=[{"return_object":False}, {"return_object":False}], thresh = 0.6, delta = 0.02, max_iter=10, initial=ini_graph)
print("final communities", communities)

#print("counts initial", consensus_com)

#print("initial consensus", initial_communities)



#test community evaluation functions

number = communities.get_number_of_communities(m)
print("number of communities leiden", number)

number = communities.get_number_of_communities(mxy)
print("number of communities alisas", number)

number = communities.get_number_of_communities(m2)
print("number of communities louvain", number)

number = communities.get_number_of_communities(m3)
print("number of communities walktrap", number)

number = communities.get_number_of_communities(c)
print("number of communities converged", number)


print("edge weights")

w, detail = communities.mean_edge_weight(m, H, weight="weight")
print(detail)

w, detail = communities.mean_edge_weight_fraction(m, H, weight="weight")
print(detail)

dist, detail = communities.get_number_of_nodes_community(m, in_detail = True)
print(dist)
#print(detail)

degree, dd = communities.average_internal_degree(m, H)
print(degree)
print(dd)

degree, dd = communities.average_internal_degree(m, H)
print("mean internal degree leiden", dd)

degree, dd = communities.average_internal_degree(mxy, H)
print("mean internal alisas", dd)

degree, dd = communities.average_internal_degree(m2, H)
print("mean internal louvain", dd)

degree, dd = communities.average_internal_degree(m3, H)
print("mean internal walktrap", dd)

degree, dd = communities.average_internal_degree(c, H)
print("mean internal converged", dd)

d, dd = communities.internal_edge_density(m, H)
print(d)
print(dd)

d, dd = communities.triangle_ratio(m, H)
print(d)
print(dd)

d, dd = communities.internal_edge_count(m, H)
print(d)
print(dd)

d, dd = communities.fraction_of_median_degree(m, H)
print(d)
print(dd)

d, dd = communities.community_density_to_graph(m, H)
print(d)
print(dd)

d, dd = communities.community_average_shortest_path(m, H, weight="weight")
print(d)
print(dd)

d, dd = communities.community_average_shortest_path_fraction(m, H, weight="weight")
print(d)
print(dd)

d, dd = communities.hub_dominace(m, H)
print(d)
print(dd)

d, dd = communities.clustering_coefficient(m, H)
print(d)
print(dd)


degree, dd = communities.clustering_coefficient(m, H)
print("cc ee leiden", dd)

degree, dd = communities.clustering_coefficient(mxy, H)
print("cc alisas", dd)

degree, dd = communities.clustering_coefficient(m2, H)
print("cc  louvain", dd)

degree, dd = communities.clustering_coefficient(m3, H)
print("cc  walktrap", dd)

degree, dd = communities.clustering_coefficient(c, H)
print("cc  converged", dd)

d, dd = communities.node_embeddedness(m, H)
print(d)
print(dd)

d, dd = communities.cut_ratio(m, H, normalized=False)
print(d)
print(dd)

d, dd = communities.outgoing_edges(m, H)
print(d)
print(dd)

d, dd = communities.conductance(m, H)
print(d)
print(dd)


d, dd = communities.max_outgoing_edge_fraction(m, H)
print(d)
print(dd)

d, dd = communities.mean_outgoing_edge_fraction(m, H)
print(d)
print(dd)

d, dd = communities.fraction_of_weak_members(m, H)
print(d)
print(dd)

d = communities.community_modularity(m, H)
print(d)

x, d, dd = communities.modular_density(m, H)
print(x)
print(d)
print(dd)
'''
