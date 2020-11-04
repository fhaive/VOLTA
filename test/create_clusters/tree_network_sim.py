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


#for each node per network compute a graph & estimate tree parameters
#create vector for each network containing all parameters for all nodes => estimate distances between
nodes = list(G1.nodes())

def helper_tree_vector(networks, nodes):
    tree_vector = []
    for network in networks:
        temp_vector = []
        for node in nodes:
            #get tree
            tree = trees.construct_binary_tree(network,  root=node, nr_trees=1, type="level")[0]

            #get tree parameters
            depth = trees.tree_depth(tree)
            temp_vector.append(depth)

            paths = trees.leave_path_metrics(tree)
            mean = paths["mean path length"]
            median = paths["median path length"]
            std = paths["std path length"]
            skw = paths["skw path length"]
            kurt = paths["kurtosis path length"]
            alt = paths["altitude"]
            alt_mag = paths["altitude magnitude"]
            ext = paths["total exterior path length"]
            ext_mag = paths["total exterior magnitude"]

            temp_vector.append(mean)
            temp_vector.append(median)
            temp_vector.append(std)
            temp_vector.append(skw)
            temp_vector.append(kurt)
            temp_vector.append(alt)
            temp_vector.append(alt_mag)
            temp_vector.append(ext)
            temp_vector.append(ext_mag)

            asy = trees.tree_asymmetry(tree, trees.number_of_leaves(tree))
            asymmetry = asy["asymmetry"]
            temp_vector.append(asymmetry)

            branching = trees.strahler_branching_ratio(tree)
            mean = branching["mean branching ratio"]
            median = branching["median branching ratio"]
            std = branching["std branching ratio"]
            skw = branching["skw branching ratio"]
            kurt = branching["kurtosis branching ratio"]

            temp_vector.append(mean)
            temp_vector.append(median)
            temp_vector.append(std)
            temp_vector.append(skw)
            temp_vector.append(kurt)

            ext = trees.exterior_interior_edges(tree)
            ee = ext["EE"]
            ei = ext["EI"]
            ee_mag = ext["EE magnitude"]
            ei_mag = ext["EI magnitude"]

            temp_vector.append(ee)
            temp_vector.append(ei)
            temp_vector.append(ee_mag)
            temp_vector.append(ei_mag)

        tree_vector.append(temp_vector)


    return tree_vector



tree_vector = helper_tree_vector(networks, nodes)

with open("tree_vector.pckl", "wb") as f:
    pickle.dump(tree_vector, f, protocol=4)


def helper_tree_sim(networks, nodes):

    results_percentage =  np.zeros((len(networks), len(networks)))
    results_correlation =  np.zeros((len(networks), len(networks)))
    results_smc =  np.zeros((len(networks), len(networks)))
    results_jaccard =  np.zeros((len(networks), len(networks)))
   
    index_list = []
    for index, x in np.ndenumerate(results_percentage):
        temp = (index[1], index[0])
        if temp not in index_list and index not in index_list:
            index_list.append(index)


    for index in index_list:
        n1 = index[0]
        n2 = index[1]

        p = []
        c = []
        s = []
        
        j = []

        for node in nodes:
            t1 = trees.construct_binary_tree(networks[n1],  root=node, nr_trees=1, type="level")[0]
            t2 = trees.construct_binary_tree(networks[n2],  root=node, nr_trees=1, type="level")[0]


            per, u = trees.tree_node_level_similarity(t1, t2, type="percentage")
            cor, u = trees.tree_node_level_similarity(t1, t2, type="correlation")
            
            smc, u = trees.tree_node_level_similarity(t1, t2, type="smc")
            jac, u = trees.tree_node_level_similarity(t1, t2, type="jaccard")


            p.append(per)
            c.append(cor)
            s.append(smc)
            j.append(jac)


        mean_p = statistics.mean(p)
        mean_c = statistics.mean(c)
        mean_s = statistics.mean(s)
        mean_j = statistics.mean(j)

        results_percentage[n1][n2] = mean_p
        results_percentage[n2][n1] = mean_p

        results_correlation[n1][n2] = mean_c
        results_correlation[n2][n1] = mean_c

        results_smc[n1][n2] = mean_s
        results_smc[n2][n1] = mean_s

        results_jaccard[n1][n2] = mean_j
        results_jaccard[n2][n1] = mean_j

    return results_percentage, results_smc, results_correlation, results_jaccard


results_percentage, results_smc, results_correlation, results_jaccard = helper_tree_sim(networks, nodes)


with open("tree_percentage.pckl", "wb") as f:
    pickle.dump(results_percentage, f, protocol=4)


with open("tree_smc.pckl", "wb") as f:
    pickle.dump(results_smc, f, protocol=4)

with open("tree_correlation.pckl", "wb") as f:
    pickle.dump(results_correlation, f, protocol=4)

with open("tree_jaccard.pckl", "wb") as f:
    pickle.dump(results_jaccard, f, protocol=4)



            



            

            





            







#for each node compare network trees and estimate distance matrices