import networkx as nx
import pandas as pd
import csv
import random
import sys
sys.path.insert(1, '../../distances/')
import trees
import pickle
import treelib as bt
sys.path.insert(1, '../../')
import communities


with open("/home/alisa/phd/projects/2019/inform/neo_code/repository/graphAlgorithms/test/test_distance_graph.pckl", "rb") as f:
    G = pickle.load(f)
X = nx.generators.classic.circulant_graph(100, [1,5])

#create tree
#returns list
ts = trees.construct_tree(X, root=None, nr_trees=1, type="level")

t1 = ts[0]
t2 = ts[1]
t1.show()
#t1.show()
#t2.show()

ts, loops = trees.construct_tree(X, root=None, nr_trees=1, type="cycle", cycle_weight = "betweenness_max", initial_cycle_weight=False)

t3 = ts[0]
#t2 = ts[1]

t3.show()
#print(loops[0])

mean, per = trees.tree_node_level_similarity(t1, t2, type="percentage")
print(mean)
print(per)

mean, per = trees.tree_node_level_similarity(t1, t2, type="correlation")
print(mean)
print(per)

#smc just returns same as jaccard ni this case
mean, per = trees.tree_node_level_similarity(t1, t2, type="smc")
print(mean)
print(per)

mean, per = trees.tree_node_level_similarity(t1, t2, type="jaccard")
print(mean)
print(per)

print(trees.number_of_leaves(t1))

paths = trees.leave_path_metrics(t1)
print(paths)

asy = trees.tree_asymmetry(t1, trees.number_of_leaves(t1))
print(asy)

str = trees.strahler_branching_ratio(t1)
print(str)

branching = trees.exterior_interior_edges(t1)
print(branching)
