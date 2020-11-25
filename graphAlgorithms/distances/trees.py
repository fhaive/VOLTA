"""
Converts a graph object into a (binary) tree, which can be used to compare networks structural makeup based on known tree distance measures.
This module is build upon treelib https://treelib.readthedocs.io/en/latest/treelib.html.

"""

import pandas as pd
import glob
import sys
import os
import datetime
import math
import networkx as nx
import collections
import matplotlib.pyplot as plt
import random
import treelib as bt
import statistics
from scipy.stats import kurtosis, skew, kendalltau
from .node_edge_similarities import *
from operator import itemgetter




def construct_tree(H,  root=None, nr_trees=1, type="level", edge_attribute="weight", cycle_weight = "max", initial_cycle_weight=True):
	'''
	Function to construct a binary tree from a graph object. This can reduce complexity since number of edges is reduced.

	Parameters:
		H (networkX graph object):
		root (None, int or str): provide node ID which should be selected as root of the tree. If None then a random node is selected.
		nr_trees (int): specifies how many trees are constructed. If root is None then multiple random trees can be created.
		type (str): if is "level" then a hierarchical tree is created. Paths from the root indicate how far each nodes are from the root node. Edge weights are not considered.
					if is "cycle" then a hierarchical tree is created where each node represents a cycle in the graph. 
						Tree leaves are the original cycles in the graph and are merged into larger cycles through edge removal until all have been merged into a single cycle.
						This method can be helpful to categorize cyclic graphs. The root parameter is not considered when this option is selected and only cyclic structures in the graph are considered.
		edge_attribute (str): name of the edge weights to be considered if type = "cycle".
		cycle_weight (str): sets how cycles are merged i.e. which edges are removed to merge cycles into larger ones.
							if is "max" then the edge with the highest weight is removed first. if is "min" then the edge with the smalles weight is removed first.
							if is "betweenness_max" the the edge with the highest betweenness value is removed first.
							if is "betweenness_min" the edge with the lowest betweenness value is removed first.
		initial_cycle_weight (boolean): if True the initial cycle basis is estimated based on edge weights with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.minimum_cycle_basis.html#networkx.algorithms.cycles.minimum_cycle_basis
										if False the initial cycles are estimated based on steps only with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis

	Returns:
		trees (list): list of treelib objects
		cycles (list): if level = "cycle" then additional a list of cycle IDs to edge mappings is returned
		

	'''
	G = H.copy()

	trees = []
	if type == "level":
		
		for i in range(nr_trees):
			if root is None:
				root_tree = random.choice(list(G.nodes()))
			else:
				root_tree = root

			tree = bt.Tree()

			#root
			tree.create_node(root_tree, root_tree)

			#create child nodes
			parents = [root_tree]
			in_tree = [root_tree]

			
			#get all one step neighbors for current node

			while len(parents) > 0:
				parents_temp = []
				for par in parents:

					if par in list(G.nodes()):
						children = G.neighbors(par)

						for child in children:
							if child not in in_tree:

								tree.create_node(child, child, parent=par)				
								#add to new parent list
								parents_temp.append(child)
								in_tree.append(child)



				parents = parents_temp

			trees.append(tree)
		return trees

	elif type == "cycle":
		print("CAREFUL this only works if you have a cyclic graph!!! else only the structures containing cycles will be considered, this may result in a very small tree only containing some of the original nodes")
		l = []
		for i in range(nr_trees):
			#get all initial cycles in the graph
			X = G.copy()
			loops, relationship, all_loops = __create_loop_relationships__(X, initial_cycle_weight, cycle_weight, edge_attribute)

			if loops is not None:
				tree = __build_tree_hierarchical__(loops, relationship)
			else:
				tree = None

			trees.append(tree)
			l.append(all_loops)
		return trees, l

	else:
		print("type is not implemented")



		return trees

def __build_tree_hierarchical__(loops, relationship):
	"""
	helper function to create a hierarchical tree based on return values of create_loop_relationships()
	Input
		dict of loops contained in a graph (or any other structure)

		dict of relationships between the structures contained in loops

	Output
		a treelib tree
	"""
	#loops contains loop ID and corresponding edges/nodes in Graph

	#build list out of stage information with each child/ leave from left to right 
	print("building tree...")
	z= 1
	#get root node
	print(relationship)
	temp = sorted(relationship.keys())[-z]

	#if root has no children jump to next node
	while relationship[temp] == []:
		z = z+1
		temp = sorted(relationship.keys())[-z]


	#construct tree

	tree = bt.Tree()

	#root
	tree.create_node(temp, temp)

	#create child nodes
	parents = [temp]
	in_tree = [temp]
	while len(parents) > 0:
		parents_temp = []
		for par in parents:

			children = relationship[par]

			for child in children:
				if child is not None:

					tree.create_node(child, child, parent=par)				
					#add to new parent list
					parents_temp.append(child)
					in_tree.append(child)

		parents = parents_temp

	#build tree out of list
	#tree = bt.build(nodes)
	print("tree built")
	return tree

def __create_loop_relationships__(G,initial_cycle_weight, cycle_weight, edge_attribute):
	"""
	helper function to determine child parent relationship for loops (or any other structural components)

	Input
		networkx graph G

		edge_attribute str, the name of the edge weight to be considered for type cycle

		cycle_weight
			the weight based on which edge are removed
			max: edge with the highest edge weight
			min: edge with the lowest edge weight
			betweenness_max: edge with the highest betweenness value
			betweenness_min: edge with the lowest betweenness value

		initial_cycle_weight
			if true then the initial cycle basis is estimated based on edge weight 
				with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.minimum_cycle_basis.html#networkx.algorithms.cycles.minimum_cycle_basis
			else the initial cycles are estimated based on steps only
				with https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis

		Output
			dict of final loops

			dict of relationships

			dict of all loops

	"""
	if initial_cycle_weight:
		cycles = nx.minimum_cycle_basis(G)
	else:
		cycles = nx.cycle_basis(G)
	#print("cycles", cycles)
	#convert cycles into dict & edge list
	cycles_dict = __convert_cycles__(cycles)
	#print("cycles", cycles_dict)

	#construct dict to save cycles and give them id which is saved in tree
	#append each new created loop to dict
	save_loops = {}
	all_loops = {}
	for i in range(len(cycles_dict)):
		all_loops[i] = cycles_dict[i]
		save_loops[i] = cycles_dict[i]
	
	#2nd dict that stores loop id and  child nodes in list
	#initialize leave nodes
	relationship = {}
	for loop in all_loops:
		relationship[loop] = [None, None]

	#merged cycles - stores cycles already merged
	merged_nodes = []

	print("start creating hierarchical tree")

	while len(G.edges()) > 0:

		#community weight functions are used to find the most valuable edge

		if cycle_weight == "max":
			to_remove = __by_weight__(G, w_max=True, attribute=edge_attribute)
		
		elif cycle_weight =="min":
			to_remove = __by_weight__(G, w_max=False, attribute=edge_attribute)

		elif cycle_weight =="betweenness_max":
			to_remove = __by_centrality__(G, w_max=True, attribute=edge_attribute, type="betweenness")

		elif cycle_weight == "betweenness_min":
			to_remove = __by_centrality__(G, w_max=False, attribute=edge_attribute, type="betweenness")

		else:
			print("cycle weight not known, please select another one")
			return None
			break

		#remove edge from graph
		G.remove_edge(to_remove[0], to_remove[1])

		children = []
		cycle = []
		for loop_ID, loop in all_loops.items():
		#compare edges (independent of direction (x,y) == (y,x))
			#print(loop)
			#print(to_remove)
			if ((to_remove in loop) or ((to_remove[1], to_remove[0]) in loop)):
							
				children.append(loop_ID)
				merged_nodes.append(loop_ID)
				
				#new cycle is made up of all child edges except the removed edge
				for edge in loop:
					if ((edge not in cycle) and ((edge[1], edge[0]) not in cycle)):
						if edge != to_remove and edge != (to_remove[1], to_remove[0]):
							cycle.append(edge)

		new_key = int(sorted(save_loops.keys())[-1]) + 1
		all_loops[new_key] = cycle
		save_loops[new_key] = cycle
		
		#and remove merged loops
		for child in children:
			del all_loops[child]
		
		#add parent child connection to relationship
		new_stage = int(sorted(relationship.keys())[-1]) + 1
		#find ID of children

		relationship[new_key] = children

	print("finished edge removal")

	return all_loops, relationship, save_loops

def __by_weight__(G, w_max=True, attribute="weight"):
    """
    helper function to find the most valuable edge 
    Input

        networkx graph G

        w_max
            if true edge with the highest weight attribute is returned
            else edge with the smallest edge attribute is returned

        attribute
            str of edge attribute name
    """
    if w_max:
        u, v, w = max(G.edges(data=attribute), key=itemgetter(2))
    else:
        
        u, v, w = min(G.edges(data=attribute), key=itemgetter(2))

    return (u, v)


def __by_centrality__(G, w_max=True, attribute="weight", type="betweenness"):
    """
    helper function to find the most valuable edge 
    returns the edge with the highest/ lowest  score

    Input

        networkx graph G

        w_max
            if true edge with the highest weight attribute is returned
            else edge with the smallest edge attribute is returned

        attribute
            str of edge attribute name

        type
            what centrality measure should be used
            options
                betweenness : based on betweenness centrality
                current_flow_betweenness :  based on current flow betweenness centrality
                load : based on load centrality 

    """
    if type == "betweenness":
        centrality = nx.edge_betweenness_centrality(G, weight=attribute)

    elif type == "current_flow_betweenness":
        centrality = nx.edge_current_flow_betweenness_centrality(G, weight=attribute)
    
    elif type == "load":
        centrality = nx.algorithms.centrality.edge_load_centrality(G)


    else:
        print("method not implemented, please define your own function")
        return None


    if w_max:
        return max(centrality, key=centrality.get)
    else:
        return min(centrality, key=centrality.get)


				
def __convert_cycles__(cycles):
	"""
	helper function of construct_binary_tree() when tree is constructed based on cycles
	it takes the output of nx.minimum_cycle_basis(G) or nx.cycle_basis(G)
	and transforms it into an edge list

	Input
		list of sublists contianing node IDs of cycles

	Output
		dict, where each key represents a cycle ID & contains list of edge tuples which construct the cycles
	"""

	cnt = 0
	cycles_dict = {}
	for cycle in cycles:
		edges = []
		for i in range(len(cycle)):
			if i == 0:
				initial = cycle[i]
				current = cycle[i]

			elif i == len(cycle) -1:
				#last one in list
				edge = (current, cycle[i])
				edges.append(edge)
				edge = (cycle[i], initial)
				current = cycle[i]
				edges.append(edge)

			else:
				edge = (current, cycle[i])
				current = cycle[i]
				edges.append(edge)
		cycles_dict[cnt] = edges
		cnt = cnt +1

	return cycles_dict
		

def tree_node_level_similarity(t1, t2, type="percentage"):
	"""
	Computes the similarity of nodes (based on their level in two rooted trees t1 and t2).

	Parameters:
		t1 (treelib tree object):
		t2 (treelib tree object)
		type (str): defines comparison method. If type = "percentage" then for each level the percentage of shared nodes based on the max possible (length of smaller) shared nodes is estimated.
					If type = "correlation" then Kendall rank correlation is estimated. Node rankings are estimated based on their level in the trees. If there are unequal number of nodes a subset of the larger one is selected.
					If type = "smc" then the smc distance for each level is estimated.
					If type ) "jaccard" then the jaccard similarity for each level is estimated.
		
	Returns:
		mean similarity (list): mean similarity scores for each level
		all scores (list):
		if type = "correlation" then kendall tau (float) and its corresponding p-val (float) are returned instead.
		
	"""

	t1_nodes = t1.all_nodes()
	t2_nodes = t2.all_nodes()

	t1_level = {}
	t2_level = {}

	max_level1 = 0
	max_level2 = 0

	for node in t1_nodes:
		depth = t1.depth(node)
		t1_level[node.tag] = depth

		if depth > max_level1:
			max_level1 = depth

	for node in t2_nodes:
		depth = t2.depth(node)
		t2_level[node.tag] = depth

		if depth > max_level2:
			max_level2 = depth

	#print("tree max levels/ depths are", max_level1, max_level2)
	#print(t1_level)
	#print(t2_level)
	if type=="percentage":
		percentages = []
		for i in range(max(max_level1, max_level2)+1):
			temp1 = [k for k,v in t1_level.items() if float(v) == i]
			temp2 = [k for k,v in t2_level.items() if float(v) == i]

			#if one is 0 p is set to 0
			if len(temp1) < 1 or len(temp2) < 1:
				print("level is 0, p is set to 0")
				p = 0

			else:
				if len(temp1) > len(temp2):
					shared = list(set(temp1).intersection(set(temp2)))

				else:
					shared = list(set(temp2).intersection(set(temp1)))
				#print(shared)
				p = percentage_shared(shared, temp1, temp2, penalize=False)

			percentages.append(p)

		mean_percentage = statistics.mean(percentages)

		return mean_percentage, percentages

	elif type == "correlation":
		#compute kendall between ranked node lists based on level ranking
		ranked1 = []
		ranked2 = []

		for i in range(max(max_level1, max_level2)+1):
			temp1 = [k for k,v in t1_level.items() if float(v) == i]
			temp2 = [k for k,v in t2_level.items() if float(v) == i]

			#since kendall requires lists to be of same length
			#for each layer only same amount of nodes are taken into account

			m=min(len(temp1), len(temp2))
			
			ranked1.append(sorted(temp1)[:m])
			ranked2.append(sorted(temp2)[:m])

		ranked1 = [item for sublist in ranked1 for item in sublist]
		ranked2 = [item for sublist in ranked2 for item in sublist]
		#print("ranked1", ranked1)

		tau, p = kendalltau(ranked1, ranked2, nan_policy="omit")

		return tau, p
	

	elif type =="smc":
		#estimate smc distance of node lists

		smc = []
		for i in range(max(max_level1, max_level2)+1):
			temp1 = [k for k,v in t1_level.items() if float(v) == i]
			temp2 = [k for k,v in t2_level.items() if float(v) == i]

			#if one is 0 p is set to 0
			if len(temp1) < 1 or len(temp2) < 1:
				s = 0

			else:
				

				s = __compute_smc_level__(temp1, temp2)

			smc.append(s)

		mean_smc = statistics.mean(smc)

		return mean_smc, smc




	elif type=="jaccard":
		jaccard = []
		for i in range(max(max_level1, max_level2)+1):
			temp1 = [k for k,v in t1_level.items() if float(v) == i]
			temp2 = [k for k,v in t2_level.items() if float(v) == i]

			#if one is 0 p is set to 0
			if len(temp1) < 1 or len(temp2) < 1:
				j = 0

			else:
				if len(temp1) > len(temp2):
					shared = list(set(temp1).intersection(set(temp2)))

				else:
					shared = list(set(temp2).intersection(set(temp1)))

				j = calculate_jaccard_index(shared, temp1, temp2, similarity=True)

			jaccard.append(j)

		mean_jaccard = statistics.mean(jaccard)

		return mean_jaccard, jaccard




def __compute_smc_level__(list1, list2):
	"""
	helper function of tree_node_level_similarity()
		smc adapted to tree levels
		if a node is in both levels it is counted as a match if it is not in both levels it is counted as a mismatch

	Input
		it is assumed that list1 and list2 are nodes contained in the same level

	Output
	float, smc score
	"""
	match = 0
	no_match = 0

	checked = []

	for k in range(len(list1)):
		if list1[k] not in checked:
			if list1[k] in list2:
				match = match + 1
			else:
				no_match = no_match + 1

			checked.append(list1[k])

	for k in range(len(list2)):
		if list2[k] not in checked:
			if list2[k] in list1:
				match = match + 1
			else:
				no_match = no_match + 1

			checked.append(list2[k])

	smc = match / (match + no_match)
	return smc

		



def tree_depth(t):
	"""
	Returns depth of a tree

	Parameters:
		t (treelib tree object):

	Returns:
		tree depth (int):
	"""
	return t.depth()

def number_of_leaves(t):

	"""
	Returns the number of leaves in t.

	Parameters:
		t (treelib tree object):

	Returns:
		leaves (int):
	"""

	return (len(t.leaves()))

def leave_path_metrics(t):
	'''
	Estimates the root - leave pathlength distribution.

	Parameters:
		t (treelib tree object):

	Returns:
		distribution (dict): keys are mean path length, median path length, std path length, skw path length, kurtosis path length, altitude, altitude magnitude, total exterior path length, total exterior magnitude
	'''
	path_to_leaves = t.paths_to_leaves()
	nr_leaves=number_of_leaves(t)
	length = []
	for path in path_to_leaves:
		length.append(len(path))
	if len(length) > 1:
		avg_path = statistics.mean(length)
		median_path = statistics.median(length)
		std_path = statistics.stdev(length)	
		skw_path = skew(length)
		kurt_path = kurtosis(length)
		altitude = sorted(length)[-1]
		total_exterior_path_length = sum(length)
		altitude_mag = altitude/nr_leaves
		total_exterior_mag = altitude/nr_leaves

	else:
		print("less than two paths found, no distribution can be estimated")
		avg_path = None
		median_path = None
		std_path = None
		skw_path = None
		kurt_path = None
		altitude = None
		total_exterior_path_length = None
		altitude_mag = None
		total_exterior_mag = None


	return {"mean path length":avg_path, "median path length": median_path, "std path length":std_path, "skw path length":skw_path, "kurtosis path length":kurt_path, "altitude":altitude, "altitude magnitude":altitude_mag, "total exterior path length":total_exterior_path_length, "total exterior magnitude":total_exterior_mag}

def __partition_symmetry__(subtree):
	"""
	Estimates tree asymmetry based on all possible subtrees.
	helper function of tree_asymmetry

	Input
		treelib object

	Output
		asymmetry, degree of tree
	"""
	#split subtree into its 2 subtrees
	current_root = subtree.root

	#degree of subtree
		
	degree = len(subtree.leaves())
	#get children (new roots)
	children = subtree.children(current_root)

	child_1 = None
	child_2 = None

	if len(children) > 0:
		child_1 = children[0].tag
	if len(children) > 1:
		child_2 = children[1].tag
		
	tree_1 = None
	tree_2 = None

	#build new subtrees
	if child_1 is not None:
		tree_1 = bt.Tree(subtree.subtree(child_1), deep = True)
	if child_2 is not None:
		tree_2 = bt.Tree(subtree.subtree(child_2), deep = True)

	degree_1 = 0
	degree_2 = 0
	#get degree
	if tree_1 is not None:

		degree_1 = len(tree_1.leaves())

	if tree_2 is not None:

		degree_2 = len(tree_2.leaves())

	if (degree_1 >= degree_2) and (degree_1 != 0):
		asymmetry = (degree_1 - degree_2) / degree_1
	elif (degree_2 >= degree_1) and (degree_2 != 0):
		asymmetry = (degree_2 - degree_1) / degree_2
	#shouldnt be assigned 
	else:
		asymmetry = 0


	return degree * asymmetry, degree

def tree_asymmetry(t, nr_leaves):
	"""
	Estimates tree asymmetry of a tree based on asymmetry of all possible subtrees
	

	Parameters:
		t (treelib tree object):
		nr_leaves (int): number of leave nodes contained in t.

	Returns:
		asymmetry (dict): keys are asymmetry, degree asymmetry (for each subtree)
	"""

	#weighted average of degree of all subpartitions (number of leaves)

	#for all nodes (except leave nodes) calculate subtree and asymmetry of this tree
	leaves = []
	for leave in t.leaves():
		leaves.append(leave.tag)
		
	total_weight = 0 
	total_asymmetry = 0
	degree_asymmetry = {}
	for node in t.nodes:
		#if not a leave node

		if node not in leaves:
			asymmetry, weight = __partition_symmetry__(bt.Tree(t.subtree(node), deep = True))
			total_asymmetry = total_asymmetry + asymmetry
			total_weight = total_weight + weight
			degree_asymmetry[weight] = (1/weight)*asymmetry
	if total_weight > 0:
		asymmetry = (1/total_weight) * total_asymmetry
	else:
		asymmetry = total_asymmetry

	return {"asymmetry":asymmetry, "degree asymmetry":degree_asymmetry}

def __set_strahler_number__(b_tree):
	"""
	helper function of strahler_branching_ratio(t) by initializing strahler numbers for each edge

	Input
		treelib object

	Output
		branching segments,
		treelib object
	"""

	updated_children = []
	for leave in b_tree.leaves():
		#set strahler number 1
		b_tree.update_node(leave.tag, data=1)
		updated_children.append(leave.tag)
	#update parent nodes
	#dict to save number of segments
	#continous line of same number is counted as one segment

	branching_segments = {}
	#all leave nodes are single segment of count 1
	branching_segments[1] = len(updated_children)

	while (b_tree.get_node(updated_children[0]).is_root() == False): #not root not reached
		for node in updated_children:
			#get strahler number and number of sibling
			strahler1 = b_tree.nodes[node].data

			#get sibling node

			sibling = b_tree.siblings(node)
			if len(sibling) > 0: #has sibling, max 1
				sib_id = sibling[0].identifier
				strahler2 = b_tree.nodes[sib_id].data

				#print(strahler2)
			else:
				strahler2 = 0
				sib_id = None

			#if sibling number not set jump 
			if strahler2 is not None:
				#strahler parent
				if strahler1 == strahler2:
					strahler_parent = strahler1 + 1
					#save in dict
					#check if key already exists
					if strahler_parent in branching_segments:
						temp = int(branching_segments[strahler_parent]) + 1
						branching_segments[strahler_parent] = temp
					else:
						branching_segments[strahler_parent] = 1
				#continous line of same number so counted as continous segment
				elif strahler1 > strahler2:
					strahler_parent = strahler1
				else:
					strahler_parent = strahler2

				#get parent and update if not root
				NoneType = type(None)
				if isinstance(b_tree.parent(node), NoneType) == False:
					parent = b_tree.parent(node).identifier
					b_tree.update_node(parent, data=strahler_parent)

					#add parent to list and remove children
					updated_children.append(parent)
				updated_children.remove(node)
				if sib_id is not None and sib_id in updated_children:
					updated_children.remove(sib_id)
					
	return branching_segments, b_tree

def strahler_branching_ratio(t):
	"""
	Calculates the strahler branching ratio of a tree t.

	Parameters:
		t (treelib tree object):

	Returns:
		barnching distribution (dict): keys are mean branching ratio, median branching ratio, std branching ratio, skw branching ratio, kurtosis branching ratio
	"""

	branching, new_tree = __set_strahler_number__(t)
	ratio = []
	key = list(branching.keys())
	for i in range(len(branching.keys())-1):
		
		ratio.append(int(branching[key[i]]) / int(branching[key[i+1]]) )
	if len(ratio)>1:
		avg_ratio = statistics.mean(ratio)
		median_ratio = statistics.median(ratio)
		std_ratio = statistics.stdev(ratio)	
		skw_ratio = skew(ratio)
		kurt_ratio = kurtosis(ratio)
	else:
		print("less than two parameters in ratio, no distribution can be estimated")
		avg_ratio = None
		median_ratio = None
		std_ratio = None
		skw_ratio = None
		kurt_ratio = None


	return {"mean branching ratio":avg_ratio, "median branching ratio":median_ratio, "std branching ratio":std_ratio, "skw branching ratio":skw_ratio, "kurtosis branching ratio":kurt_ratio}

def __exterior_interior__(t):
	"""
	Calculates the number of external & internal edges
	helper function of exterior_interior_edges()

	Input
		treelib object

	Output
		count of external and internal edges
	"""


	#number of external and external/ internal edges
	#use data from strahler (if both 1 then EE) if one large 1 then EI
	EE = 0
	EI = 0

	branching, b_tree = __set_strahler_number__(t)

	visited = []
	for node in b_tree.leaves():

		if node.identifier not in visited:
			#get parent and sibling of node
			sibling = b_tree.siblings(node.identifier)
			if len(sibling) > 0: #has sibling, max 1
				sib_id = sibling[0].identifier
			else:
				sib_id = None

			#parent = b_tree.parent(node.identifier).identifier

			#if siblings have same strahler number they are EE edges
			if sib_id is not None:
				if b_tree.get_node(node.identifier).data == b_tree.get_node(sib_id).data:
					EE = EE +1
				else:
					EI = EI+1
				visited.append(sib_id)

			visited.append(node.identifier)
	return EE, EI


def exterior_interior_edges(t):
	"""
	Estimates the number of exterior (EE) & interior (IE) edges and their magnitude

	Parameters:
		t (treelib tree object):

	Returns:
		exterior / interior edges (dict): keys are EE, EI, EE magnitude, EI magnitude
	"""
	ee, ei = __exterior_interior__(t)

	nr_leaves = number_of_leaves(t)
	ee_mag = ee/nr_leaves
	ei_mag = ei/nr_leaves


	return {"EE":ee, "EI":ei, "EE magnitude":ee_mag, "EI magnitude":ei_mag}


