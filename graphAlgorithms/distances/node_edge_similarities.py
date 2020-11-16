"""
functions to estimate between network simialrities based on nodes & edges
"""

import networkx as nx
import numpy as np
import os
import ast
import sys
import matplotlib.pyplot as plt
import pandas as pd
import gmatch4py as gm
import seaborn as sns
#!pip3 install dask[bag]
import dask.bag as db
import asyncio
import scipy
import operator
import statistics
import pickle


def percentage_shared(shared, list1, list2, penalize=False, weight="length"):
    """
    estimates percenatge of shared edges between two networks
    calculates percentage of shared edges based on max possible shared exdges = len of smaller list

    Input
        list of shared edges & list of edges of 2 networks
            node ids need to be consistet between networks
            non of the lists is allowed to contain duplicate values
        
        if penalice = True the score is penaliced by weight value based on difference in list length
            weight value can be length or int
                length will penalize based on difference between edges
                weight will set larger weight to penalizing based on lenght difference
            in order to estimate if one list is subset of other set penalize to False

    Output
        float
    
    """

    ll1 = len(list1)
    ll2 = len(list2)

    ls = len(shared)

    p = ls / min(ll1, ll2)

    if not penalize:

        return p
    else:
        if type(weight) != type(1):
            if weight == "length":

                if len(list1) != len(list2):

                    return p * (1 / (abs(len(list1) - len(list2))))
                else:
                    print("both lists have same length, not penalized")
                    return p

        elif type(weight) == type(1):
            return (p * (1 / weight * (abs(len(list1) - len(list2)))))
        else:
            print("wrong weight will return non penalized score")
            return (p)


async def percentage_shared_async(shared, list1, list2, index, penalize=False, weight="length"):

    """
    
    same as percentage shared but runs in parallel
    s. percentage shared for parameter explanation

    index is used to save results in original order
    """

    ll1 = len(list1)
    ll2 = len(list2)

    ls = len(shared)

    p = ls / min(ll1, ll2)

    if not penalize:

        return (index, p)
    else:
        if type(weight) != type(1):
            if weight == "length":

                if len(list1) != len(list2):

                    return (index, p * (1 / (abs(len(list1) - len(list2)))))
                else:
                    print("both lists have same length, not penalized")
                    return (index, p)

        elif type(weight) == type(1):
            return (index, p * (1 / weight * (abs(len(list1) - len(list2)))))
        else:
            print("wrong weight will return non penalized score")
            return (index, p)


def calculate_jaccard_index(shared, list1, list2, similarity=True):
    """
    calculates jaccard index 
    
    Input
        list of shared edges & list of edges of 2 networks
            node ids need to be consistet between networks
            non of the lists is allowed to contain duplicate values
        if similarity = False will return jaccard distance
    Output
        float
    """

    ll1 = len(list1)
    ll2 = len(list2)
    ls = len(shared)


    try:
        j = ls / (ll1 + ll2 - ls)
    except:
        j = None

    if similarity:
        return (j)
    else:
        if j is not None:
            return (1 - j)
        else:
            return (j)


async def calculate_jaccard_index_async(shared, list1, list2,  index, similarity=True):
    """
    parallel version of calculate_jaccard_index()

    """

    ll1 = len(list1)
    ll2 = len(list2)
    ls = len(shared)

    try:
        j = ls / (ll1 + ll2 - ls)
    except:
        j = None

    if similarity:
        return (index, j)
    else:
        if j is not None:
            return (index, 1 - j)
        else:
            return (index, j)


async def calculate_jaccard_index_and_percentage_async(shared, list1, list2,  index, similarity=True, penalize=False, weight="length"):
    """
    parallel version of calculate_jaccard_index_and_percentage()

    """

    j = calculate_jaccard_index(shared, list1, list2, similarity=similarity)

    p = percentage_shared(shared, list1, list2, penalize=penalize, weight=weight)

    return (index, j, p)


def calculate_jaccard_index_and_percentage(shared, list1, list2, similarity=True, penalize=False, weight="length"):
    """
    returns percentage of shared edges and jaccard index

    based on calcualte_jaccard_index() & percentage_shared()
    refer to parent functions for parameters

    Output
        tuple of jaccard index & percentage value

    """

    j = calculate_jaccard_index(shared, list1, list2, similarity=similarity)

    p = percentage_shared(shared, list1, list2, penalize=penalize, weight=weight)

    return (j, p)


def shared_elements_multiple(lists, in_async=True, labels=None, percentage=False, jaccard=True, jaccard_similarity=True, penalize_percentage=False, weight_penalize="length", is_file=True):
    """
    mother function to estimate similarities between a list of networks
    
        estimates % of shared edges/nodes between all edge list pairs/ node lists and / or jaccard similarity/ index

    Input
        assumes lists is tuple or list containing all edge lists to be compared

        percentage, jaccard state which similarity values should be estimated

        for performance recompute edgelists into int lists with map_edge_to_id() or map_node_to_id() for node comparisons
            only works if weight = False

        in_async
            runs in parallel

        if is_file then lists is list of pickled network locations instead

    Output
        similarity matrix between all layers, that can then be plottet as a heatmap if desired

    

    """

    if in_async:

        # build result matrix
        if percentage and jaccard:
            result1 = np.zeros((len(lists), len(lists)))
            result2 = np.zeros((len(lists), len(lists)))
            # split into chunks
            tasks = []
            loop = asyncio.new_event_loop()

            computed = []
            for index, x in np.ndenumerate(result1):
                # print(index)
                if index not in computed:
                    # call shared percentage
                    if is_file:
                        with open(lists[index[0]], "rb") as f:
                            l1 = pickle.load(f)
                        with open(lists[index[1]], "rb") as f:
                            l2 = pickle.load(f)
                    else:
                        l1 = lists[index[0]]
                        l2 = lists[index[1]]
                    
                    re = tasks.append(loop.create_task(calculate_jaccard_index_and_percentage_async(shared_edges(l1, l2), l1, l2, index, similarity=jaccard_similarity,  penalize=penalize_percentage, weight=weight_penalize)))

                    computed.append(index)
                    computed.append((index[1], index[0]))
            loop.run_until_complete(asyncio.wait(tasks))

            loop.close()
            # print(tasks)
            # use tasks (returns index and result as tuple) to fill result matrix
            for r in tasks:

                index = r.result()[0]
                jaccard = r.result()[1]
                percentage = r.result()[2]

                result1[index[0]][index[1]] = jaccard
                result1[index[1]][index[0]] = jaccard

                result2[index[0]][index[1]] = percentage
                result2[index[1]][index[0]] = percentage

            return result1, result2

        else:
            result = np.zeros((len(lists), len(lists)))
            # split into chunks
            tasks = []
            loop = asyncio.new_event_loop()

            computed = []
            for index, x in np.ndenumerate(result):
                print(index)
                if index not in computed:
                    # call shared percentage
                    if is_file:
                        with open(lists[index[0]], "rb") as f:
                            l1 = pickle.load(f)
                        with open(lists[index[1]], "rb") as f:
                            l2 = pickle.load(f)
                    else:
                        l1 = lists[index[0]]
                        l2 = lists[index[1]]

                    if jaccard:
                        re = tasks.append(loop.create_task(calculate_jaccard_index_async(shared_edges(l1, l2), l1, l2,  index, similarity=jaccard_similarity)))

                    else:
                        re = tasks.append(loop.create_task(percentage_shared_async(shared_edges(l1, l2), l1, l2, index, penalize=penalize_percentage, weight=weight_penalize)))
                    #result[index[0]][index[1]] = j
                    #result[index[1]][index[0]] = j
                    computed.append(index)
                    computed.append((index[1], index[0]))
            loop.run_until_complete(asyncio.wait(tasks))

            loop.close()
            # print(tasks)
            # use tasks (returns index and result as tuple) to fill result matrix
            for r in tasks:
                index = r.result()[0]
                value = r.result()[1]

                result[index[0]][index[1]] = value
                result[index[1]][index[0]] = value

    else:

        if percentage and jaccard:
            result1 = np.zeros((len(lists), len(lists)))
            result2 = np.zeros((len(lists), len(lists)))

            computed = []
            for index, x in np.ndenumerate(result1):

                if is_file:
                        with open(lists[index[0]], "rb") as f:
                            l1 = pickle.load(f)
                        with open(lists[index[1]], "rb") as f:
                            l2 = pickle.load(f)
                else:
                        l1 = lists[index[0]]
                        l2 = lists[index[1]]
                # print(index)
                if index not in computed:
                    # call shared percentage

                    re = calculate_jaccard_index_and_percentage(shared_edges(l1, l2), l1, l2, similarity=jaccard_similarity,  penalize=penalize_percentage, weight=weight_penalize)

                    computed.append(index)
                    computed.append((index[1], index[0]))

                    jaccard = re[0]
                    percentage = re[1]

                    result1[index[0]][index[1]] = jaccard
                    result1[index[1]][index[0]] = jaccard

                    result2[index[0]][index[1]] = percentage
                    result2[index[1]][index[0]] = percentage

            return result1, result2

        else:
            result = np.zeros((len(lists), len(lists)))
            # split into chunks

            computed = []
            for index, x in np.ndenumerate(result):
                print(index)
                if index not in computed:
                    if is_file:
                        with open(lists[index[0]], "rb") as f:
                            l1 = pickle.load(f)
                        with open(lists[index[1]], "rb") as f:
                            l2 = pickle.load(f)
                    else:
                        l1 = lists[index[0]]
                        l2 = lists[index[1]]
                    # call shared percentage
                    if jaccard:
                        re = calculate_jaccard_index(shared_edges(l1, l2), l1, l2, similarity=jaccard_similarity)

                    else:
                        re = percentage_shared(shared_edges(l1, l2), l1, l2, penalize=penalize_percentage, weight=weight_penalize)
                    #result[index[0]][index[1]] = j
                    #result[index[1]][index[0]] = j
                    computed.append(index)
                    computed.append((index[1], index[0]))

                    value = re[0]

                    result[index[0]][index[1]] = value
                    result[index[1]][index[0]] = value

        return result


def shared_edges(list1, list2):
    """
    calculates number of shared edges between list1 and list2

    Input
        lists need to be reformatted to ids first as returned by construct_mapped_edges
    Output
        list of shared items


    """
    #shared = []

    # test out faster method based on intersection
    l1 = set(list1)
    l2 = set(list2)

    shared = l1.intersection(l2)

    return shared


def get_sorensen_coefficient(jaccard):
    """
    estimates sorensen coefficient based on jaccard

    Input
        jaccard is jaccard index matrix as returned by shared_edges_multiple()

    Output
        matrix
    """
    matrix = jaccard.copy()

    for index, j in np.ndenumerate(matrix):

        # call shared percentage
        s = (2*j) / (1+j)
        matrix[index[0]][index[1]] = s
        matrix[index[1]][index[0]] = s

    return matrix


def map_edge_to_id(edges, mapping={}, next_value=0):
    """
    to speed up computation each edge is mapped to an id (undirected)
    so that only int lists have to be compared

    Input
        edge list
        mapping: dict, allows to use same mapping for multiple networks
        provide next_value if mapping is provided to keep consistent

    Output
        mapping and next value
    """

    for edge in edges:
        e1 = str(edge[0]) + ","+str(edge[1])
        e2 = str(edge[1]) + ","+str(edge[0])
        val = None
        f_e1 = False
        f_e2 = False
        if e1 in mapping.keys():
            # get its value
            val = mapping[e1]
            f_e1 = True
        if e2 in mapping.keys():
            if val is None:
                # get value and add e1
                val = mapping[e2]
                mapping[e1] = val
            f_e2 = True
        if not f_e2 and f_e1:
            mapping[e2] = val
        if not f_e2 and not f_e1:
            mapping[e1] = next_value
            mapping[e2] = next_value
            next_value = next_value + 1

    return mapping, next_value


def construct_mapped_edge(mapping, edges):

    """
    helper function
    """
    new_edges = []

    for edge in edges:
        e1 = str(edge[0]) + ","+str(edge[1])
        # 2 = str(edge[1]) +","+str(edge[0])
        val = mapping[e1]
        new_edges.append(val)

    return new_edges


def compute_kendall_tau(list1, list2, usage="all", x=10):
    """
    scipy implemenation of kendalltau
        only works with lists of same size => use shared sublists or top bottom x

    Input
        provide ranked lists as returned by sort_edge_list

        usage set which (sub)lists should be used in case input lists are different sizes
        if usage all will take all lists but requires lists to be same size
        if usage top top x will be taken from both lists
        if usage bottom bottom x will be taken
        if usage shared shared edges in both lists will be compared

    Output
        returns value between 1 & -1 and p-value

    
    """
    if len(list1) != len(list2):
        print("input lists have different length")
        if usage != "all":
            if usage == "top":
                l1 = list1[:x]
                l2 = list2[:x]
            elif usage == "bottom":
                l1 = list1[-x:]
                l2 = list2[-x:]
            elif usage == "shared":

                '''
                for item in list1:
                    if item in list2 and item not in l1:
                        l1.append(item)
                for item in list2:
                    if item in list1 and item not in l2:
                        l2.append(item)
                '''
                l1 = set(list1).intersection(set(list2))
                l2 = l1

            else:
                print("non specified parameter, will default to top 10")
                l1 = list1[:10]
                l2 = list2[:10]

            tau, p = scipy.stats.kendalltau(l1, l2)

        else:
            print("please set usage to top, bottom or shared if lists are of different length will default to top 10")
            l1 = list1[:10]
            l2 = list2[:10]
            tau, p = scipy.stats.kendalltau(l1, l2)
    else:
        tau, p = scipy.stats.kendalltau(list1, list2)

    return tau, p


def sort_edge_list(edges, mapping):
    """
    sorts edge lists based on weight attribute and returns sorted list of edge ids

    Input
        edges needs to be edge list of sublists were each sublists is in following format
            [gene1, gene2, weight]
        mapping is edge id mapping as returned by map_edge_to_id()

    Output
        list of ranked endges
    """

    edge_weight_mapping = {}
    sorted_edges = []

    for edge in edges:

        e = (str(edge[0]) + ","+str(edge[1]))

        edge_weight_mapping[e] = edge[2]

    # sort dict by weight

    s = list(sorted(edge_weight_mapping.items(), key=operator.itemgetter(1), reverse=True))

    # print(s)

    # construct sorted edge list
    for i in s:
        # get edge id from mapping
        id = mapping[i[0]]
        sorted_edges.append(id)
    return sorted_edges


def compute_hamming(list1, list2):
    """
    estimates hamming distance
        based on scipy

    Input
        requires lists to be binary were 1 indicates edge is in layer and 0 not
            lists need to be in format as returned by compute_binary_layer() 
        or can be list of edge ids sorted based on weight as returned by sort_edge_list()
            which also ensures that all lists have the same lenght

    Output
        float
    """
    return scipy.spatial.distance.hamming(list1, list2)


def compute_edit_distance(list1, list2):
    """
    computes edit distance
        based on https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    Input
        requires lists to be binary were 1 indicates edge is in layer and 0 not
        lists need to be in format as returned by compute_binary_layer() 
        or can be list of edge ids sorted based on weight as returned by sort_edge_list()
            which also ensures that all lists have the same lenght
    Output
        float
    """

    if len(list1) > len(list2):
        list1, list2 = list2, list1

    distances = range(len(list1) + 1)
    for i2, c2 in enumerate(list2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(list1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def compute_simple_matching_coefficient(list1, list2):
    """
    computes smc
        counts matching and non matching positions

    Input
        requires lists to be binary were 1 indicates edge is in layer and 0 not
        lists need to be in format as returned by compute_binary_layer() 
            which also ensures that all lists have the same lenght

    Output
        float

    
    """
    match = 0
    no_match = 0

    for k in range(len(list1)):
        if list1[k] == list2[k]:
            match = match + 1
        else:
            no_match = no_match + 1

    smc = match / (match + no_match)
    return smc


def build_similarity_matrix_for_binary_and_ranked(lists, compute="kendall", kendall_usage="all", kendall_x=10):
    """
    build similarity matrix for distance measures that take ranked list or binary list

    Input
        function options are 
            compute_kendall_tau() : kendall
            compute_hamming() : hamming
            compute_edit_distance() : ed
            compute_simple_matching_coefficient()': smc

        lists is tuple of input lists
            if kendall need to be sorted as returned by sort_edge_list()
            if hamming, ed or smc lists need to be binary as returned by compute_binary_layer()

        refer to function for more details and parameter values


    Output
        tuple of similarity matrices, were for hamming second matrix contains the pvalues 
            for all others second matrix is empty
    """

    print("check if kendall x is small enough ")

    ml = len(min(lists, key=lambda i: len(i)))
    if ml < kendall_x:
        print("update x")
        kendall_x = int(ml - ml * 0.4)
        print("new x "+str(kendall_x))

    result = np.zeros((len(lists), len(lists)))
    result2 = np.zeros((len(lists), len(lists)))

    computed = []
    for index, x in np.ndenumerate(result):
        print(index)
        if index not in computed:

            # call function

            if compute == "kendall":
                # returns tau and p value

                tau, p = compute_kendall_tau(lists[index[0]], lists[index[1]], usage=kendall_usage, x=kendall_x)

                result2[index[0]][index[1]] = p
                result2[index[1]][index[0]] = p

                result[index[0]][index[1]] = tau
                result[index[1]][index[0]] = tau

            elif compute == "hamming":
                h = compute_hamming(lists[index[0]], lists[index[1]])

                result[index[0]][index[1]] = h
                result[index[1]][index[0]] = h

                result2 = []

            elif compute == "ed":
                ed = compute_edit_distance(lists[index[0]], lists[index[1]])

                result[index[0]][index[1]] = ed
                result[index[1]][index[0]] = ed

                result2 = []

            elif compute == "smc":
                smc = compute_simple_matching_coefficient(
                    lists[index[0]], lists[index[1]])

                result[index[0]][index[1]] = smc
                result[index[1]][index[0]] = smc

                result2 = []

            else:
                print("usage value not supported")

                result = []
                result2 = []

            computed.append(index)
            computed.append((index[1], index[0]))

    if compute == "kendall":

        return result, result2, kendall_x
    else:
        return result, result2


def compute_binary_layer(shared_edges, layers=None):
    """
    computes binary representation of edge_layer representation

    Input
        shared edges is dict as returned by compute_shared_layers()

        layers is list of str which need to be the same one as used in shared_edges

    Output
        list of sublists for each layer specified in layers
        edges are ordered the same way as provided in shared edges & can be match to id by index comparison
    
    
    """
    res = []
    if layers is not None:
        for layer in layers:
            # loop thorugh dict
            temp = []
            for key in shared_edges.keys():
                if layer in shared_edges[key]:
                    temp.append(1)
                else:
                    temp.append(0)
            res.append(temp)

    else:
        print("please provide layer names")

    return res


def to_distance(m):
    """
    takes similarity matrix and returns distance matrix (1- x)
    """
    matrix = m.copy()
    computed = []
    for index, x in np.ndenumerate(matrix):

        if index not in computed:
            # call shared percentage
            d = 1 - x
            matrix[index[0]][index[1]] = d
            matrix[index[1]][index[0]] = d
        computed.append(index)
        computed.append((index[1], index[0]))

    return matrix


def calculate_ged(graphs, edge_attribute=None, node_attribute=None, delete_node=1, delete_edge=1, add_node=1, add_edge=1):
    """
    estimates GED based on networkx and GMatch4py

    Input
        graphs has to be list of networkx graph objects that will be compared
            if GED should be estimated for edges only then G & H need to contain exactly the same node set
            else node differences are calculated as well

        if edge or node attributes are set they will be taken into account

        weights for grah operations can be set

    Output
        unnormalized distance matrix, similarity matrix and distance matrix
    """

    ged = gm.GraphEditDistance(delete_node, add_node, delete_edge, add_edge)
    ged.set_attr_graph_used(node_attribute, edge_attribute)
    result = ged.compare(graphs, None)

    return result, ged.similarity(result), ged.distance(result)


async def construct_dict(edges, label):
    """
    helper function of compute_shared_layers()
    """
    result = {}

    for edge in edges:
        if edge not in result.keys():
            result[edge] = [label]

    return result


def mergeDict(dict1, dict2):
    """
    helper function of compute_shared_layers() to merge two dicts into one dict
    """
    # Merge dictionaries and keep values of common keys in list
    res = {**dict1, **dict2}
    for key, value in res.items():
        if key in dict1 and key in dict2:
            res[key].append(dict1[key][0])

    return res


def compute_shared_layers(lists, labels, mapping=None, weight=False, is_file=False, in_async=True):
    """
    computes in how many and which layers/ networks each edge/nodes occures

    Input
        lists is tuple of edge lists which need to be encoded into numbers 
            with map_edge_to_id() and construct_mapped_edge() first

        labels is tuple of labels to be used / displayed for each layer in same order as lists

        if weight = True edge weight will be stored in dict as well

        mapping is mapping returned by map_edge_to_id() for all layers
            if provided gene ids will be replaced by gene names

        if is_file then lists is list of network locations

        if in_async run in async


    Output
        list of shared edgs/nodes

    """

    #shared_edges = {}
    if in_async:
        tasks = []
        loop = asyncio.new_event_loop()

        for edges, label in zip(lists, labels):

            if is_file:
                #read from file
                with open(edges, "rb") as f:
                    edges = pickle.load(f)
            # construct for each edge dict and later merge into final
            # so calculation can be parallelized
            print(label)

            r = tasks.append(loop.create_task(construct_dict(edges, label)))

        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()

    

        # merge all dicts into one

        # edges occuring in multiple layers are merged into list attributes

        for i in range(len(tasks)):
            r = tasks[i]
            if i == 0:
                s_edges = r.result()

            else:

                temp_dict = r.result()
                #print("shared edges", shared_edges)
                #print("temp", temp_dict)
                for key in temp_dict:
                    s_edges.setdefault(key, []).append(temp_dict[key][0])
                
                #shared_edges = mergeDict(cur, temp_dict)
                #print("shared after", s_edges)


    else:
        result = {}
        for edges, label in zip(lists, labels):

            if is_file:
                #read from file
                with open(edges, "rb") as f:
                    edges = pickle.load(f)
            
            print(label)
            
            for edge in edges:
                if edge not in result.keys():
                    result[edge] = [label]
                else:
                    result.setdefault(edge, []).append(label)

        s_edges = result

    return s_edges


async def get_shared_layers(index, edge_count):

    res = []

    current_edges = edge_count[index]

    for edge in edge_count.keys():
        combined = len(list(set(current_edges).intersection(edge_count[edge])))

        res.append(combined)

    return (index, res)


def build_share_matrix(d):

    """
    build gene matrix were value indicates in how many layers/networks this node occures
        
    Input
        output of compute_shared_layers()

    Output
        matrix
    """

    max_key = max(d.keys())

    print("build matrix of size " + str(max_key ** 2))

    #result = np.zeros((max_key, max_key), dtype=np.int8)
    result = np.arange(max_key+1).tolist()
    print("allocated")
    computed = []
    counter = 0

    tasks = []
    loop = asyncio.new_event_loop()

    for index in range(len(result)):
        counter = counter + 1
        if counter % 1000 == 0:
            print("computed " + str(counter) + "out of " + str(max_key))

            # index directly corresponds to edge ids in d

        # print(index)
        t = tasks.append(loop.create_task(get_shared_layers(index, d)))

    loop.run_until_complete(asyncio.wait(tasks))

    loop.close()

    # edges occuring in multiple layers are merged into list attributes
    print("merging")
    for r in tasks:

        row = r.result()[0]
        data = r.result()[1]

        result[row] = data

    return np.array(result)


def get_shared_edges(l1, l2, name=False, mapping=None):
    """
    calcualtes shared edges between two lists/ networks

    Input
        l1 and l2 have to be id lists as returned by construced_mapped_edge()

        if name = True edge node names instead of id is returned this requires mapping to be not none 
            mapping needs to be edge id mapping as returned by map_edge_to_id()

    Output
        list
    """

    if not name:
        return list(set(l1).intersection(l2))
    else:
        res = []
        if mapping is not None:

            for id in list(set(l1).intersection(l2)):
                res.append(return_edge_from_id(id, mapping))
        else:
            print("please provide mapping")

        return res


def return_edge_from_id(id, mapping):
    """
    Input
        id is edge id queried for

        mapping is edge id mapping as returned by map_edge_to_id()
    Output
        list
    """

    return [item for item, i in mapping.items() if i == id]


def return_layer_from_id(id, layers):
    """
    Input
        id is edge id queried for

        layers is edge id layer mapping as returned by compute_shared_layers()

    Output
        layer name/ network name
    """

    return layers[id]


def return_top_layers(x, layers, direction="large"):
    """
    Input
        x is number of returned entries

        layers is edge id layer mapping as returned by compute_shared_layers()

        direction is if smallest or highest should be returned
            either larger or small

    Output
        list
    """

    if direction == "large":
        s = sorted(layers, key=lambda k: len(layers[k]), reverse=True)
    else:
        s = sorted(layers, key=lambda k: len(layers[k]))

    result = []
    counter = 0
    for k in s:
        counter = counter + 1
        if counter < x+1:
            result.append((k, layers[k]))

    return result


def map_node_to_id(edges, mapping={}, next_value=0):
    """
    to speed up computation each node is mapped to an id (undirected)
    so that only int lists have to be compared

    Input
        weighted edge list of graph in list of lists format

        mapping is created mapping, needs to be provided if mapping needs to be consistent between mulitple networks
            next value also needs to be set as returned by map_node_to_id()

    Output
        mapping, next_value
    """

    for edge in edges:
        nodes = [edge[0], edge[1]]

        for node in nodes:

            if node not in mapping.keys():
                # get its value

                mapping[node] = next_value

                next_value = next_value + 1

    return mapping, next_value


def construct_mapped_node(mapping, edges):

    new_nodes = []
    for edge in edges:
        nodes = [edge[0], edge[1]]
        for node in nodes:

            val = mapping[node]
            new_nodes.append(val)

    return new_nodes


def convert_to_dict(l):
    """
    convert list l of format [(x,y)] to dict of form {x:y}
    """

    new_dict = {}

    for element in l:
        new_dict[element[0]] = element[1]

    return new_dict


def sort_dict(d):
    """
    sort dict d
    """
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}


def list_to_mapping(d, mapping, as_str=True):
    """
    helper function of sort_node_list()
    d is dict of sorted nodes & mapping is node id mapping
    """
    new_list = []
    for n in d.keys():
        if as_str:
            id = mapping[str(n)]
        else:
            id = mapping[n]
        new_list.append(id)

    return new_list


def sort_node_list(Graph, mapping, degree=False, degree_centrality=False, closeness_centrality=False, betweenness=False, k=None, xx=2, as_str=True):
    """
    sorts edge lists based on weight attribute and returns sorted list of nodes ids

    Input
        Graph must be networkx object

        mapping is edge id mapping as returned by map_node_to_id()

        as_str indicates if node names are strings or int

        sort after estimates importance value to be sorted after:
        if multiple values are set to true linear combination with equal weight between both is calculated 
        combined is calculated based on position value not weight value => no bias
            - degree
            - degree_centrality
            - closeness_centrality - no approximation available - very expensive for larger graphs
            - betweenness

        to speed up use approximation of betweeness
            k is approximation for betweenness if k is set non None, else all nodes are used for calculation
            k is percentage value of what subset of graph should be used for calculation - e.g. 50% k=0.5

    if k is not None x determines how often random sampling is performed and values are averaged in order to estimate more accurate approximation

    Output
        returns dict of lists in order of degree, degree_centrality, closeness_centrality, betweenness, combined
            if set to False empty list is returned at this position

    
    """
    cnt = 0
    # only used in case multiple are mapped
    x = []
    values_saved = {}
    if degree:
        # type list
        weighted_degree = list(Graph.degree())
        # convert to dict
        # print(weighted_degree)
        weighted_d = convert_to_dict(weighted_degree)
        # print(weighted_d)
        values_saved["degree"] = weighted_d
        sorted_d = sort_dict(weighted_d)
        # print(sorted_d)
        # print(mapping)
        # convert keys to list and map to ids from mapping
        s_d = list_to_mapping(sorted_d, mapping, as_str=as_str)
        x.append(s_d)
        cnt = cnt + 1
        print(cnt)
    else:
        s_d = []
    if degree_centrality:
        # normalized version of degree
        # type dict
        weighted_dc = nx.degree_centrality(Graph)
        values_saved["dc"] = weighted_dc
        sorted_dc = sort_dict(weighted_dc)
        # convert keys to list and map to ids from mapping
        s_dc = list_to_mapping(sorted_dc, mapping, as_str=as_str)
        x.append(s_dc)
        cnt = cnt + 1
        print(cnt)
    else:
        s_dc = []
    if closeness_centrality:
        # nx version very slow calculate based on adjacency matrix

        # order is based on G.nodes() => index of G.nodes() is row index of A
        id_G = list(Graph.nodes())
        A = nx.to_numpy_matrix(Graph)
        D = scipy.sparse.csgraph.floyd_warshall(
            A, directed=False, unweighted=False)
        n = D.shape[0]
        closeness_centrality = {}
        for r in range(0, n):

            cc = 0.0

            possible_paths = list(enumerate(D[r, :]))
            shortest_paths = dict(
                filter(lambda x: not x[1] == np.inf, possible_paths))

            total = sum(shortest_paths.values())
            n_shortest_paths = len(shortest_paths) - 1.0
            if total > 0.0 and n > 1:
                s = n_shortest_paths / (n - 1)
                cc = (n_shortest_paths / total) * s
            closeness_centrality[id_G[r]] = cc

        # average shortest path from v to all other nodes
        # type dict
        weighted_cc = closeness_centrality
        values_saved["cc"] = weighted_cc
        # print(weighted_cc)
        sorted_cc = sort_dict(weighted_cc)
        # convert keys to list and map to ids from mapping
        s_cc = list_to_mapping(sorted_cc, mapping, as_str=as_str)
        x.append(s_cc)
        cnt = cnt + 1
        print(cnt)
    else:
        s_cc = []
    if betweenness:
        # fraction of shortest paths that pass through v
        # type dict
        # if k is not None betweenness is calculated x times with new random nodes and all results are averaged to get best optimazion

        if k is not None:
            temp = []
            #print("nodes in graph " +str(len(list(Graph.nodes()))))
            kk = int(len(list(Graph.nodes())) * k)
            # print("k="+str(kk))
            for i in range(xx):
                weighted_b = nx.betweenness_centrality(
                    Graph, normalized=True, k=kk)
                temp.append(weighted_b)
                #print("len res in loop")
                # print(len(weighted_b))
            # average values
            # print(temp)
            df = pd.DataFrame(temp)
            weighted_b = dict(df.mean())
            
            #print("res mean")
            # print(len(weighted_b))

        else:
            weighted_b = nx.betweenness_centrality(Graph, normalized=True)

        values_saved["bet"] = weighted_b
        sorted_b = sort_dict(weighted_b)
        # convert keys to list and map to ids from mapping
        s_b = list_to_mapping(sorted_b, mapping, as_str=as_str)
        x.append(s_b)
        cnt = cnt + 1
        print(cnt)
    else:
        s_b = []

    if cnt > 1:
        print("average position is calculated")
        # get first itme that is non empty
        # all should contain same nodes so enough to loop through one list to get all nodes of this graph
        mean_position = {}
        median_position = {}

        for node in x[0]:
            # get all indices
            i = []
            for element in x:
                i.append(element.index(node))
            # average
            #i = i / len(x)
            # save
            #print("average", node, i)
            mean_position[node] = statistics.mean(i)
            median_position[node] = statistics.median(i)


        # convert to sorted list
        sorted_mean = list(sort_dict(mean_position).keys())
        sorted_median = list(sort_dict(median_position).keys())
        # print(sorted_average)
    else:
        sorted_mean = []
        sorted_median = []

    result = {"degree": s_d, "dc": s_dc, "cc": s_cc,"betweenness": s_b, "average_mean": sorted_mean, "average_median": sorted_median}

    if cnt < 1:
        print("no weights are selected please set at least one value to True")

    return result, values_saved
