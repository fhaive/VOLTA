"""
Functions to estimate between network simialrities based on nodes & edges.
"""

import networkx as nx
import numpy as np
import os
import ast
import sys
import matplotlib.pyplot as plt
import pandas as pd
#import gmatch4py as gm
#import seaborn as sns

import dask.bag as db
import asyncio
import scipy
import operator
import statistics
import pickle


def percentage_shared(shared, list1, list2, penalize=False, weight="length"):
    """
    Function to estimate the percenatge of shared edges/nodes between two networks.
    Calculates the percentage of shared edges based on the maximum possible shared edges = len of smaller list.

    Parameters:
        shraed (list): list of shared edges or nodes. Node IDs need to be consistent between the lists. Duplicate edges in a network are not allowed.
        list1 (list): list of edges or nodes in Network 1. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        list2 (list): list of edges or nodes in Network 2. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        penalize (boolean): if True then the score is penalized based on the difference in list lengths (difference in size of the two networks). 
                Percentage score p (of shared edges) is penalized based on p * (1 / weight * (abs(len(list1) - len(list2)))).
        weight (str or float): method used when penalize is True. If penalize = "length" then weight is 1. If penalize is float then weight can be custom set.
       
    Returns:
        shared percentage (float): percentage of shared edges/ nodes between the input graphs
    
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


async def __percentage_shared_async__(shared, list1, list2, index, penalize=False, weight="length"):

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
    Calculates jaccard index between two networks.
    

    Parameters:
        shraed (list): list of shared edges or nodes. Node IDs need to be consistent between the lists. Duplicate edges in a network are not allowed.
        list1 (list): list of edges or nodes in Network 1. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        list2 (list): list of edges or nodes in Network 2. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        similarity (boolean): if True returns jaccard similarity else returns jaccard distance.
    Returns:
        jaccard similarity/ distance (float): jaccard index or distance between the input lists
    
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


async def __calculate_jaccard_index_async__(shared, list1, list2,  index, similarity=True):
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


async def __calculate_jaccard_index_and_percentage_async__(shared, list1, list2,  index, similarity=True, penalize=False, weight="length"):
    """
    parallel version of __calculate_jaccard_index_and_percentage__()

    """

    j = calculate_jaccard_index(shared, list1, list2, similarity=similarity)

    p = percentage_shared(shared, list1, list2, penalize=penalize, weight=weight)

    return (index, j, p)


def __calculate_jaccard_index_and_percentage__(shared, list1, list2, similarity=True, penalize=False, weight="length"):
    """
    Wrapper function to return jaccard index and shared percentage.

    
    Parameters:
        shraed (list): list of shared edges or nodes. Node IDs need to be consistent between the lists. Duplicate edges in a network are not allowed.
        list1 (list): list of edges or nodes in Network 1. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        list2 (list): list of edges or nodes in Network 2. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        similarity (boolean): if True returns jaccard similarity else returns jaccard distance.
        penalize (boolean): if True then the score is penalized based on the difference in list lengths (difference in size of the two networks). 
                Percentage score p (of shared edges) is penalized based on p * (1 / weight * (abs(len(list1) - len(list2)))).
        weight (str or float): method used when penalize is True. If penalize = "length" then weight is 1. If penalize is float then weight can be custom set.
       
    Returns:
        jaccard index and percentage (tuple of floats):

    """

    j = calculate_jaccard_index(shared, list1, list2, similarity=similarity)

    p = percentage_shared(shared, list1, list2, penalize=penalize, weight=weight)

    return (j, p)


def shared_elements_multiple(lists, in_async=True, labels=None, percentage=False, jaccard=True, jaccard_similarity=True, penalize_percentage=False, weight_penalize="length", is_file=True):
    """
    Wrapper function to estimate similarities between a list of networks. 
    Estimates % of shared edges/nodes between all edge list pairs/ node lists and / or jaccard similarity/ index

    Parameters:
        lists (list): list of lists containing network edges or nodes to be compared.
        in_async (boolean): if True runs in asynchronous mode.
        labels (list or None): list of network labels in same order as lists.
        percentage (boolean): if True percentage value is calculated.
        jaccard (bolean): if True jaccard distance/similarity is calculated.
        jaccard_similarity (boolean): if True returns jaccard similarity else returns jaccard distance.
        penalize_percentage (boolean): if True then the score is penalized based on the difference in list lengths (difference in size of the two networks). 
                Percentage score p (of shared edges) is penalized based on p * (1 / weight * (abs(len(list1) - len(list2)))).
        weight_penalize (str or float): method used when penalize is True. If penalize = "length" then weight is 1. If penalize is float then weight can be custom set.
        is_file (boolean): if True then lists is list of pickled locations of data described in lists.
       
    Returns:
        similarity (numpy matrix or tuple of numpy matrices): similarity matrix between all networks. If one parameter is estimated returns one matrix only.
            If both parameters are estimated returns jaccard matrix, percentage matrix

    

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
                    
                    re = tasks.append(loop.create_task(__calculate_jaccard_index_and_percentage_async__(shared_edges(l1, l2), l1, l2, index, similarity=jaccard_similarity,  penalize=penalize_percentage, weight=weight_penalize)))

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
                        re = tasks.append(loop.create_task(__calculate_jaccard_index_async__(shared_edges(l1, l2), l1, l2,  index, similarity=jaccard_similarity)))

                    else:
                        re = tasks.append(loop.create_task(__percentage_shared_async__(shared_edges(l1, l2), l1, l2, index, penalize=penalize_percentage, weight=weight_penalize)))
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

                    re = __calculate_jaccard_index_and_percentage__(shared_edges(l1, l2), l1, l2, similarity=jaccard_similarity,  penalize=penalize_percentage, weight=weight_penalize)

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

                    value = re

                    result[index[0]][index[1]] = value
                    result[index[1]][index[0]] = value

        return result


def shared_edges(list1, list2):
    """
    Calculates shared edges/ nodes between two networks

    Parameters:
        list1 (list): list of edges or nodes in Network 1. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
        list2 (list): list of edges or nodes in Network 2. Node IDs need to be consistent between the networks (lists). Duplicate edges in a network are not allowed.
    Returns:
        shared items (list): list of items shared between input graphs


    """
    #shared = []

    # test out faster method based on intersection
    l1 = set(list1)
    l2 = set(list2)

    shared = l1.intersection(l2)

    return shared


def get_sorensen_coefficient(jaccard):
    """
    Estimates sorensen coefficient based on jaccard index

    Parameters:
        jaccard (numpy matrix): numpy matrix filled with jaccard indices as returned by shared_edges_multiple()

    Returns:
        sorensens (numpy matrix): sorensen coefficient estimated from jaccard index
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
    Maps undirected edges to an ID.

    Parameters:
        edges (list): edge list
        mapping (dict): either empty dict can be provided or already populated one, when the same mapping should be used for multiple networks.
        next_value (int): if mapping is provided, then the last id set needs to be provided.

    Returns:
        mapping (dict): key is edge and value is assigned ID
        next value (int): can be used if multiple networks are mapped to the same IDs
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

def reverse_node_edge_mapping(mapping):
    """
    Reverses the mapping object. Switches keys and values of the mapping object. This allows easier retrival of original node IDs from the mapping ID.

    Parameters:
        mapping (dict): node/ edge to ID mapping as returned by map_node_to_id()/ map_edge_to_id().
    
    Returns:
        reversed mapping (dict): keys are mapped IDs and values are original node/ edge IDs.
    """

    return {value:key for key, value in mapping.items()}

def construct_mapped_edge(mapping, edges):

    """
    Creates list from assigned edge IDs. Can be used to reverse the mapping.

    Parameters:
        mapping (dict): edge to ID mapping as returned by map_edge_to_id()
        edges (list): list of edges
    Returns:
        ids (list): list of assigned IDs in same order as edges
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
    Computes Kendall Rank Correlation between two ranked lists (based on SciPy). This can be edges ranked by weight or other attributed or nodes ranked by some attributes, such as degree.
    Input lists need to have same length or usage option other than "all" needs to be selected.

    Parameters:
        list1 (list): list of ranked edge or node IDs as returned by sort_edge_list().
        list2 (list): list of ranked edge or node IDs as returned by sort_edge_list().
        usage (str): sets what parto of the two lists should be compared.
            If "all" then all items are considered but list1 and list2 need to be of same length.
            If "top" then the top x items will be considered.
            If "bottom" then the last x items will be considered.
            If "shared" then only items occuring in list1 and list2 will be considered.
        x (int): needs to be set when usage = top or bottom. Indicates how many items are considered in the rank correaltion.

    Returns:
        kendall tau (float): kendell tau coefficient between the input lists
        p value (float): p-value of the correlation

    
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
    Sorts edge lists based on weight attribute and returns sorted list of edge ids.

    Parameters:
        edges (list): list of sublists in following format  [gene1, gene2, weight]
        mapping (dict): edge to ID mapping as returned by map_edge_to_id()

    Returns:
        ranked edges (list): list of sorted edge IDs
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
    Computes hamming distance (based on SciPy)

    Parameters:
        list1 (list): list of binary values as returned by compute_binary_layer(). 1 indicating if edge/node is present in that network and 0 if not. 
        list2 (list): list of binary values as returned by compute_binary_layer(). 1 indicating if edge/node is present in that network and 0 if not. 
    Returns:
        hamming distance (float): hamming distance between the input lists
    """
    return scipy.spatial.distance.hamming(list1, list2)


def compute_edit_distance(list1, list2):
    """
    Computes edit distance (based on https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python)

    Parameters:
        list1 (list): list of binary values as returned by compute_binary_layer(). 1 indicating if edge/node is present in that network and 0 if not. 
        list2 (list): list of binary values as returned by compute_binary_layer(). 1 indicating if edge/node is present in that network and 0 if not. 
    Returns:
        edit distance (float): eidt distance between the input lists
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
    Computes SMC.

    Parameters:
        list1 (list): list of binary values as returned by compute_binary_layer(). 1 indicating if edge/node is present in that network and 0 if not. 
        list2 (list): list of binary values as returned by compute_binary_layer(). 1 indicating if edge/node is present in that network and 0 if not. 
    Returns:
        SMC (float): SMC coefficient between input lists

    
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
    Wrapper function to build a similarity/ distance matrix from different distance measures.
    
    Parameters:

        lists (list): list of lists, each sublist representing a network. If compute = "Kendall" sublists need to be ranked lists 
            else sublists need to be binary lists. 1 indicating if edge/node is present in that network and 0 if not.
        compute (str):
            "kendall" computes Kendall Rank Coefficient based on compute_kendall_tau()
            "hamming" computes hamming distance based on compute_hamming()
            "ed" computes edit distance based on compute_edit_distance()
            "smc" computes SMC based on compute_simple_matching_coefficient()
        kendall_usage (str): sets what parto of the two lists should be compared.
            If "all" then all items are considered but list1 and list2 need to be of same length.
            If "top" then the top x items will be considered.
            If "bottom" then the last x items will be considered.
            If "shared" then only items occuring in list1 and list2 will be considered.
        kendall_x (int): needs to be set when usage = top or bottom. Indicates how many items are considered in the rank correaltion. If x is larger than one of the lists it will be set automtically to a quarter less than the shortest list.

    Returns:
        similairty matrix (numpy matrix): matrix containing the kendall correlation values for each input pair
        p value matrix (numpy matrix): this matrix only contains values if  compute = hamming or compute = kendall
        used x (int): if kendall then a third value is returned indicating what portion of the lists have been used to estimate the rank correlation value
        
    """

    print("check if kendall x is small enough ")

    ml = len(min(lists, key=lambda i: len(i)))

    if compute == "kendall":
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
    Computes binary representation of edges/nodes.

    Parameters:
        shared_edges (dict): as returned by compute_shared_layers()
        layers (list): list containing network names. Needs to be the same as used in shared_edges()

    Returns:
        binary edges (list): list of sublists containing binary edge/ node representation for each layer specified in layers. 
                Edges/ Nodes are ordered the same way as provided in shared_edges.
    
    
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
    Converts a similarity matrix to a distance matrix (1- x).

    Parameters:
        m (numpy matrix): similarity matrix
    Returns:
        distance matrix (numpy matrix): converted distance matrix
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





async def __construct_dict__(edges, label):
    """
    helper function of compute_shared_layers()
    """
    result = {}

    for edge in edges:
        if edge not in result.keys():
            result[edge] = [label]

    return result


def __mergeDict__(dict1, dict2):
    """
    helper function of compute_shared_layers() to merge two dicts into one dict
    """
    # Merge dictionaries and keep values of common keys in list
    res = {**dict1, **dict2}
    for key, value in res.items():
        if key in dict1 and key in dict2:
            res[key].append(dict1[key][0])

    return res


def compute_shared_layers(lists, labels, is_file=False, in_async=True):
    """
    Computes in how many and in which layers/ networks each edge/node occures.

    Parameters:
        lists (list): list of edge/node IDs as encoded by map_edge_to_id()/ map_node_to_id() and construct_mapped_edge().
        labels (list): list of network labels. Needs to be in the same order as lists.
        is_file (boolean): if True then lists needs to contain the pickled locations to the node/ edge lists instead.
        in_async (boolean): if True then run in async

    Returns:
        shared edges/nodes (dict): key is edge/ node ID and value is label ID

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

            r = tasks.append(loop.create_task(__construct_dict__(edges, label)))

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
                
                #shared_edges = __mergeDict__(cur, temp_dict)
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


async def __get_shared_layers__(index, edge_count):

    res = []

    current_edges = edge_count[index]

    for edge in edge_count.keys():
        combined = len(list(set(current_edges).intersection(edge_count[edge])))

        res.append(combined)

    return (index, res)


def build_share_matrix(d):

    """
    Converts shared layer information into a matrix.
        
    Parameters:
        d (dict): output of compute_shared_layers()

    Returns:
        shared items (numpy matrix): matrix
    """

    max_key = max(d.keys())

    print("build matrix of size " + str(max_key ** 2))

    #result = np.zeros((max_key, max_key), dtype=np.int8)
    result = np.arange(max_key+1).tolist()
    print("allocated")
    
    counter = 0

    tasks = []
    loop = asyncio.new_event_loop()

    for index in range(len(result)):
        counter = counter + 1
        if counter % 1000 == 0:
            print("computed " + str(counter) + "out of " + str(max_key))

            # index directly corresponds to edge ids in d

        # print(index)
        t = tasks.append(loop.create_task(__get_shared_layers__(index, d)))

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
    Calcualtes shared edges/ nodes between two lists/ networks.

    Paramters:
        l1 (list): list of node/edge ids as returned by construced_mapped_edge().
        l2 (list): list of node/edge ids as returned by construced_mapped_edge().
        name (boolean): if True ID values are replaced by original node/edge names. If Ture mapping needs to be not None.
        mapping (dict or None): edge /node to ID mapping as returned by map_edge_to_id() / map_node_to_id().

    Returns:
        shared items (list): list of shared items between input lists
    """

    if not name:
        return list(set(l1).intersection(l2))
    else:
        res = []
        if mapping is not None:

            for id in list(set(l1).intersection(l2)):
                res.append(__return_edge_from_id__(id, mapping))
        else:
            print("please provide mapping")

        return res


def __return_edge_from_id__(id, mapping):
    """
    Converts mapped node/edge back to original name.

    Parameters:
        id is edge id queried for

        mapping is edge id mapping as returned by map_edge_to_id()/ map_node_to_id()
    Output
        list
    """

    return [item for item, i in mapping.items() if i == id]


def return_layer_from_id(id, layers):
    """
    Fetches layers specific node/edge ID is present in.

    Parameters:
        id (int): edge/ node ID queried for as set in map_edge_to_id()/ map_node_to_id()
        layers (dict): as returned by compute_shared_layers()

    Returns:
        layer name/ network name (list): as queried for
    """

    return layers[id]


def return_top_layers(x, layers, direction="large"):
    """
    Fetches nodes/ edges that are present in the most / least networks.

    Parameters:
        x (int): number of to be returned entries
        layers (dict): as returned by compute_shared_layers()
        direction (str): if direction = "large", the x items occuring in the most networks are returned. if direction = "small" the x items occuring in the least networks are returned.

    Returns:
        items (list): list of layer IDs as queried for
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
    Maps nodes to an ID.

    Parameters:
        edges (list): edge list
        mapping (dict): either empty dict can be provided or already populated one, when the same mapping should be used for multiple networks.
        next_value (int): if mapping is provided, then the last id set needs to be provided.

    Returns:
        mapping (dict): key is node ID and value is mapped ID
        next value (int): can be used if multiple networks are mapped to the same IDs
    """

    for edge in edges:
        nodes = [edge[0], edge[1]]

        for node in nodes:

            if node not in mapping.keys():
                # get its value

                mapping[node] = next_value

                next_value = next_value + 1

    return mapping, next_value


def __construct_mapped_node__(mapping, edges):

    new_nodes = []
    for edge in edges:
        nodes = [edge[0], edge[1]]
        for node in nodes:

            val = mapping[node]
            new_nodes.append(val)

    return new_nodes


def convert_to_dict(l):
    """
    Convert list l of format [(x,y)] to dict of form {x:y}.

    Parameters:
        l (list): input list of tuples
    
    Returns:
        convertion (dict): where key is first tuple entry and value is second tuple entry
    """

    new_dict = {}

    for element in l:
        new_dict[element[0]] = element[1]

    return new_dict


def __sort_dict__(d, reverse=True):
    """
    sorts dict d by value.

    Parameters:
        d (dict):
    
    Returns:
        sorted (dict):
    """
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}


def __list_to_mapping__(d, mapping, as_str=True):
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


def sort_node_list(Graph, mapping, degree=False, degree_centrality=False, closeness_centrality=False, betweenness=False, weight=None, k=None, xx=2, as_str=True):
    """
    Sorts node lists based on weight attribute and returns sorted list of nodes ids.

    Parameters:
        Graph (networkX graph object): graph to estimate on.
        mapping (dict): is id mapping as returned by map_node_to_id()
        as_str (boolean): if True node IDs are assumed to be string else int.
        degree (boolean): if True nodes are sorted after degree.
        degree_centrality (boolean): if True nodes are sorted after degree centrality.
        closeness_centrality (boolean): if True nodes are sorted after closeness centrality.
        betweenness (boolean): if True nodes are sorted after betweenness.
        If multiple values are set to True a combined ranking is calculated
        weight (str or None): for weighted networks name of edge attribute to be used. If None all edges are considered equal.
            Instead of node degree strength of degree will be calculated if not None, betweenness centrality will be calculated based on
            weighted edges as well as closeness centrality (weight is distance). Has no impact on degree centrality.
        k (float [0,1] or None): approximation of betweenness, if float then k percentage of nodes are used to approximate the betweenness values. If None all nodes are used.
        xx (int): state how often random sampling is performed if k is not None
        as_str (boolean): if True keys in mapping are assumed to be str (are the same as graph node IDs). If False they are assumed to be int.
        
    Returns:
        sorted (dict): keys are degree, dc, cc, betweenness, average_mean and average_median, values are list of ranked node ids (index of list is rank and value is Node ID as provided in mapping). If key is set to False an empty list is returned.
        
    
    """
    cnt = 0
    # only used in case multiple are mapped
    x = []
    values_saved = {}
    if degree:
        # type list
        weighted_degree = list(Graph.degree(weight=weight))
        # convert to dict
        # print(weighted_degree)
        weighted_d = convert_to_dict(weighted_degree)
        # print(weighted_d)
        values_saved["degree"] = weighted_d
        sorted_d = __sort_dict__(weighted_d)
        # print(sorted_d)
        # print(mapping)
        # convert keys to list and map to ids from mapping
        s_d = __list_to_mapping__(sorted_d, mapping, as_str=as_str)
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
        sorted_dc = __sort_dict__(weighted_dc)
        # convert keys to list and map to ids from mapping
        s_dc = __list_to_mapping__(sorted_dc, mapping, as_str=as_str)
        x.append(s_dc)
        cnt = cnt + 1
        print(cnt)
    else:
        s_dc = []
    if closeness_centrality:
        # nx version very slow calculate based on adjacency matrix

        # order is based on G.nodes() => index of G.nodes() is row index of A
        id_G = list(Graph.nodes())
        A = nx.to_numpy_matrix(Graph, weight=weight)
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
        sorted_cc = __sort_dict__(weighted_cc)
        # convert keys to list and map to ids from mapping
        s_cc = __list_to_mapping__(sorted_cc, mapping, as_str=as_str)
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
                    Graph, normalized=True, k=kk, weight=weight)
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
        sorted_b = __sort_dict__(weighted_b)
        # convert keys to list and map to ids from mapping
        s_b = __list_to_mapping__(sorted_b, mapping, as_str=as_str)
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
        sorted_mean = list(__sort_dict__(mean_position, reverse=False).keys())
        sorted_median = list(__sort_dict__(median_position, reverse=False).keys())
        # print(sorted_average)
    else:
        sorted_mean = []
        sorted_median = []

    result = {"degree": s_d, "dc": s_dc, "cc": s_cc,"betweenness": s_b, "average_mean": sorted_mean, "average_median": sorted_median}

    if cnt < 1:
        print("no weights are selected please set at least one value to True")

    return result, values_saved
