"""
contains functions to create single and flat multilayer networks build on networkx

as well as functions to estimate missing edges in order to construct complete networks
"""

import networkx as nx
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
import statistics
import dask.bag as db
import pickle
import asyncio
import sys
#sys.path.insert(1, '../supporting/')
#import pickle4reducer
import multiprocessing as mp



def construct_single_layer_network(edge_list, isolated_nodes=None):
    """
    builds weighted networkx graph out of edge list. 
    duplicate nodes/ edges are not added, your list can containe duplicate nodes & edges (last given value will overwrite all previous ones)

    Input
        edge lists needs to be list of lists were each sublists is in format [node 1, node 2, weight, distance]
            distance value is optional
        isolated_nodes (optional): list of node ids
            if isolated_nodes is given nodes without edges are added to graph, as provided in isolated_nodes

    Output
        return networkx graph object
    
    """

    G = nx.Graph()
    #added = []

    for edge in edge_list:

        # add node
        #print("edge", edge)
        G.add_node(edge[0], label=edge[0])
        # added.append(edge[0])

        G.add_node(edge[1], label=edge[1])
        # added.append(edge[1])

        # add edge
        if len(edge) > 3:
            G.add_edge(edge[0], edge[1], weight=edge[2], distance=edge[3], weight_str=str(edge[2]), distance_str=str(edge[3]))
        else:

            G.add_edge(edge[0], edge[1], weight=edge[2])

    if isolated_nodes is not None:
        for node in isolated_nodes:
            G.add_node(node, label=node)

    return G



def build_complete_network(H, take_weight=None, edge_attribute="weight", method="dijkstra", in_async=True, in_multi=False, max_step = 10, max_weight=1, penalize_automatic=False, nr_chunks=10, nr_processes=10, infer=True, manual_distance=1, manual_sim=0):
    """
    takes a graph and estimates weight of all missing edges based on shortest path & normalizes edges

    if network contains isolated nodes they are removed

    Input
        H, networkx object: assumes weights to be distances
        
        take weight (optional) states if shortest path is based on weight of edge or not
            if none all edges are treated equal and only steps are counted 
            else take_weight needs to be str of name of edge weight

        edge_attribute
            str, name of graph edge weight attribute which is used to estimate new edge weights based on shortest paths
        
        method (optional) method how shortest path is calculated, options are
            dijkstra or "bellman-ford"

        

        in async (optional)
            if code is run asyncronous
        
        in_multi (optional)
            if code is run as multiprocess

            if true
                nr_chunks & nr_processes set as how many tasks & on how many cores code is run

        if infer
            if not penalize_automatic: if path does not exist between 2 nodes distance is set to max_step*max weight
            if penalize_automatic = True it is set max_step*longest_shortest_path_in_graph*max_weight
        else value is set after normalization to manual_distance/ manual_sim
        
    Output
        returns three networkx graphs
            graph containing distance & similarity edge weights
            distance graph
            similarity graph
        
    """

    G = H.copy()
    
    
    isolates = list(nx.isolates(G))
    if len(isolates) > 0:
        print("Graph has isolate nodes, removing nodes ", len(isolates))
        
        G.remove_nodes_from(isolates)
    
    
    #get missing edges
    non_edges = list(nx.non_edges(G))
    
    print("graph edges missing ", len(non_edges))
    
    #get shortest path between these nodes to estimate new edge_weight
    

    if in_multi:
        print("running on multiple cores")
        #split non edges into chunks and run on multiple cores
        print("call get chunks")
        #chunks = list(get_chunks(non_edges,nr_chunks))
        chunks = list(np.array_split(np.array(non_edges), nr_chunks))



        print("splitting into chunks successfull start multiprocesses")

        
        
        func = partial(estimate_shortest_path, G, method=method, edge_attribute=edge_attribute, in_async=in_async, max_step = max_step, max_weight=max_weight, penalize_automatic=penalize_automatic, infer=infer)

        pool = Pool(processes=nr_processes)
    
        result = pool.map(func, chunks)

        print("missing edges estimated terminate multiprocesses")
        pool.terminate()
        pool.join()

        #combine result into new_edges
        print("merging result ")
        temp_edges = []
        temp_non_path = []
        max_path = []

        for r in result:
            if len(r["edges"]) > 0:
                temp_edges.append(r["edges"])
            if len(r["no_path"]) > 0:
                temp_non_path.append(r["no_path"])
                max_path.append(r["longest_path"])


        new_edges = [j for i in temp_edges for j in i]

        if len(temp_non_path) > 0:
            no_edges_temp = [j for i in temp_non_path for j in i]
            pen_weight = max(max_path)
            print("longest shortest path is of length ", pen_weight)
            if infer:
                if penalize_automatic:
                    print("non path edges are set to distance ", float(max_step*max_weight*pen_weight))
                else:
                    print("non path edges are set to distance ", float(max_step*max_weight))
                for i in no_edges_temp:
                    new_edges.append([i[0], i[1], float(max_step*max_weight*pen_weight)])

            else:
                print("non path edges are set to distance ", 1)
                for i in no_edges_temp:
                    new_edges.append([i[0], i[1], 1])

    
    else:

        print("running on one core")
        result = estimate_shortest_path(G, non_edges, edge_attribute=edge_attribute, method=method, in_async=in_async, max_step = max_step, max_weight=max_weight, penalize_automatic=penalize_automatic, infer=infer)
        new_edges = result["edges"]
        no_edges_temp = result["no_path"]
        pen_weight = result["longest_path"]
        
    
        print("longest shortest path is of length ", pen_weight)
        if infer:
            if penalize_automatic:
                print("non path edges are set to distance ", float(max_step*max_weight*pen_weight))
            else:
                print("non path edges are set to distance ", float(max_step*max_weight))
            for i in no_edges_temp:
                new_edges.append([i[0], i[1], float(max_step*max_weight*pen_weight)])

        else:
            for i in no_edges_temp:
                new_edges.append([i[0], i[1], float(max_step*max_weight*pen_weight)])

    if infer:
        print("adding new edges to Graph")

        for item in new_edges:
            if len(item) >= 3:
                G.add_edge(item[0], item[1])
                G[item[0]][item[1]][edge_attribute] = item[2]
            else:
                print("wrong length skip", item)


    print("normalize edges and transform into edge list")
    
    temp_edge_list = []
    
    #convert to edge list to be able to normalize & calculate similarity values
    temp_edges = nx.get_edge_attributes(G, name=edge_attribute)
    
    for key in temp_edges.keys():
        
        temp_edge_list.append([key[0], key[1], temp_edges[key]])
        
        
    #normalize 
    
    distance_edges, similarity_edges, combined_edges = create_normalized_edge_list_and_similarity_list(temp_edge_list) 
        
    C =  construct_single_layer_network(combined_edges)

    D =  construct_single_layer_network(distance_edges)

    S =  construct_single_layer_network(similarity_edges)

    if not infer:
        print("adding new edges to Graphs with distance/ sim", manual_distance, manual_sim)

        for item in new_edges:
            if len(item) >= 3:
                D.add_edge(item[0], item[1])
                D[item[0]][item[1]][edge_attribute] = manual_distance
                S.add_edge(item[0], item[1])
                S[item[0]][item[1]][edge_attribute] =  manual_sim
                C.add_edge(item[0], item[1],  weight=manual_sim, distance = manual_distance, weight_str=str(manual_sim), distance_str=str(manual_distance))
            else:
                print("wrong length skip", item)

    return C,D,S 


def estimate_shortest_path(G, non_edges, edge_attribute="weight", take_weight=None, method="dijkstra", max_step = 10, max_weight=1, penalize_automatic=False, in_async=True, infer=True):

    """
    child function of build_complete_network
        s. build_complete_network() for parameter explanation

    estimates shortest path for all edges stored in non_edges, which is list of non existing edges in G

    Input
        refer to build_complete_network()

    Output:
        dict
    """

    print("estimating shoretst path")
    try:
        non_edges = non_edges.tolist()
    except:
        non_edges=non_edges
    
    new_edges = []

    temp_edges = [] #used to save edges were no path exists to set to max

    longest_path = 1

    #run in parallel

    if in_async:
        print("start async")
        tasks =  []
        loop = asyncio.new_event_loop()

        for i in range(len(non_edges)):          
        
            
            edge=non_edges[i]

            r = tasks.append(loop.create_task(calc_estimate_shortest_path(G, edge, edge_attribute=edge_attribute, take_weight=take_weight, method=method, max_step = max_step, max_weight=max_weight, penalize_automatic=penalize_automatic, infer=infer)))

        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()

        
        #merge all dicts into one 
        
        #edges occuring in multiple layers are merged into list attributes
        print("async finished merging results")
        path_lengths = []
        for r in tasks:
        
            new_edges.append(r.result()[0])
            if len(r.result()[1]) > 0:
                temp_edges.append(r.result()[1])
            path_lengths.append(r.result()[2])

        longest_path = max(path_lengths)

    else:
        for i in range(len(non_edges)):
            
        
            
            edge=non_edges[i]
            
            #print(edge)
            
            if nx.has_path(G, edge[0], edge[1]):
                #if shortest path cannot be found weight is set to max_step * max_weight
                shortest_path = nx.shortest_path(G, source=edge[0], target=edge[1], weight=take_weight, method=method)

                #estimate new edge weight by adding up weights of paths
                #returns list of nodes
                previous_node = None
                target_node = edge[1]
                start_node = edge[0]

                if len(shortest_path) > longest_path:
                    longest_path = len(shortest_path)

                value = 0

                #print("shortest_path ", shortest_path)


                for i in range(len(shortest_path)):

                    current_node = shortest_path[i]

                    if current_node == start_node:
                        previous_node = current_node

                    if current_node == target_node:
                        #reached end

                        weight = G[previous_node][current_node][edge_attribute]
                        #print("last weight", weight)
                        #update new weight
                        value = value + weight

                        #print("edge completed ", value)
                        new_edges.append([start_node, target_node, value])

                    else:
                        #estimate path value
                        if previous_node != current_node:
                            #get edge weight
                            weight = G[previous_node][current_node][edge_attribute]
                            #print("weight", weight)
                            #update new weight
                            value = value + weight

                            previous_node=current_node
                        #else:
                        #    print("start node start calculation")

            else:
                #print("no path between nodes weight is set to max ", edge[0], edge[1], float(max_step*max_weight))
                if infer:
                    if not penalize_automatic:
                        new_edges.append([edge[0], edge[1], float(max_step*max_weight)])
                    else:
                        temp_edges.append([edge[0], edge[1]])

                else:
                    temp_edges.append([edge[0], edge[1]])

    
        



    return {"edges": new_edges, "no_path": temp_edges, "longest_path":longest_path}

async def calc_estimate_shortest_path(G, edge, edge_attribute="weight", take_weight=None, method="dijkstra", max_step = 10, max_weight=1, penalize_automatic=False, infer=True):

    """
    async version to estimate shortest path between two nodes
    helper function of build_complete_network() & estimate_shortest_path()

    Input
        refer to build_complete_network() & estimate_shortest_path()

    Output
        returns tuple of new edges 
    """


    new_edge = []
    temp_edge = []
    path_length = 0


    if nx.has_path(G, edge[0], edge[1]):
            #if shortest path cannot be found weight is set to max_step * max_weight
            shortest_path = nx.shortest_path(G, source=edge[0], target=edge[1], weight=take_weight, method=method)

            #estimate new edge weight by adding up weights of paths
            #returns list of nodes
            previous_node = None
            target_node = edge[1]
            start_node = edge[0]

            path_length = len(shortest_path) 

            value = 0

            #print("shortest_path ", shortest_path)


            for i in range(len(shortest_path)):

                current_node = shortest_path[i]

                if current_node == start_node:
                    previous_node = current_node

                if current_node == target_node:
                    #reached end

                    weight = G[previous_node][current_node][edge_attribute]
                    #print("last weight", weight)
                    #update new weight
                    value = value + weight

                    #print("edge completed ", value)
                    new_edge = [start_node, target_node, value]

                else:
                    #estimate path value
                    if previous_node != current_node:
                        #get edge weight
                        weight = G[previous_node][current_node][edge_attribute]
                        #print("weight", weight)
                        #update new weight
                        value = value + weight

                        previous_node=current_node
                    #else:
                    #    print("start node start calculation")

    else:
        #print("no path between nodes weight is set to max ", edge[0], edge[1], float(max_step*max_weight))
        if infer:
            if not penalize_automatic:
                new_edge = [edge[0], edge[1], float(max_step*max_weight)]
            else:
                temp_edge = [edge[0], edge[1]]

        else:
            temp_edge = [edge[0], edge[1]]


    return (new_edge, temp_edge, path_length)


def create_normalized_edge_list_and_similarity_list(edge_list, normalized=True, distance=True, combined=True):
    """
    helper function of build_complete_network() to normalize edge scores
    
    assumes scores in edge list are distance values (if similarity values take into account that returned distance and similarity results are switched)

    Input
        edge list as list of sublists where each sublist is in format [node1, node2, weight]

    Output
        returns 3 lists of sublists
            normalized distance edge list
            normalized similarity edge list
            combined edge list
        in format as edge_list where each sublists is [node 1, node 2, distance] or [node 1, node 2, similarity] or [node 1, node 2, distance, similarity]

    
    """
    # print(edge_list)

    gene1 = [item[0] for item in edge_list]
    gene2 = [item[1] for item in edge_list]
    score = [item[2] for item in edge_list]

    # normalize to be between 0 & 1

    print("normalizing score of length ", len(score))

    normalized = normalize(score)

    print("update edges ", len(gene1))

    normalized_edge_list = []
    similarity_edge_list = []
    combined_edge_list = []

    for k in range(len(gene1)):
        if k % 10000 == 0:
            print("normalizing " + str(k))

        similarity = 1 - float(normalized[k])
        if normalized:
            normalized_edge_list.append([gene1[k], gene2[k], normalized[k]])
        else:
            normalized_edge_list = None
        if distance:
            similarity_edge_list.append([gene1[k], gene2[k], similarity])
        else:
            similarity_edge_list = None
        if combined:
            combined_edge_list.append(
                [gene1[k], gene2[k], normalized[k], similarity])
        else:
            combined_edge_list = None

    return normalized_edge_list, similarity_edge_list, combined_edge_list


def normalize(a, offset=0.0001):
    """
    normalization function

    Input
        a is list of floats

        offset offset to be added to values to avoid 1 & 0 values after normalization

    Output
        list of floats in same order as a

    
    """

    # add small offset to not fully reach value 1
    min_val = min(a) - offset

    max_val = max(a) + offset

    norm = [(val-min_val) / (max_val-min_val) for val in a]

    return norm


def construct_multilayer_network_matrix(layers, weights):
    """
    function to create a flat multilayer network, while propagating each layers edge weight
        merges layers based on adjacency matrices
        matrices need to have same dimensions for all layers and need to be ordered the same way

    Input
        layers
            list of adjacency matrices, one for each layer
            numpy matrices

            use reformat_G_to_adj_matrix to convert graph object into adjacency matrix
        weights
            list of float
            weights are used to scale layers, need to be in same order as layers

            weights can be set manually or estimated based on functions provided in scaling.py
                this allows to individually weight and estimate the "importance"/ "quality" of individual layers

    
    Output
        two adjacency matrices, with normalized and unnormalized edge weights
    """

    if len(layers) != len(weights):
        print("layers and weights have different dimensions")
        return None, None
    else:
        # multiply matrices with scaling factor
        unnormalized = np.zeros((layers[0].shape))
        for i in range(len(layers)):
            unnormalized = unnormalized + weights[i] * layers[i]

        print("creating normalized")
        normalized = (unnormalized - np.min(unnormalized)) / \
            (((np.max(unnormalized)+0.0001)-np.min(unnormalized)))

        return unnormalized, normalized


def reformat_G_to_adj_matrix(G, gene_list=None):
    """
    converts a graph object into an adjacency matrix
    based on networkx 

    Input
        G
            networkx graph object

        gene_list (optional)
            in order to keep all layers consistent provide gene_list, which is list of node ids for all layers
                need to be same list for all layers!! even if it contains missing nodes

            if None then ouput of G.nodes() will be used
    Output
        numpy matrix
    
    """

    # add missing genes
    if gene_list is not None:
        for gene in gene_list:
            if gene not in list(G.nodes()):
                G.add_node(gene)

    # get adjacency matrix

    adj_matrix = nx.to_numpy_matrix(G, nodelist=gene_list)

    return adj_matrix


def reformat_edge_list(edge_list):
    """
    helper function to separate edges from their corresponding weight values

    Input
        edge_list
            list of sublists [n1, n2, weight]

    Output
        two lists containing edges and scores in same order as edge_list
    
    """

    # reformats edge lists into two lists containing edges and corresponding scores

    if edge_list is not None:
        edge_list_edge = [[edge[0], edge[1]] for edge in edge_list]
        edge_list_score = [edge[2] for edge in edge_list]

    else:
        edge_list_edge = None
        edge_list_score = None

    return edge_list_edge, edge_list_score


def relable_nodes(H):
    """
    function needed when converting networkx object to igraph 
    igraph does not allow to have "gaps in node ids",
    therefore ids are relabled after isolated nodes are removed to have consistent labeling

    Input
        networkx graph object

    Output
        function returns two dicts and the relabled graph object where isolated nodes have been removed
            the first one containing the mapping to create the new graph, the second one that can be used to revers the labeling

    """

    G = H.copy()
    # remove isolates
    G.remove_nodes_from(list(nx.isolates(G)))

    new_ids = {}
    reverse = {}
    # get all nodes and relable
    nodes = list(G.nodes())
    count = 0
    for node in nodes:
        new_ids[node] = count
        reverse[count] = node

        count = count + 1

    # update graph
    G = nx.relabel_nodes(G, new_ids, copy=False)

    # to reverse run nx.relabel_nodes(G, reverse, copy=False)
    return new_ids, reverse, G


def reverse_relable_nodes(H, mapping):
    """
    function to reverse relabeling of nodes as performed by relable_nodes

    Input
        networkx graph object
        reverse mapping as returned by relable_nodes as second item

    Output
        relabled graph object (isolated nodes are still removed but node names are consistent with original graph)
    """

    G = H.copy()

    G = nx.relabel_nodes(G, mapping, copy=False)

    return G
