'''
Contains some plotting functions.
'''


import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches
import numpy as np
from scipy.stats.stats import pearsonr 
import networkx as nx
import random
import volta.communities as communities


def plot_heatmap(matrix, xlabels=None, ylabels=None, size=(10,8), cmap="bone", annotation=False):

    """
    Plots a heatmap. Return figure fig can be saved with fig.savefig(). For parameters refer to the matplotlib documentation.

    Parameters:
        matrix (numpy matrix): matrix containing the values to be plotted
        xlabels (list or None): list of tick labels to be used for the x axis
        ylabels (list or None): list of tick labels to be used for the y axis
        size (tuple): plot size to be used
        cmap (matplotlib colormap): can be created colormap or name of a pre-defined color map
        annotation (boolea): if True then cell values are plotted

    Returns:
        fig (matplotlib figure): heatmap object
    """

    if xlabels is None:
        xlabels="auto"
    if ylabels is None:
        ylabels = "auto"

    if cmap is None:
        cmap = sns.cubehelix_palette(start=7, rot=0, dark=0.2, light=0.8, reverse=False, as_cmap=True)

    fig, ax = plt.subplots(figsize=size)  

    sns.heatmap(matrix, annot=annotation, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap=cmap)

    return fig




def plot_hierarchical_clustering(matrix, xlabels=None, ylabels=None, size=(10,8), cmap="bone"):

    """
    Plots a clustermap. Return figure fig can be saved with fig.savefig(). For parameters refer to the matplotlib documentation.

    Parameters:
        matrix (numpy matrix): matrix containing the values to be plotted
        xlabels (list or None): list of tick labels to be used for the x axis
        ylabels (list or None): list of tick labels to be used for the y axis
        size (tuple): plot size to be used
        cmap (matplotlib colormap): can be created colormap or name of a pre-defined color map

    Returns:
        fig (matplotlib figure): cluster-heatmap object
    """

    if xlabels is None:
        xlabels="auto"
    if ylabels is None:
        ylabels = "auto"

    if cmap is None:

        cmap = sns.cubehelix_palette(start=7, rot=0, dark=0.2, light=0.8, reverse=False, as_cmap=True)

    

    return sns.clustermap(matrix, annot=False, figsize=size, xticklabels=xlabels, yticklabels=ylabels, cmap=cmap)

    
def plot_clustering_heatmap(clusters, matrix, labels, cmap="bone", size=(10,8)):

    """
    Plots a clustering on top of a provided distance matrix.

    Parameters:
        clusters (array): containing the cluster IDs, needs to be in the same order as labels
        matrix (matrix): distance or similarity matrix that has been provided as input for the clustering
        labels (list): list of labels to be used for plot. Needs to be in the same order as clusters
        cmap (matplotlib colormap): can be created colormap or name of a pre-defined color map
        size (tuple): figuresize

    Returns:
        fig (matplotlib figure): of clustering
    """



    c_sorted = np.sort(clusters)
    c_sorted

    borders = []

    for cl in list(Counter(c_sorted).keys()):
        found = False
        end = False
        for ii in range(len(c_sorted)):
            i = c_sorted[ii]
            if i == cl and not found:
                borders.append(ii)
                found = True
                
            
                
    borders.append(len(c_sorted))

    #get indices of the items

    inds = np.argsort(clusters)
    for n, f in enumerate(borders[:-1]):
            i = inds[f:borders[n + 1]]
            
            cco = i[matrix[np.ix_(i, i)].mean(axis=1).argsort()[::-1]]
            
            inds[f:borders[n + 1]] = cco
            
            
            
    #create plot

    fig, ax = plt.subplots(1, 1, figsize=size)
    data = matrix[np.ix_(inds, inds)]

    if cmap is None:
        cmap = sns.cubehelix_palette(start=7, rot=0, dark=0.2, light=0.8, reverse=False, as_cmap=True)

    coll = ax.pcolormesh(data,  cmap=cmap)
    #ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))


    #ax.invert_yaxis()

    for n, e in enumerate(np.diff(borders)):
            ax.add_patch(matplotlib.patches.Rectangle((borders[n], borders[n]),
                                        e, e, fill=False, linewidth=3,
                                        edgecolor="white"))

            

    #sorted labels for plotting labels
    sorted_labels = []
    for ind in inds:
        sorted_labels.append(labels[ind])
        
        
    ax.set_yticks(np.arange(len(sorted_labels)) + 0.5)
    ax.set_yticklabels(sorted_labels)

    #get tick locations
    loc = []
    for i in range(len(borders)-1):
        loc.append(np.mean([borders[i], borders[i+1]]))


    ax.set_xticks(loc)
    ax.set_xticklabels(labels=list(Counter(c_sorted).keys()))


    fig.colorbar(coll)

    return fig


def plot_agreement_matrix(clusterings, xlabels=None, ylabels=None, size=(10,8), cmap="bone", annotation=True):
    """
    Calculates the agreement matrix between provided clusterings, which is plotted as a heatmap. For each item pair a value is estimated how often
    this item is grouped together between the provided algorithms.

    Parameters:
        clusterings (list): list of clusterings. Clusterings need to be lists and items need to be in same order between the provided
            clusterings.
        xlabels (list or None): list of tick labels to be used for the x axis
        ylabels (list or None): list of tick labels to be used for the y axis
        size (tuple): plot size to be used
        cmap (matplotlib colormap): can be created colormap or name of a pre-defined color map
        annotation (boolea): if True then cell values are plotted

    Returns:
        fig (matplotlib figure): heatmap object
        aggreement matrix (matrix): containing the agreement values
    """

    agg =  np.zeros((len(clusterings[0]), len(clusterings[0])))

    for index, x in np.ndenumerate(agg):
        cnt = 0
        for c in clusterings:
            if c[index[0]] == c[index[1]]:
                cnt = cnt + 1
                
        agg[index[0]][index[1]] = cnt



    f = plot_heatmap(agg, xlabels=xlabels, ylabels = ylabels, size=size, cmap=cmap, annotation=annotation)


    return f, agg




def plot_correlation_clusterings(clusterings, xlabels=None, ylabels=None, size=(10,8), cmap="bone"):
    """
    Calculates the pearson correlation between pairs of cluster vectors and plots a cluster map.

    Parameters:
        clusterings (list): list of clusterings. Clusterings need to be lists and items need to be in same order between the provided
            clusterings. Individual items need to be int.
        xlabels (list or None): list of tick labels to be used for the x axis
        ylabels (list or None): list of tick labels to be used for the y axis
        size (tuple): plot size to be used
        cmap (matplotlib colormap): can be created colormap or name of a pre-defined color map
        

    Returns:
        fig (matplotlib figure): heatmap object
        correaltion matrix (matrix): containing the correaltion values
    """

    cor =  np.zeros((len(clusterings), len(clusterings)))

    for index, x in np.ndenumerate(cor):
        c, p = pearsonr(clusterings[index[0]], clusterings[index[1]])
        cor[index[0]][index[1]] = c



    f = plot_hierarchical_clustering(cor, xlabels=xlabels, ylabels=ylabels, size=size, cmap=cmap)


    return f, cor


def plot_graph(G, pos=None, with_labels=False, node_color="#A0A0A0", edge_color="#A0A0A0",node_size=1000, width=2.0, node_border="black", figsize=(5,5)):
    """
    Plots a Graph object.
    
    Parameters:
        G (networkx graph object): graph to be plotted
        pos (pos or None): node positions as returned by networkx position functions. If None
            position based on a spring layeout is calculated.
        with_labels (boolean): if True node labels are plotted.
        node_color (string or list): if string needs to be hex code of node color to be used. If it is a list
            it needs to be in the same order as G.nodes() and a color needs to be assigned for each node.
        edge_color (string or list): is string needs to be hex code of edge color to be used. If it is a list
            it needs to be in the same order as G.edges() and color needs to be assigned for each edge.
        node_size (int): size of nodes to be plotted.
        width (float): edge width to be plotted.
        node_border (string): hex code of node border to be plotted.
        figsize (tuple): dimension of to be plotted figure
        
    Returns:
        fig (matplotlib figure): figure
        pos (dict): used node positining for plots
    
    """
    


    fig = plt.figure(figsize=figsize) 
    
    if pos is None:
        pos = nx.spring_layout(G)

    f = nx.draw(G, pos, with_labels = with_labels, node_size=node_size, node_color=node_color, edge_color=edge_color, width=width, edgecolors=node_border)

    return fig, pos





def plot_communities(G, com, pos=None, with_labels=False, node_color=None, edge_color="#A0A0A0",node_size=1000, width=2.0, node_border="black", figsize=(5,5)):
    """
    Plots a Graph object.
    
    Parameters:
        G (networkx graph object): graph to be plotted
        com (dict): where node ID is key and value is list of communities this node belongs to. 
            Only the first node assignment will be considered during color selection. If a default dict is returned by
            the selected community algorithms it needs to be transformed with dict(defaultdict).
        pos (pos or None): node positions as returned by networkx position functions. If None
            position based on a spring layeout is calculated.
        with_labels (boolean): if True node labels are plotted.
        node_color (list):  If it is a list it needs to be in the same order as G.nodes() 
            and a color needs to be assigned for each node, which should correspond to its community assignment.
            If it is None random colors for each community are generated.
        edge_color (string or list): is string needs to be hex code of edge color to be used. If it is a list
            it needs to be in the same order as G.edges() and color needs to be assigned for each edge.
        node_size (int): size of nodes to be plotted.
        width (float): edge width to be plotted.
        node_border (string): hex code of node border to be plotted.
        figsize (tuple): dimension of to be plotted figure
        
    Returns:
        fig (matplotlib figure): figure
        pos (dict): used node positining for plots
        colors (dict): where id is community ID and value is assigned color
    
    """
    
    if node_color is None:
        #number of communities / colors to generate
        n = communities.get_number_of_communities(com)

        colors = {}
        for i in range(n):
            #generate colors

            random_number = random.randint(0,16777215)
            hex_number = str(hex(random_number))
            color ='#'+ hex_number[2:]
            while len(color) < 7:
                color = color + "0"
            colors[i] = color


        node_color = []
        for node in G.nodes():
            node_color.append(colors[com[node][0]])


    fig = plt.figure(figsize=figsize) 
    
    if pos is None:
        pos = nx.spring_layout(G)

    f = nx.draw(G, pos, with_labels = with_labels, node_size=node_size, node_color=node_color, edge_color=edge_color, width=width, edgecolors=node_border)
    
    
    #generate a dummy plot in order to add a color ledgend
    
    for v in range(n):
        plt.scatter([],[], c=colors[v], label='Community{}'.format(v))
        
        
    fig.legend()

    return fig, pos, colors