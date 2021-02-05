'''
Contains some simple plotting functions.
'''


import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches
import numpy as np


def plot_heatmap(matrix, xlabels=None, ylabels=None, size=(10,8), cmap="bone"):

    """
    Plots a heatmap. Return figure fig can be saved with fig.savefig(). For parameters refer to the matplotlib documentation.

    Parameters:
        matrix (numpy matrix): matrix containing the values to be plotted
        xlabels (list or None): list of tick labels to be used for the x axis
        ylabels (list or None): list of tick labels to be used for the y axis
        size (tuple): plot size to be used
        cmap (matplotlib colormap): can be created colormap or name of a pre-defined color map

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

    sns.heatmap(matrix, annot=False, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap=cmap)

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
        fig (matplotlib figure): heatmap object
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
        fig (matplotlib figure):
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