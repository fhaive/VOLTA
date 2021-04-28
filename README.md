# VOLTA: adVanced mOLecular neTwork Analysis

VOLTA is a Python network analysis package, suited for different types of networks but with a focus on co-expression 
network analysis. The goal of VOLTA is to provide all functionalities needed for a comprehensive network analysis and comparison
in a single package. Its aim is to expose all functionalities in order to allow users to build their own analysis pipelines and 
adjust functionalities as needed. Additional complete analysis pipelines are provided in the form of Jupyter Notebooks.


## How do I get set up?

### Dependencies

- NumPy >=1.17.* 
- Matplotlib >=3.0.* 
- statsmodels >= 0.11.* 
- Scikit-learn >=0.21.* 
- Brain Connectivity Toolbox for Python >= 0.5.* https://github.com/aestrivex/bctpy
- NetworkX >=2.5 
- CDLIB >= 0.1.8 
- SciPy >=1.3.* 
- leidenalg >= 0.7  https://github.com/vtraag/leidenalg
- seaborn >=0.9.* 
- pandas >=0.25 
- Markov Clustering >=0.0.6.*  https://github.com/GuyAllard/markov_clustering 
- Cython https://cython.org/ 
- netneurotools >= 0.2.* https://github.com/netneurolab/netneurotools
- PyClustering == 0.9.3 
- python-louvain >= 0.13 
- treelib >= 1.6.* https://github.com/caesar0301/treelib
- pyintergraph https://gitlab.com/luerhard/pyintergraph
- Partition Quality == 0.0.7 https://github.com/GiulioRossetti/partition_quality
- ANGEL https://github.com/GiulioRossetti/ANGEL 



### Prerequisites

- Install python-igraph . Based on your system there may be specific Prerequisites to fulfill. Refer to the igraph instructions for these.
- For windows users please install the optional requirements as instructed here: https://github.com/GiulioRossetti/cdlib , else some of the community algorithms will not be available

VOTLA has been tested on Linux, Mac and Windows systems for Python 3.6+.

### Install

- download repository (wget https://github.com/fhaive/VOLTA.git)
- cd into the repository
- pip3 install .

Depending on your system you may receive an "invalid command bdist_wheel" error. You can fix this by installing the wheel package.
pip3 install wheel


## Usage

- import volta 

In the jupyternotebooks folder different examples on how to use VOLTA as well as complete analysis pipleines for co-expression 
network analysis are provided. 
VOLTA requires networks to be provided in the form of NetworkX graph objects. A Jupyter Notebook file is provided, showing how
different network formats can be converted into the required graph object.

For detailed function information refer to the documentation, which is provided as HTML in the html folder.

More information about VOLTA can be found in the corresponding publication: 


### How to cite VOLTA


### License

VOLTA is published under a GNU GENERAL PUBLIC LICENSE Version 3 license. Individual parts may underly a differet license, for this please refer to the package providers.
