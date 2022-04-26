# California road map analysis

## Introduction
The codes here presented were used as a visual first step in the data analysis of the California road map. The codes have four main purposes:
- To visualize the main features distribution of a network.
- To plots information related of the evolution of the features within the size increasing of the network.
- To have a visual comparison with an Erdős–Rényi graph.
- To generate a network which try to emulate the distributions of the features considered. 

The data used were the ones found at the page [Stanford Network Analysis Project](http://snap.stanford.edu/data/roadNet-CA.html) , where there is furthermore data information. The useful data can be found in this repository in the file 'roadnet-ca.txt', which contains couples of nodes linked together. The network features taken into account in this study are the nodes degree, the betweenness centrality, closeness centrality and the nodes clustering coefficient.

In any case these scripts can be used on any dataset representing a network.

## Quick instruction: how to use
As a first step the repository has to be copied and before invoking any commands one has to move inside the new directory using the following commands
```bash
 git clone https://github.com/campaniguido/California-road-map-analysis
 cd California-road-map-analysis
```
All codes must be executed inside a python environment, thus it could be necessary to add `python` before the instructions shown. All the simulations need to set before some parameters which are stored in the files `parameters.py` and `parameters_Copycat.py`. In these files default values are already given.

To launch the simulations related to the reduced California road network the command line to invoke  is `.\road_map.py`, the outputs are stored in csv files in a new directory whose name is the one set by the variable `name_simulation` in the file `parameters.py`.

To visualize the data of the California road network simulation the command line is `.\plot_road_map.py`, the outputs are stored in same directory of the csv data.

To launch the simulation to generate a network which tries to emulate the reduced California road map the command line  is `.\Copycat.py`, the outputs are stored in csv files in a new directory whose name is the one set by the variable `name_simulation` in the file `parameters_Copycat.py`.

To visualize the data plots of this simulation the command line  to invoke is `.\Copycat_plot.py`,the outputs are stored in same directory of the copycat csv data.

## Network basic theory

Let’s define a network as a set of elements(N) that have some linking relations among
them. Each element is called node. If two nodes are linked together there will be an
edge. Essentially, a graph G= (V, L) is a combination of these two sets: a set V of
N nodes and a set L of E links connecting the corresponding nodes.

In order to classify the nodes in a graph is it important to study the connection
among them and how to get from one to another. So a walk is the sequence of all
nodes and edges between 2 nodes and a path is a walk with all distinct nodes and
links. Instead, a closed path is called cycle.

The next step is to define the adjacency matrix A (NxN), in which a<sub>ij</sub>=1 if there
is a link between vertices i and j, otherwise, a<sub>ij</sub>=0.

The features taken into account are the node degree, the betweenness centrality, the closeness centrality and clustering coefficient.
### Degree
The degree k of the node v is the number of edges adjacent to it.

   ![degree](https://user-images.githubusercontent.com/79851796/157245914-5eb2a636-9946-4c1b-aca5-2fb8e1f2aae8.PNG)

### Betweenness centrality
Betweenness centrality of a node v is defined as:


![BC](https://user-images.githubusercontent.com/79851796/157246105-ef6a322c-d5d5-4c10-a962-507cc0a46ca8.PNG)

Where V is the set of nodes, σ(s,t) is number of shortest paths from s to t, and σ(s,t|v)
is the number of those same paths which pass through node v. If s=t, σ(s,t)=1 and
if ∈ s, t, σ(s, t|v) = 0 .

### Closeness centrality
The closeness centrality of a node u is the reciprocal of the average shortest path
distance to u over all n-1 reachable nodes weighted by the size of the component it belongs to.

  ![CC](https://user-images.githubusercontent.com/79851796/157246120-44c9089a-8fde-4dcc-b84d-7f53e59f2af5.PNG)


### Clustering coefficient
For unweighted graphs, the clustering of a node u is the fraction of possible triangles
through that node that exist:

  ![Clustering](https://user-images.githubusercontent.com/79851796/157246145-8c5ce342-2b2b-4543-a93b-c14ba7b4dd87.PNG)



### Erdős–Rényi graph
The Erdős–Rényi graph is a kind of random graph in which the set of N node are
linked together randomly. Starting from a set with no edges, all the possible couple of nodes are taken into account and between each of them a linkage can be formed
with a probability p. So given p, the number N of node and the total number of
edges E, the probability to get a specified graph is it equal to:

  ![ERG](https://user-images.githubusercontent.com/79851796/157246198-3d64a4b2-41ac-48fe-a31f-273772d4b4f8.PNG)

## The project

As It was said in the introduction the code here presented was used as a fundamental first step for the visualization of the features of the California road map, to visually compare it with an Erdős–Rényi graph and to generate a network which tries to emulate the characteristics of the real one. In the following rows we explain with more details how the four main scripts(road_map.py, plot_road_map.py, Copycat.py Copycat_plot.py) work, for more details about why this procedure was chosen we suggest reading the report Campani-Califonia_road_map.pdf.

## The libraries used 
- math
- matplotlib
- networkx
- numpy
- pandas
- pytest
- random
- os

### The California road map and ERG simulation (road_map.py)

The file road_map.py contains the code useful to read the file of all the couples of nodes, deleting any bugs in the dataset, it calculates the degree, betweeness centrality, closeness centrality and clustering of each node and simulates how these features evolves with the increasing of the network size. Finally it also calculates the distribution of the features of an ER graph with the same number of nodes and same link density.

In the first lines it stores the data related to all the couple of nodes of the graph. The position of the  data file, `file_position`, is one of the parameters that can be changed, the default one is the path related to this repository. Then the code generate a network using only the first `number_of_edge` expressed in the `parameters.py`, so it is not mandatory to take into account all the links of the network. The links are chosen in the order in which they are presented in the file. 

The next part of the code calculates the distribution of degree, betweenness centrality, closeness centrality and clustering and it assigns at each node a community.
All this data are stored  in a csv file with the name `data_road`. The name of the directory where the file is stored,`name_simulation`, is a parameter that the user could change. The default name is setting using as a name the number of edges and the seed used. Then the code stores the data related to the evolution of the cumulative distribution of the features within the increasing of the network size. So the cumulative distribution are calculated n times at each step the number of links of the network taken into account increase till reaching the fixed number. The number of the steps of this process are the ones of the variables `n_step` and `n_step_degree` (this one is only for the degree feature), these value can be modify by the user in the `parameters.py`. In the same way at each step it also calculates the main values of each feature distribution and its standard deviation. The data are stored in the same directory as csv file the following name: `degree_evolution`, `BC_evolution`, `CC_evolution`, `Clustering_evolution`.

In the last part of the script it generates an ER graph with the same number of nodes and same link density and it store a csv file, named `ERG_data`, the measure of the features of each node in the same way of the road map.

### California road map and ERG plot (plot_road_map.py)

It provides the plot of the data in the directory selected by the user. The default one is the same of the last simulation launched. More specifically, it provides histograms of the four features both for the road map and the ER graph and scatter plot of the community number vs the feature value and the node degree vs each one of the other features values. In the scatterplot the colours are related to the different nodes communities. 

![Degree distribution](https://user-images.githubusercontent.com/79851796/157253652-1a60cfa2-d2f1-41f6-8dd4-b53b70df7f60.png)

![Betweeness_Centrality vs Degree](https://user-images.githubusercontent.com/79851796/157253900-194c8e30-900b-4de6-b309-a346929b79b0.png)


Finally in the last part of the code it is possible to visualize the evolution of the mean value of  the distributions and the distributions their selves within the increase of the network size. For the degree measure instead of the distribution of it, the evolution of the ratios of the nodes degree is plotted.

![MeanCloseness](https://user-images.githubusercontent.com/79851796/157258843-45640789-6c36-4a85-b85e-3b2214c36172.png)

![Closeness centralitycumulative-convergence](https://user-images.githubusercontent.com/79851796/157258887-be8bb6db-ebaa-4ab6-a2ea-f8b473ea7aa9.png)

![ratio of eachdegree](https://user-images.githubusercontent.com/79851796/157258919-c2f9cb7b-5bfe-4638-8e78-4ed14b0dc302.png)

The plots can be just visualized setting `save_fig` equal to False or they can be saved setting it equal to True and fixing the extention of the file in the `extention` variable.

### Copycat simulation (Copycat.py)
The aim of these lines of code is to generate a network of the same size of the network under study and that which tries to emulate the features distributions of the starting network. Histograms and scatterplots of it are generated to a have a visual comparison with the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme described in road_map.py. The parameters to fix are in the file `parameters_Copycat.py`. Since the network under study is a road map it has to follow some topological rule and the position of the node must play a preeminent role in the linkage process. Unfortunately in the page [Stanford Network Analysis Project](http://snap.stanford.edu/data/roadNet-CA.html) no meta-information of points coordinates is given. To overcome this problem the networkx function spring_layout is exploited to predict the position of the nodes, the distance distribution is binned and the number of bins are to fix by the user in the variable `n_bin`. If one uses another dataset and he has a nodes coordinates map can use it instead of the one calculated here. From the information about the position of the nodes and their links a linking probability distribution in function of the distance is created.

Finally, a graph with the same node and same position of the road map or reduced road map is generated but with no linking. The links are generated by a stochastically process which exploit the position of the nodes and the linking pdf.   

Then, it is possible to make a correction on the graph. The correction breaks the links of the node with high degree until the Copycat max degree is minor or equal the one of the road maps and equalize the degree ratio of the strongest nodes until they are lower than the ‘real’ one. Next, a while loop starts until there is no more isolated nodes. At each loop the degree ratios higher or lower than the one expected are corrected and links are added to the isolated nodes following a maximum probability rule attachment or linking them to the nearest node. The number of links and the degree of the target node are chosen in order to reproduce the degree distribution of the California network. Finally, it is possible to merge together small components.

The features measures of each node are stored in a csv file, with the name `data_Copycat`. The file is saved in a directory whose name can be set by the user modifying the variable name_simulation in the parameters_Copycat file. The default setting name is related to number of links fixed in the simulation and seed number.

### Copycat plot (Copycat_plot.py)

It provides plots of the data in the directory selected by the user. The default one is the same the last simulation launched. More specifically, it provides histograms of the four features for the Copycat graph and scatter plots of the community number vs the feature value and the node degree vs  each one of the other features values. In the scatterplots the colours are related to the different communities. 
At the end of the process the code provides histograms and scatter plots of the features of the Copycat network.

The plots can be just visualized setting `save_fig` equal to False or they can be saved setting it equal to True and fixing the extention of the file in the `extention` variable.


## Other files

Campani-Califonia_road_map.pdf:
- it contains all the numerical results derived by the plots generated by the scripts and a discussion of them and of the analysis methods.
                                           
roadnet-ca.txt: 
- it contains couples of numbers representing couples of the California road map                     
        
                  
function.py: 
- it contains all the function used in the scripts for the simulations

function_plot.py:
- it contains all the function used in the scripts for the plots

test.py:
- it contains all the function tests
  in order to perform the tests, it is just necessary to invoke the pytest module: pytest


## Useful links

[Stanford Network Analysis Project](http://snap.stanford.edu/data/roadNet-CA.html)

## Contacts
Author:

Guido Campani

guido.campani@studio.unibo.it




