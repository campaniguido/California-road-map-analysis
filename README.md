# California Road map analysis

## Introduction
The codes here presented were used as a visual first step in the data analysis of the California road map. The codes have four main purposes:
- To visualize the main features distribution of a network.
- To plots information related of the evolution of the features within the size increasing of the network.
- To have a visual comparison with an Erdős–Rényi graph.
- To generate a network which try to emulate the distributions of the features considered. 

The data used were the ones found at the page [Stanford data - California Road Map](http://snap.stanford.edu/data/roadNet-CA.html) , where furthermore data information is presented. The useful data can be found in this repository in the file 'roadnet-ca.txt', which contains couples onfnodes linked together. The network features taken into account in this study are the nodes degree, their betweenness and closeness centrality and the nodes clustering coefficient.

In any case these codes can be used on any dataset representing a network.

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
The degree k of th node v is the number of edges adjacent to it.

   ![degree](https://user-images.githubusercontent.com/79851796/157245914-5eb2a636-9946-4c1b-aca5-2fb8e1f2aae8.PNG)

### Betweenness centrality
Betweenness centrality of a node v is defined as:


![BC](https://user-images.githubusercontent.com/79851796/157246105-ef6a322c-d5d5-4c10-a962-507cc0a46ca8.PNG)

Where V is the set of nodes, σ(s,t) is number of shortest path from s to t, and σ(s,t|v)
is the number of those same paths which pass trough node v. If s=t, σ(s,t)=1 and
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

As It was said in the introduction the code here presented was used as a fundamental first step for the visualization of the features of California road map, the visual comparison with an Erdős–Rényi graph and the generation of a network which try to emulate the characteristics of the real one. In the following rows we explain with more details how the three parts of the code work, for more details about why this procedure was chosen we suggest to read the report Campani-Califonia_road_map.pdf.

## The libraries used 
- math
-  matplotlib
-  networkx
-   numpy
-    pandas
-    pytest
-   random

### California road map  (main.py)

The file main.py contains the code useful to read the file roadnet-ca.txt, deleting some bugs in the dataset, to make plots/histograms of the main features of the whole/reduced graph and  to visualize the features distributions and their evolution within size increasing of the graph
               


One of the first problem was related to the size of the file. With the machine we were provided it was impossible to analyse the whole network. Indeed, betweenness and closeness centrality are very time demanding measures. So we decided to analyse just a reduce graph trying to find a right compromise between the time cost of the procedure and a map size which locally conserved some of the features property. 

So the first two parts of the code read the file roadnet-ca.txt, put the couple of numbers in a numpy.ndarray, sort the list of couples in an ascending order following the first element of the couple and create a graph of a given number of edge. This number represent a parameter to choose, higher the number smaller will be the difference between this reduced graph and the whole graph.  

The next part of the code calculates the distribution of degree, betweenness centrality, closeness centrality and, exploiting the networkx function algorithms.community.modularity_max.greedy_modularity_communities, it assigns at each node a community.

Then, the next bunch of code plots the histograms and scatterplots of the four features distributions. In the scatterplot the colours are related to the different communities. 

![Degree distribution](https://user-images.githubusercontent.com/79851796/157253652-1a60cfa2-d2f1-41f6-8dd4-b53b70df7f60.png)

![Betweeness_Centrality vs Degree](https://user-images.githubusercontent.com/79851796/157253900-194c8e30-900b-4de6-b309-a346929b79b0.png)




Finally in the last part of the code it is possible to visualize the evolution of the mean value of  the distributions and the distributions their selves within the increase of the network size. For the degree measure instead of the distribution of it the evolution of the ratios of the nodes degree is plotted.

![MeanCloseness](https://user-images.githubusercontent.com/79851796/157258843-45640789-6c36-4a85-b85e-3b2214c36172.png)

![Closeness centralitycumulative-convergence](https://user-images.githubusercontent.com/79851796/157258887-be8bb6db-ebaa-4ab6-a2ea-f8b473ea7aa9.png)

![ratio of eachdegree](https://user-images.githubusercontent.com/79851796/157258919-c2f9cb7b-5bfe-4638-8e78-4ed14b0dc302.png)



Summing up, the parameters of interest to change in the main.py are:
- The file_position, if one is interested in having the same kind of data visualization for a different set of data. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
- The number of edges to take into account.
- The option to save or not the plots and the type of extension

### Erdős–Rényi graph (ERG.py)

The aim of these lines of code is to reproduce an Erdős–Rényi graph of the same size of the network under study and with the same linking probability. Histograms and scatterplots of the random map are generated in order to a have a visual comparison with the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme previously described. Then, exploiting the function fast_gnp_random_graph an Erdős–Rényi graph is generating with the same linking probability of the reduced network. The next blocks of the codes reproduce histograms and scatterplots in the same way of the main.py


Summing up, the parameters of interest to change in the ERG.py are:
- the file_position, if one is interested to have a different starting network. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
- The number of edges to considered.
- The option to save or not the plots and the type of extension

### Copycat (Copycat.py)
The aim of these lines of code is to generate a network of the same size of the network under study and that which try to emulate the features distributions of the starting network. Histograms and scatterplots of it are generated to a have a visual comparison with the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme described in main.py. Since the network under study is a road map it has to follow some topological rule and the position of the node must play a preeminent role in the linkage process. Unfortunately in page [Stanford data - California Road Map](http://snap.stanford.edu/data/roadNet-CA.html) no meta-information of points coordinates is given. To overcome this problem the networkx function spring_layout is exploited to predict the position of the nodes. If one uses another dataset and he has a nodes coordinates map can use it instead of the one calculated here. From the information about the position of the nodes and their links a linking probability distribution in function of the distance is created. If one wants to use his own pdf can just substitute it to the one calculated here.

Finally, a graph with the same node and same position of the road map or reduced road map is generated but with no linking. The links are generated by a stochastically process which exploit the position of the nodes and the linking pdf. After this linking process it is possible to visualize the features  histograms.  

Then, it is possible to make a correction on the graph. Briefly this correction is made because with this procedure nodes in the middle have much more connections than the one in the periphery and this behaviour doesn’t fit the real map. In the report further information are presents. The correction breaks the links of the node with high degree until the Copycat max degree is minor or equal the one of the road maps and equalize the degree ratio of the strongest nodes until they are lower than the ‘real’ one. Next, a while loop starts until there is no more isolated nodes. At each loop the degree ratios higher than the one expected are corrected and links are added to the isolated nodes following a maximum probability rule attachment or linking them to the nearest node. The number of links and the degree of the target node are chosen in order to reproduce the degree distribution of the California network. Finally, it is possible to merge together small components.


At the end of the process the code provides histograms and scatter plots of the features of the Copycat network.


Summing up, the parameters of interest that can be changed in the Copycat.py are:
- the file_position, if one is interested to have a different starting network. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
- The number of edges to take into account.
- The option to save or not the plots and the type of extension
- a map distribution of the nodes of the starting graph and/or a map distribution of the nodes of the Copycat graph, otherwise the code itself gives one.
- a probability distribution that can be used as pdf in creation of the links, otherwise the code itself gives one.

## Other files

Campani-Califonia_road_map.pdf:
- it contains all the numerical results derived by the plots generated by the code and a discussion of them and of the analysis method.
                                           
roadnet-ca.txt: 
- it contains couples of numbers representing couples of conected roads                     
        
                  
function.py: 
- it contains all the function used in the codes

test.py:
- it contains all the function tests
## Useful links

[Stanford data - California Road Map](http://snap.stanford.edu/data/roadNet-CA.html)

## Contacts
Author:

Guido Campani

guido.campani@studio.unibo.it




