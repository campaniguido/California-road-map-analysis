# California Road map analysis

# Introduction
The codes here presented were used as a visual first step in the data analysis of the California road map. The codes have four main purposes:
-To visualize the main features distribution of a network.
-To plots information related of the evolution of the features within the size increasing of the network.
-To have a visual comparison with an Erdős–Rényi graph.
-To generate a network which try to emulate the distributions of the features considered. 

The data used were the ones found at the page http://snap.stanford.edu/data/roadNet-CA.html, where furthermore data information is presented. The useful data can be found in this repository in the file 'roadnet-ca.txt', which contains couples on nodes linked together. The network features taken into account in this study are the nodes degree, their betweenness and closeness centrality and the nodes clustering coefficient.

In any case these codes can be used on any dataset representing a network.

# Network basic theory

Let’s define a network as a set of elements(N) that have some linking relations among
them. Each element is called node. If two nodes are linked together there will be an
edge. Essentially, a graph G= (V, L) is a combination of these two sets: a set V of
N nodes and a set L of E links connecting the corresponding nodes.

In order to classify the nodes in a graph is it important to study the connection
among them and how to get from one to another. So a walk is the sequence of all
nodes and edges between 2 nodes and a path is a walk with all distinct nodes and
links. Instead, a closed path is called cycle.

The next step is to define the adjacency matrix A (NxN), in which aij=1.If there
is a link between vertices i and j, otherwise, aij=0.

The features taken into account are the node degree, the betweenness centrality, the closeness centrality and clustering coefficient.
# Degree
The degree k of th node v is the number of edges adjacent to it.

![degree](https://user-images.githubusercontent.com/79851796/156943628-a1d5a713-032b-4848-80f0-42bbf99cfdbf.PNG)

# Betweenness centrality
Betweenness centrality of a node v is defined as:

![BC](https://user-images.githubusercontent.com/79851796/156981731-fa18e7af-9ccb-4f00-9e42-d6e7abe88a69.PNG)

Where V is the set of nodes, σ(s,t) is number of shortest path from s to t, and σ(s,t|v)
is the number of those same paths which pass trough node v. If s=t, σ(s,t)=1 and
if ∈ s, t, σ(s, t|v) = 0 .

# Closeness centrality
The closeness centrality of a node u is the reciprocal of the average shortest path
distance to u over all n-1 reachable nodes weighted by the size of the component it belongs to.
![CC](https://user-images.githubusercontent.com/79851796/156981745-e946cf30-67fd-4743-8492-090dcdcd952e.PNG)

# Clustering coefficient
For unweighted graphs, the clustering of a node u is the fraction of possible triangles
through that node that exist:
![Clustering](https://user-images.githubusercontent.com/79851796/156981754-04980d79-ab9c-4fa0-8fef-3c8cf83fbbbe.PNG)


# Erdős–Rényi graph
The Erdős–Rényi graph is a kind of random graph in which the set of N node are
linked together randomly. Starting from a set with no edges, all the possible couple of nodes are taken into account and between each of them a linkage can be formed
with a probability p. So given p, the number N of node and the total number of
edges E, the probability to get a specified graph is it equal to:

![ERG](https://user-images.githubusercontent.com/79851796/156984349-f02f401e-8912-4c9f-8868-db461e24df7e.PNG)

# The project

As It was said in the introduction the code here presented was used as a fundamental first step for the visualization of the features of California road map, the visual comparison with an Erdős–Rényi graph and the generation of a network which try to emulate the characteristics of the real one. In the following rows we explain with more details how the three parts of the code work, for more details about why we choose this procedure was chosen we suggest to read the report Campani-Califonia_road_map.pdf.

Qualche parola sulla libreria usata

# main.py

main.py: it contains the code useful to read the file roadnet-ca.txt, deleting some bugs in the dataset,
                  to make plots/histograms of the main features of the whole/reduced graph and
                  to visualize the features distribution and their evolution within size increasing of the graph
               


One of the first problem was related to the size of the file. With the machine we were provided it was impossible to analyse the whole network. Indeed, betweenness and closeness centrality are very time demanding measures. So we decided to analyse just a reduce graph trying to find a right compromise between the time cost of the procedure and a map size which locally conserved some of the features property. 

So the first two parts of the code read the file roadnet-ca.txt, put the couple of numbers in a numpy.ndarray, sort the list of couples in an ascending order following the first element of the couple and create a graph of a given number of edge. This number represent a parameter to choose, higher the number smaller will be the difference between this reduced graph and the whole graph.  

The next part of the code calculates the distribution of degree, betweenness centrality, closeness centrality and, exploiting the networkx function algorithms.community.modularity_max.greedy_modularity_communities, it assigns at each node a community.

Then, the next bunch of code plots the histograms and scatterplots of the four features distributions. In the scatterplot the colours are related to the different communities. 





Finally in the last part of the code it is possible to visualize the evolution of the mean value of  the distributions and the distributions their selves within the increase of the network size. For the degree measure instead of the distribution of it the evolution of the ratios of the nodes degree are plotted.

Summing up, the parameters of interest to change in the main.py are:
-the file_position, if one is interested in having the same kind of data visualization for a different set of data. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
-The number of edges to take into account.
-The option to save or not the plots and the type of extension

# Erdős–Rényi graph

The aim of these lines of code is to reproduce an Erdős–Rényi graph of the same size of the network under study and with the same linking probability. Histograms and scatterplots of the random map are generated in order to a have a visual comparison with the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme previously described. Then, exploiting the function fast_gnp_random_graph an Erdős–Rényi graph is generating with the same linking probability of the reduced network. The next blocks of the codes reproduce histograms and scatterplots in the same way of the main.py


Summing up, the parameters of interest to change in the ERG.py are:
-the file_position, if one is interested to have a different starting network. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
-The number of edges to considered.
-The option to save or not the plots and the type of extension

#Copycat
The aim of these lines of code is to generate a network of the same size of the network under study and that which try to emulate the features distributions of the starting network. Histograms and scatterplots of it are generated to a have a visual comparison with the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme described in main.py. Since the network under study is a road map it has to follow some topological rule and the position of the node must play a preeminent role in the linkage process. Unfortunately in page http://snap.stanford.edu/data/roadNet-CA.html no meta-information of points coordinates is given. To overcome this problem the networkx function spring_layout is exploited to predict the position of the nodes. If one uses another dataset and he has a nodes coordinates map can use it instead of the one calculated here. From the information about the position of the nodes and their links a linking probability distribution in function of the distance is created. If one wants to use his own pdf can just substitute it to the one calculated here. Finally, a graph with the same node and same position of the road map or reduced road map is generated but with no linking. The links are generated by a stochastically process which exploit the position of the nodes and the linking pdf. After this linking process it is possible to visualize the features  histograms.  Then, it is possible to make a correction on the graph. Briefly this correction is made because with this procedure nodes in the middle have much more connections the one in the periphery and this behaviour doesn’t fit the real map. In the report further information are presents. The correction breaks the links of the node with high degree until the Copycat max degree is minor or equal the one of the road maps and equalize the degree ratio of the strongest nodes until they are lower than the ‘real’ one. Then a while loop starts until there is no more isolated nodes. At each degree ratio higher than the expected one are corrected, and links are added to the isolated nodes following a maximum probability rule attachment or linking them to the nearest node. The number of links and the degree of the target node are chosen in order to reproduce the degree distribution of the California network. Finally, it is possible to merge together small components.
At the end of the process the code provides histograms and scatter plots of the features of the Copycat network.
Summing up, the parameters of interest that can be changed in the Copycat.py are:
-the file_position, if one is interested to have a different starting network. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
-The number of edges to take into account.
-The option to save or not the plots and the type of extension
- a map distribution of the nodes of the starting graph and/or a map distribution of the nodes of the Copycat graph, otherwise the code itself gives one.
-a probability distribution that can be used as pdf in creation of the links, otherwise the code itself gives one.



The file Campani-Califonia_road_map.pdf contains all the numerical results derived by the plots generated by the code and a discussion of them.


                                             
roadnet-ca.txt: it contains couples of numbers representing couples of conected roads

main.py: it contains the code useful to read the file roadnet-ca.tx, deleting some bugs in the dataset,
                  to make plots/histograms of the main features of the whole/reduced graph and
                  to visualize the features distribution and their evolution within size increasing of the graph
                  
                  
ERG.py: it contains the code useful to create an Erdos-Rény graph with a linkg probability equal to the one of a the roadnet
                  and to make plots/histograms of the main features of this g
                  
raph
                  
Copycat.py: it contains the code useful to create a network which try to copy the features distribution of a given network.
                     and to make plots/histograms of the main features of this copynetwork.
                     
        
                  
function.py: contains all the function used in the three .py files and it also contains the tests of the functions

The libraries used are: math, matplotlib, networkx, numpy, pandas, pytest, random.



