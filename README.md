 <script src="ASCIIMathML.js"></script>

# California Road map analisys

# Introduction
The codes here presented were used as a visual first step in the data analisys of the California road map. The codes have four main pourposes:
-To vizualize the main features distribution of a network.
-To plots informations related of the evolution of the features within the size increasing of the network.
-To have a visual comparison with an Erdős–Rényi graph.
-To generate a network which try to emulate the distribuiton of the features taken into account. 

The data used were the ones found at the page http://snap.stanford.edu/data/roadNet-CA.html, where furtheremore data information are presented. The useful data can be found in this repository in the file 'roadnet-ca.txt', which contains couples on nodes linked together. The network features taken into account in this study are the nodes degree, their betweeness and closeness centrality and the nodes clustering coefficient.

In any case these codes can be used on any dataset representing a network.

# Network basic theory

Let’s define a network as a set of elements(N) that have some linking relations among
them. Each element is called node. If two node are linked together there will be an
edge. Essentially, a graph G= (V, L) is a combination of these two sets; a set V of
N nodes and a set L of E links connecting the corresponding nodes.

In order to classify the nodes in a graph is it important to study the connection
among them and how to get from one to another. So a walk is the sequence of all
nodes and edges between 2 nodes and a path is a walk with all distinct nodes and
links. Instead a closed path is called cycle.

The next step is to define the adjacency matrix A (NxN), in which aij=1.If there
is a link between vertices i and j, otherwise, aij=0.

The features taken into account are the node degree, the betweeness centrality, the closeness centrality and clustering coefficient.
# Degree
The degree k of th node v is the number of edges adjacent to it.

![degree](https://user-images.githubusercontent.com/79851796/156943628-a1d5a713-032b-4848-80f0-42bbf99cfdbf.PNG)

# Betweeness centrality
Betweeness centrality of a node v is defined as:

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

As It was said in the introduction the code here presented was used as a fondamental first step for the visualization of the features of California road map, the visual comparison with an Erdős–Rényi graph and the generation of a network which try to emulate the characteristics of the real one. In the following rows we explain with more details how the three parts of the code work, for more details about why we choose this procedure was chosen we suggest to read the report Campani-Califonia_road_map.pdf.

Qualche parola sulla libreria usata

# main.py

main.py: it contains the code useful to read the file roadnet-ca.txt, deleting some bugs in the dataset,
                  to make plots/histograms of the main features of the whole/reduced graph and
                  to visualize the features distribution and their evolution within size increasing of the graph
               


One of the first problem was related to the size of the file. With the machine we were provided it was impossible to analyze the whole network. Indeed, betweeness and closeness centrality are very time demanding measures. So we decided to analyze just a reduce graph trying to find a right comprimise between the time cost of the procedure and a map size which locally conserved some of the features property. 

So the first two parts of the code read the file roadnet-ca.txt, put the couple of numbers in a numpy.ndarray, sort the list of couples in an ascending order following the first element of the couple and create a graph of a given number og edge. This number represent a parameter to choose, higher the number smaller will be the difference between this reduced graph and the whole graph.  

The next part of the code calculates the distribution of degree,betweeness centrality, closeness centrality and, exploiting the networkx function algorithms.community.modularity_max.greedy_modularity_communities, it assignes at each node a community.

Then, the next bunch of code plots the histograms and scatterplots of the four features distributions. In the scatterplot the colours are related to the different communities. 
![cc-freq](https://user-images.githubusercontent.com/79851796/157109211-5b485aa3-0fec-40ce-af21-75fb11ce493d.png)

![Betweeness_Centrality vs Degree](https://user-images.githubusercontent.com/79851796/157111300-13c47ea0-427c-4edc-9f7d-bc9ab905f123.png)


Finally in the last part of the code it is possible to vizualize the evolution of the maen value of of the distributions and the distributions their selves within the increase of the network size. For the degree measure instead of the distribution of it the evolution of the ratios of the nodes degree are plotted.

Reasuming the parameters of interest to change in the main.py are:
-the file_position, if one is interested in having the same kind of data visualization for a differente set of data. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
-The number of edges to take into account.
-The option to save or not the plots and the type of extension

# Erdős–Rényi graph

The aim of these lines of code is to reproduce an Erdős–Rényi graph of the same size of the network under study and with the same linking probability. Histograms and scatterplots of the random map are generated in order to a have a visual comparison vith the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme previously described. Then, exploitng the function fast_gnp_random_graph an Erdős–Rényi graph is generating with the same linking probability of the reduced network. The next blocks of the codes reproduce histograms and scatterplots in the same way of the main.py


Reasuming the parameters of interest to change in the ERG.py are:
-the file_position, if one is interested to have a different starting network. The file should be organized as a table nx2 in which each row represents a couple of linked nodes.
-The number of edges to take into account.
-The option to save or not the plots and the type of extension

#Copycat
The aim of these lines of code is to generate an a network of the same size of the network under study and that which try to emulate the features distributions of the starting network. Histograms and scatterplots of it are generated in order to a have a visual comparison with the network under study.

In the first lines of the codes a reduced network of the California map is generated, following the scheme described in main.py. Since the network under study is a road map it has to follow some topological rule and the position of the node must play a preminent role in the linkage process. Unfortunatly in page http://snap.stanford.edu/data/roadNet-CA.html no meta-information of points coordinates is given. To overcome this problem the networkx function spring_layout is exploited to predict the position of the nodes. If one uses another dataset and he has a nodes coordinates map can use it instead of the one calculated here. From the information about the position of the nodes and thei links a linking probability distribution in function of the distance is created. If one wants to use his own pdf can just substitute it to the one calculated here. Finally a graph with the same node and same position of the road map or reduced road map is generated but with no linking. The links are generated by a stocastical process which exploit the position of the nodes and the linking pdf.


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



