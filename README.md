 <script src="ASCIIMathML.js"></script>

# California Road map analisys

The codes here presented were used as a visual first step in the data analisys of the California road map. The codes have four main pourposes:
-To vizualize the main features of the California road map.
-To plots informations related of the evolution of the features within the size increasing of the network.
-To have a visual comparison with an Erdős–Rényi graph.
-To generate a network which try to emulate the distribuiton of the features taken into account. 

The data used were the ones found at the page http://snap.stanford.edu/data/roadNet-CA.html, where furtheremore data information are presented. The useful data can be found in this repository in the file 'roadnet-ca.txt'. The network features taken into account in this study are the nodes degree, their betweeness and closeness centrality and the nodes clustering coefficient.

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

sum_(i=1)^n i^3=((n(n+1))/2)^2
$\sum_{i=1}^{10} t_i$
$$e^x=\sum_{i=0}^\infty \frac{1}{i!}x^i$$
The degree k of th node v is the number of edges adjacent to it.
k(v) = X
\sum_{i=1}^{10} t_i
j∈V
avj (2.1)
Betweeness centrality
Betweeness centrality of a node v is defined as:
B(v) = X
s,t∈V
σ(s, t|v)
σ(s, t)
(2.2)
Where V is the set of nodes, σ(s,t) is number of shortest path from s to t, and σ(s,t|v)
is the number of those same paths which pass trough node v. If s=t, σ(s,t)=1 and
if ∈ s, t, œ(s, t|v) = 0 .
Closeness centrality
The closeness centrality of a node u is the reciprocal of the average shortest path
distance to u over all n-1 reachable nodes weighted by the size of the component it
5
2. Theory
Figure 2.1: Erdős–Rényi network
belongs to.
C(u) = n − 1
N − 1
n − 1
Pn−1
v=1 d(v, u)
(2.3)
Clustering coefficient
For unweighted graphs, the clustering of a node u is the fraction of possible triangles
through that node that exist,
cu =
2T(u)
deg(u)deg(u) − 1
(2.4)
where T(u) is the number of triangles through node u and deg(u) is the degree of u.
2.1.1 Giant component
Another important feature to study in a network is the so called giant component.A
"component" is a subgroup of the graph nodes in which any two vertices are connected to each other by paths, and which is connected to no additional vertices in
the rest of the graph. A network has a "giant component", if it contains a finite
fraction of the entire graph’s vertices almost every node is reachable from almost
every other.
2.1.2 Erdős–Rényi graph
The Erdős–Rényi graph is a kind of random graph in which the set of N node are
linked together randomly. Starting from a set with no edges, all the possible couple
6
2. Theory
of nodes are taken into account and between each of them a linkage can be formed
with a probability p. So given p, the number N of node and the total number of
edges E, the probability to get a specified graph is it equal to:
P = p
E
(1 − p)
(
N
2 )−E



In any case these codes can be used on any dataset representing a network.
The file Campani-Califonia_road_map.pdf contains all the numerical results derived by the plots generated by the code and a discussion of them.


                                             In the file introduction there is a briefly explanation of all the data analysis procedure.
                                             
roadnet-ca.txt: it contains couples of numbers representing couples of conected roads

main.py: it contains the code useful to read the file roadnet-ca.tx, deleting some bugs in the dataset,
                  to make plots/histograms of the main features of the whole/reduced graph and
                  to visualize the features distribution
ERG.py: it contains the code useful to create an Erdos-Rény graph with a linkg probability equal to the one of a the roadnet
                  and to make plots/histograms of the main features of this graph
                  
Copycat.py: it contains the code useful to create a network which try to copy the features distribution of a given network.
                     and to make plots/histograms of the main features of this copynetwork.
                     
         
                  
function.py: contains all the function used in the three .py files and it also contains the tests of the functions

The libraries used are: math, matplotlib, networkx, numpy, pandas, pytest, random.