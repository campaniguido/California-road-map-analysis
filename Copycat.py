import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import function as fn
import pandas as pd
import random
import matplotlib.cm as cm


file=pd.read_table('C:/Users/Guido/Desktop/guido/Complex_networks/python/California/roadnet-ca.txt')
file=np.array(file)
file=fn.Divide_value(file)
#%%


number_of_edge=1000
edges=fn.Edge_list(file, number_of_edge)
G=nx.Graph(edges)
G=fn.Sorted_graph(G)
#%%
G_node_dct=fn.G_node_dct(G)
G=nx.relabel_nodes(G,G_node_dct, copy=True) #non so perchè ma è fondamentale questa funzione per funzione comunità'
edges=list(G.edges())


#%%main features
Strenght=np.array(list(G.degree))
Closeness_Centrality=np.array(list((nx.closeness_centrality(G)).items()))
Betweeness_Centrality=np.array(list((nx.betweenness_centrality(G)).items()))
Clustering=np.array(list((nx.clustering(G)).items()))

community=nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
community=fn.Unfreeze_into_list(community)
community_color=fn.Set_community_number(G, community)
colors = (cm.CMRmap(np.linspace(0, 1, max(community_color))))

#%% histogram
'''
colors=(cm.Pastel2(np.linspace(0, 0.8, time_step)))
n, bins, patches=plt.hist(Strenght_G_piccolo[:,1],color=colors[3], bins=10)
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.title("Degree distribution-Reduced graph 5000 nodes")
#plt.savefig("strenght-freq.pdf", dpi=1000)
plt.show()

n, bins, patches=plt.hist(Betweeness_Centrality_G_piccolo[:,1],color=colors[3],bins=100)
plt.xlabel("Betweeness centrality")
plt.ylabel("Number of nodes")
plt.title("Betweeness Centrality distribution-Reduced graph 5000 nodes")
plt.savefig("BC-frec-piccolo.pdf", dpi=500)
plt.show()

n, bins, patches=plt.hist(Closeness_Centrality_G_piccolo[:,1],color=colors[3],bins=100)
plt.xlabel("Closeness centrality")
plt.ylabel("number of nodes")
plt.title("Closeness Centrality distribution-Reduced graph 5000 nodes")
#plt.savefig("cc-freq-piccolo.png", dpi=500)
plt.show()

n, bins, patches=plt.hist(Clustering_G_piccolo[:,1],color=colors[3],bins=100)
plt.xlabel("Clustering")
plt.ylabel("number of nodes")
plt.title("Clustering distribution-Reduced graph 5000 nodes")
plt.savefig("Clustering_piccolo.pdf", dpi=500)
plt.show()


colors=(cm.Pastel1(np.linspace(0.01, 0.9, time_step)))
n, bins, patches=plt.hist(K[:,1],color=colors[3], density=True, stacked=True, bins=7)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Average nearest neighbor distribution total")
plt.savefig("Knn-freq.png", dpi=1000)
plt.show()
            
            


#node_size=(Betweeness_Centrality[:,1]+0.99)**80
#width=(np.array(list(Salient_dct.values()))+0.32)**3
#width=(np.array(list(Betweeness_Centrality_edges.values()))+0.975)**35'''
'''nx.draw(G_piccolo,node_size=2,font_size=6)
plt.title("Road prova")
plt.show(G_piccolo)'''

#%%scatter
'''
colors=(cm.magma(np.linspace(0, 1, len(comunity_color_CNM_piccolo))))
fig, ax = plt.subplots()
scatter=ax.scatter(comunity_color_CNM_piccolo,Strenght_G_piccolo[:,1],c=comunity_color_CNM_piccolo,s=2)
plt.xlabel("Comunity")
plt.ylabel("Degree")
plt.title("Degree_G_5k vs comunities")
#plt.savefig("strength_5k vs com.pdf", dpi=500)
plt.show()

plt.scatter(comunity_color_CNM_piccolo,Clustering_G_piccolo[:,1],c=comunity_color_CNM_piccolo,s=2)
plt.xlabel("Comunity")
plt.ylabel("Clustering ")
plt.title("Comunity vs Clustering G_5k ")
plt.savefig("Cluster_5k vs com.pdf", dpi=500)
plt.show()

plt.scatter((comunity_color_CNM_piccolo),Closeness_Centrality_G_piccolo[:,1],c=comunity_color_CNM_piccolo,s=1)
plt.xlabel("Comunity")
plt.ylabel("Closeness Centrality ")
plt.title(" Comunity vs Closeness Centrality G_5k")
plt.savefig("CC_5k vs com.pdf", dpi=500)
plt.show()

plt.scatter(comunity_color_CNM_piccolo,Betweeness_Centrality_G_piccolo[:,1],c=comunity_color_CNM_piccolo,s=1)
plt.xlabel("Comunity")
plt.ylabel("Betweeness Centrality")
plt.title(" Comunity vs Betweeness Centrality G_5K")
plt.savefig("BC_5k vs com.pdf", dpi=500)
plt.show()



#_______________-confronto con Strenght_G_piccolo______

plt.scatter(Betweeness_Centrality_G_piccolo[:,1],Strenght_G_piccolo[:,1],c=comunity_color_CNM_piccolo,s=2)
plt.ylabel("Degree")
plt.xlabel("Betweeness centrality")
#for i in (range(len(Strenght_G_piccolo[:,1]))):
    #plt.annotate(i, (Strenght[:,1][i], Betweeness_Centrality_G_piccolo[:,1][i]))
plt.title("Betweeness Centrality G_5kvs Degree G_5k")
plt.savefig("BC_5k vs deg.pdf", dpi=500)
plt.show()

plt.scatter(Closeness_Centrality_G_piccolo[:,1],Strenght_G_piccolo[:,1],c=comunity_color_CNM_piccolo,s=2)
plt.ylabel("Degree")
plt.xlabel("Closeness centrality")
#for i in (range(len(Strenght[:,1]))):
    #plt.annotate(i, (Strenght[:,1][i], Closeness_Centrality_G_piccolo[:,1][i]))
plt.title("Closeness G_5k vs Degree G_5k ")
plt.savefig("CC_5k vs deg.pdf", dpi=500)
plt.show()'''

#%%COPYCAT PRIMO STEP UNENDO SOLO A PARTIRE DALLA DISTANZA

map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
dct_dist_link=fn.Dct_dist_link(edges,map_dct)
dct_dist=fn.Dct_dist(G=G, map_dct=map_dct)
nstep=50
step=max(dct_dist_link.values())/nstep

distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
distance_linking_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,distance_frequency)



Copy_map=nx.Graph()
Copy_map.add_nodes_from(list(G.nodes))
Copy_map=fn.Add_edges_from_map(Copy_map, map_dct, distance_linking_probability)

    
    
#%% parentesi COPYMAP_
G_piccolo=nx.Graph(Copy_map)      

    
Strenght_piccolo_=np.array(list(G_piccolo.degree))
K_piccolo_=np.array(list((nx.average_neighbor_degree(G_piccolo)).items()))
Closeness_Centrality_piccolo_=np.array(list((nx.closeness_centrality(G_piccolo)).items()))
Betweeness_Centrality_piccolo_=np.array(list((nx.betweenness_centrality(G_piccolo)).items()))
Clustering_piccolo_=np.array(list((nx.clustering(G_piccolo)).items()))

community_piccolo=nx.algorithms.community.modularity_max.greedy_modularity_communities(G_piccolo)
community_piccolo=fn.Unfreeze_into_list(community)
community_color_piccolo=fn.Set_community_number(G_piccolo, community)
color_piccolo = (cm.CMRmap(np.linspace(0, 1, max(community_color_piccolo))))
            
#%%   #%%Histogram       

n, bins, patches=plt.hist(Strenght_piccolo_[:,1],color=colors[2], bins=10)
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.title("Degree distribution-Copycat graph 5000 nodes")
#plt.savefig("strenght-freq-copycat.png", dpi=1000)
plt.show()

n, bins, patches=plt.hist(Betweeness_Centrality_piccolo_[:,1],color=colors[2])
plt.xlabel("Betweeness centrality")
plt.ylabel("Number of nodes")
plt.title("Betweeness centrality distribution-Copycat graph 5000 nodes")
#plt.savefig("BC-frec-copycat.png", dpi=500)
plt.show()

n, bins, patches=plt.hist(Closeness_Centrality_piccolo_[:,1],color=colors[3])
plt.xlabel("Closeness centrality")
plt.ylabel("number of nodes")
plt.title("Closeness Centrality distribution-Reduced graph 5000 nodes")
#plt.savefig("cc-freq-piccolo.png", dpi=500)
plt.show()

n, bins, patches=plt.hist(Clustering_piccolo_[:,1],color=colors[3],bins=100)
plt.xlabel("Clustering")
plt.ylabel("number of nodes")
plt.title("Clustering distribution-Reduced graph 5000 nodes")
plt.savefig("Clustering_copy_cat.png", dpi=500)
plt.show()







#%% Copycat degree correction

Copycat=fn.Copymap_degree_correction(Copy_map, G, map_dct, step, max(dct_dist_link.values()), distance_linking_probability)

#%%
'''Strenght_G_copy_cat=np.array(list(G_copy_cat.degree))

n, bins, patches=plt.hist(Strenght_G_copy_cat[:,1],bins=20)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("streght frequency-copy_cat")
plt.show()'''


Strenght_G_copy_cat=np.array(list(Copycat.degree))
K_G_copy_cat=np.array(list((nx.average_neighbor_degree(Copycat)).items()))
Closeness_Centrality_G_copy_cat=np.array(list((nx.closeness_centrality(Copycat)).items()))
Betweeness_Centrality_G_copy_cat=np.array(list((nx.betweenness_centrality(Copycat)).items()))
Clustering_G_copy_cat=np.array(list((nx.clustering(Copycat)).items()))

community_Copycat=nx.algorithms.community.modularity_max.greedy_modularity_communities(Copycat)
community_Copycat=fn.Unfreeze_into_list(community_Copycat)
community_color_Copycat=fn.Set_community_number(Copycat, community_Copycat)
colors = (cm.CMRmap(np.linspace(0, 1, max(community_color_Copycat))))
#%%
            

n, bins, patches=plt.hist(Strenght_G_copy_cat[:,1],color=colors[2], bins=10)
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.title("Degree distribution-Copycat graph 5000 nodes")
#plt.savefig("strenght-freq-copycat.png", dpi=1000)
plt.show()

n, bins, patches=plt.hist(Betweeness_Centrality_G_copy_cat[:,1],color=colors[2],bins=100)
plt.xlabel("Betweeness centrality")
plt.ylabel("Number of nodes")
plt.title("Betweeness centrality distribution-Copycat graph 5000 nodes")
#plt.savefig("BC-frec-copycat.png", dpi=500)
plt.show()

n, bins, patches=plt.hist(Closeness_Centrality_G_copy_cat[:,1],color=colors[2],bins=100)
plt.xlabel("Closeness centrality")
plt.ylabel("number of nodes")
plt.title("Closeness Centrality distribution-Reduced graph 5000 nodes")
#plt.savefig("cc-freq-copycat.png", dpi=500)
plt.show()

n, bins, patches=plt.hist(Clustering_G_copy_cat[:,1],color=colors[2],bins=100)
plt.xlabel("Clustering")
plt.ylabel("number of nodes")
plt.title("Clustering distribution-Reduced graph 5000 nodes")
plt.savefig("Clustering_copy_cat.png", dpi=500)
plt.show()


#%% scatter plot


fig, ax = plt.subplots()
scatter=ax.scatter(community_color_Copycat,Strenght_G_copy_cat[:,1],c=community_color_Copycat,s=2)
plt.xlabel("Comunity")
plt.ylabel("Degree")
plt.title("Degree Copycat vs comunities")
plt.savefig("strength_copy vs com.pdf", dpi=500)
plt.show()

plt.scatter(community_color_Copycat,Clustering_G_copy_cat[:,1],c=community_color_Copycat,s=2)
plt.xlabel("Comunity")
plt.ylabel("Clustering_G_copy_cat")
plt.title("Comunity vs Clustering_G_copy_cat ")
plt.savefig("Cluster_copy vs com.pdf", dpi=500)
plt.show()

plt.scatter((community_color_Copycat),Closeness_Centrality_G_copy_cat[:,1],c=community_color_Copycat,s=1)
plt.xlabel("Comunity")
plt.ylabel("Closeness_Centrality_G_copy_cat")
plt.title(" Comunity vs Closeness Centrality")
plt.savefig("CC_copy vs com.pdf", dpi=500)
plt.show()

plt.scatter(community_color_Copycat,Betweeness_Centrality_G_copy_cat[:,1],c=community_color_Copycat,s=1)
plt.xlabel("Comunity")
plt.ylabel("Betweeness_Centrality_G_copy_cat")
plt.title(" Comunity vs Betweeness_Centrality_G_copy_cat")
plt.savefig("BC_copy vs com.pdf", dpi=500)
plt.show()



#_______________-confronto con Strenght_G_copy_cat______

plt.scatter(Betweeness_Centrality_G_copy_cat[:,1],Strenght_G_copy_cat[:,1],c=community_color_Copycat,s=2)
plt.ylabel("Degree")
plt.xlabel("Betweeness centrality")
#for i in (range(len(Strenght_G_copy_cat[:,1]))):
    #plt.annotate(i, (Strenght[:,1][i], Betweeness_Centrality_G_copy_cat[:,1][i]))
plt.title("Betweeness Centrality vs Degree Copycat")
plt.savefig("BC_copy vs DEC.pdf", dpi=500)
plt.show()

plt.scatter(Closeness_Centrality_G_copy_cat[:,1],Strenght_G_copy_cat[:,1],c=community_color_Copycat,s=2)
plt.ylabel("Degree")
plt.xlabel("Closeness centrality")
#for i in (range(len(Strenght[:,1]))):
    #plt.annotate(i, (Strenght[:,1][i], Closeness_Centrality_G_copy_cat[:,1][i]))
plt.title("closeness vs Degree Copycat ")
plt.savefig("CC_copy vs DEG.pdf", dpi=500)
plt.show()