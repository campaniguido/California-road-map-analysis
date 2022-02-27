import networkx as nx
import numpy as np
import function as fn
import pandas as pd
import matplotlib.cm as cm

#%%

file=pd.read_table('https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt')
file=np.array(file)
file=fn.Divide_value(file)
#%%itialize the graph


number_of_edge=1000
edges=fn.Edge_list(file, number_of_edge)
G=fn.SuperGraph(edges)
G.Sorted_graph()
G.Relable_nodes() #non so perchè ma è fondamentale questa funzione per funzione comunità'
edges=list(G.edges())


#%%main features network
Strenght=np.array(list(G.degree))
Closeness_Centrality=np.array(list((nx.closeness_centrality(G)).items()))
Betweeness_Centrality=np.array(list((nx.betweenness_centrality(G)).items()))
Clustering=np.array(list((nx.clustering(G)).items()))

community=nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
community=fn.Unfreeze_into_list(community)
community_color=fn.Set_community_number(G, community)
colors = (cm.CMRmap(np.linspace(0, 1, max(community_color))))





#%%COPYCAT network. First step: linking following the distance attachment rule

map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
dct_dist_link=fn.Dct_dist_link(edges,map_dct)
dct_dist=fn.Dct_dist(G=G, map_dct=map_dct)
nstep=50
step=max(dct_dist_link.values())/nstep

distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
distance_linking_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,distance_frequency)
Copy_map=fn.SuperGraph()
Copy_map.add_nodes_from(list(G.nodes))
Copy_map=fn.Add_edges_from_map(Copy_map, map_dct, distance_linking_probability)

    
    
#%% features COPYMAP   

  
Strenght_Copy_map=np.array(list(Copy_map.degree))
K_Copy_map=np.array(list((nx.average_neighbor_degree(Copy_map)).items()))
Closeness_Centrality_Copy_map=np.array(list((nx.closeness_centrality(Copy_map)).items()))
Betweeness_Centrality_Copy_map=np.array(list((nx.betweenness_centrality(Copy_map)).items()))
Clustering_Copy_map=np.array(list((nx.clustering(Copy_map)).items()))

community_Copy_map=nx.algorithms.community.modularity_max.greedy_modularity_communities(Copy_map)
community_Copy_map=fn.Unfreeze_into_list(community)
community_color_Copy_map=fn.Set_community_number(Copy_map, community)
color_Copy_map = (cm.CMRmap(np.linspace(0, 1, max(community_color_Copy_map))))
            
#%%   #%%Histogram   Copymap    

fn.Hist_plot(Strenght_Copy_map[:,1],color_Copy_map[5], 'Degree distribution Copymap')
fn.Hist_plot(Betweeness_Centrality_Copy_map[:,1],color_Copy_map[5], 'Betweeness_Centrality distribution Copymap')
fn.Hist_plot(Closeness_Centrality_Copy_map[:,1],color_Copy_map[5], 'Closeness_Centrality distribution Copymap')
fn.Hist_plot(Clustering_Copy_map[:,1],color_Copy_map[5], 'Clustering distribution Copymap') 


#%% Copycat degree correction

Copycat=fn.Copymap_degree_correction(Copy_map, G, map_dct, max(dct_dist_link.values()), distance_linking_probability, Merge=True)

#%% Features Copycat


Strenght_Copycat=np.array(list(Copycat.degree))
K_Copycat=np.array(list((nx.average_neighbor_degree(Copycat)).items()))
Closeness_Centrality_Copycat=np.array(list((nx.closeness_centrality(Copycat)).items()))
Betweeness_Centrality_Copycat=np.array(list((nx.betweenness_centrality(Copycat)).items()))
Clustering_Copycat=np.array(list((nx.clustering(Copycat)).items()))

community_Copycat=nx.algorithms.community.modularity_max.greedy_modularity_communities(Copycat)
community_Copycat=fn.Unfreeze_into_list(community_Copycat)
community_color_Copycat=fn.Set_community_number(Copycat, community_Copycat)
color_Copycat = (cm.CMRmap(np.linspace(0, 1, max(community_color_Copycat))))
#%% Histogram Copycat
fn.Hist_plot(Strenght_Copycat[:,1],color_Copycat[5], 'Degree distribution Copycat')
fn.Hist_plot(Betweeness_Centrality_Copycat[:,1],color_Copycat[5], 'Betweeness_Centrality distribution Copycat')
fn.Hist_plot(Closeness_Centrality_Copycat[:,1],color_Copycat[5], 'Closeness_Centrality distribution Copycat')
fn.Hist_plot(Clustering_Copycat[:,1],color_Copycat[5], 'Clustering distribution Copycat') 

#%% Scatter plot copycat


fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Clustering_Copycat[:,1], 'Clustering_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Closeness_Centrality_Copycat[:,1], 'Closeness_Centrality_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Betweeness_Centrality_Copycat[:,1], 'Betweeness_Centrality_Copycat', list(community_color_Copycat.values()))

#_______________-confronto con strenght______

fn.Scatter_plot(Betweeness_Centrality_Copycat[:,1], 'Betweeness_Centrality_Copycat' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(Closeness_Centrality_Copycat[:,1], 'Closeness_Centrality_Copycat' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(Betweeness_Centrality_Copycat[:,1], 'Betweeness_Centrality_Copycat' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))


