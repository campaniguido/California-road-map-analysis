import networkx as nx
import numpy as np
import function as fn
import pandas as pd
import matplotlib.cm as cm


#%% file reading
'''You can put here any other file path where your data are located,
   the data have to represent the n edge of your network and should be in a format (n,2) '''
   
file_position='https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt'

file=pd.read_table(file_position)
file=np.array(file)
file=fn.Divide_value(file)
#%%Graph initialization

'''Put here the number of edges you want to consider starting from the first couple'''
number_of_edge=1000
edges=fn.Edge_list(file, number_of_edge)
G=fn.SuperGraph(edges)
G.Sorted_graph()
G.Relable_nodes()
edges=list(G.edges())






#%%COPYCAT network. First step: linking following the distance attachment rule

map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
dct_dist_link=fn.Dct_dist_link(edges,map_dct)
dct_dist=fn.Dct_dist(G=G, map_dct=map_dct)
"The nstep parameter fixes the number of bin of the linking pobability distribution"
nstep=50
step=max(dct_dist_link.values())/nstep

distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
distance_linking_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,distance_frequency)
Copy_map=fn.SuperGraph()
Copy_map.add_nodes_from(list(G.nodes))
Copy_map=fn.Add_edges_from_map(Copy_map, map_dct, distance_linking_probability)

    
    
#%% features COPYMAP   

  
Strenght_Copy_map=np.array(list(Copy_map.degree))
Closeness_Centrality_Copy_map=np.array(list((nx.closeness_centrality(Copy_map)).items()))
Betweeness_Centrality_Copy_map=np.array(list((nx.betweenness_centrality(Copy_map)).items()))
Clustering_Copy_map=np.array(list((nx.clustering(Copy_map)).items()))

community_Copy_map=nx.algorithms.community.modularity_max.greedy_modularity_communities(Copy_map)
community_Copy_map=fn.Unfreeze_into_list(community_Copy_map)
community_color_Copy_map=fn.Set_community_number(Copy_map, community_Copy_map)
color_Copy_map =[cm.CMRmap(0.46),cm.CMRmap(0.08)]
            
#%%   #%%Histogram   Copymap    
'''Here an example to save the plot with a given extention'''
fn.Hist_plot(Strenght_Copy_map[:,1],color_Copy_map[1], 'Degree distribution Copymap', save_fig=True, extention='pdf')

'''Here the same plot but in red is not saved'''
fn.Hist_plot(Strenght_Copy_map[:,1],color_Copy_map[0], 'Degree distribution Copymap', save_fig=False, extention='pdf')

fn.Hist_plot(Betweeness_Centrality_Copy_map[:,1],color_Copy_map[0], 'Betweeness_Centrality distribution Copymap')
fn.Hist_plot(Closeness_Centrality_Copy_map[:,1],color_Copy_map[0], 'Closeness_Centrality distribution Copymap')
fn.Hist_plot(Clustering_Copy_map[:,1],color_Copy_map[0], 'Clustering distribution Copymap') 


#%% Copycat degree correction
max_linking_distance=max(dct_dist_link.values())
Copycat=fn.Copymap_degree_correction(Copy_map, G, map_dct,max_linking_distance, distance_linking_probability, Merge=True)

#%% Features Copycat


Strenght_Copycat=np.array(list(Copycat.degree))
K_Copycat=np.array(list((nx.average_neighbor_degree(Copycat)).items()))
Closeness_Centrality_Copycat=np.array(list((nx.closeness_centrality(Copycat)).items()))
Betweeness_Centrality_Copycat=np.array(list((nx.betweenness_centrality(Copycat)).items()))
Clustering_Copycat=np.array(list((nx.clustering(Copycat)).items()))

community_Copycat=nx.algorithms.community.modularity_max.greedy_modularity_communities(Copycat)
community_Copycat=fn.Unfreeze_into_list(community_Copycat)
community_color_Copycat=fn.Set_community_number(Copycat, community_Copycat)
color_Copycat =[cm.CMRmap(0.46),cm.CMRmap(0.08)]
#%% Histogram Copycat
fn.Hist_plot(Strenght_Copycat[:,1],color_Copycat[0], 'Degree distribution Copycat')
fn.Hist_plot(Betweeness_Centrality_Copycat[:,1],color_Copycat[0], 'Betweeness_Centrality distribution Copycat')
fn.Hist_plot(Closeness_Centrality_Copycat[:,1],color_Copycat[0], 'Closeness_Centrality distribution Copycat')
fn.Hist_plot(Clustering_Copycat[:,1],color_Copycat[0], 'Clustering distribution Copycat') 

#%% Scatter plot copycat


fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Clustering_Copycat[:,1], 'Clustering_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Closeness_Centrality_Copycat[:,1], 'Closeness_Centrality_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(list(community_color_Copycat.values()), 'Community' , Betweeness_Centrality_Copycat[:,1], 'Betweeness_Centrality_Copycat', list(community_color_Copycat.values()))

#_______________-confronto con degree______

fn.Scatter_plot(Betweeness_Centrality_Copycat[:,1], 'Betweeness_Centrality_Copycat' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(Closeness_Centrality_Copycat[:,1], 'Closeness_Centrality_Copycat' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))
fn.Scatter_plot(Betweeness_Centrality_Copycat[:,1], 'Betweeness_Centrality_Copycat' , Strenght_Copycat[:,1], 'Degree_Copycat', list(community_color_Copycat.values()))


