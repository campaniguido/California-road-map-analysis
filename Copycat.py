import networkx as nx
import numpy as np
import function as fn
import pandas as pd
#import parameters_Copycat as paramsC
import pp as paramsC
import random as rn

#%%seed

seed=paramsC.seed
rn.seed(seed)

#%% New directory creation
fn.Directory_creation(paramsC.name_simulation)
#%% file reading

file_position=paramsC.file_position
file=pd.read_table(file_position)
file_no_nan_raw=fn.Erase_nan_row(file)
edges_info=fn.Divide_value(file_no_nan_raw)


#%% Graph initialization


number_of_edge=paramsC.number_of_edge
edges=fn.Edge_list(edges_info, number_of_edge)
G=fn.SuperGraph(edges)
G.Sorted_graph()
G.Relable_nodes()
edges=list(G.edges())





#%%COPYCAT network. First step: linking following the distance attachment rule

map_dct=nx.spring_layout(G, dim=2, seed=seed)
dct_dist_link=fn.Dct_dist_link(edges,map_dct)
dct_dist=fn.Dct_dist(G=G, map_dct=map_dct)
step=max(dct_dist_link.values())/paramsC.nstep
distance_frequency=fn.Node_distance_frequency(dct_dist,paramsC.nstep,step)       
distance_link_frequency=fn.Node_distance_frequency(dct_dist_link,paramsC.nstep,step) 

distance_linking_probability=fn.Conditional_probability(distance_link_frequency,step,distance_frequency)
Copy_map=fn.SuperGraph()
Copy_map.add_nodes_from(list(G.nodes))
fn.Add_edges_from_map(Copy_map, dct_dist, distance_linking_probability)


#%% Copycat degree correction
max_linking_distance=max(dct_dist_link.values())
Copycat=fn.Copymap_degree_correction(Copy_map, G, map_dct,max_linking_distance, distance_linking_probability, Merge=True)

#%% Features Copycat


Degree_Copycat=np.array(list(Copycat.degree))
Closeness_Centrality_Copycat=np.array(list((nx.closeness_centrality(Copycat)).items()))
Betweeness_Centrality_Copycat=np.array(list((nx.betweenness_centrality(Copycat)).items()))
Clustering_Copycat=np.array(list((nx.clustering(Copycat)).items()))

community_Copycat_frozen=nx.algorithms.community.modularity_max.greedy_modularity_communities(Copycat)
community_Copycat=fn.Unfreeze_into_list(community_Copycat_frozen)
community_color_Copycat=fn.Set_community_number(Copycat, community_Copycat)




data_Copycat={'Degree':Degree_Copycat[:,1],'Closeness_Centrality':Closeness_Centrality_Copycat[:,1],'Betweeness_Centrality':Betweeness_Centrality_Copycat[:,1],'Clustering':Clustering_Copycat[:,1],'community':list(community_color_Copycat.values())}
data_Copycat=pd.DataFrame(data_Copycat)
data_Copycat.to_csv(r''+paramsC.path_to_save_data + '\data_Copycat.csv')



