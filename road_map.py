# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:42:39 2022

@author: Guido Campani
""" 

#%% Libraries
import networkx as nx
import pandas as pd
import numpy as np
import function as fn
import parameters as params







#%% file reading

file_position=params.file_position
file=pd.read_table(file_position)
file_no_nan_raw=fn.Erase_nan_row(file)
edges_info=fn.Divide_value(file_no_nan_raw)


#%% Graph initialization


number_of_edge=params.number_of_edge
edges=fn.Edge_list(edges_info, number_of_edge)
G=fn.SuperGraph(edges)
G.Sorted_graph()
G.Relable_nodes()



#%% Main features
Degree=np.array(list(G.degree))
Closeness_Centrality=np.array(list((nx.closeness_centrality(G)).items()))
Betweeness_Centrality=np.array(list((nx.betweenness_centrality(G)).items()))
Clustering=np.array(list((nx.clustering(G)).items()))

community=nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
community=fn.Unfreeze_into_list(community)
community_color=fn.Set_community_number(G, community)


data={'Degree':Degree[:,1],'Closeness_Centrality':Closeness_Centrality[:,1],'Betweeness_Centrality':Betweeness_Centrality[:,1],'Clustering':Clustering[:,1],'community':list(community_color.values())}
data=pd.DataFrame(data)
data.to_csv(r''+params.path_to_save_data + '\data_road.csv')

#%% Main features size-evolution 
step=int(len(G.edges)/params.n_step_degree)
degree_size, degree_ratio_size_evolution,degree_mean=fn.Size_evolution(G,step,'degree')

evolution=pd.DataFrame(degree_ratio_size_evolution,columns=list(range(len(G.Degree_ratio()))))
mean=pd.DataFrame(degree_mean,columns=['mean','std'])
size=pd.DataFrame(degree_size,columns=['size'])

degree_evolution=pd.concat((size,mean,evolution), axis=1)
degree_evolution.to_csv(r''+params.path_to_save_data + '\degree_evolution.csv')


step=int(len(G.edges)/params.n_step) 

BC_size, BC_time_evolution,BC_mean=fn.Size_evolution(G,step,'betweenness_centrality')
evolution=pd.DataFrame(BC_time_evolution)
size=pd.DataFrame(BC_size,columns=['size'])
mean=pd.DataFrame(BC_mean,columns=['mean','std'])
BC_evolution=pd.concat((size,mean,evolution),axis=1)
BC_evolution.to_csv(r''+params.path_to_save_data + '\BC_evolution.csv')


Clustering_size, Clustering_time_evolution,Clustering_mean=fn.Size_evolution(G,step,'clustering')  
evolution=pd.DataFrame(Clustering_time_evolution)
size=pd.DataFrame(Clustering_size,columns=['size'])
mean=pd.DataFrame(Clustering_mean,columns=['mean','std'])
Clustering_evolution=pd.concat((size,mean,evolution),axis=1)
Clustering_evolution.to_csv(r''+params.path_to_save_data + '\Clustering_evolution.csv')


CC_size, CC_time_evolution,CC_mean=fn.Size_evolution(G,step,'closeness_centrality')
evolution=pd.DataFrame(CC_time_evolution)
size=pd.DataFrame(CC_size,columns=['size'])
mean=pd.DataFrame(CC_mean,columns=['mean','std'])
CC_evolution=pd.concat((size,mean,evolution),axis=1)
CC_evolution.to_csv(r''+params.path_to_save_data + '\CC_evolution.csv')
 

#%% ERG Graph initialization

edge_probability=2*G.number_of_edges()/((len(G)-1)*(len(G)))
ERG=nx.fast_gnp_random_graph(len(G),edge_probability, seed=None, directed=False)
ERG=fn.SuperGraph(ERG)
ERG.Sorted_graph()
ERG.Relable_nodes()

#%% main ERG features

Degree_ERG=np.array(sorted(list(ERG.degree)))
Closeness_Centrality_ERG=np.array(sorted(list((nx.closeness_centrality(ERG)).items())))
Betweeness_Centrality_ERG=a=np.array(sorted(list((nx.betweenness_centrality(ERG)).items())))
Clustering_ERG=np.array(sorted(list((nx.clustering(ERG)).items())))

community_ERG=nx.algorithms.community.modularity_max.greedy_modularity_communities(ERG)
community_ERG=fn.Unfreeze_into_list(community_ERG)
community_color_ERG=fn.Set_community_number(G, community_ERG)

data_ERG={'Degree':Degree_ERG[:,1],'Closeness_Centrality':Closeness_Centrality_ERG[:,1],'Betweeness_Centrality':Betweeness_Centrality_ERG[:,1],'Clustering':Clustering_ERG[:,1],'community':list(community_color_ERG.values())}
data_ERG=pd.DataFrame(data_ERG)
data_ERG.to_csv(r''+params.path_to_save_data + '\data_ERG.csv')
