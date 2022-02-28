# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:42:39 2022

@author: Guido
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:46:39 2021

@author: Guido
"""

#%% Libraries
import networkx as nx
import pandas as pd
import numpy as np
import function as fn
import matplotlib.cm as cm
import matplotlib.pyplot as plt





#%% file reading
file=pd.read_table('https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt')
file=np.array(file)
file=fn.Divide_value(file)

#%% Graph initialization
number_of_edge=5000
edges=fn.Edge_list(file, number_of_edge)
G=fn.SuperGraph(edges)
G.Sorted_graph()
G.Relable_nodes()



#%% Main features
Strenght=np.array(list(G.degree))
Closeness_Centrality=np.array(list((nx.closeness_centrality(G)).items()))
Betweeness_Centrality=np.array(list((nx.betweenness_centrality(G)).items()))
Clustering=np.array(list((nx.clustering(G)).items()))

community=nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
community=fn.Unfreeze_into_list(community)
community_color=fn.Set_community_number(G, community)
colors = (cm.CMRmap(np.linspace(0, 1, max(community_color))))

#%% Network Visualization
Betweeness_Centrality_edges=nx.edge_betweenness_centrality(G)
node_size=np.exp((Betweeness_Centrality[:,1]+1)*100)
width=(np.array(list(Betweeness_Centrality_edges.values()))+0.95)
nx.draw(G,node_size=node_size,font_size=6, node_color=list(community_color.values()),width=width)
plt.title("Road California")
plt.show(G)






#%% Histograms

fn.Hist_plot(Strenght[:,1],colors[5], 'Degree distribution')
fn.Hist_plot(Betweeness_Centrality[:,1],colors[5], 'Betweeness_Centrality distribution')
fn.Hist_plot(Closeness_Centrality[:,1],colors[5], 'Closeness_Centrality distribution')
fn.Hist_plot(Clustering[:,1],colors[5], 'Clustering distribution')

#%% Scatterplot



fn.Scatter_plot(list(community_color.values()), 'Community' , Strenght[:,1], 'Degree', list(community_color.values()))
fn.Scatter_plot(list(community_color.values()), 'Community' , Clustering[:,1], 'Clustering', list(community_color.values()))
fn.Scatter_plot(list(community_color.values()), 'Community' , Closeness_Centrality[:,1], 'Closeness_Centrality', list(community_color.values()))
fn.Scatter_plot(list(community_color.values()), 'Community' , Betweeness_Centrality[:,1], 'Betweeness_Centrality', list(community_color.values()))

#_______________-confronto con strenght______

fn.Scatter_plot(Betweeness_Centrality[:,1], 'Betweeness_Centrality' , Strenght[:,1], 'Degree', list(community_color.values()))
fn.Scatter_plot(Closeness_Centrality[:,1], 'Closeness_Centrality' , Strenght[:,1], 'Degree', list(community_color.values()))
fn.Scatter_plot(Betweeness_Centrality[:,1], 'Betweeness_Centrality' , Strenght[:,1], 'Degree', list(community_color.values()))



   

    

        
#%% Main features size-evolution 

degree_size, degree_ratio_size_evolution,degree_mean=fn.Size_evolution(G,len(G)**0.5,'degree')
fn.Feature_ratio_evolution(degree_size,degree_ratio_size_evolution, 'degree')
    
BC_size, BC_time_evolution,BC_mean=fn.Size_evolution(G,np.log(len(G))**2.8,'betweenness_centrality')
fn.Feature_mean_evolution(BC_size,BC_mean, 'BC_mean', save_fig=False)
fn.Feature_cumulative_evolution(BC_time_evolution, 'Betweeness centrality')


Clustering_size, Clustering_time_evolution,Clustering_mean=fn.Size_evolution(G,np.log(len(G))**2.8,'clustering')  
fn.Feature_mean_evolution(Clustering_size,Clustering_mean, 'Clustering_mean')
fn.Feature_cumulative_evolution(Clustering_time_evolution, 'Clustering')


CC_size, CC_time_evolution,CC_mean=fn.Size_evolution(G,np.log(len(G))**2.8,'closeness_centrality')   
fn.Feature_mean_evolution(CC_size,CC_mean, 'CC_mean', save_fig=False)
fn.Feature_cumulative_evolution(CC_time_evolution, 'Closeness centrality')

