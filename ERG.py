# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:21:01 2021

@author: Guido
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import function as fn

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


#%% ER Graph initialization

edge_probability=2*G.number_of_edges()/((len(G)-1)*(len(G)))
ERG=nx.fast_gnp_random_graph(len(G),edge_probability, seed=None, directed=False)
ERG=fn.SuperGraph(ERG)
ERG.Sorted_graph()
ERG.Relable_nodes()

#%% main ERG features



Strenght_ERG=np.array(sorted(list(ERG.degree)))
Closeness_Centrality_ERG=np.array(sorted(list((nx.closeness_centrality(ERG)).items())))
Betweeness_Centrality_ERG=a=np.array(sorted(list((nx.betweenness_centrality(ERG)).items())))
Clustering_ERG=np.array(sorted(list((nx.clustering(ERG)).items())))


community_ERG=nx.algorithms.community.modularity_max.greedy_modularity_communities(ERG)
community_ERG=fn.Unfreeze_into_list(community_ERG)
community_color_ERG=fn.Set_community_number(G, community_ERG)
colors_ERG = (cm.CMRmap(np.linspace(0, 1, max(community_color_ERG))))

#%% Histograms

fn.Hist_plot(Strenght_ERG[:,1],colors_ERG[5], 'Degree_ERG distribution')
fn.Hist_plot(Betweeness_Centrality_ERG[:,1],colors_ERG[5], 'Betweeness_Centrality_ERG distribution')
fn.Hist_plot(Closeness_Centrality_ERG[:,1],colors_ERG[5], 'Closeness_Centrality_ERG distribution')
fn.Hist_plot(Clustering_ERG[:,1],colors_ERG[5], 'Clustering_ERG distribution')

#%% Scatterplot



fn.Scatter_plot(list(community_color_ERG.values()), 'Community' , Strenght_ERG[:,1], 'Degree', list(community_color_ERG.values()))
fn.Scatter_plot(list(community_color_ERG.values()), 'Community' , Clustering_ERG[:,1], 'Clustering', list(community_color_ERG.values()))
fn.Scatter_plot(list(community_color_ERG.values()), 'Community' , Closeness_Centrality_ERG[:,1], 'Closeness_Centrality', list(community_color_ERG.values()))
fn.Scatter_plot(list(community_color_ERG.values()), 'Community' , Betweeness_Centrality_ERG[:,1], 'Betweeness_Centrality', list(community_color_ERG.values()))

#_______________-confronto con strenght______

fn.Scatter_plot(Betweeness_Centrality_ERG[:,1], 'Betweeness_Centrality' , Strenght_ERG[:,1], 'Degree', list(community_color_ERG.values()))
fn.Scatter_plot(Closeness_Centrality_ERG[:,1], 'Closeness_Centrality' , Strenght_ERG[:,1], 'Degree', list(community_color_ERG.values()))
fn.Scatter_plot(Betweeness_Centrality_ERG[:,1], 'Betweeness_Centrality' , Strenght_ERG[:,1], 'Degree', list(community_color_ERG.values()))



   

    



