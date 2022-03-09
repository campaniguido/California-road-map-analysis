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
import matplotlib.cm as cm






#%% file reading
'''You can put here any other file path where your data are located,
   the data have to represent the n edge of your network and should be in a format (n,2) '''
   
file_position='https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt'
file=pd.read_table(file_position)
file=np.array(file)
file=fn.Divide_value(file)

#%% Graph initialization

'''Put here the number of edge you want to consider starting from the first couple'''
number_of_edge=5000
edges=fn.Edge_list(file, number_of_edge)
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
color=[cm.CMRmap(0.46),cm.CMRmap(0.08)]


#%% Histograms
'''Here an example to save the plot with a given extention'''
fn.Hist_plot(Degree[:,1],color[1], 'Degree distribution',True,'png')

'''Here the same plot but in red is not saved'''
fn.Hist_plot(Degree[:,1],color[0], 'Degree distribution')

fn.Hist_plot(Betweeness_Centrality[:,1],color[0], 'Betweeness_Centrality distribution')
fn.Hist_plot(Closeness_Centrality[:,1],color[0], 'Closeness_Centrality distribution')
fn.Hist_plot(Clustering[:,1],color[0], 'Clustering distribution')

#%% Scatterplot


'''Here an example to save the plot with a given extention'''
fn.Scatter_plot(list(community_color.values()), 'Community' , Degree[:,1], 'Degree', list(community_color.values()),True,'pdf')
'''Here the same plot is not saved'''
fn.Scatter_plot(list(community_color.values()), 'Community' , Degree[:,1], 'Degree', list(community_color.values()))

fn.Scatter_plot(list(community_color.values()), 'Community' , Clustering[:,1], 'Clustering', list(community_color.values()))
fn.Scatter_plot(list(community_color.values()), 'Community' , Closeness_Centrality[:,1], 'Closeness_Centrality', list(community_color.values()))
fn.Scatter_plot(list(community_color.values()), 'Community' , Betweeness_Centrality[:,1], 'Betweeness_Centrality', list(community_color.values()))

#_______________features vs degree______
fn.Scatter_plot(Betweeness_Centrality[:,1], 'Betweeness_Centrality' , Degree[:,1], 'Degree', list(community_color.values()),True,'png')
fn.Scatter_plot(Closeness_Centrality[:,1], 'Closeness_Centrality' , Degree[:,1], 'Degree', list(community_color.values()))
fn.Scatter_plot(Betweeness_Centrality[:,1], 'Betweeness_Centrality' , Degree[:,1], 'Degree', list(community_color.values()))



   

    

        
#%% Main features size-evolution 

'''the step parameter can be changed if you want to consider more or less point in the size increasing process'''
step=int(len(G.edges)/50)

degree_size, degree_ratio_size_evolution,degree_mean=fn.Size_evolution(G,step,'degree')
fn.Feature_ratio_evolution(degree_size,degree_ratio_size_evolution, 'degree',save_fig=False)

'''the step parameter can be changed if you want to consider more or less point in the size increasing process'''
step=int(len(G.edges)/10)   
 
BC_size, BC_time_evolution,BC_mean=fn.Size_evolution(G,step,'betweenness_centrality')
fn.Feature_mean_evolution(BC_size,BC_mean, 'Betweenness', save_fig=False)
fn.Feature_cumulative_evolution(BC_time_evolution, 'Betweeness centrality')

'''the step parameter can be changed if you want to consider more or less point in the size increasing process'''
step=int(len(G.edges)/10)

Clustering_size, Clustering_time_evolution,Clustering_mean=fn.Size_evolution(G,step,'clustering')  
fn.Feature_mean_evolution(Clustering_size,Clustering_mean, 'Clustering',save_fig=False)
fn.Feature_cumulative_evolution(Clustering_time_evolution, 'Clustering',save_fig=False)

'''the step parameter can be changed if you want to consider more or less point in the size increasing process'''
step=int(len(G.edges)/10)

CC_size, CC_time_evolution,CC_mean=fn.Size_evolution(G,step,'closeness_centrality')   
fn.Feature_mean_evolution(CC_size,CC_mean, 'Closeness', save_fig=False)
fn.Feature_cumulative_evolution(CC_time_evolution, 'Closeness centrality',save_fig=False)

