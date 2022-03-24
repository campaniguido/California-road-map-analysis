# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 23:23:32 2022

@author: Guido
"""

#%% Libraries
import pandas as pd
import numpy as np
import function as fn
import matplotlib.cm as cm
import parameters as params

#%% Histograms
color=[cm.CMRmap(0.46),cm.CMRmap(0.08)]
data_road=pd.read_csv(params.path_to_save_data+'\data_road.csv')
fn.Hist_plot(data_road.Degree,color[0],'Degree distribution',params.save_fig, params.extention)
fn.Hist_plot(data_road.Betweeness_Centrality,color[0], 'Betweeness_Centrality distribution',params.save_fig, params.extention)
fn.Hist_plot(data_road.Closeness_Centrality,color[0], 'Closeness_Centrality distribution',params.save_fig, params.extention)
fn.Hist_plot(data_road.Clustering,color[0], 'Clustering distribution',params.save_fig, params.extention)

#%% Scatterplot

fn.Scatter_plot(data_road.community, 'Community' , data_road.Degree, 'Degree', data_road.community,params.save_fig, params.extention)
fn.Scatter_plot(data_road.community, 'Community' , data_road.Clustering, 'Clustering', data_road.community,params.save_fig, params.extention)
fn.Scatter_plot(data_road.community, 'Community' , data_road.Closeness_Centrality, 'Closeness_Centrality', data_road.community,params.save_fig, params.extention)
fn.Scatter_plot(data_road.community, 'Community' , data_road.Betweeness_Centrality, 'Betweeness_Centrality', data_road.community,params.save_fig, params.extention)

#_______________features vs degree______
fn.Scatter_plot(data_road.Betweeness_Centrality, 'Betweeness_Centrality' , data_road.Degree, 'Degree', data_road.community,params.save_fig, params.extention)
fn.Scatter_plot(data_road.Closeness_Centrality, 'Closeness_Centrality' , data_road.Degree, 'Degree', data_road.community,params.save_fig, params.extention)
fn.Scatter_plot(data_road.Betweeness_Centrality, 'Betweeness_Centrality' , data_road.Degree, 'Degree', data_road.community,params.save_fig, params.extention)



   

    

        
#%% Main features size-evolution 

degree_evolution=np.array(pd.read_csv(params.path_to_save_data+'\degree_evolution.csv'))
fn.Feature_ratio_evolution(degree_evolution[:,0],degree_evolution[:,4:], 'degree',params.save_fig, params.extention)



 
BC_evolution=np.array(pd.read_csv(params.path_to_save_data+'\BC_evolution.csv'))
fn.Feature_mean_evolution(BC_evolution[:,1],BC_evolution[:,2:4], 'Betweenness',params.save_fig, params.extention)
fn.Feature_cumulative_evolution(BC_evolution[:,4:], 'Betweeness centrality',params.save_fig, params.extention)

Clustering_size, Clustering_time_evolution,Clustering_mean=fn.Size_evolution(G,step,'clustering')  
fn.Feature_mean_evolution(Clustering_size,Clustering_mean, 'Clustering',params.save_fig, params.extention)
fn.Feature_cumulative_evolution(Clustering_time_evolution, 'Clustering',params.save_fig, params.extention)


CC_size, CC_time_evolution,CC_mean=fn.Size_evolution(G,step,'closeness_centrality')   
fn.Feature_mean_evolution(CC_size,CC_mean, 'Closeness',params.save_fig, params.extention)
fn.Feature_cumulative_evolution(CC_time_evolution, 'Closeness centrality',params.save_fig, params.extention)
