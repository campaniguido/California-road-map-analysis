# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:58:33 2022

@author: Guido
"""
import pandas as pd
import function as fn
import matplotlib.cm as cm
import parameters as params

#%% Histogram Copycat

data_Copycat=pd.read_csv(params.path_to_save_data+'\data_Copycat.csv')

color=[cm.CMRmap(0.46),cm.CMRmap(0.08)]
fn.Hist_plot(data_Copycat.Degree,color[1], 'Degree_Copycat distribution', params.save_fig, params.extention)
fn.Hist_plot(data_Copycat.Betweeness_Centrality,color[1], 'Betweeness_Centrality_Copycat distribution', params.save_fig, params.extention)
fn.Hist_plot(data_Copycat.Closeness_Centrality,color[1], 'Closeness_Centrality_Copycat distribution', params.save_fig, params.extention)
fn.Hist_plot(data_Copycat.Clustering,color[1], 'Clustering_Copycat distribution', params.save_fig, params.extention)

#%% Scatterplot


fn.Scatter_plot(data_Copycat.community, 'Community' , data_Copycat.Degree, 'Degree', data_Copycat.community,params.save_fig, params.extention)
fn.Scatter_plot(data_Copycat.community, 'Community' , data_Copycat.Clustering, 'Clustering', data_Copycat.community,params.save_fig, params.extention)
fn.Scatter_plot(data_Copycat.community, 'Community' , data_Copycat.Closeness_Centrality, 'Closeness_Centrality', data_Copycat.community,params.save_fig, params.extention)
fn.Scatter_plot(data_Copycat.community, 'Community' , data_Copycat.Betweeness_Centrality, 'Betweeness_Centrality', data_Copycat.community,params.save_fig, params.extention)

#_______________features vs degree______
fn.Scatter_plot(data_Copycat.Betweeness_Centrality, 'Betweeness_Centrality' , data_Copycat.Degree, 'Degree', data_Copycat.community,params.save_fig, params.extention)
fn.Scatter_plot(data_Copycat.Closeness_Centrality, 'Closeness_Centrality' , data_Copycat.Degree, 'Degree', data_Copycat.community,params.save_fig, params.extention)
fn.Scatter_plot(data_Copycat.Betweeness_Centrality, 'Betweeness_Centrality' , data_Copycat.Degree, 'Degree', data_Copycat.community,params.save_fig, params.extention)

