# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:58:33 2022

@author: Guido
"""
import pandas as pd
import function_plot as fplot
import matplotlib.cm as cm
import parameters_Copycat as paramsC



#%% Histogram Copycat

data_Copycat=pd.read_csv(paramsC.path_to_save_data+'\data_Copycat.csv')

color=[cm.CMRmap(0.46)]
fplot.Hist_plot(data_Copycat.Degree,color[0], 'Degree_Copycat distribution',paramsC.file_to_plot, paramsC.save_fig, paramsC.extention)
fplot.Hist_plot(data_Copycat.Betweeness_Centrality,color[0], 'Betweeness_Centrality_Copycat distribution',paramsC.file_to_plot, paramsC.save_fig, paramsC.extention)
fplot.Hist_plot(data_Copycat.Closeness_Centrality,color[0], 'Closeness_Centrality_Copycat distribution',paramsC.file_to_plot, paramsC.save_fig, paramsC.extention)
fplot.Hist_plot(data_Copycat.Clustering,color[0], 'Clustering_Copycat distribution',paramsC.file_to_plot, paramsC.save_fig, paramsC.extention)

#%% Scatterplot


fplot.Scatter_plot(data_Copycat.community, 'Community_Copycat' , data_Copycat.Degree, 'Degree', data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)
fplot.Scatter_plot(data_Copycat.community, 'Community_Copycat' , data_Copycat.Clustering, 'Clustering', data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)
fplot.Scatter_plot(data_Copycat.community, 'Community_Copycat' , data_Copycat.Closeness_Centrality, 'Closeness_Centrality', data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)
fplot.Scatter_plot(data_Copycat.community, 'Community_Copycat' , data_Copycat.Betweeness_Centrality, 'Betweeness_Centrality', data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)

#_______________features vs degree______
fplot.Scatter_plot(data_Copycat.Betweeness_Centrality, 'Betweeness_Centrality Copycat' , data_Copycat.Degree, 'Degree', data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)
fplot.Scatter_plot(data_Copycat.Closeness_Centrality, 'Closeness_Centrality Copycat' , data_Copycat.Degree, 'Degree', data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)
fplot.Scatter_plot(data_Copycat.Betweeness_Centrality, 'Betweeness_Centrality Copycat' ,data_Copycat.Closeness_Centrality, 'Closeness_Centrality' , data_Copycat.community,paramsC.file_to_plot,paramsC.save_fig, paramsC.extention)

