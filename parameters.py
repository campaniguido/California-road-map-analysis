# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:00:48 2022

@author: Guido
"""
import os

#%% Simulation parameters
seed=3

#path where the data, related to the linked nodes, are stored 
file_position='https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt'
#numer of edges to take into account
number_of_edge=5028
#name of the simulation file
name_simulation='road_'+str(number_of_edge)+'_'+str(seed)
#path where the simulation data will be stored
path_to_save_data=os.getcwd()+'\\'+name_simulation

#number of steps in the size evoution simultation
n_step_degree=50
n_step=10

#%%Plot parameters
#path where the simulation data to plot are stored and where it will save the plot
file_to_plot=os.getcwd()+'\\'+name_simulation
#bool-> save_fig=True it saves the plot in the directory file_to_plot ||| save_fig=False it just shows the plot
save_fig=True
#extention of the plot files
extention='png'




