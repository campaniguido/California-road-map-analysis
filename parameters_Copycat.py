# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:54:33 2022

@author: Guido
"""

import os

seed=3

#path where the data are stored 
file_position='https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt'
#numer of edges of the simulated network
number_of_edge=5000
#name of the simulation file
name_simulation='Copycat'+str(number_of_edge)
#path where the data will be stored
path_to_save_data=os.getcwd()+'\\'+name_simulation


#path where the simulation data to plot are stored and where it will save the plot
file_to_plot=os.getcwd()+'\\'+name_simulation

#bool-> save_fig=True it saves the plot in the directory file_to_plot ||| save_fig=False it just shows the plot
save_fig=True
extention='png'
#number of bins of the distance distribution
nstep=50


