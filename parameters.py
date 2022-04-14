# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:00:48 2022

@author: Guido
"""
import os



file_position='https://raw.githubusercontent.com/campaniguido/Software_and_Computing/main/roadnet-ca.txt'
number_of_edge=5020
name_simulation='road_'+str(number_of_edge)
path_to_save_data=os.getcwd()+'\\'+name_simulation
seed=3


file_to_plot=os.getcwd()+'\\'+name_simulation
save_fig=False
extention='png'
n_step_degree=50
n_step=10



