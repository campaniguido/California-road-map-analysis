# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:05:25 2022

@author: Guido
"""

import numpy as np
import function as fn
import networkx as nx
import pytest


#%%Ct  test sorted_graph (2)
def test_sorted_graph_nodes():
    '''It builds a graph in which the nodes are in a random order and
       it tests if the function sorts the nodes in ascending way

    '''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[5,4],[8,4],[5,6],[14,14]])
    G.Sorted_graph()
    assert list(G.nodes)[0]==1
    assert list(G.nodes)[1]==3
    assert list(G.nodes)[2]==4
    assert list(G.nodes)[3]==5
    assert list(G.nodes)[4]==6
    assert list(G.nodes)[5]==8
    assert list(G.nodes)[6]==14   

def test_sorted_graph_edges():
    
    '''It builds a graph in which the nodes are in a random order and
       it tests if the function sorts the edges in ascending way following the first element of the edege

    '''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[5,4],[8,4],[5,6],[14,14]])
    G.Sorted_graph()
    assert list(G.edges)[0]==(1,8)
    assert list(G.edges)[1]==(3,8)
    assert list(G.edges)[2]==(4,5)
    assert list(G.edges)[3]==(4,8)
    assert list(G.edges)[4]==(5,6)
    assert list(G.edges)[5]==(14,14)
     

#%%Ct  test_Relable_nodes


def test_Relable_nodes_len():
    '''It builds a graph in which the nodes are in a random order and
       it tests the len of the graph is conserved also after the application of the function'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    LEN=len(G)
    G.Relable_nodes()
    assert LEN==len(G)
    

def test_Relable_nodes_sorted():
    '''It builds a graph in which the nodes are in a random order and
       it tests if the function sorts the nodes in ascending way

    '''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[5,4],[8,4],[5,6],[14,14]])
    G.Relable_nodes()
    assert list(G.nodes)[0]==0
    assert list(G.nodes)[1]==1
    assert list(G.nodes)[2]==2
    assert list(G.nodes)[3]==3
    assert list(G.nodes)[4]==4
    assert list(G.nodes)[5]==5
    assert list(G.nodes)[6]==6 
    
def test_Relable_nodes_sorted_abstract():
    '''It builds a graph in which the nodes are in a random order and
       it tests if the function sorts the nodes in ascending way

    '''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    G.Relable_nodes()
    nodes=list(G.nodes())
    for i in range(1,len(G)):
        assert nodes[i-1]<nodes[i]
        
def test_Relable_nodes_no_hole():
    '''It builds a graph in which the nodes are in a random order and
       it tests there is no jump in the numbers  labels'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    G.Relable_nodes()
    nodes=list(G.nodes())
    assert nodes[0]==0
    assert nodes[1]==1
    assert nodes[2]==2
    assert nodes[3]==3
    assert nodes[4]==4
    assert nodes[5]==5
    assert nodes[6]==6

        
def test_Relable_nodes_corrispondence():
    '''It builds a graph in which the nodes are in a random order and
        it looks if changing the name of the labels the neighbours node are still the same '''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    G.Relable_nodes()
    assert list(G.neighbors(0))==[5]
    assert list(G.neighbors(1))==[5]
    assert list(G.neighbors(2))==[5, 3]
    assert list(G.neighbors(3))==[2, 4]
    assert list(G.neighbors(4))==[3]
    assert list(G.neighbors(5))==[0, 1, 2]
    assert list(G.neighbors(6))==[6]
    
#%%Ct  test_Degree_dct (5)

def test_Degree_list_corrispondence():
    '''It builds a graph in which the nodes are in a random order and
       it verifies if  the function Degree_dct() builds a dictionary in which
       at each degree values is associeted the nodes with that degree value.'''
       
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=G.Degree_dct()
    keys=list(strenght_dct.keys())
    '''because node 2 it is also connected with its self
    and the degree function count it as 2 links'''
    assert G.degree(2)-2==keys[0]
    assert G.degree(3)==keys[1]
    assert G.degree(6)==keys[1]
    assert G.degree(8)==keys[1]
    assert G.degree(9)==keys[1]
    assert G.degree(10)==keys[1]
    assert G.degree(5)==keys[2]
    assert G.degree(4)==keys[2]
    assert G.degree(7)==keys[2]
    assert G.degree(1)==keys[5]


def test_Degree_list_elements():
    '''It builds a graph in which the nodes and links are in a random order
       The Degree_dct function creates a dictionary and
       it looks if the values of the dictionary are the same of the of the graph nodes'''
       
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=G.Degree_dct()
    values=[]
    for i in strenght_dct:
        values+=list(strenght_dct[i])
    assert sorted(values)==list(range(1,11))
    
def test_Degree_list_max_degree():
    '''It builds a graph in which the nodes and links are in a random order
       The Degree_dct function creates a dictionary and 
       it verify the highest key value is equal to the maximum degree of the graph'''
       
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=G.Degree_dct()
    keys=list(strenght_dct.keys())
    assert max(keys)==5
    
def test_Degree_list_selfconnected_nodes():
    
    '''It builds a a graph in wich some of the node are connected
       only to theirselves or to theriselves and to another node
       Then it looks if the function Degree_dct does not count as a link
       the node its self in the case of an selfconnected node'''
       
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[3,3]])
    strenght_dct=G.Degree_dct()
    assert strenght_dct[0]==[2]
    assert strenght_dct[1]==[1,3]
    
def test_empty_high_key():
    '''It builds a graph with a node connected to its self and three nodes completed connected
       Then the function Degree_dct generates a dictionary, it verifies that
       the len of last key values are grater than zero '''
    G=fn.SuperGraph()
    G.add_edges_from([[1,1],[2,3],[3,4],[2,4]])
    strenght_dct=G.Degree_dct()
    max_key=max(list(strenght_dct.keys()))
    assert len(strenght_dct[max_key])!=0
    
#%%Ct   test_Degree_ratio (4)
def test_Degree_ratio_length():
    '''It builds a graph in which the nodes and links are in a random order
       It tests the length of the graph is conserved also after the application 
       of the function Degree_ratio()'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degree_ratio=G.Degree_ratio()
    assert len(degree_ratio)==6
    
def test_Degree_ratio_I_axiom():
    '''It builds a graph in which the nodes and links are in a random order
      Then the Degree_ratio() provides a frequency discrete distribution.
      This distribution must obbeys verifies the probability I° axiom:
      all the frequency must be equal or grater than zero'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degree_ratio=G.Degree_ratio()
    for i in degree_ratio:
        assert i>=0
        
def test_Degree_ratio_II_axiom():
    '''It builds a graph in which the nodes and links are in a random order
      Then the Degree_ratio() provides a frequency discrete distribution.
      This distribution must obbeys verifies the probability II° axiom:
       the sum of all the frequency must be equal to one'''
       
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degree_ratio=G.Degree_ratio()
    assert 0.99999<sum(degree_ratio)<1
    
def test_Degree_ratio_III_axiom():
    '''It builds a graph in which the nodes and links are in a random order
      Then the Degree_ratio() provides a frequency discrete distribution.
      This distribution must obbeys verifies the probability III° axiom:
      The frequency of the union of more indipendent events is equal to the sum
      the frequency of the events '''
    
    G=fn.SuperGraph()
    G.add_edges_from([[0,0],[1,2],[2,3]])
    degree_ratio=G.Degree_ratio()
    assert degree_ratio[0]+degree_ratio[1]==3/4
    assert degree_ratio[0]+degree_ratio[2]==1/2
    assert degree_ratio[1]+degree_ratio[2]==3/4

            


#%%  tests Divide value (5)
  
def test_divide_value():
    '''The dataset file represents couples of numbers, but in the row 1 and 2 
       the couples of number are not in two different columns. The tests verify that
       Divide_value puts the two numbers in the proper way, so one number per cel'''
       
    file=[['0 ','134.0'],['45643 3456',np.nan],[np.nan,'34 5']]
    file=fn.Divide_value(file)
    assert list(file[1])==['45643', '3456']
    assert list(file[2])==['34', '5']

def test_divide_value_one_space():
    '''Giving a data set, with dimension (5,2) and with some of the data written as str
       variable with space (" ") in, it tests if after the application of the function
        Divide_value no spaces are left.'''
        
    file=[['0 ','134.0'],['45643 3456',np.nan],[np.nan,'34 5'],[np.nan,np.nan],[3,4.0]]
    file=fn.Divide_value(file)
    for i in range(2):
        for j in range(len(file)):
            if type(file[j,i])==str:
                for k in range(len(file[j,i])):
                    assert file[j,i][k]!=' '

                    
def test_divide_value_more_space():
    '''Giving a data set, with dimension (5,2) and with some of the data written as str
       variable with more than one space (" ") in, it tests if after the application of the function
        Divide_value no spaces are left.'''
    
    file=[[0,134.0],['45643            3456',np.nan],[np.nan,'34     5'],[3,4.0]]
    file=fn.Divide_value(file)
    for i in range(2):
        for j in range(len(file)):
            if type(file[j,i])==str:
                for k in range(len(file[j,i])):
                    
                    assert file[j,i][k]!=' '
                    
def test_divide_value_nothing_to_divide():
    '''In the datasete file the first element file[0,0] ends with a space but there
    is not other number after the space to put in the next column (file[0,1]), indeed,
    the next column it is occupied by a number. It is expected that
    the function Divide_value will conserve the 2.0 value'''
    file=[['1 ', 2.0],['3', '4'],['5', '6']]
    file=fn.Divide_value(file)
    assert file[0,1]==2.0                   
                    

def test_divide_value_shape():
    '''The dimension of variable file given to the function Divide_value is
    of the type (n,3) and not (n,2) an Exceotion is eexpected to raise'''
    file=[[0,134.0,4],['45643 3456',np.nan,5],[np.nan,'34 5',7],[3,4.0,8]]
    with pytest.raises(Exception):
        fn.Divide_value(file)

#%%  test Erase_nan_row

                    
def test_erase_nan_row_leght():
    ''' It test if the file len after the application of  Erase_nan_row is corrected"
    '''
    file=[['0','134.0'],['45643',56.7],[np.nan,np.nan],[3,4.0]]
    file_corrected=fn.Erase_nan_row(file)
    assert len(file_corrected)==3
    
    
def test_erase_nan_row_no_nan(): 
    '''Given a data set file with multiple rows full filled with nan it verifies the
    Erase_nan_row gives back the same file but with no nan row '''
    file=[['0','134.0'],[np.nan,np.nan],['45643',56.7],[np.nan,np.nan],[3,4.0],[np.nan,np.nan]]
    file_corrected=fn.Erase_nan_row(file)
    assert list(file_corrected[0])==['0','134.0']
    assert list(file_corrected[1])==['45643',56.7]
    assert list(file_corrected[2])==[3,4.0]
    assert len(file_corrected)==3

                    

            

#%%  tests Edge_list (5)


def test_edge_list_int():
    '''Given a file of couples of numbers expressed in different ways, 
       it tests if all the elements composing the Edge_list output are integers'''
       
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    edges=fn.Edge_list(file,3)
    for j in range(len(edges)):
        for k in range(2):
            assert type(edges[j][k])==int
                    
def test_edge_list_length():
    '''Given a file of couples of numbers expressed in different ways, 
       it looks if the output length of the the function is the one expressed
       by the variable number of edges'''
       
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=3
    edges=fn.Edge_list(file,number_of_edges)
    assert len(edges)==number_of_edges
    
def test_edge_list_shape():
    '''Given a file of couples of numbers expressed in different ways, 
       it looks if the output of the function has a shape equal to n*2'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=2
    edges=fn.Edge_list(file,number_of_edges)
    for j in range(len(edges)):
        assert len(edges[j])==2
#Negative test
def test_edge_list_too_long():
    '''Given a file of couples of numbers expressed in different ways,
    it tests that the function raise an Exception when the number_of_edges is
    major than the length of the file'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=5
    with pytest.raises(Exception):
        fn.Edge_list(file,number_of_edges)
        
def test_edge_list_input_shape():
    '''Given a file of couples of numbers expressed in different ways,
    it tests that the function raise an Exception when the shape of the input
    is not n*2 '''    
    file=[[0,134.0,88],['45643',' 3456',7],[3,4.0,'33']]
    number_of_edges=5
    with pytest.raises(Exception):
        fn.Edge_list(file,number_of_edges)

def test_edge_list_order():
    '''Given a file of couples of numbers expressed in different ways and in random order,
    it tests the output has an aschending order in the first element of the couple'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0],[1,2]]
    number_of_edges=4
    edges=fn.Edge_list(file,number_of_edges)
    assert edges[0]==[0, 134]
    assert edges[1]==[1, 2]
    assert edges[2]==[3, 4]
    assert edges[3]==[45643, 3456]
    



  
#%%  tests Unfreeze_into_list(2)

def test_Unfreeze_into_list_is_a_list_1():
    '''It builds a list of frozenset and tuple as an input for the function,
     it verifies the output items are all lists'''
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    list_=[A,B,C,D]
    list_unfrozen=fn.Unfreeze_into_list(list_)
    for i in range(len(list_unfrozen)):
        assert(type(list_unfrozen[i])==list)

def test_Unfreeze_into_list_is_a_list_2():
    '''It builds a dictionary of frozenset and tuple as an input for the function,
     it verifies the output items are all lists'''
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    dct_={0:A,1:B,2:C,3:D}
    dct_unfrozen=fn.Unfreeze_into_list(dct_)
    for i in range(len(dct_unfrozen)):
        assert(type(dct_unfrozen[i])==list)


#%%   test_Set_community_number (6)

def test_Set_community_number_corrispondence(): 
    '''It builds a list of list of elements belonging to the same community,
    then it builds a graph  using all the elements of the list with random links.
    It apllies the Set_community_number and verifies the output dictionary of the function
    match correctly all the nodes of the graph with the community they belong to'''
    
    community=[[1,25,6,9],[5,14],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,25],[2,2],[5,6],[4,8],[9,1],[14,1],[1,5],[4,7],[1,7]])
    community_number=fn.Set_community_number(G, community)
    assert community_number[1]==0
    assert community_number[25]==0
    assert community_number[6]==0
    assert community_number[9]==0
    assert community_number[5]==1
    assert community_number[14]==1
    assert community_number[7]==2
    assert community_number[2]==3
    assert community_number[4]==3
    assert community_number[8]==3
    
        
def test_Set_community_number_length():
    '''It builds a list of list of elements belonging to the same community,
    then it builds a graph  using all the elements of the list with random links.
    It apllies the Set_community_number and it tests the length of the output is
    the same of the one of the Graph'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=fn.Set_community_number(G, community)
    assert len(community_number)==len(G)
    

        
def test_Set_community_number_doubble():
    '''It builds a list of list of elements belonging to the same community,
    then it builds a graph  using all the elements of the list with random links.
    It apllies the Set_community_number and it tests if the node are
    in just one community. e.g.:the four is in two different communities'''
    
    community=[[1,3,6,9],[5,10],[7,4],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(ValueError):
        fn.Set_community_number(G, community)

def test_Set_community_number_missing_number():
    '''It builds a list of list of elements belonging to the same community,
    then it builds a graph  using all the elements of the list with random links.
    It apllies the Set_community_number and it tests if all the node are in the community variable. e.g.:the eight is missing'''
    community=[[1,3,6,9],[5,10],[7],[2,4]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(ValueError):
        fn.Set_community_number(G, community)

def test_Set_community_number_extra_number():
    '''It builds a list of list of elements belonging to the same community,
    then it builds a graph  using all the elements of the list with random links.
    It apllies the Set_community_number and it tests if the community has only the nodes labels of the graph. e.g.:the 999 is not a node number'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        fn.Set_community_number(G, community)

def test_Set_community_number_all_wrong():
    '''It builds a list of list of elements belonging to the same community,
    then it builds a graph  using all the elements of the list with random links.
    It apllies the Set_community_number and it tests tests the behaviours for the three previous mistakes'''
    community=[[1,3,6,4],[5,10],[7],[2,4,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(ValueError):
        fn.Set_community_number(G, community)
            
    
#%%   test_fill_with_zeros (3)

def test_fill_with_zeros_length_conservation():
    '''It builds a list of list and it verifies that after the application of Fill_with_zeros
    the length is the same
    '''
    
    a=[[1,2],[3,5,6,7],[4,5,6]]
    fn.Fill_with_zeros(a)
    assert len(a)==3
    
def test_fill_with_zeros_length_variation():
    '''It builds a list of list and it verifies that after the application of Fill_with_zeros
    the length of each elements is the same of the longest one
    '''
    
    a=[[1,2],[3,5,6,7],[4,5,6]]
    fn.Fill_with_zeros(a)
    assert len(a[0])==4
    assert len(a[1])==4 
    assert len(a[2])==4
    
def test_fill_with_zeros_output():
    '''It builds a list of list and it verifies that after the application of Fill_with_zeros
    the zeros are put in the right place.
    '''
    
    a=[[1,2],[3,5,6,7],[4,5,6]]
    fn.Fill_with_zeros(a)
    assert (a[0])==[1, 2, 0, 0]
    assert (a[1])==[3, 5, 6, 7]
    assert (a[2])==[4, 5, 6, 0]
    
    
#%%   test_size_evolution (3)
def test_size_evolution_size():
    '''It builds a Graph linking randomly 7 nodes, it applies the Size_evolution function and
    it verifies the three output length it is equal to the number of step of the process.
    the test is done for both the two kind of features of the function'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size, evolution,evolution_mean=fn.Size_evolution(G, step, feature)
    assert len(size)==nstep
    assert len(evolution)==nstep 
    assert len(evolution_mean)==nstep
    
    feature='betweenness_centrality'
    size, value_size_evolution,value_size_evolution_mean=fn.Size_evolution(G, step, feature)
    assert len(size)==nstep
    assert len(evolution)==nstep 
    assert len(evolution_mean)==nstep
    
    
def test_size_evolution_size_increasing_size():
    '''It builds a Graph linking randomly 7 nodes, it applies the Size_evolution function and
    it verifies the elements of the output 'size' increases at each step
    the test is done for both the two kind of features of the function'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size=fn.Size_evolution(G, step, feature)[0]
    for i in range(nstep-1):
        assert size[i]<size[i+1]
    
    feature='betweenness_centrality'
    size, value_size_evolution,value_size_evolution_mean=fn.Size_evolution(G, step, feature)
    for i in range(nstep-1):
        assert size[i]<size[i+1]

def test_size_evolution_size_degree_constant_len():
    '''It builds a Graph linking randomly 7 nodes, it applies the Size_evolution function and
    it verifies the elements 'distribution_evolution'  are of the same lenght'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    feature='degree'
    distribution_evolution=fn.Size_evolution(G, step, feature)[1]
    assert len(distribution_evolution[0])==4
    assert len(distribution_evolution[1])==4
    assert len(distribution_evolution[2])==4
    assert len(distribution_evolution[3])==4
    assert len(distribution_evolution[4])==4
    assert len(distribution_evolution[5])==4

    

#%%   test_List_dist_link (5)

    
def test_List_dist_link_length():
    '''It builds an undirect nx.Graph object with 3 nodes linking them randomly and reapiting some
    edges reversing the couple order of some of them (e.g.: (1,2), (2,1))
    or creating some autolink edges (e.g.: (1,1)). Then it build a topografical map of the nodes
    and applies the function Dct_dist_link to calculate the euclidean distance among linked nodes.
    Finally it verifies in the keys of th ouput distance dictionary are not present symmetric object (e.g.: (1,2), (2,1))
    and it removes autoedges (e.g.: (1,1))
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    assert list(dct_dist_link.keys())==[(3, 1), (1, 2)]

def test_List_dist_link_non_negativity():
    '''It builds an undirect nx.Graph object with 6 nodes linking them randomly.
    Then it build a topografical map of the nodes and applies the function Dct_dist_link
    to calculate the euclidean distance among linked nodes.
    Finally it verifies if the values the ouput distance dictionary are posives'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
        
def test_List_dist_link_simmetry():
    '''It builds an undirect nx.Graph object with 3 nodes completed connected.
    Then it build a topografical map of the nodes.
    Then it builds two list of edges of the graph in which the order of the couple of linked nodes
    is switched.
    Finally it applies the function Dct_dist_link to calculate the euclidean distance
    among linked nodes for the two symmetric set and it tests the outpu values are the same'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[2,3],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    edges=list(G.edges())
    segde=[]
    for i in range(len(edges)):
        segde.append((edges[i][1],edges[i][0]))                     
    
    dct_dist_link1=fn.Dct_dist_link(edges, map_dct)
    dct_dist_link2=fn.Dct_dist_link(segde, map_dct)
    assert list(dct_dist_link2.values())==list(dct_dist_link1.values())


def test_List_dist_link_triangular_inequality_unit_test():
    '''It builds a nx.Graph object with 3 nodes completed connected.
    Then it build a topografical map of the nodes and applies the function Dct_dist_link
    to calculate the euclidean distance among linked nodes. Finally it test the triangular 
    inequality for all the possible path'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[2,3]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    assert dct_dist_link[(1,2)]<=dct_dist_link[(2,3)]+dct_dist_link[(1,3)]
    assert dct_dist_link[(1,3)]<=dct_dist_link[(2,3)]+dct_dist_link[(1,2)]
    assert dct_dist_link[(2,3)]<=dct_dist_link[(1,2)]+dct_dist_link[(1,3)]
                
def test_List_dist_link_triangular_inequality_abstract_test():
    '''It is the abstract version of th unit test'''
    
    '''It builds a nx.Graph object with 6 nodes linking them randomly.
    Then it build a topografical map of the nodes and applies the function Dct_dist_link
    to calculate the euclidean distance among linked nodes. Finally it test the triangular 
    inequality for all the possible path'''
    
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    a=list(G.nodes())
    for node in list(G.nodes()) :

        a.remove(node)
        for next_node in a:
            all_simple_paths=sorted(list(nx.all_simple_paths(G,node,next_node)),key=len)
            if len(all_simple_paths)>0:
                if len(all_simple_paths[0])==2 and len(all_simple_paths)>1:
                    dist=dct_dist_link[(all_simple_paths[0][0],all_simple_paths[0][1])]
                    for i in range(1,len(all_simple_paths)):
                        dist2=0
                        for j in range(len(all_simple_paths[i])-1):
                            if edges.count((all_simple_paths[i][j],all_simple_paths[i][j+1]))==1:
                                dist2+=dct_dist_link[(all_simple_paths[i][j],all_simple_paths[i][j+1])]
                            else:
                                dist2+=dct_dist_link[(all_simple_paths[i][j+1],all_simple_paths[i][j])]
                assert dist<dist2                   
            
            
            
  
#%%   test_Dct_dist (3)
def test_List_dist_length():
    '''It builds an undirect nx.Graph object with 4 nodes linking them randomly and reapiting some
        edges reversing the couple order of some of them (e.g.: (1,2), (2,1)) or creating some 
        autolink edges (e.g.: (1,1)). Then it build a topografical map of the nodes
        and applies the function Dct_dist to calculate the euclidean distance among nodes.
        Finally it verifies in the keys of th ouput distance dictionary are not present symmetric object (e.g.: (1,2), (2,1))
        nor autoedges (e.g.: (1,1), but only and all the distances among the nodes)
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1), (1,4)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G, map_dct)
    assert list(dct_dist.keys())==[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

def test_List_dist_non_negativity():
    '''It builds an undirect nx.Graph object with 6 nodes linking them randomly.
       Then it build a topografical map of the nodes and applies the function Dct_dist_link
       to calculate the euclidean distance among nodes.
       Finally it verifies if the values the ouput distance dictionary are posives'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist(G, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
        
def test_List_dist_simmetry():
    '''It builds an undirect nx.Graph object with 3 nodes completed connected.
    Then it build a topografical map of the nodes.
    Then it builds two list of edges of the graph in which the order of the couple of linked nodes
    is switched.
    Finally it applies the function Dct_dist_link to calculate the euclidean distance
    among nodes for the two symmetric set and it tests the output values are the same'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[3,4],[1,3],[2,3],[2,4]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    edges=[[1,2],[3,4],[1,3],[2,3],[2,4]]
    segde=[]
    for i in range(len(edges)):
        segde.append((edges[i][1],edges[i][0]))                     
    G_reverse=nx.Graph()
    G_reverse.add_nodes_from([4,3,2,1])
    G_reverse.add_edges_from(segde)
    dct_dist_link1=fn.Dct_dist(G, map_dct)
    dct_dist_link2=fn.Dct_dist(G_reverse, map_dct)
    assert dct_dist_link1[(1,2)]==dct_dist_link2[(2,1)]
    assert dct_dist_link1[(1,3)]==dct_dist_link2[(3,1)]
    assert dct_dist_link1[(1,4)]==dct_dist_link2[(4,1)]
    assert dct_dist_link1[(2,3)]==dct_dist_link2[(3,2)]
    assert dct_dist_link1[(2,4)]==dct_dist_link2[(4,2)]
    assert dct_dist_link1[(3,4)]==dct_dist_link2[(4,3)]
    
    
    
    
def test_List_dist_triangular_inequality_unit_test():
    '''It builds a nx.Graph object with 3 nodes linking them randomly.
    Then it build a topografical map of the nodes and applies the function Dct_dist_link
    to calculate the euclidean distance among linked nodes. Finally it test the triangular 
    inequality for all the possible distances'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[2,3]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G, map_dct)
    assert dct_dist[(1,2)]<=dct_dist[(2,3)]+dct_dist[(1,3)]
    assert dct_dist[(1,3)]<=dct_dist[(2,3)]+dct_dist[(1,2)]
    assert dct_dist[(2,3)]<=dct_dist[(1,2)]+dct_dist[(1,3)]
    
    
def test_List_dist_triangular_inequality_abstract():
    '''It builds a nx.Graph object with 6 nodes linking them randomly.
    Then it build a topografical map of the nodes and applies the function Dct_dist_link
    to calculate the euclidean distance among linked nodes. Finally it test the triangular 
    inequality for all the possible distances'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G, map_dct)
    a=list(G.nodes())
    for node in list(G.nodes()) :

        a.remove(node)
        for next_node in a:
            all_simple_paths=sorted(list(nx.all_simple_paths(G,node,next_node)),key=len)
            if len(all_simple_paths)>0:
                if len(all_simple_paths[0])==2 and len(all_simple_paths)>1:
                    dist=dct_dist[(all_simple_paths[0][0],all_simple_paths[0][1])]
                    for i in range(1,len(all_simple_paths)):
                        dist2=0
                        for j in range(len(all_simple_paths[i])-1):
                            if edges.count((all_simple_paths[i][j],all_simple_paths[i][j+1]))==1:
                                dist2+=dct_dist[(all_simple_paths[i][j],all_simple_paths[i][j+1])]
                            else:
                                dist2+=dct_dist[(all_simple_paths[i][j+1],all_simple_paths[i][j])]
                assert dist<dist2

#%%   test_Node_distance_frequency (3)

   
def test_Node_distance_frequency_I_axiom():
    '''Given a set of distances it builds a discrete distribution of frequency for the distances,
    exploting the Node_distance_frequency and it verifies each frequency is positive'''
    dct_dist={(1, 2): 0.178,
              (1, 3): 0.162,
              (1, 4): 0.150,
              (1, 5): 0.303,
              (1, 6): 1.162,
              (2, 3): 0.302,
              (2, 4): 0.149,
              (2, 5): 0.161,
              (2, 6): 1.124,
              (3, 4): 0.184,
              (3, 5): 0.367,
              (3, 6): 1.317,
              (4, 5): 0.184,
              (4, 6): 1.258,
              (5, 6): 1.245}
    step=0.026346986731500856
    nstep=50
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    for i in node_distance_frequency:
        assert i>=0
        
def test_Node_distance_frequency_II_axiom():
    '''Given a set of distances it builds a discrete distribution of frequency for the distances,
        exploting the Node_distance_frequency and it verifies that the normalized sum of each
        frequency is equal to one'''
    dct_dist={(1, 2): 0.178,
              (1, 3): 0.162,
              (1, 4): 0.150,
              (1, 5): 0.303,
              (1, 6): 1.162,
              (2, 3): 0.302,
              (2, 4): 0.149,
              (2, 5): 0.161,
              (2, 6): 1.124,
              (3, 4): 0.184,
              (3, 5): 0.367,
              (3, 6): 1.317,
              (4, 5): 0.184,
              (4, 6): 1.258,
              (5, 6): 1.245}
    step=0.026346986731500856
    nstep=50
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    assert 0.99999<sum(node_distance_frequency)<1
    
def test_Node_distance_frequency_III_axiom():
    '''Given a set of distances it builds 2 discrete distributions of frequency for the distances,
        exploting the Node_distance_frequenc. The second distribution has bins two times largerr than the first,
        so it represents the sum of two group of distances.
        It verifies that the frequency of couples of distances of the first distribuition 
        is equal to the frequancy of one event of second.'''
    dct_dist={(1, 2): 0.178,
              (1, 3): 0.162,
              (1, 4): 0.150,
              (1, 5): 0.303,
              (1, 6): 1.162,
              (2, 3): 0.302,
              (2, 4): 0.149,
              (2, 5): 0.161,
              (2, 6): 1.124,
              (3, 4): 0.184,
              (3, 5): 0.367,
              (3, 6): 1.317,
              (4, 5): 0.184,
              (4, 6): 1.258,
              (5, 6): 1.245}
    step=0.026346986731500856
    nstep=14
    node_distance_frequency_1=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    
    step=2*0.026346986731500856
    nstep=7
    node_distance_frequency_2=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    for i in range(0,14,2):
        assert node_distance_frequency_1[i]+node_distance_frequency_1[i+1]==node_distance_frequency_2[int(i/2)]


#%%   test_Conditional_probability (4)

   
def test_conditional_probability_I_axiom():
    '''Given the binned frequency of an events and the frequency of them which are 'important', it calculates
    the probability to have an important event if an event is recorded. Then it verifies that the each elements
    of the ouput probability is grater or equal to 0 but minor than 1'''
    events_frequency=[0,0,2,8,24,0,]
    important_event_frequency=[0,0,1,2,3,0]
    step=1
    conditional_probability_dct=fn.Conditional_probability(important_event_frequency,step,events_frequency)
    conditional_probability=list(conditional_probability_dct.values())
    for i in conditional_probability:
        assert 0<=i<=1
        

def test_Conditional_probability_bin_value():
    '''Given the binned frequency of an events and the frequency of them which are 'important', it calculates
    the probability to have an important event if an event is recorded. Then it verifies that the value of each bin of the
    probability dictionary is corrected.'''
    events_frequency=[0,0,2,8,24,0,]
    important_event_frequency=[0,0,1,2,3,0]
    step=1
    conditional_probability_dct=fn.Conditional_probability(important_event_frequency,step,events_frequency)
    bins=list(conditional_probability_dct.keys())
    assert  bins==[1, 2, 3, 4, 5, 6]

#%%   test_Copymap_linking (3)


def test_Add_edges_from_map_nodes_number():
    '''Given a graph with random links, a dictionary of the distances among nodes and the conditional probabilities to have
    a links in function of the distance,it builds a new graph with same nodes of the old one
    and adds edges exploiting the function Add_edges_from_map.
    Finally it tests the number of nodes is the same of the old one
    '''
      
    G=nx.Graph()
    G.add_edges_from([[0,2],[0,1]])
    dct_dist={(0,1): 1,
              (0,2): 1,
              (1,2): 3}
    probability={1: 0.9, 2: 0.5, 3: 0.1}
    Copy_map=nx.Graph()
    Copy_map.add_nodes_from(list(G.nodes))
    fn.Add_edges_from_map(Copy_map, dct_dist,probability)
    assert list(Copy_map.nodes())==list(G.nodes)
    
def test_Add_edges_from_map_completed():
    '''Given a graph with 4 nodes randomly connected, a dictionary of the distances among nodes and the conditional probabilities to have
    a links in function of the distance equal to one for all the possible distance,it builds
    the graph Copy_map exploiting the function Add_edges from_map.
    Finally it tests that the new graph Copy_map is fully conected
    '''
      
    G=nx.Graph()
    G.add_edges_from([[0,1],[1,2],[2,3],[1,3]])
    dct_dist={(0,1): 1,
              (0,2): 2,
              (0,3): 3,
              (1,2): 1,
              (1,3): 2,
              (2,3): 1}
        
    probability_distribution={1: 1, 2: 1, 3: 1}
    Copy_map=fn.SuperGraph()
    Copy_map.add_nodes_from(list(G.nodes))
    fn.Add_edges_from_map(Copy_map, dct_dist,probability_distribution)
    assert list(Copy_map.edges)==[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def test_Add_edges_from_map_Bernulli_trials():
    '''Given a graph with 3 nodes randomly connected, a dictionary of the distances among nodes and the conditional probabilities to have
    a links in function of the distance,it builds 1000 the graph Copy_map exploiting 
    the function Add_edges from_map. It test that the average number of link in function of the distance
    follows the conditonal probabilities.
    '''
      
    G=nx.Graph()
    G.add_edges_from([[0,1],[1,2]])
    dct_dist={(0,1): 1,
              (0,2): 2,
              (1,2): 1}
        
    link_prob={1: 0.4, 2: 0.8}
    prob_link1=0
    prob_link2=0
    trials=1000
    
    for i in range(trials):
        Copy_map=fn.SuperGraph()
        Copy_map.add_nodes_from(list(G.nodes))
        fn.Add_edges_from_map(Copy_map, dct_dist,link_prob)
        edges=list(Copy_map.edges)
        prob_link1+=(edges.count((0,1))+edges.count((1,2)))/2
        prob_link2+=edges.count((0,2))
        
    prob_link1=prob_link1/trials
    prob_link2=prob_link2/trials
    
    'using the binomial variance and the central limit theorem'
    std1=(0.4*0.6/trials**0.5)**0.5
    std2=(0.8*0.2/trials**0.5)**0.5
    assert     0.4-3*std1<prob_link1<0.4+3*std1
    assert     0.8-3*std2<prob_link2<0.8+3*std2

        
#%%   test_Break_strongest_nodes (2)

def test_Break_strongest_nodes_size():
    '''It build a graph which has edges with no particular order and whose max degree
    is equal to five. It cuts edges until the max degree is two. Finally it test the number
    of nodes is conserve in the process'''
    
    edges=[(1,2), (3,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    len_before=len(G)
    fn.Break_strongest_nodes(G,2)
    len_after=len(G)
    assert len_after==len_before
    
def test_Break_strongest_nodes_maximum_value():
    '''It build a graph which has edges with no particular order and whose max degree
    is equal to five. It cuts edges until the max degree is two. 
    Finally it verify the nodes with the highest degree is under the threshold'''
    
    edges=[(1,2), (3,1),(1,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Break_strongest_nodes(G,2)
    deg=G.Degree_dct()
    assert max(deg.keys())<=2
    
def test_Break_strongest_nodes_untouchable_edges():
    '''It build a graph which has edges with no particular order and whose max degree
    is equal to three. It cuts edges until the max degree is two. So it removes one of
    the links of the node two.
    Finally it verify the edges which are not involved in the function are conserved,
    So the neighbours of node 4 '''
    
    edges=[(1,1), (1,2), (2,1), (3,4), (1,5), (5,2), (2,7),(4,6)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Break_strongest_nodes(G,2)
    assert list(G.neighbors(4))==[3,6]


#%% test_Find_mode (1)
def test_Find_mode():
    '''Given a list of values it tests that function Find_mode discriminatethe the
    position of the list with the maximum value.'''
    a=[-1,-5,-0.1,-3,]
    assert a[fn.Find_mode(a)]==-0.1
    

            
#%% test_Equalize_strong_nodes (2)

def test_Equalize_strong_nodes_size():
    '''It applies the function Equalize strong nodes to two graphs and 
    it tests length of the first graph is kept constant'''
    
    edges=[(1,2), (3,1), (6,3), (4,2), (2,3), (1,4), (1,5), (5,4), (5,3), (1,6), (5,3), (5,2), (5,6), (7,1)]    
    G_strong=fn.SuperGraph()
    G_strong.add_edges_from(edges)
    edges=[(1,2), (5,7), (1,1), (1,2), (2,1), (3,4), (4,5), (2,4), (5,6), (1,6), (1,2), (6,2), (5,6), (7,6)]
    G_weak=fn.SuperGraph()
    G_weak.add_edges_from(edges)
    len_before=len(G_strong)
    fn.Equalize_strong_nodes(G_strong, G_weak) 
    len_after=len(G_strong)
    assert len_after==len_before
    
def test_Equalize_strong_nodes_maximum_value():
    '''It builds two graphs, the first (strong) has a maximum degree equal to 6 and the second (weak) equal to 4.
    The mode of the second graph is 3. It tests that after the application of the
    function Equalize strong nodes the ratio of the node of the strong graph with degree
    major than 3 it is equal or minor the ratio of the weak graph'''
    
    edges=[(1,2), (3,1), (6,3), (4,0), (4,2), (2,3), (1,4), (1,5), (5,4), (5,3), (1,6), (5,3), (5,2), (5,6), (7,1)]    
    G_strong=fn.SuperGraph()
    G_strong.add_edges_from(edges)
    edges=[(0,1),(4,0),(1,2), (5,7), (1,2), (3,4), (4,5), (2,4), (5,6), (1,6), (6,2), (5,6), (7,6)]
    G_weak=fn.SuperGraph()
    G_weak.add_edges_from(edges)
    fn.Equalize_strong_nodes(G_strong, G_weak)
    assert len(G_strong.Degree_dct())<=len(G_weak.Degree_dct())
    assert (G_strong.Degree_ratio()[4])<=(G_weak.Degree_ratio()[4])




#%% test_Max_prob_target (3)

def test_Max_prob_target_degree():
    '''Given a graph with 5 nodes, their positions and the probability to have a link in function of the distance,
    It  exploits the function Max_prob_target to link the node '4' with a node with degree three.
    Finally it verifies the degree of the target is the one given'''
    
    edges=[(0,2),(1,2),(2,3),(4,0)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)    
    map_dct={0: np.array([0, 0]),
             1: np.array([1, 0]),
             2: np.array([2, 0]),
             3: np.array([3, 0]),
             4: np.array([4, 0]) }    
    
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())
    degree=3  
    source=4
    prob_distribution={1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}      
    target=fn.Max_prob_target (source,degree,map_dct,prob_distribution,max_dist,G)
    assert len(list(G.neighbors(target)))==degree

def test_Max_prob_target_not_its_self():
    '''Given a graph with 5 nodes, their positions and the probability to have a link in function of the distance,
    It  exploits the function Max_prob_target to link the node '0' with a node with degree 3 .
    Finally it verifies the target is not the source'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct={0: np.array([0, 0]),
             1: np.array([1, 0]),
             2: np.array([2, 0]),
             3: np.array([3, 0]),
             4: np.array([4, 0]) } 
    
    
    source=0
    degree=3
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())
    prob_distribution={1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}    
    assert source!=fn.Max_prob_target (source,degree,map_dct,prob_distribution,max_dist,G)
    
def test_Max_prob_target_is_not_its_neighbors():
    '''Given a graph with 5 nodes, their positions and the probability to have a link in function of the distance,
    It  exploits the function Max_prob_target to link the node '0' with a node with degree 3 .
    Finally it  verifies the target is not a source's neighbors'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct={0: np.array([0, 0]),
             1: np.array([1, 0]),
             2: np.array([2, 0]),
             3: np.array([3, 0]),
             4: np.array([4, 0]) } 
    source=0
    degree=2
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())
    prob_distribution={1: 0.4, 2: 1, 3: 0.4, 4: 0.4}   
    neighbors_0=list(G.neighbors(0))
    assert neighbors_0!=fn.Max_prob_target (source,degree,map_dct,prob_distribution,max_dist,G)


#%% test min_dis_distance_target (5)

def test_Min_distance_target_is_the_nearest():
    '''Given a graph with 10 nodes, linked with no particular order and a map of their postion.
    It links the node 0 to the nearest one with degree equal to three,
    which is not 0 or one of  its neighbors'''
    edges=[(0, 1), (0, 3), (0, 6), (0, 8), (0, 9), (1, 2), (1, 4), (1, 9), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (4, 8), (5, 7), (6, 8), (7, 9)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    
    map_dct={0: np.array([ 0, 0]),
             1: np.array([-1, -1]),
             2: np.array([-0.5, -0.2]),
             3: np.array([1, 2]),
             4: np.array([0, 5]),
             5: np.array([5, 1]),
             6: np.array([2, 1]),
             7: np.array([1, 1]),
             8: np.array([1, 1]),
             9: np.array([1, 1])}
    
    source=0
    target=fn.Min_distance_target (source,3,map_dct,G)
    
    assert target==2
    
def test_Min_distance_target_is_not_its_neighbors():
    
    '''Given a graph with 5 nodes and a map of their postion.
    It tries to links the node 0 to the nearest one with degree equal to 1, but all
    the nearest nodes with degree 1 are neighbors so it tests the target is 4'''
    edges=[(0,1),(0,2),(0,3),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    
    map_dct={0: np.array([ 0, 0]),
         1: np.array([-1, -1]),
         2: np.array([-0.5, -0.2]),
         3: np.array([1, 2]),
         4: np.array([5, 5])}
    source=0
    assert 4==fn.Min_distance_target (source,1,map_dct,G)
        
def test_Min_distance_target_is_not_its_self():
    '''Given a graph with 5 nodes, and a map of their postion.
    It tries to links the node 0 to the nearest one with degree equal to 3 but the only node
    with degree 3 is itself so it tests the target is 4'''
    edges=[(0,1),(0,2),(1,3),(0,3),(1,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    
    map_dct={0: np.array([ 0, 0]),
             1: np.array([-1, -1]),
             2: np.array([-0.5, -0.2]),
             3: np.array([1, 2]),
             4: np.array([5, 5])}
    source=0
    assert 4==fn.Min_distance_target (source,1,map_dct,G)
        
def test_Min_distance_target_degree_corrispodence():
    '''Given a graph with 5 nodes, and a map of their postion.
    It tries to links the node 0 to the nearest one with degree equal to 3.
    Finally it tests the degree of the tagat'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,1),(4,2),(4,3)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)

    map_dct={0: np.array([ 0, 0]),
             1: np.array([-1, -1]),
             2: np.array([-0.5, -0.2]),
             3: np.array([1, 2]),
             4: np.array([5, 5])}
    source=0
    target=fn.Min_distance_target (source,3,map_dct,G)
    assert len(list(G.neighbors(target)))==3
    
#%%  test Random target

def test_random_target():
    '''Given a graph with 10 nodes, linked with no particular order.
    It links the node 0 to a random node, it tests that after tha application
    of the function the target is a node of the graph'''
    edges=[(0, 1), (0, 3), (0, 6), (0, 8), (0, 9), (1, 2), (1, 4), (1, 9), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (4, 8), (5, 7), (6, 8), (7, 9)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)   
    source=0
    target=fn.Random_target(G, source)
    
    assert list(G.nodes()).count(target)==1
    
def test_Random_target_is_not_its_neighbors():
    
    '''Given a graph with 5 nodes.
    It tries to links the node 0 to a random target, all
    the nodes but the 4° are neighbors so it tests the target is the 4° '''
    edges=[(0,1),(0,2),(0,3),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    source=0
    assert 4==fn.Random_target(G, source)
        
def test_Random_target_is_not_its_self():
    '''Given a graph with 2 nodes, and a map of their postion.
    It tries to links the node 0 to a random target, but there is only another node
    so it tests the target is 4'''
    
    edges=[(0,0),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)  
    source=0
    assert 4==fn.Random_target(G, source)
        
#%%  test_Merge_small_component (2)

def test_Merge_small_component():
    '''It builds a graph with only one component of dimension three and three smaller components.
    It tests if after the apllication of the Merge function all the components 
    are bigger than the 3, the threshold'''
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    fn.Merge_small_component(G, 1,map_dct,threshold=3)
    for i in list(nx.connected_components(G)):
        assert len(i)>=3
        
def test_Merge_small_component_Exception():
    '''It builds a graph with four small components.
    It tries to merge with nodes looking for a target node with degree equal to zero, but there
    are no left so it verifies the function raise an Exception'''
    edges=[(1,2), (3,4), (6,7), (0,5)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    with pytest.raises(Exception):
        fn.Merge_small_component(G, 0,map_dct,threshold=3)

 
#%% test_Link_2_ZeroNode (3)

    

def test_Link_2_ZeroNode_reduction():
    '''Given a graoh with seven nodes, and two of them are isolated, it applies
    Link_2_ZeroNode  connected the the isolated node with two links to the rest of the graph
    it verifies the right decreasing of  '0' degree ratio
    due to the application of the function '''
    edges=[(0,2),(1,2),(2,3),(4,0),(5,5),(6,6)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)    
    map_dct={0: np.array([0, 0]),
             1: np.array([1, 0]),
             2: np.array([2, 0]),
             3: np.array([3, 0]),
             4: np.array([4, 0]),
             5: np.array([5, 0]),
             6: np.array([6, 0])}    
    
    prob_distribution={1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}  
    max_dist_link=4
    n_links=2
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , G.Degree_dct())
    degree_ratio_0_after=G.Degree_ratio()[0]
    assert 0==degree_ratio_0_after

def test_Link_2_ZeroNode_increment():
    '''Given a graoh with seven nodes, and two two of them are isolated, it applies
    Link_2_ZeroNode  connected the the isolated node with two links to the rest of the graph
    Finally it verifies the right increasing of  '2' degree ratio
    due to the application of the function '''
    
    edges=[(0,2),(1,2),(2,3),(4,0),(5,5),(6,6)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)    
    map_dct={0: np.array([0, 0]),
             1: np.array([1, 0]),
             2: np.array([2, 0]),
             3: np.array([3, 0]),
             4: np.array([4, 0]),
             5: np.array([5, 0]),
             6: np.array([6, 0])}    
    
    prob_distribution={1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}  
    max_dist_link=4
    n_links=2
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , G.Degree_dct())
    degree_ratio_2_after=G.Degree_ratio()[2]
    assert 3/7==degree_ratio_2_after


def test_Link_2_ZeroNode_constant_degree_ratio():
    '''Given a graoh with seven nodes, and two two of them are isolated,it applies
    Link_2_ZeroNode  connected the the isolated node with three links to the rest of the graph 
    Finally it verifies that the degree ratio of 1 and to remains constant after
    the application of the function '''
    
    
    edges=[(0,2),(1,2),(2,3),(4,0),(5,5),(6,6)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)    
    map_dct={0: np.array([0, 0]),
             1: np.array([1, 0]),
             2: np.array([2, 0]),
             3: np.array([3, 0]),
             4: np.array([4, 0]),
             5: np.array([5, 0]),
             6: np.array([6, 0])}    
    
    prob_distribution={1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}  
    max_dist_link=4
    n_links=3
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , G.Degree_dct())
    degree_ratio_2_after=G.Degree_ratio()[2]
    degree_ratio_1_after=G.Degree_ratio()[1]
    
    assert 1/7==degree_ratio_2_after
    assert 3/7==degree_ratio_1_after




#%% test_Remove_edge_of_degree

def test_Remove_edge_of_degree_local():
    '''Given a graph with 7 nodes and links with no particular order,
    it removes an edge with a node of a given degree (six), exploting fn.Remove_edge_of_degree.
    It verifies the ratio that had been changed it is right '''
    edges=[(1,2), (3,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2), (1,7)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    print(G.Degree_dct())
    fn.Remove_edge_of_degree(6,G)
    print(G.Degree_dct())
    assert list(G.Degree_ratio())[5]==1/7
    
def test_Remove_edge_of_degree_total():
    '''Given a graph with 7 nodes and links with no particular order,
    it removes an edge with a node of a given degree (four) exploting fn.Remove_edge_of_degree
    It verifies the ratio of all the degree are right after the application of the function'''
    edges=[(1,2), (3,1), (1,1), (1,2), (1,4), (5,4), (5,3), (1,6), (6,7), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Remove_edge_of_degree(4,G)
    
    assert (G.Degree_ratio()[0])==0
    assert (G.Degree_ratio()[1])==(2/7)
    assert round(G.Degree_ratio()[2],4)==round(3/7,4)
    assert round(G.Degree_ratio()[3],4)==round(2/7,4)
    assert len(G.Degree_ratio())==4

def test_Remove_edge_of_degree_len():
    '''Given a graph with 7 nodes and links with no particular order,
    it removes an edge with a node of a given degree (four) exploting fn.Remove_edge_of_degree
    It tests the length of the Graph does not change'''
    edges=[(1,2), (3,1), (1,1), (1,2), (1,4), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Remove_edge_of_degree(4,G)
    assert len(G)==6
 
#%% test_Equalizer_top_down(G,Copycat)

def test_Equalizer_top_down_len():
    '''
    Given two graph, G and Copycat, it applies the function fn.Equalizer_top_down,
    that removes some of the edges of Copycat. It verifies the number of nodes
    in the two graph do not changes.

    '''
    edges_G=[(0,1),(2,3),(4,5),(6,7)]
    G=fn.SuperGraph(edges_G)
    edges_copy=[(0,1),(0,2),(1,2),(1,5),(2,3),(4,5),(6,7),(8,9)]
    Copycat=fn.SuperGraph(fn.SuperGraph(edges_copy))
    fn.Equalizer_top_down(G,Copycat)
    assert len(Copycat)==10
    assert len(G)==8
    
def test_Equalizer_top_down_no_edge():
    '''
    Given two graph, G and Copycat, it applies the function fn.Equalizer_top_down,
    that removes some of the edges of Copycat in order to make the degree 
    ratio distribution of the graph Copycat lower or equal to the one og G.
    It expecteded that some of the edges of the Copycat node with degree 
    equal to two disappear.

    '''
    edges_G=[(0,0),(1,2),(2,3),(3,4),(3,5)]
    'it has only one node with degree two'
    G=fn.SuperGraph(edges_G)
    'it has two nodes with degree equal to two'
    edges_copy=[(0,0),(0,1),(1,2),(2,3),(3,4),(3,5)]
    Copycat=fn.SuperGraph(fn.SuperGraph(edges_copy))
    fn.Equalizer_top_down(G,Copycat)
    assert list(Copycat.edges()).count((0,1))==0 or list(Copycat.edges()).count((1,2))==0 or list(Copycat.edges()).count((2,3))==0
    
def test_Equalizer_top_down_right_ratio():
    '''Given two graph, G and Copycat, it applies the function fn.Equalizer_top_down,
    that removes some of the edges of Copycat in order to make each degree 
    ratio of the graph Copycat lower or equal to the one of G.
    Finally it tests if the ineauqlity of the degree ratios is the one expected
    '''
    edges_G=[(0,0),(1,2),(2,3),(3,4),(3,5)]
    G=fn.SuperGraph(edges_G)
    edges_copy=[(0,0),(0,1),(0,6),(2,4),(1,2),(2,3),(3,4),(3,5),(0,5)]
    Copycat=fn.SuperGraph(fn.SuperGraph(edges_copy))
    fn.Equalizer_top_down(G,Copycat)
    
    if len(Copycat.Degree_ratio())>=2:
        assert Copycat.Degree_ratio()[1]<=G.Degree_ratio()[1]
    if len(Copycat.Degree_ratio())>=3:
        assert Copycat.Degree_ratio()[2]<=G.Degree_ratio()[2]    
    if len(Copycat.Degree_ratio())>=4:
        assert Copycat.Degree_ratio()[3]<=G.Degree_ratio()[3]


#%% test_Equalizer_down_top(G,Copycat,map_dct,prob_distribution,max_dist_link):

'''numero nodi
    non ci sia più quell'edge (so quale tolgo)
    i ratio siano adeguati
'''
def test_Equalizer_down_top():
    
    '''Given two graph, G and Copycat, it applies the function Equalizer_down_top,
    that adds some of the edges of Copycat. It verifies the number of nodes
    in the two graph do not changes.'''
    
    edges_G=[(0,1),(0,2),(1,2),(1,5),(2,3),(4,5),(6,7),(8,9)]
    edges_copy=[(0,1),(2,3),(4,5),(6,7)]
    G=fn.SuperGraph(edges_G)
    Copycat=fn.SuperGraph(fn.SuperGraph(edges_copy))
    map_dct=nx.spring_layout(Copycat, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    prob_distribution={0.2:0.5, 0.4:0.5, 0.6:0.5, 0.8:0.5, 1:0.5,
                    1.2:0.5, 1.4:0.5, 1.6:0.5, 1.8:0.5, 2:0.5,}
    dct_dist_link=fn.Dct_dist_link(list(Copycat.edges()), map_dct)
    max_dist_link=max(dct_dist_link.values())
    fn.Equalizer_down_top(G,Copycat,map_dct,prob_distribution,max_dist_link)
    assert len(Copycat)==8
    assert len(G)==10
    
def test_Equalizer_down_top_edge_presence():
    
    '''Given two graph, G and Copycat, it applies the function Equalizer_down_top,
    that adds some of the edges of Copycat in order to make each degree 
    ratio of the graph Copycat major or equal to the one of G.
    It expecteded that an edge betwwen 0 and 1 or 0 and 4 will appear.'''
    
    edges_G=[(0,0),(0,1),(1,2),(2,3),(3,4)]
    edges_copy=[(0,0),(1,2),(2,3),(3,4)]
    G=fn.SuperGraph(edges_G)    
    Copycat=fn.SuperGraph(fn.SuperGraph(edges_copy))
    map_dct=nx.spring_layout(Copycat, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    prob_distribution={0.2:0.5, 0.4:0.5, 0.6:0.5, 0.8:0.5, 1:0.5,
                    1.2:0.5, 1.4:0.5, 1.6:0.5, 1.8:0.5, 2:0.5,}
    dct_dist_link=fn.Dct_dist_link(list(Copycat.edges()), map_dct)
    max_dist_link=max(dct_dist_link.values())
    fn.Equalizer_down_top(G,Copycat,map_dct,prob_distribution,max_dist_link)
    assert list(Copycat.edges()).count((0,1))==1  or list(Copycat.edges()).count((0,4))==1
    

def test_Equalizer_down_top_right_ratio():
    
    '''Given two graph, G and Copycat, it applies the function Equalizer_down_top,
    that adds some of the edges of Copycat in order to make each degree 
    ratio of the graph Copycat major or equal to the one of G.
    Finally it tests if the ineauqlity of the degree ratios is the one expected
    '''
    
    edges_G=[(0,0),(0,1),(0,6),(2,4),(2,3),(4,4),(3,5),(0,5),(8,7),(9,9)]
    edges_copy=[(0,0),(1,1),(2,3),(3,4),(3,5),(6,6),(7,7),(8,8),(9,9)]
    G=fn.SuperGraph(edges_G)    
    Copycat=fn.SuperGraph(fn.SuperGraph(edges_copy))
    map_dct=nx.spring_layout(Copycat, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    prob_distribution={0.2:0.5, 0.4:0.5, 0.6:0.5, 0.8:0.5, 1:0.5,
                    1.2:0.5, 1.4:0.5, 1.6:0.5, 1.8:0.5, 2:0.5,}
    dct_dist_link=fn.Dct_dist_link(list(Copycat.edges()), map_dct)
    max_dist_link=max(dct_dist_link.values())
    fn.Equalizer_down_top(G,Copycat,map_dct,prob_distribution,max_dist_link)
   
    if len(Copycat.Degree_ratio())>=2:
        assert Copycat.Degree_ratio()[1]>=G.Degree_ratio()[1]
    if len(Copycat.Degree_ratio())>=3:
        assert Copycat.Degree_ratio()[2]>=G.Degree_ratio()[2]    
    if len(Copycat.Degree_ratio())>=4:
        assert Copycat.Degree_ratio()[3]>=G.Degree_ratio()[3]

#%% test_Copymap_degree_correction(Copy_map,G,map_dct,max_dist_link,prob_distribution,Merge=False)

def test_Copymap_degree_correction():
    '''It test the final network has a degree distribution compatible with the distribution of the reference network'''
    
    G=fn.SuperGraph(nx.newman_watts_strogatz_graph(300, 3, p=0.4, seed=3))
    link_dist_prob={0.2:0.5, 0.4:0.5, 0.6:0.5, 0.8:0.5, 1:0.5,
                    1.2:0.5, 1.4:0.5, 1.6:0.5, 1.8:0.5, 2:0.5,}
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(list(G.edges),map_dct)
    new_graph=fn.SuperGraph()
    new_graph.add_nodes_from(G.nodes())
    max_dist_link=max(dct_dist_link.values())
    
    edge_probability=2*G.number_of_edges()/((len(G)-1)*(len(G)))
    ERG=fn.SuperGraph(nx.fast_gnp_random_graph(len(G),edge_probability, seed=None, directed=False))
    ERG.Sorted_graph()
    ERG.Relable_nodes()
    
    Copycat=fn.Copymap_degree_correction(ERG,G,map_dct,max_dist_link,link_dist_prob,Merge=False)
    
    for i in range(len(Copycat.Degree_ratio())):
        print(G.Degree_ratio()[i]-3*(G.Degree_ratio()[i]+0.001)**0.5,'<=',Copycat.Degree_ratio()[i],'<=',G.Degree_ratio()[i]+3*(G.Degree_ratio()[i]+0.01)**0.5)
        assert (G.Degree_ratio()[i]-3*(G.Degree_ratio()[i]+0.001)**0.5<=Copycat.Degree_ratio()[i]<=G.Degree_ratio()[i]+3*(G.Degree_ratio()[i]+0.01)**0.5)
    
               
def test_trunk_array_at_nan_len():
    '''Given a np array 4*4, the Trunk_array_at_nan trunk each row at the first nan value.
    a list of lists is expected whose rows have different len'''
    a=np.array(([0,1,np.nan,np.nan],[1,np.nan,0,0]))
    A=fn.Trunk_array_at_nan(a)
    assert len(A[0])==2
    assert len(A[1])==1
    
def test_trunk_array_at_nan_no_nan():
    '''Given a np array 4*4, the Trunk_array_at_nan trunk each row at the first nan value.
    a list of lists is expected whose rows have all the values before the nan value'
    a=np.array(([0,1,np.nan,np.nan],[1,np.nan,0,0]))
    A=fn.Trunk_array_at_nan(a)
    assert (A[0])==[0.0, 1.0]
    assert (A[1])==[1.0]
    