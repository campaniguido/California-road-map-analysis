# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:05:25 2022

@author: Guido
"""
import math
import random as rn
import numpy as np
import function as fn
import networkx as nx
import pytest
rn.seed(3)

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
    '''it tests if all the elements are integers'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    edges=fn.Edge_list(file,2)
    for j in range(len(edges)):
        for k in range(2):
            assert type(edges[j][k])==int
                    
def test_edge_list_length():
    ''' it looks if the output length is equal of number of edges'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=2
    edges=fn.Edge_list(file,number_of_edges)
    assert len(edges)==number_of_edges
    
def test_edge_list_shape():
    '''it tests if the edges shape is n*2'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=2
    edges=fn.Edge_list(file,number_of_edges)
    for j in range(len(edges)):
        assert len(edges[j])==2
#Negative test
def test_edge_list_too_long():
    ''' If the number of edges is> of the length of the file it will raise an exception'''    
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=5
    with pytest.raises(Exception):
        fn.Edge_list(file,number_of_edges)
        
def test_edge_list_input_shape():
    ''' If the shape of the file is not n*2 it will raise an exception'''    
    file=[[0,134.0,88],['45643',' 3456',7],[3,4.0,'33']]
    number_of_edges=5
    with pytest.raises(Exception):
        fn.Edge_list(file,number_of_edges)


    



  
#%%  tests Unfreeze_into_list(2)

def test_Unfreeze_into_list_is_a_list_1():
    '''It verifies the output items are lists'''
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    list_=[A,B,C,D]
    list_=fn.Unfreeze_into_list(list_)
    for i in range(len(list_)):
        assert(type(list_[i])==list)

def test_Unfreeze_into_list_is_a_list_2():
    '''it verifies the output items are lists'''
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    dct_={0:A,1:B,2:C,3:D}
    dct_=fn.Unfreeze_into_list(dct_)
    for i in range(len(dct_)):
        assert(type(dct_[i])==list)


#%%   test_Set_community_number (6)

'''voglio testare che:
    tutti i nodi abbiano la loro corrispettiva comunità
    che gli elementi diversi della comunità sono lo stesso numero dei nodi (ha senso sia fare un raise nella funzione sia testarlo in maniera positiva che negativa)
    il numero di elementi unici dei valori del dizionario sia uguale alla grandezza della community'''

def test_Set_community_number_corrispondence():   
    '''It verifies the community corrispondence is right'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=fn.Set_community_number(G, community)
    for i in list(G.nodes()):
        assert community[community_number[i]].count(i)==1
        
def test_Set_community_number_length():
    '''It tests the length of the output is the same of the one of the Graph'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=fn.Set_community_number(G, community)
    assert len(community_number)==len(G)
    

        
def test_Set_community_number_doubble():
    '''It tests if the node are in just one community. e.g.:the four is in two different communities'''
    community=[[1,3,6,9],[5,10],[7,4],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        fn.Set_community_number(G, community)

def test_Set_community_number_missing_number():
    '''It tests if all the node are in the community variable. e.g.:the eight is missing'''
    community=[[1,3,6,9],[5,10],[7],[2,4]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(ValueError):
        fn.Set_community_number(G, community)

def test_Set_community_number_extra_number():
    '''It tests if the community has only the nodes labels of the graph. e.g.:the 999 is not a node number'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        fn.Set_community_number(G, community)

def test_Set_community_number_all_wrong():
    '''It tests the behaviours for the three previous mistakes'''
    community=[[1,3,6,4],[5,10],[7],[2,4,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        fn.Set_community_number(G, community)
            
    

#%%   test_size_evolution (3)
def test_size_evolution_size():
    '''It tests the len of each output'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size, value_size_evolution,value_size_evolution_mean=fn.Size_evolution(G, step, feature)
    assert len(size)==nstep and len(value_size_evolution)==nstep and len(value_size_evolution_mean)==nstep
    
    feature='betweenness_centrality'
    size, value_size_evolution,value_size_evolution_mean=fn.Size_evolution(G, step, feature)
    assert len(size)==nstep and len(value_size_evolution)==nstep and len(value_size_evolution_mean)==nstep

def test_size_evolution_size_increasing_size():
    '''It tests the output 'size' increases at each step'''
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
    '''It tests, for the degree feature, the lenght of the value_size_evolution it is always the same'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    value_size_evolution=fn.Size_evolution(G, step, feature)[1]
    for i in range(nstep):
        assert len(value_size_evolution[i])==4
    

#%%   test_List_dist_link (4)

    
def test_List_dist_link_length():
    '''it verifies in the dictionary are not present symmetric object (e.g.: (1,2), (2,1))
    and it removes autoedges (e.g.: (1,1))
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    assert len(dct_dist_link)==2

def test_List_dist_link_non_negativity():
    '''It tests distances are not negative'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
        
def test_List_dist_link_simmetry():
    '''It tests link distance symmetry'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    segde=[]
    for i in range(len(edges)):
        segde.append((edges[i][1],edges[i][0]))                     
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link1=fn.Dct_dist_link(edges, map_dct)
    dct_dist_link2=fn.Dct_dist_link(segde, map_dct)
    assert list(dct_dist_link2.values())==list(dct_dist_link1.values())
    
def test_List_dist_link_triangular_inequality():
    '''It tests triangular inequality'''
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
    '''it verify in the dictionary are not present symmetric object (e.g.: (1,2), (2,1))
    and it doesn't remove autoedges (e.g.: (1,1))
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1), (1,4)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    list_dist=fn.Dct_dist(G, map_dct)
    assert len(list_dist)==6

def test_List_dist_non_negativity():
    '''It tests distances are not negative'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist(G, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
    
def test_List_dist_triangular_inequality():
    '''It tests triangular inequality'''
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
    '''It verifies the probability I axiom'''
    dct_dist={(1, 2): 0.17826839815610848,
              (1, 3): 0.1621369469779289,
              (1, 4): 0.15016564862477497,
              (1, 5): 0.30311264216355577,
              (1, 6): 1.1628898382687927,
              (2, 3): 0.3027628033303571,
              (2, 4): 0.14940329750489897,
              (2, 5): 0.1615649741071594,
              (2, 6): 1.1248380701452185,
              (3, 4): 0.18437027351662638,
              (3, 5): 0.367850010102952,
              (3, 6): 1.3173493365750428,
              (4, 5): 0.18411463816128404,
              (4, 6): 1.2582690118458433,
              (5, 6): 1.2459654725200846}
    step=0.026346986731500856
    nstep=50
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    for i in node_distance_frequency:
        assert i>=0
        
def test_Node_distance_frequency_II_axiom():
    '''It verifies the probability II axiom'''
    dct_dist={(1, 2): 0.17826839815610848,
              (1, 3): 0.1621369469779289,
              (1, 4): 0.15016564862477497,
              (1, 5): 0.30311264216355577,
              (1, 6): 1.1628898382687927,
              (2, 3): 0.3027628033303571,
              (2, 4): 0.14940329750489897,
              (2, 5): 0.1615649741071594,
              (2, 6): 1.1248380701452185,
              (3, 4): 0.18437027351662638,
              (3, 5): 0.367850010102952,
              (3, 6): 1.3173493365750428,
              (4, 5): 0.18411463816128404,
              (4, 6): 1.2582690118458433,
              (5, 6): 1.2459654725200846}
    step=0.026346986731500856
    nstep=50
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    assert 0.99999<sum(node_distance_frequency)<1
    
def test_Node_distance_frequency_III_axiom():
    '''It verifies the probability III axiom'''
    dct_dist={(1, 2): 0.17826839815610848,
              (1, 3): 0.1621369469779289,
              (1, 4): 0.15016564862477497,
              (1, 5): 0.30311264216355577,
              (1, 6): 1.1628898382687927,
              (2, 3): 0.3027628033303571,
              (2, 4): 0.14940329750489897,
              (2, 5): 0.1615649741071594,
              (2, 6): 1.1248380701452185,
              (3, 4): 0.18437027351662638,
              (3, 5): 0.367850010102952,
              (3, 6): 1.3173493365750428,
              (4, 5): 0.18411463816128404,
              (4, 6): 1.2582690118458433,
              (5, 6): 1.2459654725200846}
    step=0.026346986731500856
    nstep=50
    node_distance_frequency_1=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    
    dct_dist={(1, 2): 0.17826839815610848,
              (1, 3): 0.1621369469779289,
              (1, 4): 0.15016564862477497,
              (1, 5): 0.30311264216355577,
              (1, 6): 1.1628898382687927,
              (2, 3): 0.3027628033303571,
              (2, 4): 0.14940329750489897,
              (2, 5): 0.1615649741071594,
              (2, 6): 1.1248380701452185,
              (3, 4): 0.18437027351662638,
              (3, 5): 0.367850010102952,
              (3, 6): 1.3173493365750428,
              (4, 5): 0.18411463816128404,
              (4, 6): 1.2582690118458433,
              (5, 6): 1.2459654725200846}
    step=2*0.026346986731500856
    nstep=25
    node_distance_frequency_2=fn.Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    for i in range(0,50,2):
        assert node_distance_frequency_1[i]+node_distance_frequency_1[i+1]==node_distance_frequency_2[int(i/2)]


#%%   test_Link_distance_probability (4)

   
def test_Link_distance_probability_I_axiom():
    '''It verifies the probability I axiom'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    link_distance_probability=list(link_distance_conditional_probability.values())*node_distance_frequency
    for i in link_distance_probability:
        assert i>=0
        
def test_Link_distance_probability_II_axiom():
    '''It verifies the probability II axiom'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    link_distance_probability=list(link_distance_conditional_probability.values())*node_distance_frequency/len(dct_dist_link)
    assert 0.999<sum(link_distance_probability)<=1
    
def test_Link_distance_probability_III_axiom():
    '''It verifies the probability III axiom'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    link_distance_probability=list(link_distance_conditional_probability.values())*node_distance_frequency/len(dct_dist_link)
        
    nstep2=5
    step2=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency2=fn.Node_distance_frequency(dct_dist,nstep2,step2)
    link_distance_conditional_probability2=fn.Link_distance_conditional_probability(dct_dist_link,nstep2,node_distance_frequency2)
    link_distance_probability2=list(link_distance_conditional_probability2.values())*node_distance_frequency2/len(dct_dist_link)
    assert link_distance_probability[-1]+link_distance_probability[-2]==link_distance_probability2[-1]

def test_Link_distance_probability_maximum_value():
    '''It verifies the last key of probability dictionary is the maximum link distance'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    assert list(link_distance_conditional_probability.keys())[-1]==max(list(dct_dist_link.values()))
    

#%%   test_Copymap_linking (2)


def test_Add_edges_from_map_nodes_number():
    '''It tests if the nodes number is conserved'''
      
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    Copy_map=nx.Graph()
    Copy_map.add_nodes_from(list(G.nodes))
    Copy_map=fn.Add_edges_from_map(Copy_map, map_dct, link_distance_conditional_probability)
    assert list(Copy_map.nodes())==list(G.nodes)
    
def test_Add_edges_from_map_Bernoulli_trials():
    
      
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=fn.Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    
    
    Copy_map=nx.Graph()
    Copy_map.add_nodes_from(list(G.nodes))
    Copy_map=fn.Add_edges_from_map(Copy_map, map_dct, link_distance_conditional_probability)
    
    dct_dist_link_copymap=fn.Dct_dist_link(list(Copy_map.edges()), map_dct)
    
    for i in range(nstep):
        p=link_distance_conditional_probability[(i+1)*step]
        LG=0
        for value in list(dct_dist.values()) :
            if  i*step<value<(i+1)*step:
                LG+=1
        LCopymap=0
        for value in list(dct_dist_link_copymap.values()) :
            if  i*step<value<(i+1)*step:
                LCopymap+=1
              
        assert LG*p-3*(p*(1-p)*LG)**0.5<=LCopymap<=LG*p+3*(p*(1-p)*LG)**0.5
    

        
#%% test_Break_strongest_nodes (2)

def test_Break_strongest_nodes_size():
    '''it tests len of the graph is kept constant'''
    
    edges=[(1,2), (3,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    len_before=len(G)
    fn.Break_strongest_nodes(G,2)
    len_after=len(G)
    assert len_after==len_before
    
def test_Break_strongest_nodes_maximum_value():
    '''it verify the nodes with the highest degree is under the threshold'''
    
    edges=[(1,2), (3,1),(1,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Break_strongest_nodes(G,2)
    deg=G.Degree_dct()
    assert max(deg.keys())<=2
    
def test_Break_strongest_nodes_untouchable_edges():
    '''it verify the edges which are not involved in the function are conserved'''
    
    edges=[(1,1), (1,2), (2,1), (3,4), (1,5),(6,7), (5,2), (2,7),(10,11)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Break_strongest_nodes(G,3)
    assert list(G.neighbors(4))==[3]


#%% test_Find_mode (1)
def test_Find_mode():
    '''the value calculated in the mode is the max'''
    a=[-1,-5,-0.1,-3,]
    assert a[fn.Find_mode(a)]==max(a)
    

            
#%% test_Equalize_strong_nodes (2)

def test_Equalize_strong_nodes_size():
    '''it tests length of the first graph is kept constant'''
    
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
    '''it verify if the  degree ratio of highest node of the first graph is lower than the second one'''
    
    edges=[(1,2), (3,1), (6,3), (4,2), (2,3), (1,4), (1,5), (5,4), (5,3), (1,6), (5,3), (5,2), (5,6), (7,1)]    
    G_strong=fn.SuperGraph()
    G_strong.add_edges_from(edges)
    edges=[(1,2), (5,7), (1,1), (1,2), (2,1), (3,4), (4,5), (2,4), (5,6), (1,6), (1,2), (6,2), (5,6), (7,6)]
    G_weak=fn.SuperGraph()
    G_weak.add_edges_from(edges)
    fn.Equalize_strong_nodes(G_strong, G_weak) 
    dct_degree_strong=G_strong.Degree_dct()
    degree_ratio_strong=G_strong.Degree_ratio()
    degree_ratio_weak=G_weak.Degree_ratio()
    mode=fn.Find_mode(degree_ratio_weak)
    i=max(dct_degree_strong.keys())
    while i>mode:
        assert degree_ratio_strong[i]<=degree_ratio_weak[i]
        i=i-1


#%% test_Max_prob_target (3)

def test_Max_prob_target_degree():
    ''''It verifies the degree of the target is the one given'''
    edge_probability=0.4
    n_node=100
    G=fn.SuperGraph(nx.fast_gnp_random_graph(n_node,edge_probability, seed=None, directed=False))
    
    G.Relable_nodes()
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    strenght_dct=G.Degree_dct()
    degree=edge_probability*n_node
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())
    step=max_dist/10    
    prob_distribution={}
    for i in range(10):
        prob_distribution[step*(i+1)]=0.4        
    target=fn.Max_prob_target (source,strenght_dct,degree,map_dct,prob_distribution,max_dist,G)
    assert len(list(G.neighbors(target)))==degree

def test_Max_prob_target_not_its_self():
    '''It verifies the taget is not the source'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    G.Relable_nodes()
    map_dct={0: np.array([ 0, 0]),
         1: np.array([-1, -1]),
         2: np.array([-0.5, -0.2]),
         3: np.array([1, 2]),
         4: np.array([5, 5])}
    
    
    source=0
    strenght_dct=G.Degree_dct()
    degree=3
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())

    step=max_dist/10    
    prob_distribution={}
    for i in range(10):
        prob_distribution[step*(i+1)]=0.4  
    for i in range(100):
        assert source!=fn.Max_prob_target (source,strenght_dct,degree,map_dct,prob_distribution,max_dist,G)
    
def test_Max_prob_target_is_not_its_neighbors():
    '''it verifies the target is not a source's neighbors'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    G.Relable_nodes()
    map_dct={0: np.array([ 0, 0]),
         1: np.array([-1, -1]),
         2: np.array([-0.5, -0.2]),
         3: np.array([1, 2]),
         4: np.array([5, 5])}
    source=0
    strenght_dct=G.Degree_dct()
    degree=2
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())

    step=max_dist/10    
    prob_distribution={}
    for i in range(10):
        prob_distribution[step*(i+1)]=0.4  
    for i in range(100):
        assert source!=fn.Max_prob_target (source,strenght_dct,degree,map_dct,prob_distribution,max_dist,G)


#%% test_Min_distance_target (5)

def test_Min_distance_target_is_the_nearest():
    '''It tests the target is the nearest node which is not itself or one of its neighbor'''
    edges=[(0, 1), (0, 3), (0, 6), (0, 8), (0, 9), (1, 2), (1, 4), (1, 9), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (4, 8), (5, 7), (6, 8), (7, 9)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    G.Relable_nodes()
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
    degree_dct=G.Degree_dct()
    neighbour_0=list(G.neighbors(0))
    target=fn.Min_distance_target (source,degree_dct,3,map_dct,neighbour_0)
    
    assert target==2
    
def test_Min_distance_target_is_not_its_self():
    '''It tests the target is not its self'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,4)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    G.Relable_nodes()
    map_dct={0: np.array([ 0, 0]),
         1: np.array([-1, -1]),
         2: np.array([-0.5, -0.2]),
         3: np.array([1, 2]),
         4: np.array([5, 5])}
    source=0
    degree_dct=G.Degree_dct()
    neighbour_0=list(G.neighbors(0))
    for i in range(100):
        assert source!=fn.Min_distance_target (source,degree_dct,3,map_dct,neighbour_0)
        
def test_Min_distance_target_is_not_its_neighbors():
    '''It test the target is not one of its neighbors'''
    edges=[(0,1),(0,2),(1,3),(1,2),(0,0)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    G.Relable_nodes()
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_dct=G.Degree_dct()
    neighbour_0=list(G.neighbors(0))
    for i in range(100):
        assert 2!=fn.Min_distance_target (source,degree_dct,2,map_dct,neighbour_0)
        
def test_Min_distance_target_degree_corrispodence():
    '''It tests the target degree is the one chosen'''
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,1),(4,2),(4,3)]
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    G.Relable_nodes()
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_dct=G.Degree_dct()
    neighbour_0=list(G.neighbors(0))
    target=fn.Min_distance_target (source,degree_dct,3,map_dct,neighbour_0)
    assert len(list(G.neighbors(target)))==3
    
 
#%%  test_Merge_small_component (2)

def test_Merge_small_component():
    '''It verifies all the components are bigger than the threshold'''
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    fn.Merge_small_component(G, 1,map_dct,threshold=3)
    for i in list(nx.connected_components(G)):
        assert len(i)>=3
        
def test_Merge_small_component_Exception():
    '''It test if raises an exception for empty list'''
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    with pytest.raises(Exception):
        fn.Merge_small_component(G, 0,map_dct,threshold=3)

 
#%% test_Link_2_ZeroNode (3)

    

def test_Link_2_ZeroNode_reduction():
    rn.seed(3)
    '''It verifies the right decreasing of  '0' degree ratio'''
    edges=[(0, 3), (0, 2), (0, 8), (0, 9), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 8), (2, 8), (8, 9)]
    
    G=fn.SuperGraph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(edges)
    G.Relable_nodes()
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    max_dist_link=max(dct_dist_link.values())
    nstep=10
    step=max_dist_link/nstep
    dct_dist=fn.Dct_dist(G, map_dct)
    distance_frequency=fn.Node_distance_frequency(dct_dist, nstep, step)
    prob_distribution=fn.Link_distance_conditional_probability(dct_dist_link, nstep, distance_frequency)
    
    max_dist_link=max(dct_dist_link.values())
    degree_dct=G.Degree_dct()
    degree_ratio_0_before=G.Degree_ratio()[0]
    n_links=4
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , degree_dct)
    degree_dct=G.Degree_dct()
    degree_ratio_0_after=G.Degree_ratio()[0]
    assert degree_ratio_0_before==degree_ratio_0_after+2/10

def test_Link_2_ZeroNode_increment():
    '''It verifies the right incrasing of  target degree ratio'''
    edges=[(0, 3), (0, 2), (0, 8), (0, 9), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 8), (2, 8), (8, 9)]
    
    G=fn.SuperGraph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(edges)
    G.Relable_nodes()
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    max_dist_link=max(dct_dist_link.values())
    nstep=10
    step=max_dist_link/nstep
    dct_dist=fn.Dct_dist(G, map_dct)
    distance_frequency=fn.Node_distance_frequency(dct_dist, nstep, step)
    prob_distribution=fn.prob_distribution=fn.Link_distance_conditional_probability(dct_dist_link, nstep, distance_frequency)
    
    
    degree_dct=G.Degree_dct()
    degree_ratio_4_before=G.Degree_ratio()[4]
    n_links=4
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , degree_dct)
    degree_dct=G.Degree_dct()
    degree_ratio_4_after=G.Degree_ratio()[4]
    assert degree_ratio_4_before==degree_ratio_4_after-2/10

def test_Link_2_ZeroNode_constant_degree_ratio():
    '''It tests that the degree ratio between the source and the target degree do not change'''
    edges=[(0, 3), (0, 2), (0, 8), (0, 9), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 8), (2, 8), (8, 9)]
    
    G=fn.SuperGraph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(edges)
    G.Relable_nodes()
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    max_dist_link=max(dct_dist_link.values())
    nstep=10
    step=max_dist_link/nstep
    dct_dist=fn.Dct_dist(G, map_dct)
    distance_frequency=fn.Node_distance_frequency(dct_dist, nstep, step)
    prob_distribution=fn.prob_distribution=fn.Link_distance_conditional_probability(dct_dist_link, nstep, distance_frequency)
    
    
    degree_dct=G.Degree_dct()
    degree_ratio_before=G.Degree_ratio()
    n_links=4
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , degree_dct)
    degree_dct=G.Degree_dct()
    degree_ratio_after=G.Degree_ratio()
    for i in range (1,4):
        assert degree_ratio_before[i]==degree_ratio_after[i]



#%% test_Remove_edge_of_degree

def test_Remove_edge_of_degree_local():
    '''It verifies the ratio that had been changed it is right '''
    edges=[(1,2), (3,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2), (1,7)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Remove_edge_of_degree(6,G)
    assert list(G.Degree_ratio())[5]==1/7
    
def test_Remove_edge_of_degree_total():
    ''' It verifies the ratio of all the degree are right after the application of the function'''
    edges=[(1,2), (3,1), (1,1), (1,2), (1,4), (5,4), (5,3), (1,6), (6,7), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Remove_edge_of_degree(4,G)
    assert list(G.Degree_ratio())==[0, 2/7, 3/7, 2/7]

def test_Remove_edge_of_degree_len():
    '''It tests the length of the Graph does not change'''
    edges=[(1,2), (3,1), (1,1), (1,2), (1,4), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    fn.Remove_edge_of_degree(4,G)
    assert len(G)==6
 

#%%Copymap_degree_correction(Copy_map,G,map_dct,max_dist_link,prob_distribution,Merge=False)

def test_Copymap_degree_correction():
    '''It test the final network has a degree distribution compatible with the distribution of the reference network'''
    
    G=fn.SuperGraph(nx.newman_watts_strogatz_graph(300, 3, p=0.4, seed=3))
    edge_probability=2*G.number_of_edges()/((len(G)-1)*(len(G)))
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=fn.Dct_dist_link(list(G.edges),map_dct)
    dct_dist=fn.Dct_dist(G=G, map_dct=map_dct)
    nstep=50
    
    step=max(dct_dist_link.values())/nstep

    distance_frequency=fn.Node_distance_frequency(dct_dist,nstep,step)
    distance_linking_probability=fn.Link_distance_conditional_probability(dct_dist_link,nstep,distance_frequency)


    ERG=nx.fast_gnp_random_graph(len(G),edge_probability, seed=None, directed=False)
    ERG=fn.SuperGraph(ERG)
    ERG_map_dct=nx.spring_layout(ERG, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    ERG_dct_dist_link=fn.Dct_dist_link(list(ERG.edges),map_dct)
    max_dist_link=max(ERG_dct_dist_link.values())
    
    ERG=fn.Copymap_degree_correction(ERG,G,ERG_map_dct,max_dist_link,distance_linking_probability,Merge=False)
    
    for i in range(len(ERG.Degree_ratio())):
        print(G.Degree_ratio()[i]-2*(G.Degree_ratio()[i])**0.5,'<=',ERG.Degree_ratio()[i],'<=',G.Degree_ratio()[i]+2*(G.Degree_ratio()[i])**0.5)
        assert (G.Degree_ratio()[i]-2*(G.Degree_ratio()[i])**0.5<=ERG.Degree_ratio()[i]<=G.Degree_ratio()[i]+2*(G.Degree_ratio()[i])**0.5)
    
               
 