# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:17:05 2021

@author: Guido
"""
# %%
##################################################################
#
#           ALL MY FUNCTION
#
######################################################################################

import random as rn
import networkx as nx
import pandas as pd
import numpy as np
import pytest
import function as fn
import math

rn.seed(3)
#%%1  Divide value
def Divide_value(file):
    '''It takes an array nx2 which represents the graph edges. It looks for any case in which
    two linked nodes are written in the same cell in a string variable and put the two variables
    in the two columns of the same row. It substitutes any full nan row with another row
    
    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    file : TYPE
        DESCRIPTION.

    '''
    file=pd.DataFrame(file)
    file=np.array(file)
    
    if file.shape[1] != 2:
        raise Exception('file shape should be with axis=1 equal to 2')
        
        
    for k in range(2):
        for i in range(len(file)):
            if type(file[i,k])==str:            
                j=0
                while j <(len(file[i,k])):
                    if file[i,k][j]==' ':
                        file[i]=[file[i,k][:j],file[i,k][j+1:len(file[i,k])]]
                    j=j+1
            elif math.isnan(file[i,k])==True and k==0:
                if type(file[i,k+1])!=str and math.isnan(file[i,k+1]):
                    file[i]=file[i-1]                       
    return file
#%%1  tests Divide value (3)
#ci sono un sacco di casi che possono capitare rispetto a come sono
#i dati in input quanto devo lavorarci sopra potrei non avere il file 
#in colonna, potrei avere una virgola nella tring o un punto
#TEST CHE non so come FARE: 
    '''-prende sia array, sia list sia Dataframe
       -se il file entra in un'altra forma mi da errore (sono mille le forme in cui potrebbe entrare)
        questa cosa la metto nella funzione o nel test?
   '''
   
#POSITIVE TEST
def test_divide_value():
    '''it tests if there is any string variable with a space inside'''
    file=[[0,134.0],['45643 3456',np.nan],[np.nan,'34 5'],[3,4.0]]
    file=Divide_value(file)
    for i in range(2):
        for j in range(len(file)):
            if type(file[j,i])==str:
                for k in range(len(file[j,i])):
                    assert file[j,i][k]!=' '
                    
def test_divide_value_nan_cell():
    ''' it looks for any nan variable'''
    file=[[0,134.0],[np.nan,np.nan],[np.nan,'34 5'],[3,4.0]]
    file=Divide_value(file)
    for i in range(2):
        for j in range(len(file)):
            if type(file[j,i])!=str:
                assert math.isnan(file[j,i])!=True
#Negative test
def test_divide_value_shape():
    '''it tests if the data are in the shape n*2'''
    file=[[0,134.0,4],['45643 3456',np.nan,5],[np.nan,'34 5',7],[3,4.0,8]]
    with pytest.raises(Exception):
        Divide_value(file)
    
              

    
#%%2  Edge_list

def Edge_list(file, number_of_edges):
    '''
    It takes a file of couples of number and return a list of a desired lenght(number_of_edge) of couple of numbers expressed as
    integers 
    

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    number_of_edge : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    edges : TYPE
        DESCRIPTION.

    '''
    file=pd.DataFrame(file)
    file=np.array(file)
    if number_of_edges>len(file):
        raise Exception('number_of_edges must be minor of the file length')    
    if file.shape[1] != 2:
        raise Exception('file shape should be with axis=1 equal to 2')
    edges = []
    for i in range((number_of_edges)):
        edges.append([int(file[i, 0]),int(file[i, 1])])
    edges = sorted(edges)
    return edges

#%%2  tests Edge_list (5)

#POSITIVE TEST
def test_edge_list_int():
    '''it tests if all the elements are integers'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    edges=Edge_list(file,2)
    for j in range(len(edges)):
        for k in range(2):
            assert type(edges[j][k])==int
                    
def test_edge_list_length():
    ''' it looks if the fil length is equal of number of edges'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=2
    edges=Edge_list(file,number_of_edges)
    assert len(edges)==number_of_edges
    
def test_edge_list_shape():
    '''it tests if the edges shape is n*2'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=2
    edges=Edge_list(file,number_of_edges)
    for j in range(len(edges)):
        assert len(edges[j])==2
#Negative test
def test_edge_list_too_long():
    ''' If the number of edges is> of the length of the file it will raise an exception'''    
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    number_of_edges=5
    with pytest.raises(Exception):
        Edge_list(file,number_of_edges)
def test_edge_list_input_shape():
    ''' If the shape of the file is not n*2 it will raise an exception'''    
    file=[[0,134.0,88],['45643',' 3456',7],[3,4.0,'33']]
    number_of_edges=5
    with pytest.raises(Exception):
        Edge_list(file,number_of_edges)

#%%3  Sorted_graph *
def Sorted_graph(G):
    '''It takes a graph G and it orders its nodes list

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    G : TYPE
        DESCRIPTION.

    '''


    G_sorted = nx.Graph()
    G_sorted.add_nodes_from(sorted(list(G.nodes())))
    G_sorted.add_edges_from(sorted(list(G.edges())))
    G = nx.Graph(G_sorted)
    return G
#%%3  test sorted_graph (2)
def test_sorted_graph_nodes():
    edges = []
    i=0
    while i<20:
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append([a, b])
        i+=1
    G = nx.Graph()
    G.add_edges_from(edges)
    TESTER = G
    G = Sorted_graph(G)
    assert list(G.nodes()) == sorted(list(TESTER.nodes()))



def test_sorted_graph_edges():
    edges = []
    for i in range(0, 100):
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append((a, b))
    G = nx.Graph()
    G.add_edges_from(edges)
    G = Sorted_graph(G)
    vecchio = list(G.edges())[0][0]
    for i in range(1, len(G)):
        assert list(G.edges())[i][0] <= list(G.edges())[i][1]
        assert vecchio <= list(G.edges())[i][0]
        vecchio = list(G.edges())[i][0]

#%%4  G_node_dct *

def G_node_dct(G):
    '''
    It creates a dictionary. The keys are represented by the number of the node the 
    values is its increasing position in the nodes list

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    g_node_dct= {}
    list_node = sorted(list(G.nodes()))
    for i in range(len(G)):
        g_node_dct[list_node[i]] = i
    return(g_node_dct)

#%%4  test G_node_dct (2)
def test_G_node_dct_keys():
    '''it looks if the  dct keys are the nodes of the graph in the sorted order,
    in this example the node list is [1,4,3,5,19],
    in the new dictioary we want to switch 3 and 4'''
    G=nx.Graph()
    G.add_edges_from([[1,4],[3,5],[4,3],[19,1]])
    node_dct=G_node_dct(G)
    for i in range(len(G)):
        assert sorted(list(G.nodes()))[i]==list(node_dct.keys())[i]

def test_G_node_dct_value():
    '''it looks if the dct values are the order in which the nodes comes out'''
    G=nx.Graph()
    G.add_edges_from([[1,4],[3,5],[4,3],[19,1]])
    node_dct=G_node_dct(G)
    list_node = sorted(list(G.nodes()))
    for i in range(len(G)):
        assert i==(node_dct[list_node[i]])

    

#%%5  Unfreeze_into_list
def Unfreeze_into_list(comunity):
    '''It takes a variable of n elements and transform each element in a list variables
    

    Parameters
    ----------
    comunity : TYPE
        DESCRIPTION. support item assignment

    Returns
    -------
    comunity : TYPE
        DESCRIPTION.

    '''
    for i in range(len(comunity)):
        comunity[i]=list(comunity[i])
    return comunity
#%%   tests Unfreeze_into_list(2)
'''Ad esempio qua per la funzione mi andrebbe bene sia un dct che una list come elemento in entrata
    ma quanto ha senso che io verifichi per tutte le variabili che supportano item assigment'''
def test_Unfreeze_into_list_is_a_list_1():
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    list_=[A,B,C,D]
    list_=Unfreeze_into_list(list_)
    for i in range(len(list_)):
        assert(type(list_[i])==list)

def test_Unfreeze_into_list_is_a_list_2():
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    dct_={0:A,1:B,2:C,3:D}
    dct_=Unfreeze_into_list(dct_)
    for i in range(len(dct_)):
        assert(type(dct_[i])==list)
#%%6  Set_comunity

def Set_community_number(G, community):
    '''It assigns to each node to the graph a community number. Node with the same number
    are in the same community, each number cannot belongs to different communities. G is the graph,
    while each entry of the community  variable represents node of the same community. It returns 
    a dictionary: the keys are the nodes numbers, the values are the numbers of the community
    they belong to.
    

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    comunity : TYPE
        DESCRIPTION.

    Returns
    -------
    comunity_number : TYPE
        DESCRIPTION.

    '''    
    community_number={}
    tot_elem=[]
    tester=0
    for j in range(len(community)):
        tot_elem+=community[j]
        
        for i in list(G.nodes()):
            if community[j].count(i)==1:
                community_number[i]=j
                tester+=1
    if tester!=len(G):
        raise ValueError('in the community at least one node of the graph is missing')
    tot_elem=len(set(tot_elem))
    if tot_elem!=len(G):
        raise Exception('The number of elements in the community is not the same of th number of nodes')
    
    return community_number
#%%   test_Set_community_number (6)

'''voglio testare che:
    tutti i nodi abbiano la loro corrispettiva comunità
    che gli elementi diversi della comunità sono lo stesso numero dei nodi (ha senso sia fare un raise nella funzione sia testarlo in maniera positiva che negativa)
    il numero di elementi unici dei valori del dizionario sia uguale alla grandezza della community'''

def test_Set_community_number_corrispondence():    
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=Set_community_number(G, community)
    for i in list(G.nodes()):
        assert community[community_number[i]].count(i)==1
        
def test_Set_community_number_length():
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=Set_community_number(G, community)
    assert len(community_number)==len(G)
    

        
def test_Set_community_number_doubble():
    '''the four is in two different communities'''
    community=[[1,3,6,9],[5,10],[7,4],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        Set_community_number(G, community)

def test_Set_community_number_missing_number():
    '''the eight is missing'''
    community=[[1,3,6,9],[5,10],[7],[2,4]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(ValueError):
        Set_community_number(G, community)

def test_Set_community_number_extra_number():
    '''the 999 is not a node number'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        Set_community_number(G, community)

def test_Set_community_number_all_wrong():
    community=[[1,3,6,4],[5,10],[7],[2,4,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        Set_community_number(G, community)
            
    
#%%7  Degree_dct *
def Degree_dct(G):
    '''It returns a dictionary. The keys are the degree of the graph from 0 to the maximum.
    The values are all and only the nodes with the key degree
    

    Parameters
    ----------
    G : networkx.classes.graph.Graph

    Returns
    -------
    strenght_dct : TYPE
        DESCRIPTION.

    '''
    strenght_dct={}
    for key in range(int(max(np.array(list(G.degree))[:,1])+1)):
        strenght_dct[key]=[]
    for i in list(G.nodes()):
        degree=len(list(G.neighbors(i)))-list(G.neighbors(i)).count(i) 
        for key in strenght_dct:
            if degree==key:
                strenght_dct[key].append(i)
    j=0
    while j==0:
        if len(strenght_dct[max(strenght_dct.keys())])==0:
            del(strenght_dct[max(strenght_dct.keys())])
        else:
            j=1
    return strenght_dct

#%%   test_Degree_dct (5)

def test_Degree_list_corrispondence():
    '''It verifies each node has a degree equal to its key'''
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=Degree_dct(G)
    for key in strenght_dct:
        for node in strenght_dct[key]:
            assert G.degree(node)-2*list(G.neighbors(node)).count(node)==key

def test_Degree_list_length():
    '''It looks if the elements of the dictionary values are the same of the of the graph nodes'''
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=Degree_dct(G)
    values=[]
    for i in strenght_dct:
        values+=set(strenght_dct[i])
    assert sorted(values)==list(range(1,11))
    
def test_Strenght_list_max_degree():
    '''It verify the highest key value is equal to the maximum degree'''
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=Degree_dct(G)
    assert len(list(strenght_dct.keys()))==max(np.array(list(G.degree))[:,1])+1
    
def test_Strenght_list_autoconnected_nodes():
    '''It looks that the function does not count as a neighbour the node its self in the case
    of an autoconnected node'''
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[3,3]])
    strenght_dct=Degree_dct(G)
    assert strenght_dct[0]==[2]
    assert strenght_dct[1]==[1,3]
    
def test_empty_high_key():
    '''it verify the last keys doesn't have 0 length'''
    G=nx.Graph()
    G.add_edges_from([[1,1],[2,2],[5,5],[8,8],[9,9],[10,10],[10,10],[7,7],[10,10],[0,1]])
    strenght_dct=Degree_dct(G)
    assert len(list(strenght_dct.keys()))==2


    
    
    
    
#%%8  Degree ratio

def Degree_ratio(strenght_dct):
    '''From a dictionary degre:[nodes], it returns the probability distribution of the nodes degree
    

    Parameters
    ----------
    strenght_list : TYPE
        DESCRIPTION.

    Returns
    -------
    degree_ratio : TYPE
        DESCRIPTION.

    '''
    degree_ratio=[]
    for key in (strenght_dct):
        degree_ratio.append(len(strenght_dct[key]))
    degree_ratio=np.array(degree_ratio)/sum(degree_ratio)
    return degree_ratio

#%%   test_Degree_ratio (4)
def test_Degree_ratio_length():
    strenght_dct={0: [2], 1: [3, 6, 8, 9, 10], 2: [5, 4, 7], 3: [], 4: [], 5: [1]}
    degree_ratio=Degree_ratio(strenght_dct)
    assert len(degree_ratio)==len(strenght_dct)
    
def test_Degree_ratio_I_axiom():
    '''It verifies the probability I axiom'''
    strenght_dct={0: [2], 1: [3, 6, 8, 9, 10], 2: [5, 4, 7], 3: [], 4: [], 5: [1]}
    degree_ratio=Degree_ratio(strenght_dct)
    for i in degree_ratio:
        assert i>=0
        
def test_Degree_ratio_II_axiom():
    '''It verifies the probability II axiom'''
    strenght_dct={0: [2], 1: [3, 6, 8, 9, 10], 2: [5, 4, 7], 3: [], 4: [], 5: [1]}
    degree_ratio=Degree_ratio(strenght_dct)
    assert 0.99999<sum(degree_ratio)<1
    
def test_Degree_ratio_III_axiom():
    '''It verifies the probability III axiom'''
    strenght_dct={0: [2], 1: [3, 6, 8, 9, 10], 2: [5, 4, 7], 3: [], 4: [], 5: [1]}
    degree_ratio=Degree_ratio(strenght_dct)
    strenght_dct_2={0: [2,1], 1: [3, 6, 8, 9, 10, 5, 4, 7], 2: [], 3: [], 4: [], 5: []}
    degree_ratio_2=Degree_ratio(strenght_dct_2)
    assert degree_ratio[0]+degree_ratio[5]==degree_ratio_2[0]
    assert degree_ratio[1]+degree_ratio[2]==degree_ratio_2[1]
    
    
    
                        
#%%9  Time evolution (3) *
def Time_evolution(G,step, feature):
    '''
    It takes back information related to the network G feature along its size increasing.
    Starting from the graph size equal to the number of edge indicated by the step variable,
    it increases the number of edge of the step variable at each loop. 
    The outputs are three lists related to the size of the graph at each step, 
    the value of the feature of each node at each step and
    the mean value and its standard deviation of the feature at each step.
    In the case of the feature='degree' it instead of the value of each node it returns
    the degree distribution. 

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    feature : TYPE
        DESCRIPTION.

    Returns
    -------
    size : TYPE
        DESCRIPTION.
    value_time_evolution : TYPE
        DESCRIPTION.
    value_time_evolution_mean : TYPE
        DESCRIPTION.

    '''
    value_time_evolution=[]
    time_step=int(len(G.edges)/step)
    value_time_evolution_mean=[]
    edges=list(G.edges())
    size=[]
    if feature=='degree':
        for i in range(time_step):      
            G=nx.Graph(edges[:(i+1)*step])
            G=Sorted_graph(G)
            g_node_dct= G_node_dct(G)
            G=nx.relabel_nodes(G,g_node_dct, copy=True)
            value_time_evolution.append(Degree_ratio(Degree_dct(G)))
            value_time=np.array(list((getattr(nx, feature)(G))))[:,1]
            value_time_evolution_mean.append([np.mean(value_time),np.std(value_time)])
            size.append(len(G))  
        
        for i in range(len(value_time_evolution)):
            while len(value_time_evolution[i])<len(value_time_evolution[-1]):
                value_time_evolution[i]=np.append(value_time_evolution[i],0)
        value_time_evolution_mean=np.array(value_time_evolution_mean)
        value_time_evolution=np.array(value_time_evolution)
        size=np.array(size)
        return size, value_time_evolution,value_time_evolution_mean
    else:
        for i in range(time_step):
            G=nx.Graph(edges[:(i+1)*step])
            G=Sorted_graph(G)
            g_node_dct=G_node_dct(G)
            G=nx.relabel_nodes(G,g_node_dct, copy=True)
            value_time=np.array(list((getattr(nx, feature)(G)).items()))[:,1]
            value_time_evolution_mean.append([np.mean(value_time),np.std(value_time)])
            value_time_evolution.append(value_time)
            size.append(len(G))
        value_time_evolution_mean=np.array(value_time_evolution_mean)
        return size, value_time_evolution,value_time_evolution_mean
#%%   test_Time_evolution
def test_Time_evolution_size():
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size, value_time_evolution,value_time_evolution_mean=Time_evolution(G, step, feature)
    assert len(size)==nstep and len(value_time_evolution)==nstep and len(value_time_evolution_mean)==nstep
    
    feature='betweenness_centrality'
    size, value_time_evolution,value_time_evolution_mean=Time_evolution(G, step, feature)
    assert len(size)==nstep and len(value_time_evolution)==nstep and len(value_time_evolution_mean)==nstep

def test_Time_evolution_size_increasing_size():
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size=Time_evolution(G, step, feature)[0]
    for i in range(nstep-1):
        assert size[i]<size[i+1]
    
    feature='betweenness_centrality'
    size, value_time_evolution,value_time_evolution_mean=Time_evolution(G, step, feature)
    for i in range(nstep-1):
        assert size[i]<size[i+1]

def test_Time_evolution_size_degree_constant_len():
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    value_time_evolution=Time_evolution(G, step, feature)[1]
    for i in range(nstep):
        assert len(value_time_evolution[i])==4
    
#%%10 Dct_dist_link *
def Dct_dist_link(edges,map_dct):
    '''It calculates all the distances of the nodes linked together whose position is decribed by
    the dictionary map
    

    Parameters
    ----------
    edges : TYPE
        DESCRIPTION.
    map_dct : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for i in edges:
        if edges.count((i[1],i[0])):
            edges.remove(i)
        dct_dist_link={}
    for i in edges:
        x0=map_dct[i[0]]
        x1=map_dct[i[1]]
        dist=np.linalg.norm(x0-x1)
        dct_dist_link[i]=dist
    return(dct_dist_link)

#%%   test_List_dist_link (4)

    
def test_List_dist_link_length():
    '''it verify in the dictionary are not present symmetric object (e.g.: (1,2), (2,1))
    and it removes autoedges (e.g.: (1,1))
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist_link(edges, map_dct)
    assert len(dct_dist_link)==2

def test_List_dist_link_non_negativity():
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist_link(edges, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
        
def test_List_dist_link_simmetry():
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    segde=[]
    for i in range(len(edges)):
        segde.append((edges[i][1],edges[i][0]))                     
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link1=Dct_dist_link(edges, map_dct)
    dct_dist_link2=Dct_dist_link(segde, map_dct)
    assert list(dct_dist_link2.values())==list(dct_dist_link1.values())
    
def test_List_dist_link_triangular_inequality():
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist_link(edges, map_dct)
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
                        
            
            
            
    
    


#%%11 Dct_dist *
def Dct_dist(G,map_dct):
    '''
    It returns all the distance among the nodes, even the not linked one. 
    It exploits a dictionary map of the position of the nodes
    

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    map_dct : TYPE
        DESCRIPTION.

    Returns
    -------
    list_dist : TYPE
        DESCRIPTION.

    '''
    
    dct_dist={}
    list_nodes=list(G.nodes())
    for i in range(len(list_nodes)):
        j=i+1
        while j < (len(G)):
            x0=map_dct[list_nodes[i]]
            x1=map_dct[list_nodes[j]]
            dist=np.linalg.norm(x0-x1)
            dct_dist[(list_nodes[i],list_nodes[j])]=(dist)
            j=j+1
    return dct_dist

#%%   test_Dct_dist (3)
def test_List_dist_length():
    '''it verify in the dictionary are not present symmetric object (e.g.: (1,2), (2,1))
    and it doesn't remove autoedges (e.g.: (1,1))
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1), (1,4)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    list_dist=Dct_dist(G, map_dct)
    assert len(list_dist)==6

def test_List_dist_non_negativity():
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist(G, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
    
def test_List_dist_triangular_inequality():
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=Dct_dist(G, map_dct)
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
#%%12 Node_distance_frequency *
def Node_distance_frequency(dct_dist,nstep,step):
    '''It returns a binned not normalized nodes distance distribution. 
    It puts all the elements>nstep*step in the last bin 
    '''
    n=[0]*nstep
    for key in (dct_dist):
        for i in range (nstep):
            if dct_dist[key]>i*step and dct_dist[key]<(i+1)*step:
                n[i]=n[i]+1
    n[nstep-1]=n[nstep-1]+len(dct_dist)-sum(n)
    node_distance_frequency=np.array(n)
    return node_distance_frequency  
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
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
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
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
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
    node_distance_frequency_1=Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    
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
    node_distance_frequency_2=Node_distance_frequency(dct_dist,nstep,step)/len(dct_dist)
    for i in range(0,50,2):
        assert node_distance_frequency_1[i]+node_distance_frequency_1[i+1]==node_distance_frequency_2[int(i/2)]

#%%13 Link_distance_conditional_probability *

def Link_distance_conditional_probability(dct_dist_link,nstep,distance_frequency):
    step=max(dct_dist_link.values())/nstep
    n_link=[0]*nstep
    for key in (dct_dist_link):
        for i in range (nstep):
            if dct_dist_link[key]>i*step and dct_dist_link[key]<=(i+1)*step:
                n_link[i]=n_link[i]+1
        distance_link_frequency=np.array(n_link)
    link_distance_probability={}
    for i in range(len(distance_link_frequency)):
        if distance_frequency[i]!=0:
            link_distance_probability[(i+1)*step]=(distance_link_frequency[i]/(distance_frequency[i]))
        elif distance_frequency[i]==0 and distance_link_frequency[i]==0:
            link_distance_probability[(i+1)*step]=(0)
        else:
            print(i)
            raise ZeroDivisionError('so qua')
            
    return link_distance_probability
    
#%%   test_Link_distance_probability (4)

   
def test_Link_distance_probability_I_axiom():
    '''It verifies the probability I axiom'''
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
    link_distance_probability=list(link_distance_conditional_probability.values())*node_distance_frequency
    for i in link_distance_probability:
        assert i>=0
        
def test_Link_distance_probability_II_axiom():
    '''It verifies the probability II axiom'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    link_distance_probability=list(link_distance_conditional_probability.values())*node_distance_frequency/len(dct_dist_link)
    assert 0.99999<sum(link_distance_probability)<=1
    
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
    dct_dist=Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    assert list(link_distance_conditional_probability.keys())[-1]==max(list(dct_dist_link.values()))
    

#%%14 Add_edges_from_map(G,map_dct,distance_linking_probability)                           
def Add_edges_from_map(G,map_dct,distance_linking_probability):
    '''
    It creates links among nodes of graph with no links. 
    It exploits the nodes positions informations conteined in the map dictionary, map_dct,
    and the conditional distribution probability to make a link in function of the distance.
    

    Parameters
    ----------
    Copy_map : TYPE
        DESCRIPTION.
    map_dct : TYPE
        DESCRIPTION.
    prob_distribution : TYPE
        DESCRIPTION.
    list_dist_link : TYPE
        DESCRIPTION.
    nstep : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.

    Returns
    -------
    Copy_map : TYPE
        DESCRIPTION.

    '''
    i=0
    step=list(distance_linking_probability.keys())[0]
    nodes=list(G.nodes())
    while i<len(G):
        j=i+1
        #j=0
        while j<len(nodes):        
            x0=map_dct[nodes[i]]
            x1=map_dct[nodes[j]]
            dist=np.linalg.norm(x0-x1)
            if dist<=max(distance_linking_probability.keys()):
                for k in range(len(distance_linking_probability)):
                    if dist>k*step and dist<(k+1)*step:                    
                        uniform=rn.uniform(0,1)
                        if uniform<= list(distance_linking_probability.values())[k]:
                            G.add_edge(nodes[i],nodes[j])
            j=j+1
        i=i+1  
    return G

#%%   test_Copymap_linking (2)


def test_Add_edges_from_map_nodes_number():
    '''It tests if the nodes number is conserved'''
      
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=Dct_dist(G,map_dct)
    dct_dist_link=Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    Copy_map=nx.Graph()
    Copy_map.add_nodes_from(list(G.nodes))
    Copy_map=Add_edges_from_map(Copy_map, map_dct, link_distance_conditional_probability)
    assert list(Copy_map.nodes())==list(G.nodes)
    
def test_Add_edges_from_map_Bernoulli_trials():
    '''It tests if the nodes number is conserved'''
      
    G=nx.Graph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist=Dct_dist(G,map_dct)
    dct_dist_link=Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    
    
    Copy_map=nx.Graph()
    Copy_map.add_nodes_from(list(G.nodes))
    Copy_map=Add_edges_from_map(Copy_map, map_dct, link_distance_conditional_probability)
    
    dct_dist_link_copymap=Dct_dist_link(list(Copy_map.edges()), map_dct)
    
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
    
'''voglio testare che preso un  grafo il numero di linkati è uguale a più o meno il numero di grafi  
    il numero di nodi rimane lo stesso
    
'''
#%%15 Break_strong_nodes

def Break_strongest_nodes(G, threshold):
    '''
    It breaks randomly links of nodes with the highest degree
    till the node with the maximum degree has a value under or equal to the trheshold
    

    Parameters
    ----------
    G_strong : TYPE
        DESCRIPTION.
    G_weak : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    dct_degree=Degree_dct(G)
    while threshold<max(np.array(list(G.degree))[:,1]):
        source=rn.choice(list(dct_degree.values())[len(dct_degree)-1])
        node=rn.choice(list(G.neighbors(source)))
        G.remove_edge(source,node)
        dct_degree=Degree_dct(G)
        
#%% test_Break_strongest_nodes (2)
''' testo che il grafo abbia stessa grandezza, che il massimo del max degree sia sotto threshold 
'''
def test_Break_strongest_nodes_size():
    '''it tests len of the graph is kept constant'''
    
    edges=[(1,2), (3,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    len_before=len(G)
    Break_strongest_nodes(G,2)
    len_after=len(G)
    assert len_after==len_before
    
def test_Break_strongest_nodes_maximum_value():
    '''it verify the nodes with the highest degree is under the threshold'''
    
    edges=[(1,2), (3,1),(1,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    Break_strongest_nodes(G,2)
    deg=Degree_dct(G)
    assert max(deg.keys())<=2


#%%16 Find_mode(pfd) 
def Find_mode(pdf):
    '''
    It returns the x value of the maximum of function p(x). Where x represent a bin 

    Parameters
    ----------
    pdf : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    mode : TYPE
        DESCRIPTION.

    '''
    max_pdf=max(pdf)
    mode=np.nan
    for i in range(len(pdf)):
        if pdf[i]>=max_pdf:
            max_pdf=pdf[i]
            mode=i
    if math.isnan(mode)==True:
        raise ValueError('it could not find the mode')
    return mode

#%% test_Find_mode (1)
def test_Find_mode():
    a=[-1,-5,-0.1,-3]
    assert a[Find_mode(a)]==max(a)
    
    
        
#%%17 Equalize_strong_nodes
def Equalize_strong_nodes(G_strong, G_weak):
    '''
    It compares two graph. It takes the strongest node of the first graph which have a degree value major 
    then the degree mode of the second graph. It remove random links of these nodes untill the degree ratio
    of the the first graph (G_strong) of these high degree values are above the ones of the second graph(G_weak)

    Parameters
    ----------
    G_strong : TYPE
        DESCRIPTION.
    G_weak : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    dct_degree_strong=Degree_dct(G_strong)
    dct_degree_weak=Degree_dct(G_weak)
    degree_ratio_strong=Degree_ratio(dct_degree_strong)
    degree_ratio_weak=Degree_ratio(dct_degree_weak)
    
    threshold=Find_mode(degree_ratio_weak)
    
    print('w',dct_degree_weak)
    print('S', dct_degree_strong)
    i=max(dct_degree_weak.keys())
    print(i)
    if i<max(np.array(list(G_strong.degree))[:,1]):
        
        Break_strongest_nodes(G_strong, i)        
    while i>threshold:
        print(i)
        while (degree_ratio_strong[i])>(degree_ratio_weak[i]):
            print(i)
            
            source=rn.choice(dct_degree_strong[i])
            node=rn.choice(list(G_strong.neighbors(source)))
            
            G_strong.remove_edge(source,node)              
            dct_degree_strong=Degree_dct(G_strong)
            degree_ratio_strong=Degree_ratio(dct_degree_strong)            
        i=i-1
            
#%% test_Equalize_strong_nodes (2)

'''voglio verificare il numero di nodi di G_strong sia lo stesso e che i degree massimi siano minori dell'altro grafo'''

def test_Equalize_strong_nodes_size():
    '''it tests len of the first graph is kept constant'''
    
    edges=[(1,2), (3,1), (6,3), (4,2), (2,3), (1,4), (1,5), (5,4), (5,3), (1,6), (5,3), (5,2), (5,6), (7,1)]    
    G_strong=nx.Graph()
    G_strong.add_edges_from(edges)
    edges=[(1,2), (5,7), (1,1), (1,2), (2,1), (3,4), (4,5), (2,4), (5,6), (1,6), (1,2), (6,2), (5,6), (7,6)]
    G_weak=nx.Graph()
    G_weak.add_edges_from(edges)
    len_before=len(G_strong)
    Equalize_strong_nodes(G_strong, G_weak) 
    len_after=len(G_strong)
    assert len_after==len_before
    
def test_Equalize_strong_nodes_maximum_value():
    '''it verify if the  degree ratio of highest node of the first graph is lower than the second one'''
    
    edges=[(1,2), (3,1), (6,3), (4,2), (2,3), (1,4), (1,5), (5,4), (5,3), (1,6), (5,3), (5,2), (5,6), (7,1)]    
    G_strong=nx.Graph()
    G_strong.add_edges_from(edges)
    edges=[(1,2), (5,7), (1,1), (1,2), (2,1), (3,4), (4,5), (2,4), (5,6), (1,6), (1,2), (6,2), (5,6), (7,6)]
    G_weak=nx.Graph()
    G_weak.add_edges_from(edges)
    Equalize_strong_nodes(G_strong, G_weak) 
    dct_degree_strong=fn.Degree_dct(G_strong)
    dct_degree_weak=fn.Degree_dct(G_weak)
    degree_ratio_strong=fn.Degree_ratio(dct_degree_strong)
    degree_ratio_weak=fn.Degree_ratio(dct_degree_weak)
    mode=Find_mode(degree_ratio_weak)
    i=max(dct_degree_strong.keys())
    while i>mode:
        assert degree_ratio_strong[i]<=degree_ratio_weak[i]
        i=i-1

#%%18 Max_prob_target

    
def Max_prob_target (source,strenght_dct,degree,map_dct,distance_linking_probability,max_dist,G):
    x0=map_dct[source]
    max_prob=-1
    target=-5

    step=max_dist/len(distance_linking_probability)
    for i in strenght_dct[degree]:
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)
        if dist<max_dist:
            for k in range(len(distance_linking_probability)):
                if dist>k*step and dist<(k+1)*step:
                    prob=list(distance_linking_probability.values())[k]
                    if prob>max_prob and i!=source and list(G.neighbors(source)).count(i)!=1:
                        
                        max_prob=prob
                        target=i                    

    
    if target==-5:
        (print('min'))
        target=fn.Min_distance_target(source, strenght_dct,degree, map_dct, list(G.neighbors(source)))
  
    return target

#%% test_Max_prob_target (3)
'''
potrei aggiungere bernoulli trials per verificare i link ad una determinata distanza
'''
def test_Max_prob_target_degree():
    edge_probability=0.4
    n_node=100
    G=nx.fast_gnp_random_graph(n_node,edge_probability, seed=None, directed=False)
    g_node_dct=fn.G_node_dct(G)    
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    strenght_list=fn.Degree_dct(G)[edge_probability*n_node]
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())
    step=max_dist/10    
    prob_distribution={}
    for i in range(10):
        prob_distribution[step*(i+1)]=0.4        
    target=fn.Max_prob_target (source,strenght_list,map_dct,prob_distribution,max_dist,G)
    assert len(list(G.neighbors(target)))==edge_probability*n_node

def test_Max_prob_target_not_its_self():
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    strenght_list=fn.Degree_dct(G)[3]
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())

    step=max_dist/10    
    prob_distribution={}
    for i in range(10):
        prob_distribution[step*(i+1)]=0.4  
    
    with pytest.raises(ValueError):
        fn.Max_prob_target (source,strenght_list,map_dct,prob_distribution,max_dist,G)
    
def test_Max_prob_target_is_not_its_neighbors():
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    strenght_list=fn.Degree_dct(G)[2]
    dct_dist_link=fn.Dct_dist_link(list(G.edges()), map_dct)
    max_dist=max(dct_dist_link.values())

    step=max_dist/10    
    prob_distribution={}
    for i in range(10):
        prob_distribution[step*(i+1)]=0.4  
    with pytest.raises(ValueError):
        fn.Max_prob_target (source,strenght_list,map_dct,prob_distribution,max_dist,G)
        
#%%19 Min_distance_target (source,strenght_dct,degree,map_dct,source_neighbour_list)

def Min_distance_target (source,strenght_dct,degree,map_dct,source_neighbour_list):
    '''
    Given a distance map of the nodes, From a starting vertice (the source) of the graph it returns the nearest node 
    with a given degree and which is not already linked to the source

    Parameters
    ----------
    source : TYPE
        DESCRIPTION.
    strenght_list : TYPE
        DESCRIPTION.
    map_dct : TYPE
        DESCRIPTION.
    source_neighbour_list : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    target : TYPE
        DESCRIPTION.

    '''
    x0=map_dct[source]
    min_=999
    target=-5
    for i in strenght_dct[degree]:
        
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)
        print(dist,i,len(source_neighbour_list),(strenght_dct[degree]),(source_neighbour_list.count(i)!=1))
        if dist!=0 and source_neighbour_list.count(i)!=1:
            
            if dist<min_:
                
                min_=dist
                target=i
    if target<0:
        target=rn.choice(rn.choice(list(strenght_dct.values())))
        #raise ValueError('it could not find the target')
    return target

#%% test_Min_distance_target (5)

''' voglio verificare che è effettivamente il più vicino e che non è un suo vicino 
che non è se stesso e che ha il grado richiesto
se strength list vuota mi da un errore
'''

def test_Min_distance_target_is_the_nearest():
    edges=[(0, 1), (0, 3), (0, 6), (0, 8), (0, 9), (1, 2), (1, 4), (1, 9), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (4, 8), (5, 7), (6, 8), (7, 9)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=fn.G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_dct=fn.Degree_dct(G)
    neighbour_0=list(G.neighbors(0))
    target=fn.Min_distance_target (source,degree_dct,3,map_dct,neighbour_0)
    print(target)
    assert target==9
    
def test_Min_distance_target_is_not_its_self():
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_3=Degree_dct(G)[3]
    neighbour_0=list(G.neighbors(0))
    with pytest.raises(ValueError):
        Min_distance_target (source,degree_3,map_dct,neighbour_0)
        
def test_Min_distance_target_is_not_its_neighbors():
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_3=Degree_dct(G)[2]
    neighbour_0=list(G.neighbors(0))
    with pytest.raises(ValueError):
        Min_distance_target (source,degree_3,map_dct,neighbour_0)
        
def test_Min_distance_target_degree_corrispodence():
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0),(4,1),(4,2),(4,3)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_3=Degree_dct(G)[3]
    neighbour_0=list(G.neighbors(0))
    target=Min_distance_target (source,degree_3,map_dct,neighbour_0)
    assert len(list(G.neighbors(target)))==3
    
    
    
    
def test_Min_distance_target_empty_list():
    edges=[(0,1),(0,2),(0,3),(1,2),(0,0)]
    G=nx.Graph()
    G.add_edges_from(edges)
    g_node_dct=fn.G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    source=0
    degree_3=fn.Degree_dct(G)[0]
    neighbour_0=list(G.neighbors(0))
    with pytest.raises(ValueError):
        Min_distance_target (source,degree_3,map_dct,neighbour_0)
    
    
    
#%%20 Merge_small_component
def Merge_small_component(G, deg,map_dct,threshold):
    print('deg',deg)
    i=0
    all_components=list(nx.connected_components(G))
    while i < len(all_components):
    
        if len(list(all_components)[i])<threshold:
            source=rn.choice(list(all_components[i]))
            list_nodes=Degree_dct(G)[deg]
            if len(list_nodes)==0:
                raise Exception('the node list is empty')
                
            target=Min_distance_target(source,fn.Degree_dct(G),deg,map_dct,list(G.neighbors(source)))
            G.add_edge(source,target)
            
        i=i+1
        
#%%  test_Merge_small_component (2)
'''voglio verificare che funzioni e che quindi non esistano componenti con dimensione più piccola di un tot,
 voglio verificare che sia alzato un errore se la lista con un determinato grado finisce'''

def test_Merge_small_component():
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    Merge_small_component(G, 1,map_dct,threshold=3)
    for i in list(nx.connected_components(G)):
        assert len(i)>=3
        
def test_Merge_small_component_Exception():
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    with pytest.raises(Exception):
        Merge_small_component(G, 0,map_dct,threshold=3)

    
#%%21 Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G,n_links, degree_dct)
def Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G,n_links, degree_dct):
    '''It takes an isolated node of the graph and it creates n links (n=n_links) with n different nodes.
    The degree of the nodes are 0, 1, 2 ... n-1. At in the of the process the number of nodes with degree
    equal to n increase of 2 and the number of isolated nodes decrease of two. 
    The degree ratio of the other values remains constant. The attachment rule of the links follow the
    Max_prob_target function.
    

    Parameters
    ----------
    source : TYPE
        DESCRIPTION.
    map_dct : TYPE
        DESCRIPTION.
    prob_distribution : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    max_dist_link : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.
    degree_dct : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    source=rn.choice(degree_dct[0])
    for i in range(n_links):
                      
        target=fn.Max_prob_target(source,degree_dct,i,map_dct,prob_distribution,max_dist_link,G)
        print(source,target)
        G.add_edge(source, target)
        degree_dct=Degree_dct(G)
#%% test_Link_2_ZeroNode (3)

''' voglio testare che il degree ratio segue andamento voluto, che i degree dei nodi sono tutti presenti
    
'''
def test_Link_2_ZeroNode_reduction():
    edges=[(0, 3), (0, 2), (0, 8), (0, 9), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 8), (2, 8), (8, 9)]
    
    G=nx.Graph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(edges)
    g_node_dct=fn.G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    max_dist_link=max(dct_dist_link.values())
    nstep=10
    step=max_dist_link/nstep
    dct_dist=fn.Dct_dist(G, map_dct)
    distance_frequency=fn.Node_distance_frequency(dct_dist, nstep, step)
    prob_distribution=fn.Link_distance_conditional_probability(dct_dist_link, nstep, distance_frequency)
    
    max_dist_link=max(dct_dist_link.values())
    degree_dct=fn.Degree_dct(G)
    degree_ratio_0_before=fn.Degree_ratio(degree_dct)[0]
    n_links=4
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , degree_dct)
    degree_dct=fn.Degree_dct(G)
    degree_ratio_0_after=fn.Degree_ratio(degree_dct)[0]
    assert degree_ratio_0_before==degree_ratio_0_after+2/10

def test_Link_2_ZeroNode_increment():
    edges=[(0, 3), (0, 2), (0, 8), (0, 9), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 8), (2, 8), (8, 9)]
    
    G=nx.Graph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(edges)
    g_node_dct=fn.G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    max_dist_link=max(dct_dist_link.values())
    nstep=10
    step=max_dist_link/nstep
    dct_dist=fn.Dct_dist(G, map_dct)
    distance_frequency=fn.Node_distance_frequency(dct_dist, nstep, step)
    prob_distribution=fn.prob_distribution=fn.Link_distance_conditional_probability(dct_dist_link, nstep, distance_frequency)
    
    
    degree_dct=fn.Degree_dct(G)
    degree_ratio_4_before=fn.Degree_ratio(degree_dct)[4]
    n_links=4
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , degree_dct)
    degree_dct=fn.Degree_dct(G)
    degree_ratio_4_after=fn.Degree_ratio(degree_dct)[4]
    assert degree_ratio_4_before==degree_ratio_4_after-2/10

def test_Link_2_ZeroNode_constant_degree_ratio():
    edges=[(0, 3), (0, 2), (0, 8), (0, 9), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 8), (2, 8), (8, 9)]
    
    G=nx.Graph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(edges)
    g_node_dct=fn.G_node_dct(G)
    G=nx.relabel_nodes(G,g_node_dct, copy=True)
    
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    max_dist_link=max(dct_dist_link.values())
    nstep=10
    step=max_dist_link/nstep
    dct_dist=fn.Dct_dist(G, map_dct)
    distance_frequency=fn.Node_distance_frequency(dct_dist, nstep, step)
    prob_distribution=fn.prob_distribution=fn.Link_distance_conditional_probability(dct_dist_link, nstep, distance_frequency)
    
    
    degree_dct=fn.Degree_dct(G)
    degree_ratio_before=fn.Degree_ratio(degree_dct)
    n_links=4
    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G, n_links , degree_dct)
    degree_dct=fn.Degree_dct(G)
    degree_ratio_after=fn.Degree_ratio(degree_dct)
    for i in range (1,4):
        assert degree_ratio_before[i]==degree_ratio_after[i]
    
               
#%% Copymap_degree_correction():               
def Copymap_degree_correction(Copy_map,G,map_dct,step,max_dist_link,prob_distribution):
    degree_dct_G=Degree_dct(G)
    degree_ratio_G=Degree_ratio(degree_dct_G)
    Copycat=nx.Graph(Copy_map)

    Break_strongest_nodes(Copycat, max(np.array(list(G.degree()))[:,1]))
    print(G,'B')
    Equalize_strong_nodes(Copycat, G)
    print(G,'E')
     
    rn.seed(3)
    mode=Find_mode(degree_ratio_G)
    print(G,'mode')
    degree_dct_Copycat=Degree_dct(Copycat)
    degree_ratio_Copycat=Degree_ratio(Copycat)
    
    while len(degree_dct_Copycat[0])>0:
    
        print(len(degree_dct_Copycat[0]))
        source=rn.choice(degree_dct_Copycat[0])
        if(degree_ratio_G[mode])< degree_ratio_Copycat[mode]:
            Link_2_ZeroNode(source,map_dct, prob_distribution, step, max_dist_link,Copycat,mode, degree_ratio_Copycat)
            degree_ratio_Copycat=Degree_ratio(Copycat)
            degree_dct_Copycat=Degree_dct(Copycat)
              
        else:
            Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,Copycat,mode, degree_dct_Copycat)
            degree_ratio_Copycat=Degree_ratio(Copycat)
 
            
            if len(degree_dct_Copycat[0])>0:
                target=Max_prob_target(source,degree_dct_Copycat,0,map_dct,prob_distribution,max_dist_link,Copycat)
                print('source',source,'target=', target)
                Copycat.add_edge(source, target)
                degree_dct_Copycat=Degree_dct(Copycat)
                degree_ratio_Copycat=Degree_ratio(Copycat)
            else:
                target=(fn.Max_prob_target(source,degree_dct_Copycat,2,map_dct,prob_distribution,step,max_dist_link,Copycat))
                Copycat.add_edge(source, target)
                degree_dct_Copycat=Degree_dct(Copycat)
                degree_ratio_Copycat=Degree_ratio(Copycat) 
                
            
        while degree_ratio_Copycat[1]> degree_ratio_G[1]:
            source=rn.choice(degree_dct_Copycat[1])            
            target=(fn.Max_prob_target(source,degree_dct_Copycat,2,map_dct,prob_distribution,step,max_dist_link,Copycat))
            Copycat.add_edge(source, target)
            degree_dct_Copycat=Degree_dct(Copycat)
            degree_ratio_Copycat=Degree_ratio(Copycat) 
            
        while(degree_ratio_Copycat[2])> (degree_ratio_Copycat[2]):
            source=rn.choice(degree_dct_Copycat[2])
            node=rn.choice(list(Copycat.neighbors(source)))
            Copycat.remove_edge(source,node)
            degree_dct_Copycat=Degree_dct(Copycat)
            degree_ratio_Copycat=Degree_ratio(Copycat) 
            
        while degree_ratio_Copycat[5]> degree_ratio_Copycat[5]:
            source=rn.choice(degree_dct_Copycat[5])
            node=rn.choice(list(Copycat.neighbors(source)))
            Copycat.remove_edge(source,node)
            degree_dct_Copycat=Degree_dct(Copycat)
            degree_ratio_Copycat=Degree_ratio(Copycat) 
            
    Merge_small_component(Copycat,deg=1, map_dct=map_dct, threshold=3)

            
    Copycat_dct=G_node_dct(Copycat)
    Copycat=nx.relabel_nodes(Copycat,Copycat_dct, copy=True)
    return Copycat
# %% tests sorted graph

# DOMANDONA MA COME FACCIO A TESTARE IL GRAFO SE PUò ESSERE INIZIALIZZATO IN MOLTI MODI
# ma anche nelle funzioni test devo fissare un seed

'''
def test_nodes_sorted_graph():
    edges = []
    for i in range(0, 100):
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append((a, b))
    G = nx.Graph()
    G.add_edges_from(edges)
    TESTER = G
    G = sorted_graph(G)
    assert list(G.nodes()) == sorted(list(TESTER.nodes()))


# ha senso fare una funzione test così lunga?
def test_edges_sorted_graph():
    edges = []
    for i in range(0, 100):
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append((a, b))
    G = nx.Graph()
    G.add_edges_from(edges)
    G = sorted_graph(G)
    vecchio = list(G.edges())[0][0]
    for i in range(1, len(G)):
        assert list(G.edges())[i][0] <= list(G.edges())[i][1]
        assert vecchio <= list(G.edges())[i][0]
        vecchio = list(G.edges())[i][0]


def test_lenght_sorted_graph():
    edges = []
    for i in range(0, 100):
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append((a, b))
    G = nx.Graph()
    G.add_edges_from(edges)
    TESTER = G
    sorted_graph(G)
    assert len(G) == len(TESTER)'''

#%%
def edge_list_random(file, number_of_edge):
    '''mettere caso troppo alto il numero'''
    if file.shape[1] != 2:
        raise BaseException('file shape should be with axis=1 equal to 2')
    edges = []
    np.random.shuffle(file)
    for i in range((number_of_edge)):
        edges.append(tuple(list((int(file[i, 0]),int(file[i, 1])))))
    #edges = sorted(edges)
    return edges




        
def count_ratio(G, BC,start, end):
    count_ratio=0
    for i in range(len(G)):        
            if BC[i]<end and BC[i]>=start:
                count_ratio=count_ratio+1
    count_ratio=count_ratio/len(G)
    return count_ratio

       
# %%attaccare nuovi punti
'''

def Attachment_deg_list(Strenght):
    k = []
    if len(Strenght) == 0:
        return np.zeros((1, 2))
    else:
        for i in range(int(max(Strenght))+1):
            k.append(list(Strenght).count(i))
        k = np.array(k)/sum(k)
        n_strenght = np.array(range(len(k)))
        k = np.array((n_strenght, k)).transpose()
        k = np.array(sorted(k, key=lambda x: x[1]))

        attachment_deg_list = np.zeros((int(max(Strenght))+1, 2))
        for i in range(len(k)):
            attachment_deg_list[i] = [k[i, 0], sum(k[:i, 1])]
        return(np.array(attachment_deg_list))


def Attachment_deg(attachment_deg_list):
    a = rn.uniform(0, 1)
    m = 0
    for j in range(len(attachment_deg_list)-1, 1, -1):
        if (a > (attachment_deg_list[j, 1])):
            m = int(attachment_deg_list[j, 0])
            break
    return m

def min_distance_target (source,strenght_list,map_dct):
    x0=map_dct[source]
    min_=999
    target=-5
    for i in strenght_list:
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)
        if dist!=0:
            if dist<min_:
                min_=dist
                target=i
    return target

       

def max_prob_target_bis (source,strenght_list,map_dct,prob_distribution,step,max_dist,G):
    x0=map_dct[source]
    max_prob=-1
    target=-5
    second_target=-10
    third_target=-50
    for i in list(G.nodes):
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)
        if dist<max_dist:
            for k in range(50):
                if dist>k*step and dist<(k+1)*step:
                    prob=prob_distribution[k]
                    if prob>max_prob:
                        max_prob=prob
                        target=i
                        second_target=target
                        third_target=second_target

    
    if target==-5:
        min_=999
        for i in list(G.nodes):
            x1=map_dct[i]
            dist=np.linalg.norm(x0-x1)
            if dist<min_:
                min_=dist
                target=i
                second_target=target
                third_target=second_target
                
    for n in G.neighbors(source):
        if n==target:
            target=second_target
    for n in G.neighbors(source):
        if n==target:
            target=third_target
            
    if target==-5 or target==source:
        target=random.choice(list(G.nodes()))
        
    return target    




def BC_deg( strenght_list,Betweeness_Centrality):
    BC_tot=[]
    for i in strenght_list:
        BC_mean=[]
        for j in strenght_list[i]:
            BC_mean.append(Betweeness_Centrality[j,1])
        BC_tot.appen(np.mean(BC_mean), np.std(BC_mean))
    BC_tot=np.array(BC_tot)
    return BC_tot
'''
