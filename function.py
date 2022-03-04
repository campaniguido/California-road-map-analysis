# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 23:08:20 2022

@author: Guido
"""

# %%libraries
import random as rn
import networkx as nx
import pandas as pd
import numpy as np
import pytest
import function as fn
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
#%%seed

rn.seed(3)

#%% CLASS SuperGraph

class SuperGraph(nx.Graph):
    
        
    def Sorted_graph(self):
        '''It orders nodes list and edges list of the SuperGraph

        '''
        G_sorted =SuperGraph()
        G_sorted.add_nodes_from(sorted(list(self.nodes())))
        G_sorted.add_edges_from(sorted(list(self.edges())))
        self.remove_nodes_from(list(self.nodes()))
        self.add_nodes_from(sorted(list(G_sorted.nodes())))
        self.add_edges_from(sorted(list(G_sorted.edges())))
        
    def Relable_nodes(self):
        '''It orders nodes list and edges list of the SuperGraph and it renames the nodes
        in order to remove any jump in the labels (e.g.: G((5,4),(4,0)->G((0,1),(1,2)) )

        '''
        nodes=sorted(list(self.nodes()))    
        edges=list(self.edges())
        H=fn.SuperGraph()
        H.add_nodes_from(nodes)
        H.add_edges_from(edges)
        H=nx.convert_node_labels_to_integers(H)
        self.remove_nodes_from(list(self.nodes()))
        self.add_nodes_from(list(H.nodes()))
        self.add_edges_from(list(H.edges()))
            
    def Degree_dct(self):
        '''It returns a dictionary. The keys are the degree of the SuperGraph from 0 to the maximum.
        The values are all and only the nodes with the key degree'''
        
        degree_dct={}
        for key in range(int(max(np.array(list(self.degree))[:,1])+1)):
            degree_dct[key]=[]
        for i in list(self.nodes()):
            degree=len(list(self.neighbors(i)))-list(self.neighbors(i)).count(i) 
            for key in degree_dct:
                if degree==key:
                    degree_dct[key].append(i)
        j=0
        while j==0:
            if len(degree_dct[max(degree_dct.keys())])==0:
                del(degree_dct[max(degree_dct.keys())])
            else:
                j=1
        return degree_dct
    
    def Degree_ratio(self):
        '''It returns a np.array of the denity distribution of the nodes degree'''
        degree_dct=self.Degree_dct()
        degree_ratio=[]
        for key in (degree_dct):
            degree_ratio.append(len(degree_dct[key]))
        degree_ratio=np.array(degree_ratio)/sum(degree_ratio) 
        return degree_ratio
    
#%%Ct  test sorted_graph (2)
def test_sorted_graph_nodes():
    rn.seed(3)
    '''it tests the graph nodes are sorted after the application of the function'''
    edges = []
    i=0
    while i<20:
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append([a, b])
        i+=1
    G =fn.SuperGraph()
    G.add_edges_from(edges)
    TESTER = G
    G.Sorted_graph()
    assert list(G.nodes()) == sorted(list(TESTER.nodes()))
    
def test_sorted_graph_edges():
    rn.seed(3)
    '''it tests the graph edgess are sorted after the application of the function'''
    edges = []
    for i in range(0, 100):
        a = rn.randint(0, 100)
        b = rn.randint(0, 100)
        edges.append((a, b))
    G =fn.SuperGraph()
    G.add_edges_from(edges)
    G.Sorted_graph()
    vecchio = list(G.edges())[0][0]
    for i in range(1, len(G)):
        assert list(G.edges())[i][0] <= list(G.edges())[i][1]
        assert vecchio <= list(G.edges())[i][0]
        vecchio = list(G.edges())[i][0]

#%%Ct  test_Relable_nodes


def test_Relable_nodes_len():
    '''It tests the len of the graph is conserved also after the application of the function'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    LEN=len(G)
    G.Relable_nodes()
    assert LEN==len(G)
    
def test_Relable_nodes_sorted():
    '''it tests the ascending order'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    G.Relable_nodes()
    nodes=list(G.nodes())
    for i in range(1,len(G)):
        assert nodes[i-1]<nodes[i]

def test_Relable_nodes_no_hole():
    '''it tests there is no jump in the numbers labels'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    G.Relable_nodes()
    nodes=list(G.nodes())
    for i in range(len(G)):
        assert nodes[i]==i
        
def test_Relable_nodes_corrispondence():
    '''it looks for neigbours corrispondence before and after the relable'''
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
    '''It verifies each node has a degree equal to its key'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=G.Degree_dct()
    for key in strenght_dct:
        for node in strenght_dct[key]:
            assert G.degree(node)-2*list(G.neighbors(node)).count(node)==key

def test_Degree_list_length():
    '''It looks if the elements of the dictionary values are the same of the of the graph nodes'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=G.Degree_dct()
    values=[]
    for i in strenght_dct:
        values+=set(strenght_dct[i])
    assert sorted(values)==list(range(1,11))
    
def test_Strenght_list_max_degree():
    '''It verify the highest key value is equal to the maximum degree'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    strenght_dct=G.Degree_dct()
    assert len(list(strenght_dct.keys()))==max(np.array(list(G.degree))[:,1])+1
    
def test_Strenght_list_autoconnected_nodes():
    '''It looks that the function does not count as a neighbour the node its self in the case
    of an autoconnected node'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[3,3]])
    strenght_dct=G.Degree_dct()
    assert strenght_dct[0]==[2]
    assert strenght_dct[1]==[1,3]
    
def test_empty_high_key():
    '''it verify the last keys doesn't have 0 length'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,1],[2,2],[5,5],[8,8],[9,9],[10,10],[10,10],[7,7],[10,10],[0,1]])
    strenght_dct=G.Degree_dct()
    assert len(list(strenght_dct.keys()))==2
    
#%%Ct   test_Degree_ratio (4)
def test_Degree_ratio_length():
    '''It tests the length of the graph is conserved also after the application of the function'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degree_ratio=G.Degree_ratio()
    assert len(degree_ratio)==6
    
def test_Degree_ratio_I_axiom():
    '''It verifies the probability I axiom'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degree_ratio=G.Degree_ratio()
    for i in degree_ratio:
        assert i>=0
        
def test_Degree_ratio_II_axiom():
    '''It verifies the probability II axiom'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degree_ratio=G.Degree_ratio()
    assert 0.99999<sum(degree_ratio)<1
    
def test_Degree_ratio_III_axiom():
    '''It verifies the probability III axiom'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    degre_dct=G.Degree_dct()
    degree_ratio=G.Degree_ratio()
    
    for i in range(len(G)):
        j=i+1
        while j<len(degree_ratio):
            assert(degree_ratio[i]+degree_ratio[j]==np.array(len(degre_dct[i])+len(degre_dct[j]))/10)
            j=j+1
            
#%%1  Divide value
def Divide_value(file):
    '''It takes an array (n,2) which represents the graph edges. It looks for any case in which
    two linked nodes are written in the same cell in a string variable and put the two variables
    in the two columns of the same row. It substitutes any full nan row with the previous row
    
    Parameters
    ----------
    file : Dict can contain Series, arrays, constants, dataclass or list-like objects.
           If data is a dict, column order follows insertion-order. 
           If a dict contains Series which have an index defined, it is aligned by its index.
           Its shape mut be (n,2)

    Raises
    ------
    Exception
        Exception raises if the file shape is not (n,2)

    Returns
    -------
    file : np.array((n,2))

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
#%%  tests Divide value (3)
    

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

def test_divide_value_shape():
    '''it tests if the data are in the shape n*2'''
    file=[[0,134.0,4],['45643 3456',np.nan,5],[np.nan,'34 5',7],[3,4.0,8]]
    with pytest.raises(Exception):
        Divide_value(file)
    
              


#%%2  Edge_list
def Edge_list(file, number_of_edges):
    '''
    It takes a file of couples of number and return a list of a desired lenght(number_of_edge) of couple 
    of numbers expressed as  integers    

    Parameters
    ----------
    file : list, tuple,np.array, pd.DataFrame
        The file to be read.
        
    number_of_edge : Integer
        It is the number of file couples desired

    Raises
    ------
    Exception
        Exception raises if the number of edges required is major than the length of the file, or the file
        has a shape different from (n,2)

    Returns
    -------
    edges : list of shape (n,2)
        

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

#%%  tests Edge_list (5)

#POSITIVE TEST
def test_edge_list_int():
    '''it tests if all the elements are integers'''
    file=[[0,134.0],['45643',' 3456'],[3,4.0]]
    edges=Edge_list(file,2)
    for j in range(len(edges)):
        for k in range(2):
            assert type(edges[j][k])==int
                    
def test_edge_list_length():
    ''' it looks if the output length is equal of number of edges'''
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





    

    



    
        
        
#%%3  Unfreeze_into_list
def Unfreeze_into_list(comunity):
    '''It takes a (support item assignment) variable of n elements and transform each element in a list
    of variables
    

    Parameters
    ----------
    comunity : Support item assignment variable
         

    Returns
    -------
    comunity : It returns the same variable but each item is a list
        DESCRIPTION.

    '''
    for i in range(len(comunity)):
        comunity[i]=list(comunity[i])
    return comunity
#%%  tests Unfreeze_into_list(2)

def test_Unfreeze_into_list_is_a_list_1():
    '''It verifies the output items are lists'''
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    list_=[A,B,C,D]
    list_=Unfreeze_into_list(list_)
    for i in range(len(list_)):
        assert(type(list_[i])==list)

def test_Unfreeze_into_list_is_a_list_2():
    '''it verifies the output items are lists'''
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    D= (1,2)
    dct_={0:A,1:B,2:C,3:D}
    dct_=Unfreeze_into_list(dct_)
    for i in range(len(dct_)):
        assert(type(dct_[i])==list)
#%%4  Set_comunity

def Set_community_number(G, community):
    '''It assigns to each node to the graph a community number. Node with the same number
    are in the same community, each number cannot belongs to different communities. G is the graph,
    while each entry of the community  variable represents node of the same community. It returns 
    a dictionary: the keys are the nodes numbers, the values are the numbers of the community
    they belong to.
    

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    comunity : Support item assignment variable
        

    Returns
    -------
    comunity_number : dictionary
        

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
    '''It verifies the community corrispondence is right'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=Set_community_number(G, community)
    for i in list(G.nodes()):
        assert community[community_number[i]].count(i)==1
        
def test_Set_community_number_length():
    '''It tests the length of the output is the same of the one of the Graph'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    community_number=Set_community_number(G, community)
    assert len(community_number)==len(G)
    

        
def test_Set_community_number_doubble():
    '''It tests if the node are in just one community. e.g.:the four is in two different communities'''
    community=[[1,3,6,9],[5,10],[7,4],[2,4,8]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        Set_community_number(G, community)

def test_Set_community_number_missing_number():
    '''It tests if all the node are in the community variable. e.g.:the eight is missing'''
    community=[[1,3,6,9],[5,10],[7],[2,4]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(ValueError):
        Set_community_number(G, community)

def test_Set_community_number_extra_number():
    '''It tests if the community has only the nodes labels of the graph. e.g.:the 999 is not a node number'''
    community=[[1,3,6,9],[5,10],[7],[2,4,8,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        Set_community_number(G, community)

def test_Set_community_number_all_wrong():
    '''It tests the behaviours for the three previous mistakes'''
    community=[[1,3,6,4],[5,10],[7],[2,4,999]]
    G=nx.Graph()
    G.add_edges_from([[1,3],[2,2],[5,6],[4,8],[9,1],[10,1],[1,5],[4,7],[1,7]])
    with pytest.raises(Exception):
        Set_community_number(G, community)
            
    





    
    
    
    



                        
#%%5  Size evolution (3)
def Size_evolution(G,step, feature):
    '''
    It takes back information related to the network G feature along its size increasing.
    Starting from the graph size equal to the number of edge indicated by the step variable,
    it increases the number of edge of the step variable at each loop. 
    The outputs are three lists related to the size of the graph at each step, 
    the value of the feature of each node at each step and
    the mean value and its standard deviation of the feature at each step.
    In the case of the feature='degree' it returns  the degree distribution. 

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    step : integer
        
    feature : degree, closeness_centrality, betweenness_centrality, clustering

    Returns
    -------
    size : np.array of integers
        Each integer is the size of the graph at that step
    value_size_evolution : If feature=='degree' it returns a numpy.ndarray((#G.edges/step,max degree G +1)).
                           It represents the degree ratio of the graph at each step.
                           Else it returns a list #G.edges/step long. Each item of the list is a numpy.ndarray
                           of the nodes values of the selected features
        
    value_size_evolution_mean : It returns numpy.ndarray((#G.edges/step,2)). 
                                Each entry is the mean value of the degree and its std
                                
        

    '''
    value_size_evolution=[]
    n_step=int(len(G.edges)/step)
    value_size_evolution_mean=[]
    edges=list(G.edges())
    size=[]
    if feature=='degree':
        for i in range(n_step):      
            G=fn.SuperGraph(edges[:(i+1)*step])
            G.Relable_nodes()
            
            value_size_evolution.append(G.Degree_ratio())
            value_size=np.array(list((getattr(nx, feature)(G))))[:,1]
            value_size_evolution_mean.append([np.mean(value_size),np.std(value_size)])
            size.append(len(G))  
        
        for i in range(len(value_size_evolution)):
            while len(value_size_evolution[i])<len(value_size_evolution[-1]):
                value_size_evolution[i]=np.append(value_size_evolution[i],0)
        value_size_evolution_mean=np.array(value_size_evolution_mean)
        value_size_evolution=np.array(value_size_evolution)
        size=np.array(size)
        return size, value_size_evolution,value_size_evolution_mean
    else:
        for i in range(n_step):
            G=fn.SuperGraph(edges[:(i+1)*step])
            G.Sorted_graph()
            G.Relable_nodes()
            value_size=np.array(list((getattr(nx, feature)(G)).items()))[:,1]
            value_size_evolution_mean.append([np.mean(value_size),np.std(value_size)])
            value_size_evolution.append(value_size)
            size.append(len(G))
        value_size_evolution_mean=np.array(value_size_evolution_mean)
        return size, value_size_evolution,value_size_evolution_mean
#%%   test_size_evolution
def test_size_evolution_size():
    '''It tests the richt len of each output'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size, value_size_evolution,value_size_evolution_mean=Size_evolution(G, step, feature)
    assert len(size)==nstep and len(value_size_evolution)==nstep and len(value_size_evolution_mean)==nstep
    
    feature='betweenness_centrality'
    size, value_size_evolution,value_size_evolution_mean=Size_evolution(G, step, feature)
    assert len(size)==nstep and len(value_size_evolution)==nstep and len(value_size_evolution_mean)==nstep

def test_size_evolution_size_increasing_size():
    '''It tests the output 'size' increases at each step'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    size=Size_evolution(G, step, feature)[0]
    for i in range(nstep-1):
        assert size[i]<size[i+1]
    
    feature='betweenness_centrality'
    size, value_size_evolution,value_size_evolution_mean=Size_evolution(G, step, feature)
    for i in range(nstep-1):
        assert size[i]<size[i+1]

def test_size_evolution_size_degree_constant_len():
    '''It tests, for the degree feature, the lenght of the value_size_evolution it is always the same'''
    G=fn.SuperGraph()
    G.add_edges_from([[1,8],[8,3],[8,4],[5,4],[5,6],[14,14]])
    step=1
    nstep=int(len(G.edges)/step)
    feature='degree'
    value_size_evolution=Size_evolution(G, step, feature)[1]
    for i in range(nstep):
        assert len(value_size_evolution[i])==4
    
#%%6 Dct_dist_link 
def Dct_dist_link(edges,map_dct):
    '''It calculates all the distances of the nodes linked together whose position is described by
    the dictionary map
    

    Parameters
    ----------
    edges : Support item assignment variable
        It represents all the links of the network
    map_dct : Support item assignment variable
        It describes the position of the nodes

    Returns
    -------
    dct_dist_link : dictionary of all the nodes linked together. key:value= edge:distance

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
    return dct_dist_link

#%%   test_List_dist_link (4)

    
def test_List_dist_link_length():
    '''it verifies in the dictionary are not present symmetric object (e.g.: (1,2), (2,1))
    and it removes autoedges (e.g.: (1,1))
    '''
    edges=[(1,2),(3,1),(1,1),(1,2),(2,1)]    
    G=nx.Graph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist_link(edges, map_dct)
    assert len(dct_dist_link)==2

def test_List_dist_link_non_negativity():
    '''It tests distances are not negative'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    edges=list(G.edges())
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist_link(edges, map_dct)
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
    dct_dist_link1=Dct_dist_link(edges, map_dct)
    dct_dist_link2=Dct_dist_link(segde, map_dct)
    assert list(dct_dist_link2.values())==list(dct_dist_link1.values())
    
def test_List_dist_link_triangular_inequality():
    '''It tests triangular inequality'''
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
                        
            
            
            
    
    


#%%7 Dct_dist 
def Dct_dist(G,map_dct):
    '''
    It returns all the distance among the nodes, even the not linked one. 
    It exploits a dictionary map of the position of the nodes
    

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    map_dct : Support item assignment variable
        It describes the position of the nodes

    Returns
    -------
    dct_dist : dictionary of all the nodes linked together. key:value= (node_i,node_j):distance
        

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
    '''It tests distances are not negative'''
    G=nx.Graph()
    G.add_edges_from([[1,2],[1,3],[1,4],[2,4],[3,4],[4,5],[6,6],[3,1],[2,5]])
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    dct_dist_link=Dct_dist(G, map_dct)
    for i in list(dct_dist_link.values()):
        assert i>=0
    
def test_List_dist_triangular_inequality():
    '''It tests triangular inequality'''
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
#%%8 Node_distance_frequency 
def Node_distance_frequency(dct_dist,nstep,step):
    '''It returns a binned not normalized nodes distance distribution. 
    It puts all the elements>nstep*step in the last bin 
    

    Parameters
    ----------
    dct_dist : Support item assignment variable
        It describes all the distances among nodes
    nstep : integer
        
    step : integer
        

    Returns
    -------
    node_distance_frequency : numpy.ndarray
        Each entry is the frequency of elements with the same distance inside the bin of length==step.
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

#%%9 Link_distance_conditional_probability 

def Link_distance_conditional_probability(dct_dist_link,nstep,distance_frequency):
    '''
    It returns the probability density binned function of having a link at a fixed disstance 
    

    Parameters
    ----------
    dct_dist_link : Support item assignment variable
        It describes all the distances among nodes linked together
        
    nstep : integer
        
    distance_frequency : Support item assignment variable
        It represents the frequency of nodes that are in a distance equal to n*step. Where 0<n<nstep and
        step=max(dct_dist_link.values())/nstep

    Raises
    ------
    ZeroDivisionError
        It raise the error when some linked node distance is in a bin i but no node distance is present in
        in the same bin

    Returns
    -------
    link_distance_probability : TYPE
        DESCRIPTION.

    '''
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
            raise ZeroDivisionError('There is no couple of nodes with this distance')
            
    return link_distance_probability
    
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
    dct_dist=Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
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
    dct_dist=Dct_dist(G,map_dct)
    dct_dist_link=fn.Dct_dist_link(edges, map_dct)
    nstep=10
    step=max(list(dct_dist_link.values()))/nstep
    node_distance_frequency=Node_distance_frequency(dct_dist,nstep,step)
    link_distance_conditional_probability=Link_distance_conditional_probability(dct_dist_link,nstep,node_distance_frequency)
    assert list(link_distance_conditional_probability.keys())[-1]==max(list(dct_dist_link.values()))
    

#%%10 Add_edges_from_map(G,map_dct,distance_linking_probability)                           
def Add_edges_from_map(G,map_dct,distance_linking_probability):
    '''
    It creates links among nodes of graph. 
    It exploits the nodes positions informations conteined in the map dictionary, map_dct,
    and the conditional distribution probability to make a link in function of the distance.
    

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        
    map_dct : Support item assignment variable
        It describes the position of all the nodes
        
    distance_linking_probability : Support item assignment variable
        It describe the probability to have a link among two nodes at a given distance
    

    Returns
    -------
    G : networkx.classes.graph.Graph
        it returns the old graph but with the additon of the new links

    '''
    rn.seed(3)
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
    

#%%11 Break_strong_nodes

def Break_strongest_nodes(G, threshold):
    '''
    It breaks randomly links of nodes with the highest degree
    till the node with the maximum degree has a value under or equal to the trheshold

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    threshold : integer
        It is the maximum of the degree allows. all the edges of the nodes with
        a higher degree will be dissolved

    Returns
    -------
    None.

    '''
    rn.seed(3)

    
    dct_degree=G.Degree_dct()
    while threshold<max(np.array(list(G.degree))[:,1]):
        source=rn.choice(list(dct_degree.values())[len(dct_degree)-1])
        node=rn.choice(list(G.neighbors(source)))
        G.remove_edge(source,node)
        dct_degree=G.Degree_dct()
        
#%% test_Break_strongest_nodes (2)

def test_Break_strongest_nodes_size():
    '''it tests len of the graph is kept constant'''
    
    edges=[(1,2), (3,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    len_before=len(G)
    Break_strongest_nodes(G,2)
    len_after=len(G)
    assert len_after==len_before
    
def test_Break_strongest_nodes_maximum_value():
    '''it verify the nodes with the highest degree is under the threshold'''
    
    edges=[(1,2), (3,1),(1,1), (1,1), (1,2), (2,1), (1,4), (1,5), (5,4), (5,3), (1,6), (6,2), (5,2)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    Break_strongest_nodes(G,2)
    deg=G.Degree_dct()
    assert max(deg.keys())<=2


#%%12 Find_mode(pfd) 
def Find_mode(pdf):
    '''
    It returns the x value of the maximum of function p(x). Where x represent a bin 

    Parameters
    ----------
    pdf : Support item assignment variable
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
    
    
        
#%%13 Equalize_strong_nodes
def Equalize_strong_nodes(G_strong, G_weak):
    '''
    It compares two graph. It takes the strongest node of the first graph which have nodes witha degree value major 
    then the degree mode of the second graph. It remove random links of these nodes untill the degree ratio
    of the the first graph (G_strong) of these high degree values are above the ones of the second graph(G_weak)

    Parameters
    ----------
    G_strong : function.SuperGraph
        
    G_weak : function.SuperGraph
        

    Returns
    -------
    None.

    '''
    rn.seed(3)
    dct_degree_weak=G_weak.Degree_dct()
    degree_ratio_weak=G_weak.Degree_ratio()    
    threshold=fn.Find_mode(degree_ratio_weak)
    i=max(dct_degree_weak.keys())
    
    if i<max(np.array(list(G_strong.degree))[:,1]):       
        fn.Break_strongest_nodes(G_strong, i) 
        
    while i>threshold:
        
        
        while len(G_strong.Degree_ratio())>=len(G_weak.Degree_ratio()) and (G_strong.Degree_ratio()[i])>(G_weak.Degree_ratio()[i]):
            source=rn.choice(G_strong.Degree_dct()[i])
            node=rn.choice(list(G_strong.neighbors(source)))
            G_strong.remove_edge(source,node)              
            
                       
        i=i-1
            
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
    Equalize_strong_nodes(G_strong, G_weak) 
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

#%%14 Max_prob_target

    
def Max_prob_target (source,strenght_dct,degree,map_dct,distance_linking_probability,max_dist,G):
    '''
    It finds the best linking target for a surce node of the graph G. The target,
    if it is possible, has the degree chosen in the input, it's not a source's neighbour, 
    it's not its self and among the the nodes with the previous three characteristic is the most
    probable one

    Parameters
    ----------
    source : integer
        It represents the node label of the source.
        
    strenght_dct :  Support item assignment variable
        each items is the group of nodes with the same degree
        
    degree : integer
        It represents the desired degree of the target
        
    map_dct : Support item assignment variable
        Each item represents the spatial position of a node of the graph
        
    distance_linking_probability :Support item assignment variable
        It represents a binned density function distribution related to the distance, each bin distance 
        entry is the ptobability to have a link at tthat distance
        
    max_dist : integer
        It is the ditance threshold among the source and the others nodes. Node distances above it
        will not take into account
        
    G : networkx.classes.graph.Graph
        

    Returns
    -------
    target : integer
        It is the label of node chosen

    '''
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

        target=fn.Min_distance_target(source, strenght_dct,degree, map_dct, list(G.neighbors(source)))
    return target
  
    

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
    '''It verifies the taget is not the ource'''
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
       
#%%15 Min_distance_target (source,strenght_dct,degree,map_dct,source_neighbour_list)

def Min_distance_target (source,strenght_dct,degree,map_dct,source_neighbour_list):
    '''
    Given a distance map of the nodes, From a starting vertice (the source) of the graph it returns the nearest node 
    with a given degree and which is not already linked to the source.    

    Parameters
    ----------
    ----------
    source : integer
        It represents the node label of the source.
        
    strenght_dct :  Support item assignment variable
        each items is a group of nodes with the same degree
        
    degree : integer
        It represents the desired degree of the target        
        
    map_dct : Support item assignment variable
        Each item represents the spatial position of a node of the graph
        
    source_neighbour_list : list

    Returns
    -------
    target : integer
        It is the label of node chosen for the linkage

    '''
    rn.seed(3)
    x0=map_dct[source]
    min_=999
    target=-5
    'It verifies there is some node left to bind to'
    assert len(source_neighbour_list)!=len(map_dct)
    for i in strenght_dct[degree]:
        
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)
        
        if dist!=0 and source_neighbour_list.count(i)!=1:
            
            if dist<min_:
                
                min_=dist
                target=i
    while target<0:
        list_target=rn.choice(list(strenght_dct.values()))
        if len(list_target)!=0:
            target_prova=rn.choice(list_target)
            if target_prova!=source and source_neighbour_list.count(target_prova)!=1:
                target=target_prova
        
    return target

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
        assert source!=Min_distance_target (source,degree_dct,3,map_dct,neighbour_0)
        
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
        assert 2!=Min_distance_target (source,degree_dct,2,map_dct,neighbour_0)
        
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
    target=Min_distance_target (source,degree_dct,3,map_dct,neighbour_0)
    assert len(list(G.neighbors(target)))==3
    
    
    
#%%16 Merge_small_component
def Merge_small_component(G, deg,map_dct,threshold):
    '''
    It merge all the graph component smaller than a given threshold, by linking a random node of
    the component with a node of the graph with a given degre==deg

    Parameters
    ----------
    G : function.SuperGraph
        
    deg : degree of the target node
        DESCRIPTION.
        
    map_dct : Support item assignment variable
        Each item represents the spatial position of a node of the graph
        
    threshold : integer
        

    Raises
    ------
    Exception
        it raises an exception if the node list with the degree chosen is empty

    Returns
    -------
    None.

    '''
    rn.seed(3)
    i=0
    all_components=list(nx.connected_components(G))
    while i < len(all_components):
    
        if len(list(all_components)[i])<threshold:
            source=rn.choice(list(all_components[i]))
            list_nodes=G.Degree_dct()[deg]
            if len(list_nodes)==0:
                raise Exception('the node list with the degree chosen is empty')
                
            target=Min_distance_target(source,G.Degree_dct(),deg,map_dct,list(G.neighbors(source)))
            G.add_edge(source,target)
            
        i=i+1
        
#%%  test_Merge_small_component (2)
'''voglio verificare che funzioni e che quindi non esistano componenti con dimensione più piccola di un tot,
 voglio verificare che sia alzato un errore se la lista con un determinato grado finisce'''

def test_Merge_small_component():
    '''It verifies all the components are bigger than the threshold'''
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    Merge_small_component(G, 1,map_dct,threshold=3)
    for i in list(nx.connected_components(G)):
        assert len(i)>=3
        
def test_Merge_small_component_Exception():
    '''It test if raises an exception for empty list'''
    edges=[(1,2), (3,4), (6,7), (7,5), (8,8)]    
    G=fn.SuperGraph()
    G.add_edges_from(edges)
    map_dct=nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)
    with pytest.raises(Exception):
        Merge_small_component(G, 0,map_dct,threshold=3)

    
#%%17 Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G,n_links, degree_dct)
def Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G,n_links, degree_dct):
    '''
    It takes an isolated node of the graph and it creates n links (n=n_links) with n different nodes.
    The degree of the nodes are 0, 1, 2 ... n-1. At in the of the process the number of nodes with degree
    equal to n increase of 2 and the number of isolated nodes decrease of two. 
    The degree ratio of the other values remains constant. The attachment rule of the links follow the
    Max_prob_target function.

    Parameters
    ----------
    map_dct : Support item assignment variable
        Each item represents the spatial position of a node of the graph
        
    prob_distribution : Support item assignment variable
        It represents a binned density function distribution related to the distance, each bin distance 
        entry is the ptobability to have a link at tthat distance
        
    max_dist : integer
        It is the ditance threshold among the source and the others nodes. Node distances above it
        will not take into account
        
    G : function.SuperGraph
        
    n_links : integer
            number of links to create
        
    degree_dct : Support item assignment variable
        each items is the group of nodes with the same degree

    Returns
    -------
    None.

    '''
    if len(degree_dct[0])!=0:
        source=rn.choice(degree_dct[0])
        for i in range(n_links):                      
            target=fn.Max_prob_target(source,degree_dct,i,map_dct,prob_distribution,max_dist_link,G)
            G.add_edge(source, target)
            degree_dct=G.Degree_dct()
#%% test_Link_2_ZeroNode (3)
''' voglio testare che il degree ratio segue andamento voluto, che i degree dei nodi sono tutti presenti
    
'''
def test_Link_2_ZeroNode_reduction():
    rn.seed(3)
    '''It verifies the right increasing of  '0' degree ratio'''
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
    '''It verifies the right decreasing of  '4' degree ratio'''
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




#%%18 Remove_edge_of_degree 
def Remove_edge_of_degree(degree,G):
    '''
    It removes a link of a node with a given degree
    

    Parameters
    ----------
    degree : integer
            Degree of the node of which we wants to remove a link
        
    G : function.SuperGraph
        

    Returns
    -------
        None.

    ''' 
    rn.seed(3)
    degree_dct=G.Degree_dct()
    source=rn.choice(degree_dct[degree])
    node=rn.choice(list(G.neighbors(source)))
    G.remove_edge(source,node)            
    

#%% test_Remove_edge_of_degree

''' che la lunghezza del grafo sia la stessa
'''

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
    

    
               


#%%19 Copymap_degree_correction(Copy_map,G,map_dct,max_dist_link,prob_distribution,Merge=False):

def Copymap_degree_correction(Copy_map,G,map_dct,max_dist_link,prob_distribution,Merge=False):
    '''
    

    Parameters
    ----------
    Copy_map : function.SuperGraph
        Graph we want to change
        
    G : function.SuperGraph
        Model Graph
        
    map_dct : Support item assignment variable
        Each item represents the spatial position of a node of the graph         
    
    max_dist : integer
        It is the ditance threshold among the source and the others nodes. Node distances above it
        will not take into account
        
    prob_distribution : Support item assignment variable
        It represents a binned density function distribution related to the distance, each bin distance 
        entry is the ptobability to have a link at tthat distance
        
    Merge : bool, optional
        If True at the end of the process the function merge the small components.
        The default is False.

    Returns
    -------
    Copycat : function.SuperGraph
        It returns the graph with the degree ratio corrected
    '''
    
    rn.seed(3)
    Copycat=fn.SuperGraph(Copy_map)
    
    fn.Break_strongest_nodes(Copycat, max(np.array(list(G.degree()))[:,1]))
    
    fn.Equalize_strong_nodes(Copycat, G)
    
    while len(Copycat.Degree_dct()[0])>0:
        
        for i in range(len(G.Degree_ratio())-1,0,-1):
            if i< len(Copycat.Degree_ratio()):           
                while Copycat.Degree_ratio()[i]> G.Degree_ratio()[i]:            
                    fn.Remove_edge_of_degree(i, Copycat) 
                  
        for i in range(1,len(G.Degree_ratio())):
                while len(Copycat.Degree_dct()[0])>0 and Copycat.Degree_ratio()[i]< G.Degree_ratio()[i]:  
                    fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,Copycat,i, Copycat.Degree_dct())
                    
        
    if Merge==True:
         Merge_small_component(Copycat,deg=1, map_dct=map_dct, threshold=3)
                  
    fn.Break_strongest_nodes(Copycat, max(np.array(list(G.degree()))[:,1]))
    Copycat.Relable_nodes()
    
    return Copycat
#%%                         PLOT FUNCTION
#%%20 Hist_plot

def Hist_plot(distribution, color, title, save_fig=False):
    '''
    It shows the distribution histogram of an input set of data, it provides labels for the axis and  the graph.
    It can also save the plot.
    

    Parameters
    ----------
    distribution : (n,) array or sequence of (n,) arrays
        Input values, this takes either a single array or a sequence of
        arrays which are not required to be of the same length.
        
    color : color or array-like of colors or None, default: None
        Color or sequence of colors, one per dataset.  Default (``None``)
        uses the standard line color sequence.
        
    title : str
        Title of the histogram
        
    save_fig : bool, optional
        If ''True'' save a pdf file with the name title.pdf . The default is False.

    Returns
    -------
    None.

    '''
    n, bins, patches=plt.hist(distribution,color=color)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(title)
    if save_fig==True:
        plt.savefig(title+'.pdf', dpi=500)
    plt.show()

#%%21 Scatter_plot

def Scatter_plot(distribution1, name_distribution1, distribution2, name_distribution2, color, save_fig=False):
    '''
    It shows the the scatter plot of two set of input data, it provides labels for the axis and the graph.
    It can also save the plot.
    

    Parameters
    ----------
    distribution1 : float or array-like, shape (n, )
        The data positions.
        
    name_distribution1 : str
        Name of the distribution.
        
    distribution2 : float or array-like, shape (n, )
        The data positions.
        
    name_distribution2 : str
        Name of the distribution.
        
    color : array-like or list of colors or color, optional
        The marker colors. Possible values:
    
        - A scalar or sequence of n numbers to be mapped to colors using
          *cmap* and *norm*.
        - A 2D array in which the rows are RGB or RGBA.
        - A sequence of colors of length n.
        - A single color format string.
        
    save_fig : bool, optional
        If ''True'' save a pdf file with the name title.pdf . The default is False.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots()
    ax.scatter(distribution1,distribution2,c=color,s=2)
    plt.xlabel(name_distribution1)
    plt.ylabel(name_distribution2)
    plt.title(name_distribution1+ ' vs '+  name_distribution2)
    if save_fig==True:
        plt.savefig(name_distribution1+ ' vs '+  name_distribution2 +".pdf", dpi=500)
    plt.show()
       
#%%22 Feature_mean_evolution

def Feature_mean_evolution(feature_size,feature_mean, feature_name, save_fig=False):
    '''
    It shows the the scatter plot of a set of input data,  it provides labels for the axis and the graph.
    It can also save the plot.
    

    Parameters
    ----------
    feature_size :  float or array-like, shape (n, )
        The data positions.
        
    feature_mean :  float or array-like, shape (n, 2)
        In the first column there are the values, in the second the errors of them .
        
    feature_name : str
        It is the name of the values distribution
        
    save_fig : bool, optional
        If ''True'' save a pdf file with the name title.pdf . The default is False.

    Returns
    -------
    None.

    '''
    x =feature_mean
    colors = (cm.CMRmap(np.linspace(0.01, 0.9, len(x))))
    fig, ax = plt.subplots()
    ax.scatter(feature_size, list(x[:,0]),c=colors,s=10)
    ax.errorbar(feature_size,list(x[:,0]), yerr=list(x[:,1]), xerr=None,fmt='o', ecolor=colors,markersize=0)
    plt.xlabel("number of nodes")
    plt.ylabel(feature_name)
    plt.title("Mean"+feature_name)
    if save_fig==True:
        plt.savefig("Mean"+feature_name+".pdf", dpi=500)
    plt.show()
#%%23 Feature_cumulative_evolution


def Feature_cumulative_evolution(feature, feature_name, save_fig=False):
    '''
    It shows the cumulative distribution(normalized on the number of data) of n distributions of input data,
    it provides labels for the axis and the graph.
    It can also save the plot.
    

    Parameters
    ----------
    feature : n dimension array_like
        Each dimension is a different distribution of the input data
        
    feature_name : str
        name of the distribution.
        
    save_fig :  bool, optional
        If ''True'' save a pdf file with the name title.pdf . The default is False.

    Returns
    -------
    None.

    '''
    x=feature
    fig, ax = plt.subplots()
    colors = (cm.magma(np.linspace(0, 1, len(x))))
    for i in range(len(x)):
        size=len(x[i])
        values, base = np.histogram(x[i],bins=500)
        cumulative = np.cumsum(values/size)
        ax.plot(base[:-1], cumulative, c=colors[-i-1],label=size)        
    ax.set_xlabel("Values")
    ax.set_ylabel("Cumulative probability")    
    ax.legend(prop={'size': 10})
    ax.set_title('Cumulative distribution of '+  feature_name)
    if save_fig==True:
        plt.savefig("Cc-cumulative-convergence.pdf", dpi=500)
    plt.show()

#%%24 Feature_ratio_evolution


'''mettere a posto lettura file

scrivere il readme'''

def Feature_ratio_evolution(feature_position,feature_ratio, feature_name, save_fig=False):
    '''
    It plot a scatter plot: at each position it scatter a vector of n point corrisponding to the n values of,
    of each element of the feature ratio.
    it provides labels for the axis and the graph.
    It can also save the plot.
    

    Parameters
    ----------
    feature_size : 1 dimension array_like
        number label of each set of data
    
    feature_ratio :  n dimension array_like
        Each dimension represent a set of data of the same dimension
        .
    feature_name : str
        Name of the distribution.
        
    save_fig : bool, optional
        If ''True'' save a pdf file with the name title.pdf . The default is False.

    Returns
    -------
    None.

    '''
    x=feature_ratio
    colors = (cm.tab10(np.linspace(0, 1, len(x.transpose()))))
    fig, ax = plt.subplots()
    x=feature_ratio
    for i in range(len(x.transpose())):
        size=feature_position
        ax.scatter(feature_position,    x[:,i], c=[colors[i]], s=2,label='%s' %i)
        
    ax.legend(loc='upper left', shadow=True, fontsize=8)  
    plt.xlabel("number of nodes")
    plt.xlim(-30, max(size+10))
    plt.ylabel("ratio of each " + feature_name)
    plt.title("ratio of each " + feature_name + " for increasing nodes")
    if save_fig==True:
        plt.savefig("ratio of each"+ feature_name +".pdf", dpi=500)
    plt.show()