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
import function as fn
import math
import os


#%% CLASS SuperGraph

class SuperGraph(nx.Graph):
    
        
    def Sorted_graph(self):
        '''
        It orders nodes list and edges list of the SuperGraph. 
        Nodes are sorted in ascending way, 
        edges are sorting in ascending way following the first element of the couple
        

        Returns
        -------
        None.
        
        example:
            
        G=fn.SuperGraph()
        G.add_edges_from(([(9, 1), (9, 0), (9, 4), (1, 2), (4,4)]))
        G.Sorted_graph()
        G.edges(), G.nodes()
        
        >output:
        
        (EdgeView([(0, 9), (1, 2), (1, 9), (4, 4), (4, 9)]), NodeView((0, 1, 2, 4, 9)))


        '''
        G_sorted =SuperGraph()
        G_sorted.add_nodes_from(sorted(list(self.nodes())))
        G_sorted.add_edges_from(sorted(list(self.edges())))
        self.remove_nodes_from(list(self.nodes()))
        self.add_nodes_from(sorted(list(G_sorted.nodes())))
        self.add_edges_from(sorted(list(G_sorted.edges())))
        
    def Relable_nodes(self):
        '''It orders nodes list and edges list of the SuperGraph and it renames the nodes
        in order to remove any jump in the labels (e.g.: G((5,4),(4,0))->G((0,1),(1,2)) ) 
        
                                                   
        Returns
        -------
    
        None.
        
        
        example:
            
        G=fn.SuperGraph()
        G.add_edges_from(([(9, 1), (9, 0), (9, 4), (1, 2), (4,4)]))
        G.Relable_nodes()
        G.edges(), G.nodes()
        
        >output:
        
        (EdgeView([(0, 4), (1, 4), (1, 2), (3, 4), (3, 3)]), NodeView((0, 1, 2, 3, 4)))
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
        The values are all and only the nodes with the key degree.
        It doesn't take into account as a link a self connected node.
        


        
        Returns
        -------
        degree_dct : dct
            The keys are the degree of the SuperGraph,the values are all and only the nodes with the key degree
        
        example:
        G=fn.SuperGraph()
        G.add_edges_from(([(0, 1), (0, 3), (0, 4), (1, 2), (4,4)]))
        G.Degree_dct()
        
        >output:
        {0: [], 1: [3, 4, 2], 2: [1], 3: [0]}    
        '''

        
        degree_dct={}
        for key in range(int(max(np.array(list(self.degree))[:,1])+1)):
            degree_dct[key]=[]
        for i in list(self.nodes()):
            'to delete him self from the neighbors'
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

        '''It returns a np.array of the normalized frequency of the nodes degree
        
        Returns
        -------
        degree_ratio : np.array
        
        example:
        G=fn.SuperGraph()
        G.add_edges_from(([(0, 1), (0, 3), (0, 4), (1, 2), (4,4)]))
        G.Degree_ratio()
        
        >output:
        array([0. , 0.6, 0.2, 0.2])

        '''
        degree_dct=self.Degree_dct()
        degree_ratio=[]
        for key in (degree_dct):
            degree_ratio.append(len(degree_dct[key]))
        degree_ratio=np.array(degree_ratio)/sum(degree_ratio) 
        return degree_ratio

 

     
#%%1  Divide value

def Divide_value(file_to_correct):
    '''It takes an array (n,2) which represents the graph edges. It looks for any case in which
    two linked nodes are written in the same cell in a string variable and put the two variables
    in the two columns of the same row. It substitutes any full nan row with the previous row
    
    Parameters
    ----------
    file_to_correct : Dict can contain Series, arrays, constants, dataclass or list-like objects.
           If data is a dict, column order follows insertion-order. 
           If a dict contains Series which have an index defined, it is aligned by its index.
           Its shape must be (n,2)

    Raises
    ------
    Exception
        Exception raises if the file shape is not (n,2)

    Returns
    -------
    
        
    file: numpy.ndarray ((n,2))
        file with corrected entries

    '''
    file_dataFrame=pd.DataFrame(file_to_correct)
    file=np.array(file_dataFrame)
    
    if file.shape[1] != 2:
        raise Exception('file shape should be with axis=1 equal to 2')
        
    for i in range(len(file)):    
        for k in range(2):
            if type(file[i,k])==str: 
                empty_space=[]
                for char  in range(len(file[i,k])):                    
                    if file[i,k][char].isdigit()==False:
                        empty_space.append(char)
                        if char+1<len(file[i,k]) and file[i,k][char+1]!=' ':
                            file[i]=[(file[i,k][:empty_space[0]]),(file[i,k][char+1:len(file[i,k])])]

                            break
    return file


#%%2 Erase nan row
def Erase_nan_row(file_to_correct):
    '''It takes an array (n,2) which represents the graph edges. It returns a list of list of shape ((n-k,2))
       erasing all the k full nan row.
    
    

    Parameters
    ----------
    file_to_correct : Dict can contain Series, arrays, constants, dataclass or list-like objects.
           If data is a dict, column order follows insertion-order. 
           If a dict contains Series which have an index defined, it is aligned by its index.
           Its shape mut be (n,2)

    Raises
    ------
    Exception
        Exception raises if the file shape is not (n,2)

    Returns
    -------
    file: list ((n-k,2)) with  0<=k<=n
         file with no nan row
    '''
    file_data=pd.DataFrame(file_to_correct)
    file=np.array(file_data)
    
    if file.shape[1] != 2:
        raise Exception('file shape should be with axis=1 equal to 2')
    
    corrected_file=[]
    for i in range(len(file)):
        
        if type(file[i,0])!=str:
            if math.isnan(file[i,0])!=True and math.isnan(file[i,1])!=True:
             corrected_file.append([(file[i,0]),(file[i,1])])
             
        else:
            corrected_file.append([(file[i,0]),(file[i,1])]) 
            
    
    return corrected_file
    
#%%3  Edge_list
def Edge_list(file, number_of_edges):
    '''
    It takes a file of couples of number and return a list of a desired lenght(number_of_edge) of couple 
    of numbers expressed as  integers. The list has an ascending order following the first element of the couples   

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
    file_data=pd.DataFrame(file)
    file_array=np.array(file_data)
    if number_of_edges>len(file_array):
        raise Exception('number_of_edges must be minor of the file length')    
    if file_array.shape[1] != 2:
        raise Exception('file shape should be with axis=1 equal to 2')
        
    edges = []
    for i in range((number_of_edges)):
        edges.append([int(file_array[i, 0]),int(file_array[i, 1])])
    edges = sorted(edges)
    return edges

    
#%%4  Unfreeze_into_list
def Unfreeze_into_list(comunity):
    '''It takes an indexable variable of n elements and transform each element in a list
    of variables
    

    Parameters
    ----------
    comunity : indexable variable
         

    Returns
    -------
    comunity : It returns the same variable but each item is a list

    '''
    for i in range(len(comunity)):
        comunity[i]=list(comunity[i])
    return comunity

#%%5  Set_comunity

def Set_community_number(G, community):
    '''It assigns to each node to the graph a community number. Node with the same number
    are in the same community, each node cannot belongs to different communities. G is the graph,
    while each entry of the community  variable represents node of the same community. It returns 
    a dictionary: the keys are the nodes numbers, the values are the numbers of the community
    they belong to.
    

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    comunity : indexable variable. Each item contain a group of nodes.
               The union of all the group has to be the ensamble of nodes
        

    Returns
    -------
    comunity_number : dictionary key:item<->node number: community number
        

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

                        
#%%6  Fill_with_zeros

def Fill_with_zeros(list_):
    '''Given a list of list it makes all the list elements of the same length of the last one filling
     the first element of the list with zeros
    

    Parameters
    ----------
    list_ : list of list
        list.

    Returns
    -------
    None.
    
    example:
    input:
    list_=[[1,1],[3,3,3],[4,4,4,4]]
    fill_the_holes(list_)
    list_
    >output:
    [[ 1, 1, 0], [3, 3, 3, 0], [4, 4, 4, 4]]
            

    '''
    length=[]
    for i in range(len(list_)):
        length.append(len(list_[i]))
    max_len=max(length)
    
    for i in range(len(list_)):
        while len(list_[i])<max_len:
            list_[i].append(0)
            
#%%7  Size evolution
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
        
    step : int
           It represent the number of edge that will be added at each loop 
        
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
    distribution_evolution=[]
    n_step=int(len(G.edges)/step)
    evolution_mean=[]
    edges=list(G.edges())
    size=[]
    if feature=='degree':
        for i in range(n_step):      
            G=fn.SuperGraph(edges[:(i+1)*step])
            G.Relable_nodes()            
            value_size=np.array(list((getattr(nx, feature)(G))))[:,1]
            evolution_mean.append([np.mean(value_size),np.std(value_size)])
            distribution_evolution.append(list(G.Degree_ratio()))            
            size.append(len(G))   
        fn.Fill_with_zeros(distribution_evolution)   
        
    else:
        for i in range(n_step):
            G=fn.SuperGraph(edges[:(i+1)*step])
            G.Relable_nodes()            
            value_size=np.array(list((getattr(nx, feature)(G)).items()))[:,1]
            evolution_mean.append([np.mean(value_size),np.std(value_size)])
            distribution_evolution.append(value_size)
            size.append(len(G))
            
    return size, distribution_evolution,evolution_mean

#%%8 Dct_dist_link 
def Dct_dist_link(edges,map_dct):
    '''It calculates all the distances of the nodes linked together whose position is described by
    the dictionary map
    

    Parameters
    ----------
    edges : indexable variable
        It represents all the links of the network
    map_dct : indexable variable
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
        dct_dist_link[(i)]=dist
    return dct_dist_link



#%%9 Dct_dist 
def Dct_dist(G,map_dct):
    '''
    It returns all the distance among the nodes, even the not linked one. 
    It exploits a dictionary map of the position of the nodes
    

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    map_dct : indexable variable
        It describes the position of the nodes

    Returns
    -------
    dct_dist : dictionary of all the nodes linked together. key:value= (node_i,node_j):distance
        

    '''
    
    dct_dist={}
    list_nodes=list(G.nodes())
    for i in range(len(list_nodes)):

        for j  in range(i+1, len(G)):
            
            x0=map_dct[list_nodes[i]]
            x1=map_dct[list_nodes[j]]
            dist=np.linalg.norm(x0-x1)
            dct_dist[(list_nodes[i],list_nodes[j])]=(dist)
            
            
    return dct_dist


#%%10 Node_distance_frequency 
def Node_distance_frequency(dct_dist,nstep,step):
    '''It returns a binned not normalized nodes distance distribution. 
    It puts all the elements>nstep*step in the last bin 
    

    Parameters
    ----------
    dct_dist : indexable variable
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
    'in order to consider also elements > step*nstep'            
    n[nstep-1]=n[nstep-1]+len(dct_dist)-sum(n)
    node_distance_frequency=np.array(n)
    return node_distance_frequency  

#%%11 Conditional_probability 

def Conditional_probability(important_events_frequency,step, events_frequency):
    '''Given the binned frequency of an events and the frequency of them which are 'important', it calculates
    the probability to have an important event if an event is recorded. 
    

    Parameters
    ----------
    important_events_frequency : indexable
        It describes the frequency of important events for each bin
        
    step : width of the bin
        
    distance_frequency : indexable variable
        It represents the frequency of elements for each bins

    Raises
    ------
    ZeroDivisionError
        It raise the error when some linked node distance is in a bin i but no node distance is present in
        in the same bin

    Returns
    -------
    link_distance_probability : dct
        the keys are the the bin positions the values are the probabilities

    '''     
    link_distance_probability={}
    
    for i in range(len(important_events_frequency)):
        if events_frequency[i]!=0:
            link_distance_probability[(i+1)*step]=(important_events_frequency[i]/(events_frequency[i]))
        elif events_frequency[i]==0 and important_events_frequency[i]==0:
            link_distance_probability[(i+1)*step]=0
        else:
            raise ZeroDivisionError('There is no couple of nodes with this distance')
            
    return link_distance_probability
     
#%%12 Add_edges_from_map(G,map_dct,distance_linking_probability)                           
def Add_edges_from_map(G,dct_dist,dist_link_prob):
    '''
    It creates links among nodes of the graph G. 
    It exploits the nodes distances informations conteined in the distances dictionary, dct_dist,
    and the probability to make a link in function of the distance.
    

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        
    map_dct : indexable variable
        It describes the position of all the nodes
        
    distance_linking_probability : dct
        It describe the probability to have a link among two nodes at a given distance
    

    Returns
    -------
    None

    '''

    step=list(dist_link_prob.keys())[0]
    nstep=len(dist_link_prob)
    max_distance=max(dist_link_prob.keys())
    link_probabilities=list(dist_link_prob.values())
    
    for i in dct_dist: 
        dist=dct_dist[i]
        if dist<=max_distance:
            for k in range(nstep):
                if dist>k*step and dist<=(k+1)*step:                    
                    uniform=rn.uniform(0,1)
                    if uniform<= link_probabilities[k]:                        
                        G.add_edge(i[0],i[1])




#%%13 Break_strong_nodes

def Break_strongest_nodes(G, threshold):
    '''
    It breaks randomly links of nodes with the highest degree
    till the node with the maximum degree has a value under or equal to the trheshold

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        
    threshold : integer
        It is the maximum of the degree allows. The edges of a node with
        a higher degree will be dissolved until the node reaches
        a degrre value smaller than the threshold 

    Returns
    -------
    None.

    '''
    

    
    dct_degree=G.Degree_dct()
    while threshold<max(dct_degree):
        source=rn.choice(list(dct_degree.values())[len(dct_degree)-1])
        node=rn.choice(list(G.neighbors(source)))
        G.remove_edge(source,node)
        dct_degree=G.Degree_dct()
        

#%%12 Find_mode(pfd) 
def Find_mode(pdf):
    '''
    It returns the x value of the maximum of function p(x). Where x represent a bin 

    Parameters
    ----------
    pdf : indexable variable
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    mode : int
        the mode of the function

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


        
#%%13 Equalize_strong_nodes
def Equalize_strong_nodes(G_strong, G_weak):
    '''
    It compares two graph. It breaks links of the highest degree nodes of G_strong, until the maximum degree of G_strong
    is equal or minor the one of G_weak.
    Then it break links of the nodes with degree in between the degree mode of G_weak and the max degree of G_weak unitl 
    the ratio of these degree are equal or lower the ones of the weak network.

    Parameters
    ----------
    G_strong : function.SuperGraph
        
    G_weak : function.SuperGraph
        

    Returns
    -------
    None.

    '''
    
    
    degree_ratio_weak=G_weak.Degree_ratio()    
    threshold=fn.Find_mode(degree_ratio_weak)
    max_weak=len(G_weak.Degree_ratio())
    max_strong=len(G_strong.Degree_ratio())
    
    'otherwise we compare ratio of degree which do not exist in the weak graph'
    if max_weak<max_strong:       
        fn.Break_strongest_nodes(G_strong, max_weak-1) 
    i=max_weak    
    
    for i in range(max_weak-1,threshold,-1):      
        while len(G_strong.Degree_ratio())>=max_weak and (G_strong.Degree_ratio()[i])>(G_weak.Degree_ratio()[i]):

            source=rn.choice(G_strong.Degree_dct()[i])
            node=rn.choice(list(G_strong.neighbors(source)))
            G_strong.remove_edge(source,node)         

            

#%%14 Max_prob_target

    
def Max_prob_target (source,degree,map_dct,distance_linking_probability,max_dist,G):
    '''
    It finds the best linking target for a source node of the graph G. The target,
    if it is possible, has the degree chosen in the input, it's not a source's neighbour, 
    it's not its self and among the the nodes with the previous three characteristic is the most
    probable one

    Parameters
    ----------
    source : integer
        It represents the node label of the source.
        
        
    degree : integer
        It represents the desired degree of the target
        
    map_dct : indexable variable
        Each item represents the spatial position of a node of the graph
        
    distance_linking_probability :indexable variable
        It represents a binned density function distribution related to the distance, each bin distance 
        entry is the ptobability to have a link at tthat distance
        
    max_dist : integer
        It is the ditance threshold among the source and the others nodes. Node distances above it
        will not take into account
        
    G : networkx.classes.graph.Graph
        

    Returns
    -------
    target : int
        It is the label of node chosen

    '''
    x0=map_dct[source]
    max_prob=-1
    target=-5
    step=max_dist/len(distance_linking_probability)
    strenght_dct=G.Degree_dct()
    
    for i in strenght_dct[degree]:
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)
        
        if dist<max_dist:
            nstep=len(distance_linking_probability)
            for k in range(nstep):
                if dist>k*step and dist<(k+1)*step:
                    prob=list(distance_linking_probability.values())[k]
                    if prob>max_prob and i!=source and list(G.neighbors(source)).count(i)!=1:
                        max_prob=prob
                        target=i                    

                            
    '''In the case there was no nodes with the right conditions to link to'''
    if target==-5:

        target=fn.Min_distance_target(source,degree, map_dct,G)
    return target
    

    
                
        


                        
#%%15 Min_distance_target (source,strenght_dct,degree,map_dct,source_neighbour_list)

def Min_distance_target (source,degree,map_dct,G):
    '''
    Given a distance map of the nodes, From a starting vertice (the source) of the graph it returns the nearest node 
    with a given degree and which is not already linked to the source. If it is impossible a random one is chosen  

    Parameters
    ----------
    ----------
    source : integer
        It represents the node label of the source.
        
    strenght_dct :  indexable variable
        each items is a group of nodes with the same degree
        
    degree : integer
        It represents the desired degree of the target        
        
    map_dct : indexable variable
        Each item represents the spatial position of a node of the graph
        
    source_neighbour_list : list
                          list of the neighbour of the node

    Returns
    -------
    target : int
        It is the label of node chosen for the linkage

    '''
    
    x0=map_dct[source]
    min_=999
    target=-5
    'to avoids to start an endless while loop'
    source_neighbour_list=list(G.neighbors(source))
    assert len(source_neighbour_list)!=len(map_dct)
    
    for i in (G.Degree_dct())[degree]:        
        x1=map_dct[i]
        dist=np.linalg.norm(x0-x1)        
        if dist!=0 and source_neighbour_list.count(i)!=1:            
            if dist<min_:                
                min_=dist
                target=i
                
    if target<0:
        '''In order to find a target to link with even if it has a different degree 
        in respect to the one chosen'''
        target=fn.Random_target(G, source)    
        
    return target
#%%16 Random target

def Random_target(G,source):
    '''
    It links the source to a random node of the graph

    Parameters
    ----------
    G : function.SuperGraph
        It represents a graph
    source : int
        it represent the source of the link.

    Returns
    -------
    target : int
        it represent the target of the link

    '''
    target=-5
    while target<0:
        target_prova=rn.choice(list(G.nodes()))
        source_neighbour_list=list(G.neighbors(source))
        if target_prova!=source and source_neighbour_list.count(target_prova)!=1:
            target=target_prova
    return target
    



    
#%%17 Merge_small_component
def Merge_small_component(G, deg,map_dct,threshold):
    '''
    It merge all the graph component smaller than a given threshold, by linking a random node of
    the component with a node of the graph with a given degre==deg

    Parameters
    ----------
    G : function.SuperGraph
        
    deg : degree of the target node
        DESCRIPTION.
        
    map_dct : indexable variable
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
    
    all_components=list(nx.connected_components(G))
    for i in range(len(all_components)):
    
        if len(list(all_components)[i])<threshold:
            source=rn.choice(list(all_components[i]))
            list_nodes=G.Degree_dct()[deg]
            
            if len(list_nodes)==0:
                raise Exception('the node list with the degree chosen is empty')
                
            target=Min_distance_target(source,deg,map_dct,G)
            G.add_edge(source,target)
        
        

    
#%%18 Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G,n_links, degree_dct)
def Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,G,n_links, degree_dct):
    '''
    It takes an isolated node of the graph and it creates n links (n=n_links) with n different nodes.
    For each degree group, from 0 to the degree n-1,one node will be chosen as a target till n targets will be reached.
    At the end of the process the number of nodes with degree equal to n increase of 2 
    and the number of isolated nodes decrease of two. 
    The degree ratio of the other values remains constant. The attachment rule of the links follow the
    Max_prob_target function.

    Parameters
    ----------
    map_dct : indexable variable
        Each item represents the spatial position of a node of the graph
        
    prob_distribution : indexable variable
        It represents a binned density function distribution related to the distance, each bin distance 
        entry is the ptobability to have a link at tthat distance
        
    max_dist : integer
        It is the ditance threshold among the source and the others nodes. Node distances above it
        will not take into account
        
    G : function.SuperGraph
        
    n_links : integer
            number of links to create
        
    degree_dct : indexable variable
        each items is the group of nodes with the same degree

    Returns
    -------
    None.

    '''
    if len(degree_dct[0])!=0:
        source=rn.choice(degree_dct[0])
        for i in range(n_links):                      
            target=fn.Max_prob_target(source,i,map_dct,prob_distribution,max_dist_link,G)
            G.add_edge(source, target)
            degree_dct=G.Degree_dct()


#%%19 Remove_edge_of_degree 
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
    
    degree_dct=G.Degree_dct()
    source=rn.choice(degree_dct[degree])
    node=source
    while node==source:
        node=rn.choice(list(G.neighbors(source)))
    
    G.remove_edge(source,node)   
            
    
#%%20 Equalizer_top_down

def Equalizer_top_down(G,Copycat):
    '''
    It makes each degree ratio of the network Copycat equal or lower than
    the corresponding degree ratio of network G. It starts from the higer
    degree and if it needed it removes edges exploiting the fn.Remove_edge_of_degree
    The process end when the ratios have the right values, the ratio for the degree 
    0 cannot be corrected.
    
    Parameters
    ----------
    G : fn.SuperGraph
        Model graph
    Copycat : fn.SuperGraph
        Graph to modify

    Returns
    -------
    None.

    '''
    for i in range(len(G.Degree_ratio())-1,0,-1):
            while i<= len(Copycat.Degree_ratio())-1 and Copycat.Degree_ratio()[i]> G.Degree_ratio()[i]:
                fn.Remove_edge_of_degree(i, Copycat) 

#%%21 Equalizer_down_top
def Equalizer_down_top(G,Copycat,map_dct,prob_distribution,max_dist_link):
    '''
    It makes each degree ratio of the network Copycat equal or bigger than
    the corresponding degree ratio of network G. If a degree ratio is lower
    then what expected it links an isolated node exploiting the function 
    fn.Link_2_ZeroNode, in order to increas the equied degree ratio.
    The process end when no isolated nodes are left or the ratios have the 
    right values

    Parameters
    ----------
    G : fn.SuperGraph
        Model graph
    Copycat : fn.SuperGraph
        Graph to modify
    map_dct : dct
        It described the topological positions of the nodes
        
    prob_distribution : dct
                The probability to have a link among two nodes
                at a fixed distance
        
    max_dist_link : int
            The greatest distance among two linked nodes

    Returns
    -------
    None.

    '''
    
    for i in range(1,len(G.Degree_ratio())):
        if i< len(Copycat.Degree_ratio()):
            while len(Copycat.Degree_dct()[0])>0 and Copycat.Degree_ratio()[i]< G.Degree_ratio()[i]:  
                fn.Link_2_ZeroNode(map_dct, prob_distribution, max_dist_link,Copycat,i, Copycat.Degree_dct())

#%%22 Copymap_degree_correction(Copy_map,G,map_dct,max_dist_link,prob_distribution,Merge=False):


def Copymap_degree_correction(Copy_map,G,map_dct,max_dist_link,prob_distribution,Merge=False):
    '''
    It returns a network in which new links are added to the isolated node of the Copy_map 
    and some other links are removed in order to make the degree distribution of the new network similar
    to the degree distribution of G

    Parameters
    ----------
    Copy_map : function.SuperGraph
        Graph we want to change
        
    G : function.SuperGraph
        Model Graph
        
    map_dct : indexable variable
        Each item represents the spatial position of a node of the graph         
    
    max_dist : integer
        It is the ditance threshold among the source and the others nodes. Node distances above it
        will not take into account
        
    prob_distribution : indexable variable
        It represents a binned density function distribution related to the distance, each bin distance 
        entry is the probability to have a link at that distance
        
    Merge : bool, optional
        If True at the end of the process the function merge the small components.
        The default is False.

    Returns
    -------
    Copycat : function.SuperGraph
        It returns the graph with the degree ratio corrected
    '''
    

    Copycat=fn.SuperGraph(Copy_map)    
    fn.Break_strongest_nodes(Copycat, max(np.array(list(G.degree()))[:,1]))    
    fn.Equalize_strong_nodes(Copycat, G)
    
    while len(Copycat.Degree_dct()[0])>0:
        Equalizer_top_down(G,Copycat)                  
        Equalizer_down_top(G,Copycat,map_dct,prob_distribution,max_dist_link)       
        
    if Merge==True:
         Merge_small_component(Copycat,deg=1, map_dct=map_dct, threshold=3)
                  
    fn.Break_strongest_nodes(Copycat, max(np.array(list(G.degree()))[:,1]))
    Copycat.Relable_nodes()
    
    return Copycat
#%%23 Trunk_array
def Trunk_array_at_nan(array):
    '''
    It takes an array n*m dimensional (m>1) , it reads each row
    and when it finds a nan value it copies all the value of the road 
    till the nan value, excluded, into a list. At the end of the process 
    it will return a list of lists

    Parameters
    ----------
    array :np.array n*m dimensional (m>1)
        

    Returns
    -------
    new_array : list of lists
        

    '''
    new_array=[] #np.zeros([len(array),])
    for i in range(len(array)):
        cutter=len(array[i])
        j=0
        while j <len(array[i])-1 and cutter==len(array[i]):
            if math.isnan(array[i,j])==True:
                cutter=j
            j+=1
        new_array.append(list(array[i,:cutter]))
        
    return new_array

#%%24 Directory_creation(name)
def Directory_creation(name):
    '''It creates a new directory in the directory where the function in launched.
    The name of hte directory is the argument of the function. If the directory already exists it raise an exception
    

    Parameters
    ----------
    name : str
        name of the new directory.

    Raises
    ------
    Exception
        The exception raises if there is already a directory with the name chosen

    Returns
    -------
    None.

    '''
    name_simulation='\\'+name
    path = os.getcwd()+name_simulation
    # Check whether the specified path exists or not
    isExist = os.path.exists(path+name_simulation)

    if not isExist:  
      # Create a new directory because it does not exist 
      os.makedirs(path)
      
    else:
        raise Exception('This directory name has already taken')