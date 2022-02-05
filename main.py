# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:42:39 2022

@author: Guido
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:46:39 2021

@author: Guido
"""

#%%
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import function as fn
import matplotlib.cm as cm


file=pd.read_table('C:/Users/Guido/Desktop/guido/Complex_networks/python/California/roadnet-ca.txt')
file=np.array(file)
file=fn.Divide_value(file)


#%%
number_of_edge=10000
edges=fn.Edge_list(file, number_of_edge)
G=nx.Graph(edges)
G=fn.Sorted_graph(G)
edges=list(G.edges())
#%%
G=nx.convert_node_labels_to_integers(G) #non so perchè ma è fondamentale questa funzione per funzione comunità'
edges=list(G.edges())


#%%main features
Strenght=np.array(list(G.degree))
Closeness_Centrality=np.array(list((nx.closeness_centrality(G)).items()))
Betweeness_Centrality=np.array(list((nx.betweenness_centrality(G)).items()))
Clustering=np.array(list((nx.clustering(G)).items()))

community=nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
community=fn.Unfreeze_into_list(community)
community_color=fn.Set_community_number(G, community)
colors = (cm.CMRmap(np.linspace(0, 1, max(community_color))))

#%%VIsualizzazione network
Betweeness_Centrality_edges=nx.edge_betweenness_centrality(G)
node_size=np.exp((Betweeness_Centrality[:,1]+1)*100)
#width=(np.array(list(Salient_dct.values()))+0.32)**3
width=(np.array(list(Betweeness_Centrality_edges.values()))+0.95)
nx.draw(G,node_size=node_size,font_size=6, node_color=list(community_color.values()),width=width)
plt.title("Road prova")
plt.show(G)


'riga aggiunta'



#%% histogram

n, bins, patches=plt.hist(Strenght[:,1],color=colors[3], density=True, stacked=True)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Degree distribution")
#plt.savefig("strenght-freq.png", dpi=1000)
plt.show()

n, bins, patches=plt.hist(Betweeness_Centrality[:,1],color=colors[3])
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Betweeness_Centrality distribution-Reduced graph 1800 nodes")
#plt.savefig("BC-frec.png", dpi=1000)
plt.show()

n, bins, patches=plt.hist(Closeness_Centrality[:,1],color=colors[3], density=True, stacked=True)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Closeness_Centrality distribution-Reduced graph 1800 nodes")
#plt.savefig("cc-freq.png", dpi=1000)
plt.show()

n, bins, patches=plt.hist(Clustering[:,1],color=colors[3], density=True, stacked=True)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("clustering")
#plt.savefig("Mean Betweeness Centrality random.png", dpi=1000)
plt.show()

#%% SCATTERPLOT



fig, ax = plt.subplots()
scatter=ax.scatter(list(community_color.values()),Strenght[:,1],c=list(community_color.values()),s=2)
plt.xlabel("Community")
plt.ylabel("Degree")
plt.title("Degree vs comumnities")
plt.savefig("Deg scatter.pdf", dpi=500)
plt.show()


plt.scatter(list(community_color.values()),Clustering[:,1],c=list(community_color.values()),s=2)
plt.xlabel("Community")
plt.ylabel("Clustering")
plt.title(" Clustering vs communities")
plt.savefig("Clustering scatter.pdf", dpi=500)
plt.show()


plt.scatter((list(community_color.values())),Closeness_Centrality[:,1],c=list(community_color.values()),s=1)
plt.xlabel("Community")
plt.ylabel("Closeness Centrality")
plt.title(" Closeness Centrality vs communities")
plt.savefig("CC scatter.pdf", dpi=500)
plt.show()


plt.scatter(list(community_color.values()),Betweeness_Centrality[:,1],c=list(community_color.values()),s=1)
plt.xlabel("Communities")
plt.ylabel("Betweeness centrality")
plt.title("Betweeness centrality vs comunities")
plt.savefig("BC scatter.pdf", dpi=500)
plt.show()



#_______________-confronto con strenght______

plt.scatter(Betweeness_Centrality[:,1],Strenght[:,1],c=list(community_color.values()),s=2)
plt.ylabel("Degree")
plt.xlabel("Betweeness centrality")
#for i in (range(len(Strenght[:,1]))):
    #plt.annotate(i, (Strenght[:,1][i], Betweeness_Centrality[:,1][i]))
plt.title("Betweeness centrality vs degree")
plt.savefig("BC vs degree.pdf", dpi=500)
plt.show()


plt.scatter(Closeness_Centrality[:,1],Strenght[:,1],c=list(community_color.values()),s=2)
plt.ylabel("Degree")
plt.xlabel("Closeness centrality")
#for i in (range(len(Strenght[:,1]))):
    #plt.annotate(i, (Strenght[:,1][i], Closeness_Centrality[:,1][i]))
plt.title("Closeness centrality vs degree")
plt.savefig("CC vs degree.pdf", dpi=500)
plt.show()

plt.scatter(Clustering[:,1],Strenght[:,1],c=list(community_color.values()),s=2)
plt.ylabel("strenght")
plt.xlabel("clustering")
plt.title("clustering vs strenght ")
plt.show()


#__________________-confronto con CC


plt.scatter(Closeness_Centrality[:,1],Betweeness_Centrality[:,1],c=list(community_color.values()),s=(Strenght[:,1]/3)**6)
plt.ylabel("Betweeness Centrality")
plt.xlabel('closness centrality')
plt.title("Closeness_Centrality vs Betweeness_Centrality ")
plt.show()

plt.scatter(Closeness_Centrality[:,1],Clustering[:,1],c=list(community_color.values()),s=2)
plt.ylabel("Clustering")
plt.xlabel('closness centrality')
plt.title("Closeness_Centrality vs Clustering")
plt.show()


   

    
#%% Power law

k=[]
for i in range(int(max(Strenght[:,1]))+1):
               k.append(list(Strenght[:,1]).count(i))
k=np.array(k)/sum(k)
n_strenght=np.array(range(len(k)))
k=np.array((n_strenght,k)).transpose()
k=np.array(sorted(k, key=lambda x: x[1]))

attachment_deg=np.zeros((9,2))
for i in range(len(k)):
    attachment_deg[i]=[k[i,0],sum(k[:i,1])]
    
    
plt.scatter(k[:,0],k[:,1])
plt.xlabel("degree")
plt.ylabel("probability")
plt.title("power law")
plt.show()          
            
        
#%% Strength in time     

degree_size, degree_ratio_time_evolution,degree_mean=fn.Time_evolution(G,100,'degree')
colors = (cm.tab10(np.linspace(0, 1, len(degree_ratio_time_evolution.transpose()))))

fig, ax = plt.subplots()

for i in range(len(degree_ratio_time_evolution.transpose())):
    scatter=ax.scatter(degree_size,    degree_ratio_time_evolution[:,i], c=[colors[i]]*len(degree_size), s=2,label='%s' %i)
    
legend = ax.legend(loc='upper left', shadow=True, fontsize=8)  
plt.xlabel("number of nodes")
plt.xlim(-30, max(degree_size+10))
plt.ylabel("ratio of each degree")
plt.title("ratio of each degree for increasing nodes")

#plt.savefig("ratio of each degree.png", dpi=1000)
plt.show()

#%% BC in time


BC_size, BC_time_evolution,BC_mean=fn.Time_evolution(G,1000,'betweenness_centrality')    
x =BC_mean
colors = (cm.CMRmap(np.linspace(0.01, 0.9, len(x))))
fig, ax = plt.subplots()
scatter=ax.scatter(BC_size, list(x[:,0]),c=colors,s=10)

ax.errorbar(BC_size,list(x[:,0]), yerr=list(x[:,1]), xerr=None,fmt='o', ecolor=colors,markersize=0)
plt.xlabel("number of nodes")
plt.ylabel("Mean Betweeness Centrality")
plt.title("Mean Betweeness Centrality")

#plt.savefig("Mean Betweeness Centrality.png", dpi=1000)
plt.show()

x=BC_time_evolution
fig, ax = plt.subplots()
colors = (cm.magma(np.linspace(0, 1, len(BC_size))))
for i in range(len(BC_size)):
    values, base = np.histogram(x[i],bins=500)
    cumulative = np.cumsum(values/BC_size[i])
    ax.plot(base[:-1], cumulative, c=colors[-i-1],label=list(BC_size)[i])
    
ax.set_xlabel("Values")
ax.set_ylabel("Frequency")


ax.legend(prop={'size': 10})
ax.set_title('Cumulative distribution of between_centrality centrality')
#plt.savefig("Cc-cumulative-convergence.png", dpi=1000)
plt.show()


#%% clustering in time

Clustering_size, Clustering_time_evolution,Clustering_mean=fn.Time_evolution(G,1000,'clustering')  
 
x = np.array(Clustering_mean)

colors = (cm.magma(np.linspace(0.01, 0.9, len(Clustering_size))))

fig, ax = plt.subplots()
scatter=ax.scatter(Clustering_size,list(x[:,0]),c=colors,s=10)

ax.errorbar(Clustering_size,list(x[:,0]), yerr=list(x[:,1]), xerr=None,fmt='o',markersize=0, ecolor=colors)
plt.xlabel("number of  total Kn (node*1000) ")
plt.ylabel("Mean Clustering")
plt.title("Mean Clustering")

#plt.savefig("Mean Clustering.png", dpi=1000)
plt.show()

x=Clustering_time_evolution
fig, ax = plt.subplots()
colors = (cm.magma(np.linspace(0, 1, len(Clustering_size))))
for i in range(len(Clustering_size)):
    values, base = np.histogram(x[i],bins=500)
    cumulative = np.cumsum(values/Clustering_size[i])
    ax.plot(base[:-1], cumulative, c=colors[-i-1],label=list(Clustering_size)[i])
    
ax.set_xlabel("Values")
ax.set_ylabel("Frequency")


ax.legend(prop={'size': 10})
ax.set_title('Cumulative distribution of Clustering')
#plt.savefig("Cc-cumulative-convergence.png", dpi=1000)
plt.show()


#%% CC in time
#51430
CC_size, CC_time_evolution,CC_mean=fn.Time_evolution(G,1000,'closeness_centrality')   


x=CC_time_evolution
fig, ax = plt.subplots()
colors = (cm.magma(np.linspace(0, 1, len(CC_size))))
for i in range(len(CC_size)):
    values, base = np.histogram(x[i],bins=500)
    cumulative = np.cumsum(values/CC_size[i])
    ax.plot(base[:-1], cumulative, c=colors[-i-1],label=list(CC_size)[i])
    #n, bins, patches=ax.hist(x[i],bins=int(len(comunity)*2/3),density=True,stacked=False,histtype='step',color=colors[i], label=i)
ax.set_xlabel("Values")
ax.set_ylabel("Frequency")


ax.legend(prop={'size': 10})
ax.set_title('Cumulative distribution of closeness centrality')
#plt.savefig("Cc-cumulative-convergence.png", dpi=1000)
plt.show()


x = CC_mean

colors = (cm.CMRmap(np.linspace(0.01, 0.9, len(CC_size))))

fig, ax = plt.subplots()
scatter=ax.scatter(CC_size,list(x[:,0]),c=colors,s=10)

ax.errorbar(CC_size,list(x[:,0]), yerr=list(x[:,1]), xerr=None,fmt='o', ecolor=colors,markersize=0)
plt.xlabel("Steps")
plt.ylabel("Mean Coseness Centrality")
plt.title("Mean Closeness Centrality")

#plt.savefig("Mean Closeness Centrality.png", dpi=1000)
plt.show()
    

#%%

x=data_road
y=comunity

    
Mean_BC_com=[]
for i in range(len(y)):
    print(i)
    Mean_BC=[]
    for j in range(len(G)):
        if x[j,4]==i:
            Mean_BC.append(x[i,2])
    Mean_BC_com.append([np.mean(Mean_BC),i])
Mean_BC_com=np.array((Mean_BC_com))
Mean_BC_com=Mean_BC_com[Mean_BC_com[:,0].argsort()]
Mean_BC_com=np.flip(Mean_BC_com)
chi=[]
comp=[]
for i in range(len(list(BC_tot))):
    uno=(BC_tot[i])
    j=i+1
    while j < len(list(BC_tot)):
        due=(BC_tot[j])
        chi.append([chi2(due,uno)*BC_len[i],i,j])
        if(chi2(due,uno)<BC_len[i]):
            comp.append([chi2(due,uno)*BC_len[i],i,j])
        j=j+1