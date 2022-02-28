# Software_and_Computing
The pdf file Campani-Califonia_road_map.pdf: it contains the report of the complex network analysis of the californian road. 
                                             In the file introduction there is a briefly explanation of all the data analysis procedure.
                                             
The txt file roadnet-ca.txt: contains couples of numbers representing couples of conected roads

The file main.py: contains the code useful to read the file roadnet-ca.tx, deleting some bugs in the dataset,
                  to make plots/histograms of the main features of the whole/reduced graph and
                  to visualize the features distribution
                  
The file ERG. py: contains the code useful to create an Erdos-Rény graph with a linkg probability equal to the one of a the roadnet
                  and to make plots/histograms of the main features of this graph
                  
The file Copycat.py: contains the code useful to create a network which try to copy the features distribution of a given network.
                     and to make plots/histograms of the main features of this copynetwork.
                     
         
                  
The file function.py contains all the function used in the three .py files and it also contains the tests of the functions

The libraries used are: math, matplotlib, networkx, numpy, pandas, pytest, random.
