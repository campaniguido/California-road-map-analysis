import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import function as fn

def Hist_plot(distribution, color, title, path=None, save_fig=False, extention='pdf'):
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
        
    extention: str, optional
               it represent the file extension of the file to save. The default is 'pdf'
    Returns
    -------
    None.

    '''
    n, bins, patches=plt.hist(distribution,color=color)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(title)
    if save_fig==True:
        plt.savefig(path+'\\'+title+'.'+extention, dpi=100)
    else:
       plt.show()
          
    plt.close()

#%%21 Scatter_plot

def Scatter_plot(distribution1, name_distribution1, distribution2, name_distribution2, color,path=None, save_fig=False, extention='pdf'):
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
        
    extention: str, optional
               it represent the file extension of the file to save. The default is 'pdf'
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
        plt.savefig(path+'\\'+name_distribution1+ ' vs '+  name_distribution2 +"."+ extention, dpi=100)
    else:
        plt.show()
       
#%%22 Feature_mean_evolution

def Feature_mean_evolution(feature_size,feature_mean, feature_name,path=None, save_fig=False, extention='pdf'):
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
        
    extention: str, optional
               it represent the file extension of the file to save. The default is 'pdf'
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
        plt.savefig(path+'\\'+"Mean"+feature_name+"."+ extention, dpi=100)
    else:
        plt.show()
#%%23 Feature_cumulative_evolution



def Feature_cumulative_evolution(feature, feature_name,path=None, save_fig=False, extention='pdf'):
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
        
    extention: str, optional
               it represent the file extension of the file to save. The default is 'pdf'

    Returns
    -------
    None.

    '''

                    
        
    
    x=fn.Trunk_array_at_nan(feature)
    
    fig, ax = plt.subplots()
    colors = (cm.magma(np.linspace(0, 1, len(x))))
    for i in range(len(x)):
        size=len(x[i])
        values, base = np.histogram(x[i],bins=500)
        cumulative = np.cumsum(values/size)
        ax.plot(base[:-1], cumulative, c=colors[-i-1],label=size)        
    ax.set_xlabel("Values")
    ax.set_ylabel("Cumulative probability")    
    ax.legend(title="# nodes",prop={'size': 10})
    ax.set_title('Cumulative distributions of '+  feature_name)
    if save_fig==True:
        plt.savefig(path+'\\'+feature_name+"cumulative-convergence""."+ extention, dpi=100)
    else:
        plt.show()

#%%24 Feature_ratio_evolution



def Feature_ratio_evolution(feature_position,feature_ratio, feature_name,path=None, save_fig=False, extention='pdf'):
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

    extention: str, optional
               it represent the file extension of the file to save. The default is 'pdf'

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
        
    ax.legend(title="degree",loc='upper left', shadow=True, fontsize=8)  
    plt.xlabel("number of nodes")
    plt.xlim(-30, max(size+10))
    plt.ylabel("ratio of each " + feature_name)
    plt.title("ratio of each " + feature_name + " for increasing size")
    if save_fig==True:
        plt.savefig(path+'\\'+"ratio of each"+ feature_name +"."+ extention, dpi=100)
    else:
        plt.show()

