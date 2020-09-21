import matplotlib.pyplot as plt

# data visualization
def cluster_plot(data,index_list):
    plt.scatter(x=data['x1'], y=data['x2'], c=index_list)
    plt.title('K means clustering plot')
    return plt.show()
