import matplotlib.pyplot as plt

# data visualization
def cluster_plot(data,index_list):
    plt.scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], c=index_list)
    plt.title('K means clustering plot')
    return plt.show()
