from main import *
from utils import cluster_plot
import pandas as pd

dataset = pd.read_csv('kmean_df.csv')
x = input('Enter number of K:')
centroid, index_list = kmeans(dataset, int(x))
print('centroids plot as following:')
cluster_plot(dataset,index_list)