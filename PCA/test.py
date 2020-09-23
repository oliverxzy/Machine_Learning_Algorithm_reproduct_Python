import pandas as pd
from main import *
import numpy as np

df = pd.read_csv('pca_data.csv')
lowdata, redata = pca(df.as_matrix(),2)

output_Data = pd.DataFrame(lowdata)
output_Data.to_csv('pca_after_data.csv')
