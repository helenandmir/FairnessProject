import sys
import time
import math
import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt, colors
from math import sin, cos, sqrt, atan2
R=6373.0
X=[]
Y=[]
Z=[]
file_name = "../DataSet/Banks.csv"
df = pd.read_csv(file_name, usecols=["ID","X", "Y","Z","Longitude","Latitude"])
print("create X,Y,Z lists")
for id in list(df.ID):
    lon1 = math.radians(df.Longitude[id])
    lat1 = math.radians(df.Latitude[id])
    x1= R * math.cos(lat1) * math.cos(lon1)
    X.append(x1)
    y1 = R * math.cos(lat1) * math.sin(lon1)
    Y.append(y1)
    z1 = R * math.sin(lat1)
    Z.append(z1)
print("write X,Y,Z list to file")
print("###")
df1 = pd.read_csv(file_name)
df1.to_csv(file_name, index=False)
df1["X"] = X
df1.to_csv(file_name, index=False)
df1.head()
df1["Y"] = Y
df1.to_csv(file_name, index=False)
df1.head()
df1["Z"] = Z
df1.to_csv(file_name, index=False)
df1.head()

