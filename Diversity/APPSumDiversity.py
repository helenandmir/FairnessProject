import numpy as np
import matplotlib
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import numpy as np
import  math

fileA ="../DataSet/Banks.csv"
fileB ="../LongTimeData/500_num.csv"


df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)

def find_farthest_points():
    max_dis =-1
    p1 = None
    p2 = None
    for i in df1["ID"]:
        for j in dic_point_dis[i]:
            if j>max_dis:
                max_dis = j
                p1 =i
                p2 = dic_point_dis_to_point[dic_point_dis[i].index(j)]
    return p1,p2

list_centers = list(df2["ID"])
dic_point_dis ={}
dic_point_dis_to_point ={}
for i in df1["ID"]:
    dic_point_dis[i] =[]
    dic_point_dis_to_point[i] =[]
    for j in df2["ID"]:
        max_distance = -1
        farthest_point = None
        for l in [int(num) for num in df2.Ball[list_centers.index(j)].strip('{}').split(',')]:
           distance = abs(df1.X[i] - df1.X[l])
           if distance > max_distance:
                max_distance = distance
                farthest_point = l
        dic_point_dis[i].append(max_distance)
        dic_point_dis_to_point[i].append(farthest_point)
        print(dic_point_dis[i])
S =[]
while len(S) != len(df1["ID"]):
    p1,p2 = find_farthest_points();
    dic_point_dis_to_point.pop(p1)
    dic_point_dis_to_point.pop(p2)
    dic_point_dis.pop(p1)
    dic_point_dis.pop(p2)
    S.append(p1)
    S.append(p2)
