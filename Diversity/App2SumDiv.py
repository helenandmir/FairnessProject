import numpy as np
import matplotlib
import pandas as pd
from operator import itemgetter

from scipy.spatial import distance
from sklearn.neighbors import KDTree
import numpy as np
import  math
from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

fileA ="../DataSet/Point.csv"
fileB ="../LongTimeData/Point_group_250_r_for_div.csv"


df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)

dic_point_to_ball ={}
list_centers = list(df2["ID"])
all_point = list(df1["ID"])
Ball_list = list(df2["Ball"])
for i in df2["ID"]:
    for l in [int(num) for num in Ball_list[list_centers.index(i)].strip('{}').split(',')]:
        dic_point_to_ball[l] = i
pairs = {}
print("###")
dic_points_to_X ={}
for i in df2["ID"]:
    for j in Ball_list[list_centers.index(i)].strip('{}').split(','):
        dic_points_to_X[int(j)] = df1.X[all_point.index(int(j))]

dic_centers_to_X ={}
for i in df2["ID"]:
    dic_centers_to_X[i] = df1.X[i]

print("###")

sorted_dic_points_to_X = dict(sorted(dic_points_to_X.items(), key=itemgetter(1)))

S =[]
flag =0
sorted_dic_points_to_X_copy = sorted_dic_points_to_X.copy()
while len(sorted_dic_points_to_X) >= 2:
    p1 = list(sorted_dic_points_to_X.keys())[0]
    p2 = list(sorted_dic_points_to_X.keys())[len(sorted_dic_points_to_X)-1]
    j=1
    while dic_point_to_ball[p1] == dic_point_to_ball[p2] :
        if  j>len(list(sorted_dic_points_to_X.keys()))-1:
            flag =1
            break
        p1 = list(sorted_dic_points_to_X.keys())[j]
        j=j+1
    if flag ==1:
        S.append(p1)
        break
    S.append(p1)
    S.append(p2)
    for i in sorted_dic_points_to_X_copy.keys():
        if dic_point_to_ball[i] == dic_point_to_ball[p1]:
            sorted_dic_points_to_X.pop(i)
        if dic_point_to_ball[i] == dic_point_to_ball[p2]:
            sorted_dic_points_to_X.pop(i)
print("##")
sum =0
for i in range(len(S)):
    for j in range(i + 1, len(S)):
        sum =sum + abs(sorted_dic_points_to_X_copy[S[i]]-sorted_dic_points_to_X_copy[S[j]])
print(len(S))
sum2 =0
for i in range(len(list_centers)):
    for j in range(i + 1, len(list_centers)):

        sum2 =sum2 + abs(dic_centers_to_X[list_centers[i]]-dic_centers_to_X[list_centers[j]])

print("before div sum = {}".format(sum2))
print("after div sum= {}".format(sum))
print("diff = {}".format(sum-sum2))