from sklearn.neighbors import KDTree
import pandas as pd
import math
import numpy as np
import matplotlib
from Similarity import Hungarian
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import numpy as np


fileA ="../DataSet/Point.csv"
fileB="../DataSet/Point_radius_1000.csv"


fileC ="../LongTimeData/Point_group_1000_u.csv"
fileD ="../LongTimeData/Point_group_1000_u_num.csv"

fileE ="../LongTimeData/Point_group_1000_u_for_div.csv"

df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)
df3 = pd.read_csv(fileC)
df4 = pd.read_csv(fileD)
df5 = pd.read_csv(fileE)

center_list  = list(df3["ID"])


color_list = df3.columns.tolist()
color_list.remove("ID")


dic_center_ball ={}
for id in center_list:
    dic_center_ball[id] =[]
dic_num_center_ball ={}
dic_list_num_colors ={}

# create ball
list_t = list(center_list)
tuple_color = tuple(zip(list(df1.X[list_t]), list(df1.Y[list_t]), list(df1.Z[list_t])))
tree_c = KDTree(np.array(list(tuple_color)))

for i in df1.ID:
    dist_c, ind_c = tree_c.query([[df1.X[i], df1.Y[i], df1.Z[i]]], 1)
    if(i not in center_list):
       dic_center_ball[list_t[ind_c[0][0]]].append(i)
    else:
       k=list_t.index(i)
       dic_center_ball[i].append(i)



# Hngarian algo
color_list = pd.read_csv(fileC, nrows=0).columns.tolist()
color_list.remove("ID")
req_dic = {}
dic_center_and_colors = {}
for i in color_list:
    req_dic[i] = 0
for i in df3.ID:
    dic_center_and_colors[i] = df1.Colors[i]
    c = df1.Colors[i]
    req_dic[c] = req_dic[c] + 1


print("req_dic:{}".format(req_dic))


H = Hungarian.MaxSum(req_dic,fileA,fileC)
H.convert()
max_sum_list , dic_center_and_colors2=H.play_hungarian_algo()



# create new balls
print("---> new center ball <---")

dic_new_center_ball ={}
for i in dic_center_ball.keys():

    dic_new_center_ball[i] = [x for x in dic_center_ball[i] if df1.Colors[x] == dic_center_and_colors2[i]]

new_rec_colors = list(dic_center_and_colors2.values())


# new_dic_center_ball ={}
# for i in dic_center_ball:
#  if len(dic_center_ball[i]) !=0:
#      new_dic_center_ball[i]=dic_center_ball[i]


print("--->########<---")

for key in dic_new_center_ball:
    print(key, 'corresponds to', dic_new_center_ball[key])

print(dic_center_and_colors2)

print(max_sum_list)

new_list_centers = list(dic_new_center_ball.keys())
df5["ID"] = new_list_centers

for i in dic_new_center_ball:
    print("i = {}".format(i))
    print("new_list_centers.index(i) = {}".format(new_list_centers.index(i)))
    print("set(new_dic_center_ball[i]) = {}".format(set(dic_new_center_ball[i])))
    print("len = {}".format(len(set(dic_new_center_ball[i]))))
    list_a  = list(set(dic_new_center_ball[i]))
    if( len(list_a) >3000):
        half_length = len(list_a)//2
        first_half, second_half = list_a[:half_length],list_a[half_length:]
        df5["Ball1"][new_list_centers.index(i)] = set(first_half)
        df5["Ball2"][new_list_centers.index(i)] = set(second_half)
    else:
       df5["Ball1"][new_list_centers.index(i)] = set(list_a)


df5.to_csv(fileE)
df5.head()