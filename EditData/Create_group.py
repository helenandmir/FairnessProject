from sklearn.neighbors import KDTree
import pandas as pd
import math
import numpy as np
import matplotlib

import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import numpy as np
fileA ="../DataSet/Banks.csv"
fileB="../DataSet/Banks_radius_100.csv"
fileC ="../DataSet/banks_group_100_2.csv"

df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)
df3 = pd.read_csv(fileC)
center_list=[11773, 11836, 15450, 11660, 10664, 10326, 23825, 15124, 10322, 25701, 10587, 11177, 10837, 15785, 15709, 11179, 11137, 27370, 18728]
color_list = df3.columns.tolist()
color_list.remove("ID")
# df3["ID"] = center_list
# df3.to_csv(fileC)
# df3.head()


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
    dic_center_ball[list_t[ind_c[0][0]]].append(i)

for c in center_list:
    dic_num_center_ball[c] = len(dic_center_ball[c])

for c in center_list:
    dic_list_num_colors[c] =[c]
    for color in color_list:
        #dic_list_num_colors[c].append(len([j for j in dic_center_ball[c] if df1.Colors[j] ==color]))

        dic_list_num_colors[c].append(len([j for j in dic_center_ball[c] if df1.Colors[j] ==color])/dic_num_center_ball[c])

for c in center_list:
    df3.loc[center_list.index(c),:] = dic_list_num_colors[c]

df3.to_csv(fileC)
df3.head()

