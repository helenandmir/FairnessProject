from sklearn.neighbors import KDTree
import pandas as pd
import math
import numpy as np
import matplotlib

import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import numpy as np
fileA ="../DataSet/Point.csv"
fileB="../DataSet/Point_radius_100.csv"
fileC ="../DataSet/Point_group_100_3.csv"
fileD ="../DataSet/Point_group_100_3_num.csv"

df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)
df3 = pd.read_csv(fileC)
df4 = pd.read_csv(fileD)
center_list=[420316, 207773, 199630, 46424, 190295, 141898, 357737, 349666, 363769, 230919, 260938, 125313, 374495, 176704, 321712, 251958, 145344, 99553, 343784, 5450, 150026, 342383, 285857, 382270, 415984, 397976, 415216, 182544, 282667, 332032, 85172, 381207, 179635, 372816, 39019, 13081, 77065, 288965, 214996, 132422, 406214, 133978, 375269, 138601, 340156, 172510, 144738, 376578, 91179, 250283, 316436, 84363, 286811, 258537, 140950, 240960, 346510, 413311, 283641, 416162, 84836, 276263, 381239, 31471, 10816, 239682, 139733, 356135, 239177, 412409]
color_list = df3.columns.tolist()
color_list.remove("ID")
df3["ID"] = center_list
df3.to_csv(fileC)
#df3.head()
df4["ID"] = center_list
df4.to_csv(fileD)
#df4.head()

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

dic_list_num_colors2 = dic_list_num_colors.copy()
for c in center_list:
    dic_list_num_colors[c] =[c]
    dic_list_num_colors2[c] =[c]
    for color in color_list:
        dic_list_num_colors2[c].append(len([j for j in dic_center_ball[c] if df1.Colors[j] ==color]))

        dic_list_num_colors[c].append(len([j for j in dic_center_ball[c] if df1.Colors[j] ==color])/dic_num_center_ball[c])

for c in center_list:
    df3.loc[center_list.index(c),:] = dic_list_num_colors[c]
    df4.loc[center_list.index(c), :] = dic_list_num_colors2[c]

df3.to_csv(fileC)
df3.head()
df4.to_csv(fileD)
df4.head()
