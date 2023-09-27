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


fileA ="../DataSet/Banks.csv"
fileB="../DataSet/Banks_radius_500.csv"
fileC ="../LongTimeData/500.csv"
fileD ="../LongTimeData/500_num.csv"

fileE ="../DataSet/Banks_group_500_ran.csv"
fileF ="../DataSet/Banks_group_500_ran_num.csv"


df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)
df3 = pd.read_csv(fileC)
df4 = pd.read_csv(fileD)


df5 = pd.read_csv(fileE)
center_list = list(df5["ID"])
#center_list=[20406, 22701, 7730, 22883, 18516, 19648, 20338, 21343, 19041, 5211, 5668, 20490, 18324, 19844, 21427, 5797, 27005, 20051, 22834, 31709, 14475, 18294, 4838, 30302, 31607, 11775, 20442, 29169, 8175, 18315, 18429, 19755, 5434, 18867, 8081, 4361, 30998, 21070, 26909, 7457, 22600, 2886, 10489, 18479, 22645, 18165, 10253, 27878, 32727, 1484, 6379, 32742, 12277, 6658, 15166, 32987, 4597, 9634, 19962, 21865, 8436, 13096, 11728, 18697, 30083, 14635, 9116, 10413, 28588, 12382, 13844, 18881, 29646, 4496, 11723, 24120, 4430, 1993, 6478, 31410, 21731, 3856, 18389, 22676, 29135, 12526, 4678, 20545, 18644, 10538, 5148, 3063, 30285, 29728, 6209, 10525, 6107, 8957, 4313, 27497, 6459, 4309, 26379, 1743, 4970, 30746, 19051, 14318, 12977, 23815, 15561, 23058, 9311, 21634, 3839, 4818, 32311, 7370, 10535, 29362, 4737, 29414, 3915, 4432, 19226, 11219, 19358, 12290, 18295, 6253, 9705, 6273, 30994, 4515, 8051, 28267, 5717, 10333, 18903, 3053, 28871, 11362, 11673, 5681, 28573, 26426, 9995, 24486, 7073, 12859, 28039, 16045, 14304, 17768, 15032, 4007, 2539, 13453, 6778, 27058, 28937, 21146, 15450, 32982, 10042, 12223, 11692, 12460, 20623, 31673, 25328, 4977, 28925, 28582, 12461, 4329, 5493, 28983, 15718, 491, 19342, 31757, 10537, 3780, 5394, 4251, 12493, 16378, 12310, 4828, 8571, 731, 12623, 17302, 12566, 18642, 30762, 29405, 3582, 28994, 3695, 27573, 27441, 14027, 4871, 23778, 29808, 9799, 23198, 2315, 5080, 20896, 15005, 15964, 9062, 19624, 7411, 9727, 2882, 29942, 374, 18420, 9009, 27219, 9352, 24090, 24678, 18744, 10995, 21215, 9342, 5928, 10242, 8859, 6926, 30159, 12673, 15213, 9813, 27601, 15778, 26524, 6586, 5779, 19310, 12564, 23916, 8613, 13457, 26852, 29767, 4867, 6208, 5235, 12922, 15723, 3648, 15301, 13557, 12107, 28206, 4521, 10727, 30885, 15143, 27067, 23013, 24137, 18214, 5938, 8577, 25332, 15345, 31008, 27558, 26736, 16526, 16555, 9880, 35, 25720, 12135, 11266, 15855, 19231, 4574, 26495, 9803, 1726, 10851, 25999, 8244, 24239, 16318, 18017, 7597, 29611, 24354, 13972, 30602, 13946, 17177, 3401, 19228, 25706, 11136, 420, 27248, 14114, 19413, 22551, 9881, 8909, 27302, 10725, 25714, 26548, 1841, 29129, 14518, 16342, 3601, 4183, 3225, 5541, 10584, 26812, 30671, 18733, 21154, 23388, 1129, 6261, 16816, 29107, 27317, 14954, 16536, 1557, 9818, 856, 12267, 24879, 32188, 20258, 25526, 23253, 19568, 771, 25689, 15589, 31848, 32226, 19210, 23357, 24167, 21940, 11349, 29833, 24979, 16057, 345, 3985, 5789, 2622, 31379, 25827, 6412, 19414, 17627, 27791, 1790, 17161, 11450, 11547, 32914, 2199, 1358, 23181, 529, 22074, 25036, 28842, 6469, 30009, 15579, 15416, 4804, 24998, 7246, 21903, 1211, 16496, 9768, 28105, 1218, 26106, 26356, 1428, 14508, 21022, 17634, 16612, 24156, 1159, 22301, 24430, 32422, 2250, 2415, 1238, 7669, 23214, 1941, 24679, 24479, 1397, 24564, 1224, 17538, 22124, 7298, 26348, 26135, 17691, 31561, 32359, 28986, 17959, 16480, 21538, 536, 2826, 2311, 17803, 23894, 1170, 23347, 1469, 16153, 20965, 17587, 21048, 30310, 17564, 32682, 2724, 18039, 24352, 16558, 31099, 31650, 7739, 16473, 21014, 27155, 22416, 32146, 2788, 17986, 23519, 16374, 16881, 11834, 25584, 852, 7833, 15913, 16865, 13772, 1539, 2559, 32736, 31899, 16208, 2538, 32512, 18033, 31648, 22842, 16738, 26306, 1965, 30381, 24269, 20266, 32294, 29858, 909, 32595, 29290, 26960, 21011, 7700, 29329, 32011, 8914, 31813]

color_list = df3.columns.tolist()
color_list.remove("ID")
df3["ID"] = center_list
#df3.to_csv(fileC)//
#df3.head()
#df4.to_csv(fileD)//
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
    if(i not in center_list):
       dic_center_ball[list_t[ind_c[0][0]]].append(i)
    else:
       k=list_t.index(i)
       dic_center_ball[i].append(i)

# for key in dic_center_ball:
#     print(key, 'corresponds to', dic_center_ball[key])


# Hngarian algo
dff2 = pd.read_csv(fileE)
dff1 = pd.read_csv(fileA)
color_list = pd.read_csv(fileE, nrows=0).columns.tolist()
color_list.remove("ID")
req_dic = {}
dic_center_and_colors = {}
for i in color_list:
    req_dic[i] = 0
for i in dff2.ID:
    dic_center_and_colors[i] = dff1.Colors[i]
    c = dff1.Colors[i]
    req_dic[c] = req_dic[c] + 1


print("req_dic:{}".format(req_dic))

for key in dic_center_ball:
    print(key, 'corresponds to', dic_center_ball[key])
H = Hungarian.MaxSum(req_dic,fileA,fileE)
H.convert()
max_sum_list , dic_center_and_colors2=H.play_hungarian_algo()


# create new balls
print("---> new center ball <---")
# for key in dic_center_ball:
#     print(key, 'corresponds to', dic_center_ball[key])
dic_new_center_ball ={}
for i in dic_center_ball.keys():

    dic_new_center_ball[i] = [x for x in dic_center_ball[i] if df1.Colors[x] == dic_center_and_colors2[i]]

new_rec_colors = list(dic_center_and_colors2.values())

#
# for key in dic_center_ball:
#     print(key, 'corresponds to', dic_new_center_ball[key])
#for key in dic_center_and_colors2:
 #   print(key, 'corresponds to', dic_center_and_colors2[key])
new_dic_center_ball ={}
for i in dic_center_ball:
 if len(dic_center_ball[i]) !=0:
     new_dic_center_ball[i]=dic_center_ball[i]


print("--->########<---")

print(dic_center_and_colors2)

print(max_sum_list)

# new_list_centers = list(new_dic_center_ball.keys())
# df4["ID"] = new_list_centers
#
# for i in new_dic_center_ball:
#     df4["Ball"][new_list_centers.index(i)] = set(new_dic_center_ball[i])
#
#
# df4.to_csv(fileD)
# df4.head()