import Hungarian
import Max_Min
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

file1 ="../DataSet/Banks.csv"
file2 ="../DataSet/Banks_group_500_ran.csv"

def compute_similarity_before_change(df2,dic_center_and_colors):
    list_centers = list(df2.ID)
    list_colors = list(pd.read_csv(file2, nrows=0).columns.tolist())
    list_colors.remove("ID")
    matrix = np.delete(np.array(df2), 0, 1)
    list_before_change =[]
    for i in list_centers:
        color = dic_center_and_colors[i]
        list_before_change.append(matrix[list_centers.index(i),list_colors.index(color)])
    return list_before_change

def replace_center(df,dic_center_and_colors):
    list_new_centers ={}
    for k,v in dic_center_and_colors.items():
        #print(k)
        all_point = list([i for i in df.ID if df.Colors[i] ==v and i not in list_new_centers.values()])
        tuple_color = tuple(zip(list(df1.X[all_point]), list(df1.Y[all_point]), list(df1.Z[all_point])))
        tree_c = KDTree(np.array(list(tuple_color)))
        dist_c, ind_c = tree_c.query([[df.X[k], df.Y[k], df.Z[k]]], 1)
        list_new_centers[k]=all_point[ind_c[0][0]]
    return list_new_centers

def plot_result(list_before,list_hung1,list_hung2):
    labels = [i for i in range(len(list_before))]
    x = np.arange(len(list_before))  # the label locations
      # the width of the bars
    fig, ax = plt.subplots()
    width=0.3
    rects1 = ax.bar(x + 0.00, list_hung2, width, label='Hungarian2(MaxMin)', color ='r')
    rects2 = ax.bar(x + 0.3, list_hung1, width, label='Hungarian(MaxSum)',color ='b')
    rects2 = ax.bar(x + 0.6 , list_before, width, label='before change', color='g')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('similarity')
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()

    label = ['Hungarian2(MaxMin)', 'Hungarian(MaxSum)', 'before change']
    fig, ax = plt.subplots()
    x_pos1 =np.arange(3)
    list_sum = [sum(list_hung2),sum(list_hung1),sum(list_before)]
    plt.bar(x_pos1, list_sum ,color=['red', 'blue', 'green'])
    plt.xticks(x_pos1, label)
    ax.set_title('MaxSum similarity')

    fig, ax = plt.subplots()
    x_pos2 = np.arange(3)
    list_min = [min(list_hung2), min(list_hung1), min(list_before)]
    plt.bar(x_pos2, list_min, color=['red', 'blue', 'green'])
    plt.xticks(x_pos2, label)
    ax.set_title('MaxMin similarity')

    fig, ax = plt.subplots()
    x_pos3 = np.arange(3)
    list_mean = [mean(list_hung2), mean(list_hung1), mean(list_before)]
    plt.bar(x_pos3, list_mean, color=['red', 'blue', 'green'])
    plt.xticks(x_pos3, label)
    ax.set_title('average similarity')
    plt.show()

# requirements extraction
df2 = pd.read_csv(file2)
df1 = pd.read_csv(file1)
color_list = pd.read_csv(file2, nrows=0).columns.tolist()
color_list.remove("ID")
req_dic = {}
dic_center_and_colors = {}
for i in color_list:
    req_dic[i] = 0
for i in df2.ID:
    dic_center_and_colors[i] = df1.Colors[i]
    c = df1.Colors[i]
    req_dic[c] = req_dic[c] + 1


req_dic2 = req_dic.copy()

print(req_dic2)

# Before change
no_change = compute_similarity_before_change(df2,dic_center_and_colors)
max_min_no_change = min(no_change)
max_sum_no_change = sum(no_change)
avg_no_change = mean(no_change)
print("->>>before running algorithms<<<-")
print("no_change:{}".format(no_change))

print("center:{} ".format(list(df2["ID"])))
print("max_min_no_change:{}".format(max_min_no_change))
print("max_sum_no_change:{}".format(max_sum_no_change))
print("avg_no_change:{}".format(avg_no_change))

# Running Hungarian algorithm
print("->>>result of Hungarian (MaxSum) algorithms<<<-")

H = Hungarian.MaxSum(req_dic,file1,file2)
H.convert()
max_sum_list , dic_center_and_colors2=H.play_hungarian_algo()
new_centers_list = replace_center(df1,dic_center_and_colors2)
print("new center:{}".format(new_centers_list))
max_min_hung = min(max_sum_list)
max_sum_hung = sum(max_sum_list)
avg_hung = mean(max_sum_list)
print("max_sum_list:{}".format(max_sum_list))
print("max_min_hung:{}".format(max_min_hung))
print("max_sum_hung:{}".format(max_sum_hung))
print("avg_hung:{}".format(avg_hung))

# Running MaxMin algorithm
print("->>>result of Hungarian2 (MaxMin) algorithms<<<-")

M = Max_Min.MaxMin(req_dic2, dic_center_and_colors,file1,file2)
M.sort_indexs_matrix()
M.main_algo()
dic_result =M.get_result_perfect()
max_min_list=M.get_result2(dic_result)
max_min_hung2 = min(max_min_list)
max_sum_hung2 = sum(max_min_list)
avg_hung2 = mean(max_min_list)
print("max_min_list:{}".format(max_min_list))
print("max_min_hung2:{}".format(max_min_hung2))
print("max_sum_hung2:{}".format(max_sum_hung2))
print("avg_hung2:{}".format(avg_hung2))

plot_result(no_change,max_sum_list,max_min_list)