from sklearn.neighbors import KDTree
import pandas as pd
import math
import numpy as np
import matplotlib

file_name ="../DataSet/Banks.csv"
col_name = "service"
df = pd.read_csv(file_name)
print(set(list(df.Colors)))
org_type_list =df[col_name]
org_type_list = list(set(org_type_list))
print(org_type_list)
print("org_type_list len ={}".format(len(org_type_list)))


num_type = len(org_type_list)
color_list = list(matplotlib.colors.cnames.keys())[0:num_type]
print(color_list)
new_color_list=[]
for t in df[col_name]:
     new_color_list.append(color_list[org_type_list.index(t)])

df["Colors2"] = new_color_list
df.to_csv(file_name)
df.head()
