import pandas as pd
from scipy.optimize import linear_sum_assignment

fileA ="../DataSet/Banks.csv"
fileB ="../LongTimeData/Banks_group_500_ran_for_div.csv"


df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)


list_nodes_community = list(df2["ID"])
list_nodes_div_property = list(set(df1["service"]))
list_edges =[]
dic_points_to_property = dict(zip(df1["ID"],df1["service"]))

# After
print("after")
for i in df2["ID"]:
    for l in [int(num) for num in df2.Ball[list_nodes_community.index(i)].strip('{}').split(',')]:
        list_edges.append((i,dic_points_to_property[l]))


list_edges = list(set(list_edges))
# Create a matrix where the value at position (i, j) represents the distance between node i on the left side and node j on the right side
distance_matrix = [[0] * len(list_nodes_div_property) for _ in range(len(list_nodes_community))]
for row, column in list_edges:
    distance_matrix[list_nodes_community.index(row)][list_nodes_div_property.index(column)] = 1  # Assuming a distance of 1 between all nodes

# Use the Hungarian algorithm to find the maximum matching of nodes
rows, columns = linear_sum_assignment(distance_matrix)

# The maximum matching nodes in the graph
maximum_matching = [(list_nodes_community[row], list_nodes_div_property[column]) for row, column in zip(rows, columns)]

# print("The maximum matching nodes are:")
# for pair in maximum_matching:list_b
#     print(pair)
print("The maximum matching nodes are: {}".format(len(maximum_matching)))


# before
print("before")
list_edges_b =[]
list_matching_b =[]
for i in df2["ID"]:
        list_edges_b.append((i,dic_points_to_property[i]))
        list_matching_b.append(dic_points_to_property[i])
list_edges_b = list(set(list_edges_b))
list_matching_b= list(set(list_matching_b))
# Create a matrix where the value at position (i, j) represents the distance between node i on the left side and node j on the right side


print("The maximum matching nodes are: {}".format(len(list_matching_b)))