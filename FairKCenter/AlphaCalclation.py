import sys
import time
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt, colors
from math import sin, cos, sqrt, atan2, radians
from scipy.spatial import KDTree
import math
import CreateGraph
import matplotlib.pyplot as plt
import MaxMatching
fileA = "../DataSet/Listings.csv"
fileB = "../PropoData/Listings_radius_10.csv"

class FairKCenter:
    def __init__(self,req_dic,color_list,k):
        self.req_dic = req_dic  # A Dictionary that holds the constraints, the desired number of representatives from each entry
        self.col_list1 = [i for i in self.req_dic if self.req_dic[i] != 0]
        self.col_list1.append("ID")
        self.dic_id_loc = {}  # A dictionary that holds the location for each point
        #self.dic_id_dis = {}  # A dictionary that holds the list of distance from each point to key point
        self.dic_id_NR = {}  # A dictionary that holds the neighborhood radius for each point
        self.dic_close_center ={}
        self.dic_dis_to_close_center ={}

        self.dic_center_ball ={}  # A dictionary that holds the ball of each center point
        self.ball ={}
        self.dic_center_id_NR = {}  # A dictionary that holds the neighborhood radius for each center point
        self.dic_center_loc={}
        self.col_list = ["ID", "X", "Y","Z","Longitude","Latitude","Colors"]
        #self.df = pd.read_csv("zomato.csv", usecols=self.col_list)
        self.df = pd.read_csv(fileA, usecols=self.col_list)
        self.df1 = pd.read_csv(fileB,usecols=self.col_list1)
        self.df2 = pd.read_csv(fileB,usecols=["ID","NR_TYPE_ONE"])


        self.K = k #number of centers
        self.num_color ={}
        for i in color_list:
            self.num_color[i]=0

    def initialization_NR_dic(self,all_point, centers_points):
        start_time=time.time()

        non_zero_color=[i for i in self.req_dic if self.req_dic[i]!=0]

        for i in all_point:
            min = 10000
            for j in range(1,len(non_zero_color)+1):
                if self.df1.iat[i,j] <=min:
                    min=self.df1.iat[i,j]
            self.dic_id_NR[i] = max(self.df2.iat[i,1], min)
            self.dic_id_loc[i] = [self.df.X[i], self.df.Y[i], self.df.Z[i]]
            if i in centers_points:
                self.dic_center_id_NR[i] = self.dic_id_NR[i]
                self.dic_center_loc[i] = self.dic_id_loc[i]

        print('time to "initialization_NR_dic" end is {}'.format(time.time() - start_time))

        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)




    def update_dis_from_center(self):

        start_time = time.time()
        for p in list(self.df.ID):
            # arr_coordinates = np.array(self.dic_id_loc)
            tree = KDTree(np.array(list(self.dic_center_loc.values())))
            dist, ind = tree.query([[self.df.X[p], self.df.Y[p], self.df.Z[p]]], 1)
            self.dic_dis_to_close_center[p] = dist[0]
            self.dic_close_center[p] = np.array(list(self.dic_center_loc.keys()))[ind[0]]
        print('time to "update_dis_from_center" end is {}'.format(time.time() - start_time))

    def results2(self):

        start_time=time.time()
        # ##########
        # dic_alpha={}
        # for k,v in self.dic_id_NR.items():
        #     if self.dic_dis_to_close_center[k] == 0 and  self.dic_id_NR[k]==0:
        #         print("if")
        #         dic_alpha[k] = 1
        #     elif self.dic_id_NR[k] == 0:
        #         print("######################elif###########3")
        #         dic_alpha[k] = math.inf
        #     else:
        #         dic_alpha[k]=self.dic_dis_to_close_center[k]/ self.dic_id_NR[k]
        #
        # ##########
        dic_alpha = dict([(k,self.dic_dis_to_close_center[k]/ self.dic_id_NR[k]) for k, v in self.dic_id_NR.items()])
        max_key = max(dic_alpha, key=lambda x: dic_alpha[x])
        print("The point with maximum alpha is:{}".format(max_key))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_key],self.dic_id_NR[max_key]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_key],self.dic_dis_to_close_center[max_key]))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        print("#################")
        max_dis = max(self.dic_dis_to_close_center, key=self.dic_dis_to_close_center.get)
        print("The point with maximum distance from his nearest center is:{} in {}".format(max_dis, (
            self.df.Longitude[max_dis], self.df.Latitude[max_dis])))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_dis], self.dic_id_NR[max_dis]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_dis],
                                                                      self.dic_dis_to_close_center[max_dis]))

        print("#################")
        max_key = max(self.dic_id_NR, key=lambda x: self.dic_id_NR[x])
        print("The point with maximum NR is {} is NR is {}".format(max_key, self.dic_id_NR[max_key]))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_key], self.dic_id_NR[max_key]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_key],
                                                                      self.dic_dis_to_close_center[max_key]))

        print(self.dic_center_id_NR.keys())
        print('time to "results2" end is {}'.format(time.time() - start_time))

    def plot_point(self):
        x_c =list(self.df.Longitude[self.dic_center_id_NR.keys()])
        y_c = list(self.df.Latitude[self.dic_center_id_NR.keys()])
        c_c = list(self.df.Colors[self.dic_center_id_NR.keys()])
        size_c = [100]*len(x_c)
        x = list(self.df.Longitude)
        y = list(self.df.Latitude)
        c = list(self.df.Colors)
        s_p = [30]*len(x)

        plt.scatter(x, y, c=c, marker='.',s=s_p,cmap='viridis', edgecolors='gray')
        plt.scatter(x_c, y_c, c=c_c, marker='*',s= size_c,edgecolors='black')

        plt.show()

def main() :
    start_time = time.time()

    k = 10
    k_temp =k
    col_list = ["Colors"]
    df1 = pd.read_csv(fileA, usecols=col_list)


    req_dic = {'cyan': 0, 'green': 4, 'blue': 5, 'orange': 0, 'pink': 0, 'gray': 0, 'purple': 0, 'red': 1, 'yellow': 0}


    color_list = list(df1.Colors)#list(matplotlib.colors.cnames.keys())[0:num_type]
    list_centers =[13331,6757,4280,62368,13344,6580,37787,66793,63846,7827]
    fair = FairKCenter(req_dic,color_list,k)
    fair.initialization_NR_dic(list(fair.df.ID), list_centers)
    fair.update_dis_from_center()

    #fair.two_fair_k_center(0.01)


    fair.results2()
    print('time to end is {}'.format(time.time() - start_time))
    fair.plot_point()


if __name__ == '__main__':
    main()

