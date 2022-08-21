import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt, colors
from math import sin, cos, sqrt, atan2, radians

fileA = "../DataSet/Listings.csv"
fileB = "../DataSet/Listings_radius_10000.csv"
class FairKCenter:
    # global

    def __init__(self, k):
        self.dic_id_loc = {}  # A dictionary that holds the location for each point
        self.dic_id_NR = {}  # A dictionary that holds the neighborhood radius for each point
        self.dic_close_center = {}
        self.dic_dis_to_close_center = {}

        self.dic_center_ball = {}  # A dictionary that holds the ball of each center point
        self.dic_center_id_NR = {}  # A dictionary that holds the neighborhood radius for each center point
        self.dic_center_loc = {}
        self.col_list = ["ID", "X", "Y", "Z", "Longitude", "Latitude", "Colors"]
        self.df = pd.read_csv(fileA, usecols=self.col_list)
        self.df1 = pd.read_csv(fileB)
        self.K = k  # number of centers

    def initialization_NR_dic(self):
        start_time = time.time()
        self.dic_id_NR = dict(self.df1.NR_TYPE_ONE)
        self.dic_id_loc = list(np.array(list(zip(self.df.X, self.df.Y,self.df.Z))))

        print('time to "initialization_NR_dic" end is {}'.format(time.time() - start_time))

    def fair_k(self, alpha):
        balance = len(self.df.ID)/self.K
        start_time = time.time()
        #self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()
        temp_dic_id_NR2 = self.dic_id_NR.copy()

        list_NR_keys = list(temp_dic_id_NR.keys())
        while len(self.dic_center_id_NR.keys()) != self.K and len(temp_dic_id_NR.keys())!=0:
            #print(len(self.dic_center_id_NR.keys()))
            p = min(temp_dic_id_NR, key=temp_dic_id_NR.get)
            self.dic_center_id_NR[p] = self.dic_id_NR[p]
            self.dic_center_loc[p] = self.dic_id_loc[p]
            self.dic_center_ball[p] = list([])
            self.dic_dis_to_close_center[p] = 0
            temp_dic_id_NR.pop(p)

            for i in temp_dic_id_NR2:
                dis = distance.euclidean(self.dic_id_loc[i], self.dic_center_loc[p])
                if dis <= self.dic_id_NR[i] + self.dic_center_id_NR[p]:  # alpha*
                    if i in temp_dic_id_NR:
                       temp_dic_id_NR.pop(i)

            temp_dic_id_NR2 = temp_dic_id_NR.copy()
        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)

    def fair_k_2(self, alpha):
        balance = len(self.df.ID)/self.K
        start_time = time.time()
        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()

        while len(self.dic_center_id_NR.keys()) != self.K and len(temp_dic_id_NR.keys())!=0:
            #print(len(self.dic_center_id_NR.keys()))
            p = min(temp_dic_id_NR, key=temp_dic_id_NR.get)
            self.dic_center_id_NR[p] = self.dic_id_NR[p]
            self.dic_center_loc[p] = self.dic_id_loc[p]
            self.dic_center_ball[p] = list([])
            self.dic_dis_to_close_center[p] = 0
            temp_dic_id_NR.pop(p)

            ##########
            list_t = list(temp_dic_id_NR.keys())
            tuple_color = tuple(zip(list(self.df.X[list_t]), list(self.df.Y[list_t]),list(self.df.Z[list_t])))
            tree_c = KDTree(np.array(list(tuple_color)))
            n = len(list_t)
            dist_c, ind_c = tree_c.query([[self.df.X[p], self.df.Y[p],self.df.Z[p]]], n)
            for dis,i in zip(dist_c[0],ind_c[0]):
                if dis <= self.dic_id_NR[list_t[i]] +self.dic_id_NR[p]:
                    temp_dic_id_NR.pop(list_t[i])



            #########

        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)

    def two_fair_k_center(self, alpha):
        """
         The main algorithm
        :return: set of centers point
                  """
        start_time = time.time()

        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()
        list_NR_keys = list(temp_dic_id_NR.keys())
        for p in list(self.df.ID):
            self.dic_dis_to_close_center[
                p] = sys.maxsize  # Initialize each point at an infinite distance from the center
        while len(list_NR_keys) != 0:
            p = list_NR_keys[0]
            signal = 0  # 0 if p can be center point and 1 otherwise
            for s in self.dic_center_id_NR.keys():
                dis = distance.euclidean(self.dic_id_loc[s], self.dic_id_loc[p])
                if dis <= self.dic_id_NR[s] + self.dic_id_NR[p]:  # alpha*

                    list_NR_keys.remove(p)
                    signal = 1

                    break
            if signal == 0:  # update P to be new center
                self.dic_center_id_NR[p] = self.dic_id_NR[p]
                self.dic_center_loc[p] = self.dic_id_loc[p]
                self.dic_center_ball[p] = list([])
                self.dic_dis_to_close_center[p] = 0
                self.dic_close_center[p] = p
                list_NR_keys.remove(p)
        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)

    def update_dis_from_center(self):
        start_time = time.time()

        for p in list(self.df.ID):
            # arr_coordinates = np.array(self.dic_id_loc)
            tree = KDTree(np.array(list(self.dic_center_loc.values())))
            dist, ind = tree.query([[self.df.X[p], self.df.Y[p], self.df.Z[p]]], 1)
            self.dic_dis_to_close_center[p] = dist[0][0]
            self.dic_close_center[p] = np.array(list(self.dic_center_loc.keys()))[ind[0][0]]
        print('time to "update_dis_from_center" end is {}'.format(time.time() - start_time))

    def results2(self):
        self.update_dis_from_center()
        start_time = time.time()

        dic_alpha = dict([(k, self.dic_dis_to_close_center[k] / self.dic_id_NR[k]) for k, v in self.dic_id_NR.items()])
        max_key = max(dic_alpha, key=lambda x: dic_alpha[x])
        print("The point with maximum alpha is:{} in {}".format(max_key, (
        self.df.Longitude[max_key], self.df.Latitude[max_key])))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_key], self.dic_id_NR[max_key]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_key],
                                                                      self.dic_dis_to_close_center[max_key]))
        print("#################")
        max_dis = max(self.dic_dis_to_close_center, key=self.dic_dis_to_close_center.get)
        print("The point with maximum distance from his nearest center is:{} in {}".format(max_dis, (
        self.df.Longitude[max_dis], self.df.Latitude[max_dis])))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_dis], self.dic_id_NR[max_dis]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_dis],
                                                                      self.dic_dis_to_close_center[max_dis]))
        print(self.dic_center_id_NR.keys())

        print('time to "results2" end is {}'.format(time.time() - start_time))

    def save_group(self):
        df1 = pd.read_csv("virginia_new.csv", usecols=["Cuisines_Colors"])
        type_list = list(df1.Cuisines_Colors)
        req_dic = list(set(type_list))
        dic_colors = {}
        for c in req_dic:
            dic_colors[c] = list([])
        count = 0
        for i in self.dic_center_ball:
            for c in req_dic:
                dic_colors[c].append(0)
            for j in self.dic_center_ball[i]:
                dic_colors[self.df.Cuisines_Colors[j]][count] += 1
            count += 1
        print("debug")
        self.df4 = pd.read_csv("group.csv")
        for c in dic_colors:
            self.df4[str(c)] = dic_colors[c]
            self.df4.to_csv("group.csv")
            self.df4.head()

        print("debug")

    def plot_point(self):
        x_c = list(self.df.Longitude[self.dic_center_id_NR.keys()])
        y_c = list(self.df.Latitude[self.dic_center_id_NR.keys()])
        c_c = list(self.df.Colors[self.dic_center_id_NR.keys()])
        size_c = [100] * len(x_c)
        x = list(self.df.Longitude)
        y = list(self.df.Latitude)
        c = list(self.df.Colors)
        s_p = [50] * len(x)

        plt.scatter(x, y, c=c, marker='.', s=s_p, cmap='viridis', edgecolors='gray')
        plt.scatter(x_c, y_c, c=c_c, marker='*', s=size_c, edgecolors='black')
        plt.show()


def main():
    start_time = time.time()

    fair = FairKCenter(10000)
    fair.initialization_NR_dic()
    fair.fair_k_2(1.399)
    #fair.two_fair_k_center(1.399)  # 1.359
    # fair.save_group()
    fair.results2()
    print('time to end is {}'.format(time.time() - start_time))

    fair.plot_point()

    # low =1
    # high=2
    # while low <= high:
    #     mid = (low+high)/2
    #     print("alpha(mid) ={}".format(mid))
    #     fair = FairKCenter(100)
    #     fair.initialization_NR_dic(list(fair.df.ID))
    #     if fair.two_fair_k_center(mid) < fair.K:
    #         high = mid
    #     elif fair.two_fair_k_center(mid) == fair.K:
    #         fair.results2()
    #         print(fair.dic_center_id_NR.keys())
    #         print(len(fair.dic_center_id_NR.keys()))
    #         fair.plot_point()
    #         break
    #     else:
    #         low =mid


if __name__ == '__main__':
    main()
